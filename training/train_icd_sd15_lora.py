import argparse
import functools
import logging
import os
import random
import shutil
from pathlib import Path

import accelerate
import diffusers
import numpy as np
import torch
import torch.distributed as dist
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from safetensors.torch import load_file
import wandb
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from packaging import version
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import lpips
from src.datasets import get_coco_loader
from src.fid_score_in_memory import calculate_fid
from src.forward_eval import eval_inversion, log_validation_inversion, log_train_inversion
from src.lcm import DDIMSolver
from src.reverse_eval import distributed_sampling, log_validation
from src.train import (
    forward_preserve_train_step,
    forward_train_step,
    reverse_preserve_train_step,
    reverse_train_step,
    forward_alignment_step,
)
from src.utils import recover_resume_step

logger = get_logger(__name__)

def load_model(model, checkpoint_path):
    lora_state_dict = load_file(checkpoint_path)
    new_state_dict = {}
    for k in lora_state_dict.keys():
        new_key = k[5:]
        new_key = new_key.split('.')
        new_key = new_key[:-1] + ['default'] + new_key[-1:]
        new_state_dict['.'.join(new_key)] = lora_state_dict[k]
    # print(type(lora_state_dict))
    res = model.load_state_dict(new_state_dict, strict=False)
    # assert len([k for k in res.missing_keys if 'lora' in k]) == 0
    # assert len(res.unexpected_keys) == 0
    print("Missing keys:", [k for k in res.missing_keys if 'lora' in k])

def get_module_kohya_state_dict_teacher(
    module, prefix: str, dtype: torch.dtype, adapter_name: str = "default"
):
    kohya_ss_state_dict = {}
    for peft_key, weight in module.items():
        kohya_key = peft_key.replace("unet.base_model.model", prefix)
        kohya_key = kohya_key.replace("lora_A", "lora_down")
        kohya_key = kohya_key.replace("lora_B", "lora_up")
        kohya_key = kohya_key.replace(".", "_", kohya_key.count(".") - 2)
        kohya_ss_state_dict[kohya_key] = weight.to(dtype)
        # Set alpha parameter
        if "lora_down" in kohya_key:
            alpha_key = f'{kohya_key.split(".")[0]}.alpha'
            kohya_ss_state_dict[alpha_key] = torch.tensor(8).to(dtype)

    return kohya_ss_state_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    # ----------Model Checkpoint Loading Arguments----------
    parser.add_argument(
        "--pretrained_teacher_model",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained LDM teacher model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--teacher_revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained LDM teacher model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained LDM model identifier from huggingface.co/models.",
    )
    # ----------Training Arguments----------
    # ----General Training Arguments----
    parser.add_argument(
        "--output_dir",
        type=str,
        default="lcm-xl-distilled",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    # ----Logging----
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    # ----Checkpointing----
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    # ----Image Processing----
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    # ----Batch Size and Training Steps----
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.",
    )
    # ----Learning Rate----
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    # ----Optimizer (Adam)----
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    # ----Latent Consistency Distillation (LCD) Specific Arguments----
    parser.add_argument(
        "--w_min",
        type=float,
        default=5.0,
        required=False,
        help=(
            "The minimum guidance scale value for guidance scale sampling. Note that we are using the Imagen CFG"
            " formulation rather than the LCM formulation, which means all guidance scales have 1 added to them as"
            " compared to the original paper."
        ),
    )
    parser.add_argument(
        "--w_max",
        type=float,
        default=15.0,
        required=False,
        help=(
            "The maximum guidance scale value for guidance scale sampling. Note that we are using the Imagen CFG"
            " formulation rather than the LCM formulation, which means all guidance scales have 1 added to them as"
            " compared to the original paper."
        ),
    )
    parser.add_argument(
        "--forward_w_min",
        type=float,
        default=0.0,
        required=False,
        help=(
            "The minimum guidance scale value for guidance scale sampling. Note that we are using the Imagen CFG"
            " formulation rather than the LCM formulation, which means all guidance scales have 1 added to them as"
            " compared to the original paper."
        ),
    )
    parser.add_argument(
        "--forward_w_max",
        type=float,
        default=0.0,
        required=False,
        help=(
            "The maximum guidance scale value for guidance scale sampling. Note that we are using the Imagen CFG"
            " formulation rather than the LCM formulation, which means all guidance scales have 1 added to them as"
            " compared to the original paper."
        ),
    )
    parser.add_argument(
        "--num_ddim_timesteps",
        type=int,
        default=50,
        help="The number of timesteps to use for DDIM sampling.",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="l2",
        choices=["l2", "huber", 'lpips'],
        help="The type of loss to use for the LCD loss.",
    )
    
    parser.add_argument(
        "--loss_type_without_guidance",
        type=str,
        default="l2",
        choices=["l2", "huber", 'lpips'],
        help="The type of loss to use for the LCD loss.",
    )
    
    parser.add_argument(
        "--huber_c",
        type=float,
        default=0.001,
        help="The huber loss parameter. Only used if `--loss_type=huber`.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=64,
        help="The rank of the LoRA projection matrix.",
    )
    # ----Mixed Precision----
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--cast_teacher_unet",
        action="store_true",
        help="Whether to cast the teacher U-Net to the precision specified by `--mixed_precision`.",
    )
    # ----Training Optimizations----
    parser.add_argument("--single_t_per_batch", action="store_true")
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    # ----Distributed Training----
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    # ----------Validation Arguments----------
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=200,
        help="Run validation every X steps.",
    )
    parser.add_argument(
        "--evaluation_steps",
        type=int,
        default=1000,
        help="Run score calculation every X steps.",
    )
    # ----------Accelerate Arguments----------
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="lcm",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--val_prompt_path",
        type=str,
        default="./training/data/val2014.csv",
    )
    parser.add_argument(
        "--inception_path",
        type=str,
        default="./training/stats/pt_inception-2015-12-05-6726825d.pth",
    )
    parser.add_argument(
        "--coco_ref_stats_path",
        type=str,
        default="./training/stats/fid_stats_mscoco256_val.npz",
    )
    parser.add_argument(
        "--coco_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--num_endpoints",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--num_forward_endpoints",
        type=int,
        default=1,
    )
    parser.add_argument("--no_forward", action="store_true")
    parser.add_argument(
        "--start_forward_timestep",
        type=int,
        default=19,
        help="Learn forward iCD to start encoding from a slightly noise image"
    )
    parser.add_argument(
        "--preserve_loss_coef",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--cfg_distill_teacher_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--forward_model_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--backward_model_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--reverse_preserve_loss_coef",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--forward_preserve_loss_coef",
        type=float,
        default=0.0,
    )  
    parser.add_argument(
        "--cycle_loss_c",
        type=float,
        default=1.0,
    )
    parser.add_argument("--embed_guidance", action="store_true")
    parser.add_argument("--discrete_w", type=str, default=None)
    parser.add_argument("--endpoints", type=str, default=None)
    parser.add_argument("--forward_endpoints", type=str, default=None)

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


# Adapted from pipelines.StableDiffusionPipeline.encode_prompt
def encode_prompt(
    prompt_batch, text_encoder, tokenizer, proportion_empty_prompts, is_train=True
):
    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        text_inputs = tokenizer(
            captions,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(text_input_ids.to(text_encoder.device))[0]

    return prompt_embeds


class LPIPS_Loss():
    
    def __init__(self, lpips_loss_fn, args):
        self.lpips_loss_fn = lpips_loss_fn
        self.coefficient = args.cycle_loss_c
    
    def __call__(self, reconstructed_images, true_images):
        reconstructed_patches = reconstructed_images.unfold(2, 224, 110).unfold(3, 224, 110)
        reconstructed_patches = reconstructed_patches.permute(0, 2, 4, 1, 3, 5).contiguous()
        reconstructed_patches = reconstructed_patches.view(reconstructed_images.shape[0], -1, 3, 224, 224)  # (B, N_patches, C, H_patch, W_patch)
        
        
        true_patches = true_images.unfold(2, 224, 110).unfold(3, 224, 110)
        true_patches = true_patches.permute(0, 2, 4, 1, 3, 5).contiguous()
        true_patches = true_patches.view(true_images.shape[0], -1, 3, 224, 224)
        
        true_patches = true_patches.view(-1, 3, 224, 224)  # (B*N_patches, C, H, W)
        reconstructed_patches = reconstructed_patches.view(-1, 3, 224, 224)
        
        lpips_dist = self.lpips_loss_fn(reconstructed_patches, true_patches)
        
        return self.coefficient * lpips_dist.view(true_images.shape[0], 9).mean()
    
    
    

def main(args):
    assert (
        args.gradient_accumulation_steps == 1
    ), "gradient_accumulation_steps is not supported at the moment"

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        split_batches=True,  # It's important to set this to True when using webdataset to get the right number of steps for lr scheduling. If set to False, the number of steps will be devide by the number of processes assuming batches are multiplied by the number of processes
    )

    forward_accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        split_batches=True,  # It's important to set this to True when using webdataset to get the right number of steps for lr scheduling. If set to False, the number of steps will be devide by the number of processes assuming batches are multiplied by the number of processes
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed + dist.get_rank())

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # 1. Create the noise scheduler and the desired noise schedule.
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_teacher_model,
        subfolder="scheduler",
        revision=args.teacher_revision,
    )

    # The scheduler calculates the alpha and sigma schedule for us
    alpha_schedule = torch.sqrt(noise_scheduler.alphas_cumprod)
    sigma_schedule = torch.sqrt(1 - noise_scheduler.alphas_cumprod)
    solver = DDIMSolver(
        noise_scheduler.alphas_cumprod.numpy(),
        timesteps=noise_scheduler.config.num_train_timesteps,
        ddim_timesteps=args.num_ddim_timesteps,
        num_endpoints=args.num_endpoints,
        num_forward_endpoints=args.num_forward_endpoints,
    )

    # 2. Load tokenizers from SD-XL checkpoint.
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_teacher_model,
        subfolder="tokenizer",
        revision=args.teacher_revision,
        use_fast=False,
    )

    # 3. Load text encoders from SD-1.5 checkpoint.
    text_encoder = StableDiffusionPipeline.from_pretrained(
        args.pretrained_teacher_model,
        torch_dtype=torch.float16,
        variant="fp16",
    ).text_encoder

    # 4. Load VAE from SD-XL checkpoint (or more stable VAE)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_teacher_model,
        subfolder="vae",
        revision=args.teacher_revision,
    )

    # 5. Create student and teacher unets.
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_teacher_model,
        subfolder="unet",
        revision=args.teacher_revision,
        time_cond_proj_dim=512 if args.embed_guidance else None,
        low_cpu_mem_usage=False,
        device_map=None,
    )
    forward_unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_teacher_model,
        subfolder="unet",
        revision=args.teacher_revision,
        time_cond_proj_dim=512 if args.embed_guidance else None,
        low_cpu_mem_usage=False,
        device_map=None,
    )
    teacher_unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_teacher_model,
        subfolder="unet",
        revision=args.teacher_revision,
        time_cond_proj_dim=512 if args.embed_guidance else None,
        low_cpu_mem_usage=False,
        device_map=None,
        torch_dtype=torch.float16,
        variant="fp16",
    )

    if args.cfg_distill_teacher_path:
        logger.info(f"Loading teacher from {args.cfg_distill_teacher_path}")
        state_dict = torch.load(args.cfg_distill_teacher_path, map_location="cpu")

        teacher_unet.load_state_dict(state_dict)
        unet.load_state_dict(state_dict)
        forward_unet.load_state_dict(state_dict)

    unet.train()
    forward_unet.train()

    # 6. Freeze teacher vae, text_encoder, and teacher_unet
    unet.requires_grad_(False)
    forward_unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    teacher_unet.requires_grad_(False)

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(unet).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}"
        )

    # 8. Add LoRA to the student U-Net, only the LoRA projection matrix will be updated by the optimizer.
    lora_modules = [
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
        "proj_in",
        "proj_out",
        "ff.net.0.proj",
        "ff.net.2",
        "conv1",
        "conv2",
        "conv_shortcut",
        "downsamplers.0.conv",
        "upsamplers.0.conv",
        "time_emb_proj",
    ]
    lora_config = LoraConfig(r=args.lora_rank, target_modules=lora_modules)

    forward_unet = get_peft_model(forward_unet, lora_config)
    unet = get_peft_model(unet, lora_config)

    

    load_model(forward_unet, args.forward_model_path)
    load_model(unet, args.backward_model_path)


    # 9. Handle mixed precision and device placement
    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    vae.to(accelerator.device)
    if args.pretrained_vae_model_name_or_path is not None:
        vae.to(dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # Move teacher_unet to device, optionally cast to weight_dtype
    teacher_unet.to(accelerator.device)
    if args.cast_teacher_unet:
        teacher_unet.to(dtype=weight_dtype)

    # Also move the alpha and sigma noise schedules to accelerator.device.
    alpha_schedule = alpha_schedule.to(accelerator.device)
    sigma_schedule = sigma_schedule.to(accelerator.device)
    solver = solver.to(accelerator.device)

    # 10. Handle saving and loading of checkpoints
    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                unet_ = accelerator.unwrap_model(unet)
                lora_state_dict = get_peft_model_state_dict(
                    unet_, adapter_name="default"
                )
                StableDiffusionPipeline.save_lora_weights(
                    os.path.join(output_dir, "unet_lora"), lora_state_dict
                )
                unet_.save_pretrained(output_dir)

                # save weights in peft format to be able to load them back
                unet_.save_pretrained(output_dir)

                for _, model in enumerate(models):
                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            # load the LoRA into the model
            unet_ = accelerator.unwrap_model(unet)
            unet_.load_adapter(input_dir, "default", is_trainable=True)

            for _ in range(len(models)):
                # pop models so that they are not loaded again
                models.pop()

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

        def save_model_hook(models, weights, output_dir):
            if forward_accelerator.is_main_process:
                unet_ = forward_accelerator.unwrap_model(forward_unet)
                lora_state_dict = get_peft_model_state_dict(
                    unet_, adapter_name="default"
                )
                StableDiffusionPipeline.save_lora_weights(
                    os.path.join(output_dir, "unet_lora"), lora_state_dict
                )
                # save weights in peft format to be able to load them back
                unet_.save_pretrained(output_dir)

                for _, model in enumerate(models):
                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            # load the LoRA into the model
            unet_ = forward_accelerator.unwrap_model(forward_unet)
            unet_.load_adapter(input_dir, "default", is_trainable=True)

            for _ in range(len(models)):
                # pop models so that they are not loaded again
                models.pop()

        forward_accelerator.register_save_state_pre_hook(save_model_hook)
        forward_accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        forward_unet.enable_gradient_checkpointing()

    # 12. Optimizer creation
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    forward_optimizer = torch.optim.AdamW(
        forward_unet.parameters(),#
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    global_batch_size = args.train_batch_size * accelerator.num_processes
    current_step = recover_resume_step(args.output_dir)
    logger.info(f"Resume the training from {global_batch_size * current_step}")

    # Prepare COCO dataset loader
    train_dataloader = get_coco_loader(
        args, batch_size=args.train_batch_size, is_train=True
    )

    # Here, we compute not just the text embeddings but also the additional embeddings
    # needed for the SD XL UNet to operate.
    def compute_embeddings(
        prompt_batch, proportion_empty_prompts, text_encoder, tokenizer, is_train=True
    ):
        prompt_embeds = encode_prompt(
            prompt_batch, text_encoder, tokenizer, proportion_empty_prompts, is_train
        )
        return {"prompt_embeds": prompt_embeds}

    compute_embeddings_fn = functools.partial(
        compute_embeddings,
        proportion_empty_prompts=0,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
    )

    # Scheduler and math around the number of training steps.
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    forward_lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=forward_optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, lr_scheduler = accelerator.prepare(unet, optimizer, lr_scheduler)
    forward_unet, forward_optimizer, forward_lr_scheduler = forward_accelerator.prepare(
        forward_unet, forward_optimizer, forward_lr_scheduler
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    uncond_input_ids = tokenizer(
        [""] * args.train_batch_size,
        return_tensors="pt",
        padding="max_length",
        max_length=77,
    ).input_ids.to(accelerator.device)
    uncond_prompt_embeds = text_encoder(uncond_input_ids)[0]

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("forward-checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[2]))
            forward_path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            forward_accelerator.load_state(os.path.join(args.output_dir, forward_path))
            global_step = int(path.split("-")[1])
            forward_global_step = int(forward_path.split("-")[2])
            assert global_step == forward_global_step

            initial_global_step = global_step
    else:
        initial_global_step = 0
    



    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # backward model freeze
    for param in unet.parameters():
        param.requires_grad = False

    if accelerator.is_main_process:
        log_validation_inversion(
                            vae,
                            unet,
                            forward_unet,
                            args,
                            accelerator,
                            forward_accelerator,
                            weight_dtype,
                            global_step,
                            is_sdxl=False,
                            compute_embeddings_fn=compute_embeddings_fn,
                        )
    
    
    calc_patch_lpips = LPIPS_Loss(lpips.LPIPS(net='vgg').to(accelerator.device).float(), args)
     
    for batch in train_dataloader:
        image, text = batch["image"], batch["text"]

        pixel_values = image.to(accelerator.device, non_blocking=True)
        encoded_text = compute_embeddings_fn(text)

        # encode pixel values with batch size of at most 32
        latents = []
        for i in range(0, pixel_values.shape[0], 32):
            latents.append(vae.encode(pixel_values[i : i + 32]).latent_dist.sample())
        latents = torch.cat(latents, dim=0)

        latents = latents * vae.config.scaling_factor
        latents = latents.to(weight_dtype)

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]

        # 20.4.6. Sample a random guidance scale w from U[w_min, w_max] and embed it
        if args.discrete_w is not None:
            w_choices = [float(w) for w in args.discrete_w.split(",")]
            w = torch.tensor(random.choices(w_choices, k=bsz))
        else:
            w = (args.w_max - args.w_min) * torch.rand((bsz,)) + args.w_min

        w = w.reshape(bsz, 1, 1, 1)
        w = w.to(device=latents.device, dtype=latents.dtype)

        assert args.forward_w_max == args.forward_w_min == 0.0, "in the paper, we use the unguided forward process only"
        forward_w = (args.forward_w_max - args.forward_w_min) * torch.rand((bsz,)) + args.forward_w_min
        forward_w = forward_w.reshape(bsz, 1, 1, 1)
        forward_w = forward_w.to(device=latents.device, dtype=latents.dtype)

        # 20.4.8. Prepare prompt embeds and unet_added_conditions
        prompt_embeds = encoded_text.pop("prompt_embeds")
        
        if not args.no_forward:
            forward_loss = forward_train_step(
                args,
                forward_accelerator,
                latents,
                noise,
                prompt_embeds,
                uncond_prompt_embeds,
                encoded_text,
                forward_unet,
                teacher_unet,
                solver,
                forward_w,
                noise_scheduler,
                forward_optimizer,
                forward_lr_scheduler,
                weight_dtype,
            )
        else:
            forward_loss = 0
        
        forward_alignment_loss = forward_alignment_step(
            args,
            forward_accelerator,
            latents,
            prompt_embeds,
            encoded_text,
            forward_unet,
            unet,
            w,
            noise_scheduler,
            forward_optimizer,
            forward_lr_scheduler,
            vae=vae,
            lpips_loss_fn=calc_patch_lpips,
            )
        if args.forward_preserve_loss_coef > 0.0:
            forward_preserve_loss_logs = forward_preserve_train_step(
                args,
                forward_accelerator,
                latents,
                noise,
                prompt_embeds,
                uncond_prompt_embeds,
                encoded_text,
                forward_unet,
                unet,
                solver,
                w,
                forward_w,
                noise_scheduler,
                forward_optimizer,
                forward_lr_scheduler,
                weight_dtype,
            )
        else:
            forward_preserve_loss_logs = {"forward_preserve_loss": 0.0}


        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients and forward_accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1

            if accelerator.is_main_process:
                if global_step % args.checkpointing_steps == 0:
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [
                            d for d in checkpoints if d.startswith("checkpoint")
                        ]
                        checkpoints = sorted(
                            checkpoints, key=lambda x: int(x.split("-")[1])
                        )

                        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = (
                                len(checkpoints) - args.checkpoints_total_limit + 1
                            )
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(
                                f"removing checkpoints: {', '.join(removing_checkpoints)}"
                            )

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(
                                    args.output_dir, removing_checkpoint
                                )
                                shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(
                        args.output_dir, f"checkpoint-{global_step}"
                    )
                    accelerator.save_state(save_path)

                    save_path = os.path.join(
                        args.output_dir, f"forward-checkpoint-{global_step}"
                    )
                    forward_accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

                # Visualize a few validation examples
                if global_step % args.validation_steps == 0:
                    log_validation(
                        vae,
                        unet,
                        args,
                        accelerator,
                        weight_dtype,
                        global_step,
                        is_sdxl=False,
                        compute_embeddings_fn=compute_embeddings_fn,
                    )
                    if not args.no_forward: 
                        log_validation_inversion(
                            vae,
                            unet,
                            forward_unet,
                            args,
                            accelerator,
                            forward_accelerator,
                            weight_dtype,
                            global_step,
                            is_sdxl=False,
                            compute_embeddings_fn=compute_embeddings_fn,
                        )
                        
                        log_train_inversion(
                            vae,
                            unet,
                            forward_unet,
                            args,
                            accelerator,
                            forward_accelerator,
                            weight_dtype,
                            global_step,
                            is_sdxl=False,
                            compute_embeddings_fn=compute_embeddings_fn,
                        )

            # FID evaluation
            if global_step % args.evaluation_steps == 0:
                images, prompts = distributed_sampling(
                    vae,
                    unet,
                    args,
                    accelerator,
                    weight_dtype,
                    global_step,
                    is_sdxl=False,
                    compute_embeddings_fn=compute_embeddings_fn,
                )
                if accelerator.is_main_process:
                    fid_score = calculate_fid(
                        images,
                        args.coco_ref_stats_path,
                        inception_path=args.inception_path,
                    )
                    logs = {"fid": fid_score.item()}
                    accelerator.log(logs, step=global_step)
                dist.barrier()

                if not args.no_forward:
                    eval_inversion(
                        vae,
                        unet,
                        forward_unet,
                        args,
                        forward_accelerator,
                        accelerator,
                        weight_dtype,
                        global_step,
                        is_sdxl=False,
                        compute_embeddings_fn=compute_embeddings_fn,
                    )
                    
                    eval_inversion(
                        vae,
                        unet,
                        forward_unet,
                        args,
                        forward_accelerator,
                        accelerator,
                        weight_dtype,
                        global_step,
                        is_sdxl=False,
                        compute_embeddings_fn=compute_embeddings_fn,
                        is_train=True
                    )
        logs = {
            "forward_loss": forward_loss.detach().item(),
            "lr": lr_scheduler.get_last_lr()[0],
            'forward_alignment_loss': forward_alignment_loss.detach().item(),
        }
        logs.update(forward_preserve_loss_logs)
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=global_step)

        if global_step >= args.max_train_steps:
            break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        unet.save_pretrained(args.output_dir)
        lora_state_dict = get_peft_model_state_dict(unet, adapter_name="default")
        StableDiffusionPipeline.save_lora_weights(
            os.path.join(args.output_dir, "unet_lora"), lora_state_dict
        )

        forward_unet = accelerator.unwrap_model(forward_unet)
        forward_unet.save_pretrained(args.output_dir)
        lora_state_dict = get_peft_model_state_dict(
            forward_unet, adapter_name="default"
        )
        StableDiffusionPipeline.save_lora_weights(
            os.path.join(args.output_dir, "unet_lora"), lora_state_dict
        )

    accelerator.end_training()
    forward_accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
