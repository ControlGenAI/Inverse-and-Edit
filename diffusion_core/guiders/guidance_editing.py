from collections import OrderedDict, defaultdict
from typing import Callable, Dict, Optional

import torch
import numpy as np
import PIL
import gc
from diffusion_core.utils import inversion
from tqdm.auto import trange, tqdm
import torch.nn.functional as F
from diffusion_core.guiders.opt_guiders import opt_registry
from diffusion_core.diffusion_utils import latent2image, image2latent
from diffusion_core.guiders.noise_rescales import noise_rescales
from diffusion_core.utils import toggle_grad, use_grad_checkpointing

def match_stats_255(img, target_mean, target_std):
    img_mean = img.mean()
    img_std = img.std()
    if img_std < 1e-7:
        return img
    adjusted = (img - img_mean) * (target_std / img_std) + target_mean
    return np.clip(adjusted, 0, 255).astype(np.uint8)

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def predicted_origin(model_output, timesteps, boundary_timesteps, sample, prediction_type, alphas, sigmas):
    sigmas_s = extract_into_tensor(sigmas, boundary_timesteps, sample.shape)
    alphas_s = extract_into_tensor(alphas, boundary_timesteps, sample.shape)

    sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
    alphas = extract_into_tensor(alphas, timesteps, sample.shape)

    # Set hard boundaries to ensure equivalence with forward (direct) CD
    alphas_s[boundary_timesteps == 0] = 1.0
    sigmas_s[boundary_timesteps == 0] = 0.0

    if prediction_type == "epsilon":
        pred_x_0 = (sample - sigmas * model_output) / alphas  # x0 prediction
        pred_x_0 = alphas_s * pred_x_0 + sigmas_s * model_output  # Euler step to the boundary step
    elif prediction_type == "v_prediction":
        assert boundary_timesteps == 0, "v_prediction does not support multiple endpoints at the moment"
        pred_x_0 = alphas * sample - sigmas * model_output
    else:
        raise ValueError(f"Prediction type {prediction_type} currently not supported.")
    return pred_x_0

class GuidanceEditing:
    def __init__(
            self,
            model,
            config,
            device
    ):
        self.config = config
        self.model = model
        self.gs_schedule = self.config['cfg_schedule']
        self.trg_x0_predictions = []
        self.trg_latents = []
        self.device = device

        self.src_x0_predictions = []
        self.src_latents = []
        
        self.norms = defaultdict(list)
        
        toggle_grad(self.model.reverse_cons_model.unet, False)
        toggle_grad(self.model.forward_cons_model.unet, False)

        if config.get('gradient_checkpointing', False):
            use_grad_checkpointing(mode=True)
        else:
            use_grad_checkpointing(mode=False)

        self.guiders = {
            g_data.name: (opt_registry[g_data.name](**g_data.get('kwargs', {})), g_data.g_scale)
            for g_data in config.guiders
        }

        self.latents_stack = []

        self.context = None

        self.noise_rescaler = noise_rescales[config.noise_rescaling_setup.type](
            config.noise_rescaling_setup.init_setup,
            **config.noise_rescaling_setup.get('kwargs', {})
        )

        for guider_name, (guider, _) in self.guiders.items():
            guider.clear_outputs()

        self.self_attn_layers_num = config.get('self_attn_layers_num', [6, 1, 9])
        if type(self.self_attn_layers_num[0]) is int:
            for i in range(len(self.self_attn_layers_num)):
                self.self_attn_layers_num[i] = (0, self.self_attn_layers_num[i])
            

    def __call__(
            self,
            image_gt: PIL.Image.Image,
            inv_prompt: str,
            trg_prompt: str,
    ):
        self.norm_diff = None
        self.train(
            image_gt,
            inv_prompt,
            trg_prompt
        )
        del self.model.forward_cons_model
        self.model.reverse_cons_model.unet = self.model.reverse_cons_model.unet.to(self.device)
        self.trg_x0_predictions = []
        self.trg_latents = []

        self.src_x0_predictions = []
        self.src_latents = []
        edited_img = self.edit()[0][0][:]
        return match_stats_255(edited_img, self.mean, self.std)

    def train(
            self,
            image_gt: PIL.Image.Image,
            inv_prompt: str,
            trg_prompt: str,
            verbose: bool = False
    ):
        self.init_prompt(inv_prompt, trg_prompt)
        self.verbose = verbose
        self.image_gt = image_gt

        image_gt = np.array(image_gt)
        self.mean = image_gt.mean()
        self.std = image_gt.std()
        
        (image_gt, image_rec), self.inv_latents, self.uncond_embeddings = inversion.invert(
        # Playing params
        image_gt=image_gt,
        prompt=inv_prompt,
        # Fixed params
        is_cons_inversion=True,
        w_embed_dim=512, #512,
        inv_guidance_scale=0.0, #0.0,
        stop_step=50,
        solver=self.model,
        seed=10500)

        for g_name, (guider, _) in self.guiders.items():
            if hasattr(guider, 'model_patch'):
                guider.model_patch(self.model.reverse_cons_model, self_attn_layers_num=self.self_attn_layers_num)

        self.start_latent = self.inv_latents[-1].clone()

        params = {
            'model': self.model,
            'inv_prompt': inv_prompt,
            'trg_prompt': trg_prompt
        }
        for g_name, (guider, _) in self.guiders.items():
            if hasattr(guider, 'train'):
                guider.train(params)

        for guider_name, (guider, _) in self.guiders.items():
            guider.clear_outputs()

    def _construct_data_dict(
            self, latents,
            diffusion_iter,
            timestep
    ):
        uncond_emb, inv_prompt_emb, trg_prompt_emb = self.context.chunk(3)

        if self.uncond_embeddings is not None:
            uncond_emb = self.uncond_embeddings[diffusion_iter]

        data_dict = {
            'latent': latents,
            'inv_latent': self.inv_latents[-diffusion_iter - 1],
            'timestep': timestep,
            'model': self.model,
            'uncond_emb': uncond_emb,
            'trg_emb': trg_prompt_emb,
            'inv_emb': inv_prompt_emb,
            'diff_iter': diffusion_iter
        }

        with torch.no_grad():
            inv_prompt_unet = self.model.get_noise_pred(
                model=self.model.reverse_cons_model,
                latent=data_dict['inv_latent'],
                t=timestep.to(self.model.model.device),
                context=torch.cat([inv_prompt_emb]),
                tau1=0.8, tau2=0.8,
                w_embed_dim=512, 
                guidance_scale=0.0,
                dynamic_guidance=True)



        for g_name, (guider, _) in self.guiders.items():
            if hasattr(guider, 'model_patch'):
                if 'inv_inv' in guider.forward_hooks:
                    data_dict.update({f"{g_name}_inv_inv": guider.output})
                guider.clear_outputs()

        data_dict['latent'].requires_grad = True
        
        
        src_prompt_unet = self.model.get_noise_pred(
                model=self.model.reverse_cons_model,
                latent=data_dict['latent'],
                t=timestep.to(self.model.model.device),
                context=torch.cat([inv_prompt_emb]),
                tau1=0.8, tau2=0.8,
                w_embed_dim=512,
                guidance_scale=0.0,
                dynamic_guidance=True)
        

        for g_name, (guider, _) in self.guiders.items():
            if hasattr(guider, 'model_patch'):
                if 'cur_inv' in guider.forward_hooks:
                    data_dict.update({f"{g_name}_cur_inv": guider.output})
                guider.clear_outputs()

        trg_prompt_unet = self.model.get_noise_pred(
                model=self.model.reverse_cons_model,
                latent=data_dict['latent'],
                t=timestep.to(self.model.model.device),
                context=torch.cat([trg_prompt_emb]),
                tau1=1., tau2=1.,
                w_embed_dim=512,
                guidance_scale=self.gs_schedule[diffusion_iter],
                dynamic_guidance=True)
        
        for g_name, (guider, _) in self.guiders.items():
            if hasattr(guider, 'model_patch'):
                if 'cur_trg' in guider.forward_hooks:
                    data_dict.update({f"{g_name}_cur_trg": guider.output})
                guider.clear_outputs()
            
        data_dict.update({
            'trg_prompt_unet': trg_prompt_unet,
            'src_prompt_unet': src_prompt_unet,
            'inv_prompt_unet': inv_prompt_unet,
        })

        return data_dict

    def _get_noise(self, data_dict, diffusion_iter):
        backward_guiders_sum = 0.
        noises = {
            'trg': data_dict['trg_prompt_unet']
        }
    
        for name, (guider, g_scale) in self.guiders.items():
            if guider.grad_guider:
                cur_noise_pred = self._get_scale(g_scale, diffusion_iter) * guider(data_dict)
                noises[name] = cur_noise_pred
            else:
                energy = self._get_scale(g_scale, diffusion_iter) * guider(data_dict)
                if not torch.allclose(energy, torch.tensor(0.)):
                    backward_guiders_sum += energy

        if hasattr(backward_guiders_sum, 'backward'):
            backward_guiders_sum.backward()
            noises['other'] = data_dict['latent'].grad

        scales = self.noise_rescaler(noises, 0, data_dict['trg_prompt_unet'] - data_dict['inv_prompt_unet'])
        
        noise_pred = sum(scales[k] * noises[k] for k in noises)
        for g_name, (guider, _) in self.guiders.items():
            if not guider.grad_guider:
                guider.clear_outputs()
            gc.collect()
            torch.cuda.empty_cache()

        return noise_pred

    @staticmethod
    def _get_scale(g_scale, diffusion_iter):
        if type(g_scale) is float:
            return g_scale
        else:
            return g_scale[diffusion_iter]

    @torch.no_grad()
    def _step(self, noise_pred, t, s, latent):
        alpha_schedule = torch.sqrt(self.model.reverse_cons_model.scheduler.alphas_cumprod).to(self.model.model.unet.device)
        sigma_schedule = torch.sqrt(1 - self.model.reverse_cons_model.scheduler.alphas_cumprod).to(self.model.model.unet.device)
        latent = predicted_origin(
                noise_pred,
                torch.tensor([t] * len(latent), device=self.model.model.unet.device),
                torch.tensor([s] * len(latent), device=self.model.model.unet.device),
                latent,
                self.model.scheduler.config.prediction_type,
                alpha_schedule,
                sigma_schedule,
            )
        return latent
    
    def edit(self):
        latents = self.start_latent
        for i, (t, s) in enumerate(tqdm(zip(self.model.reverse_timesteps, self.model.reverse_boundary_timesteps))):
            # 1. Construct dict            
            data_dict = self._construct_data_dict(latents, i, t)
            # 2. Calculate guidance
            noise_pred = self._get_noise(data_dict, i)
            # 3. Scheduler step
            latents = self._step(noise_pred, t, s, latents)
            

        self._model_unpatch(self.model.model)
        return latent2image(latents, self.model.model)

    @torch.no_grad()
    def init_prompt(self, inv_prompt: str, trg_prompt: str):
        trg_prompt_embed = self.get_prompt_embed(trg_prompt)
        inv_prompt_embed = self.get_prompt_embed(inv_prompt)
        uncond_embed = self.get_prompt_embed("")

        self.context = torch.cat([uncond_embed, inv_prompt_embed, trg_prompt_embed])

    def get_prompt_embed(self, prompt: str):
        text_input = self.model.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.model.text_encoder(
            text_input.input_ids.to(self.model.model.device)
        )[0]

        return text_embeddings

    def sample_noised_latents(self, latent):
        all_latent = [latent.clone().detach()]
        latent = latent.clone().detach()
        for i in trange(self.model.scheduler.num_inference_steps, desc='Latent Sampling'):
            timestep = self.model.scheduler.timesteps[-i - 1]
            if i + 1 < len(self.model.scheduler.timesteps):
                next_timestep = self.model.scheduler.timesteps[- i - 2]
            else:
                next_timestep = 999

            alpha_prod_t = self.model.scheduler.alphas_cumprod[timestep]
            alpha_prod_t_next = self.model.scheduler.alphas_cumprod[next_timestep]

            alpha_slice = alpha_prod_t_next / alpha_prod_t

            latent = torch.sqrt(alpha_slice) * latent + torch.sqrt(1 - alpha_slice) * torch.randn_like(latent)
            all_latent.append(latent)
        return all_latent

    def _model_unpatch(self, model):
        def new_forward_info(self):
            def patched_forward(
                    hidden_states,
                    encoder_hidden_states=None,
                    attention_mask=None,
                    temb=None,
            ):
                residual = hidden_states

                if self.spatial_norm is not None:
                    hidden_states = self.spatial_norm(hidden_states, temb)

                input_ndim = hidden_states.ndim

                if input_ndim == 4:
                    batch_size, channel, height, width = hidden_states.shape
                    hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

                batch_size, sequence_length, _ = (
                    hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
                )
                attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

                if self.group_norm is not None:
                    hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

                query = self.to_q(hidden_states)

                ## Injection
                is_self = encoder_hidden_states is None

                if encoder_hidden_states is None:
                    encoder_hidden_states = hidden_states
                elif self.norm_cross:
                    encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

                key = self.to_k(encoder_hidden_states)
                value = self.to_v(encoder_hidden_states)

                query = self.head_to_batch_dim(query)
                key = self.head_to_batch_dim(key)
                value = self.head_to_batch_dim(value)

                attention_probs = self.get_attention_scores(query, key, attention_mask)
                hidden_states = torch.bmm(attention_probs, value)
                hidden_states = self.batch_to_head_dim(hidden_states)

                # linear proj
                hidden_states = self.to_out[0](hidden_states)
                # dropout
                hidden_states = self.to_out[1](hidden_states)

                if input_ndim == 4:
                    hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

                if self.residual_connection:
                    hidden_states = hidden_states + residual

                hidden_states = hidden_states / self.rescale_output_factor

                return hidden_states

            return patched_forward

        def register_attn(module):
            if 'Attention' in module.__class__.__name__:
                module.forward = new_forward_info(module)
            elif hasattr(module, 'children'):
                for module_ in module.children():
                    register_attn(module_)

        def remove_hooks(module):
            if hasattr(module, "_forward_hooks"):
                module._forward_hooks: Dict[int, Callable] = OrderedDict()
            if hasattr(module, 'children'):
                for module_ in module.children():
                    remove_hooks(module_)

        register_attn(model.unet)
        remove_hooks(model.unet)
