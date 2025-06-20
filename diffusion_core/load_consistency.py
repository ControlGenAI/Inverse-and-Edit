
from .utils.loading import load_models
from .utils.generation import Generator
from diffusers import DDPMScheduler

def load_consistency(device='cuda:0'):
    root = 'checkpoints'
    ldm_stable, reverse_cons_model, forward_cons_model = load_models(
    model_id="runwayml/stable-diffusion-v1-5",
    device=device,
    forward_checkpoint=f'{root}/finetuned_forward_model.safetensors',
    reverse_checkpoint=f'{root}/iCD-SD15-reverse_259_519_779_999.safetensors',
    r=64,
    w_embed_dim=512,
    teacher_checkpoint=f'{root}/sd15_cfg_distill.pt',
    )

    noise_scheduler = DDPMScheduler.from_pretrained(
    "runwayml/stable-diffusion-v1-5", subfolder="scheduler", )
    start_timestep = 0
    NUM_REVERSE_CONS_STEPS = 4
    REVERSE_TIMESTEPS = [259, 519, 779, 999]
    NUM_FORWARD_CONS_STEPS = 4
    FORWARD_TIMESTEPS = [start_timestep, 259, 519, 779]
    NUM_DDIM_STEPS = 50


    solver = Generator(
    model=ldm_stable,
    noise_scheduler=noise_scheduler,
    n_steps=NUM_DDIM_STEPS,
    forward_cons_model=forward_cons_model,
    forward_timesteps=FORWARD_TIMESTEPS,
    reverse_cons_model=reverse_cons_model,
    reverse_timesteps=REVERSE_TIMESTEPS,
    num_endpoints=NUM_REVERSE_CONS_STEPS,
    num_forward_endpoints=NUM_FORWARD_CONS_STEPS,
    max_forward_timestep_index=49,
    start_timestep=start_timestep)
    solver.model = solver.model.to(device)
    solver.forward_cons_model.unet = solver.forward_cons_model.unet.to(device)
    return solver