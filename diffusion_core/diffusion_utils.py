import torch
import numpy as np

from PIL import Image


@torch.no_grad()
def latent2image(latents, model, return_type='np'):
    latents = latents.detach() / model.vae.config.scaling_factor
    image_1 = model.vae.decode(latents)['sample'].detach()
    image_2 = model.vae.decode(latents)['sample'].detach()
    if return_type == 'np':
        image = (image_1 / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).astype(np.uint8)
    return image, image_2


@torch.no_grad()
def image2latent(image, model):
    if type(image) is Image:
        image = np.array(image)
    if type(image) is torch.Tensor and image.dim() == 4:
        latents = image
    else:
        image = torch.from_numpy(image).float() / 127.5 - 1
        image = image.permute(2, 0, 1).unsqueeze(0).to(model.device).to(model.unet.dtype)
        latents = model.vae.encode(image)['latent_dist'].mean
        latents = latents * model.vae.config.scaling_factor
    return latents

