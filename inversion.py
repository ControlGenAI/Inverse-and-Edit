from PIL import Image
from omegaconf import OmegaConf
import argparse
from collections import defaultdict
from diffusion_core import diffusion_models_registry
from diffusion_core.utils import load_512
from diffusion_core.guiders.guidance_editing import GuidanceEditing
from diffusion_core.utils import use_deterministic
from diffusion_core.load_consistency import load_consistency


def get_model(model_name, scheduler, device):
    model = diffusion_models_registry[model_name](device)
    return model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_path',
        default='./example_images/coffee.jpg',
    )
    parser.add_argument(
        '--src_prompt',
        default='a cup of coffee with drawing of tulip putted on the wooden table',
    )
    parser.add_argument(
        '--output_path',
        default='inverted_image.jpg',
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    use_deterministic()
    args = parse_args()
    device = 'cuda:0'
    solver = load_consistency(device)
    
    config = OmegaConf.create()
    config['cfg_schedule'] = [0, 0, 0, 0]
    config['guiders'] = []
    config['noise_rescaling_setup'] = {"type": 'identity_rescaler',
                                       'init_setup': None}
    image = load_512(args.image_path)
    guidance = GuidanceEditing(solver, config, device)
    result = guidance(image, args.src_prompt, 
                      args.src_prompt)
    Image.fromarray(result).save(args.output_path)