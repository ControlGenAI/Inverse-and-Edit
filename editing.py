from PIL import Image
from omegaconf import OmegaConf
import argparse
import torch
from diffusion_core.utils import load_512
from diffusion_core.guiders.guidance_editing import GuidanceEditing
from diffusion_core.utils import use_deterministic
from diffusion_core.load_consistency import load_consistency

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
        '--trg_prompt',
        default='a cup of coffee with drawing of lion putted on the wooden table',
    )
    
    parser.add_argument(
        '--config_path',
        default='./configs/consistency.yaml',
    )
    
    parser.add_argument(
        '--output_path',
        default='edited_image.jpg',
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    use_deterministic()
    args = parse_args()
    device = 'cuda:0'
    solver = load_consistency(device)
    config = OmegaConf.load(args.config_path)
    image = load_512(args.image_path)
    guidance = GuidanceEditing(solver, config, device)
    result = guidance(image, args.src_prompt, 
                      args.trg_prompt)
    Image.fromarray(result).save(args.output_path)
