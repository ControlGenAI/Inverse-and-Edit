import torch
from typing import Optional

from diffusion_core.utils.class_registry import ClassRegistry
from diffusion_core.guiders.scale_schedulers import last_steps, first_steps

opt_registry = ClassRegistry()

class BaseGuider:
    def __init__(self):
        self.clear_outputs()
    
    @property
    def grad_guider(self):
        return hasattr(self, 'grad_fn')
    
    def __call__(self, data_dict):
        if self.grad_guider:
            return self.grad_fn(data_dict)
        else:
            return self.calc_energy(data_dict)   
        
    def clear_outputs(self):
        if not self.grad_guider:
            self.output = self.single_output_clear()
    
    def single_output_clear(self):
        raise NotImplementedError()

@opt_registry.add_to_registry('cfg')
class ClassifierFreeGuidance(BaseGuider):
    def __init__(self, is_source_guidance=False):
        self.is_source_guidance = is_source_guidance
    
    def grad_fn(self, data_dict):
        prompt_unet = data_dict['inv_prompt_unet'] if self.is_source_guidance else data_dict['trg_prompt_unet']
        return prompt_unet #- data_dict['uncond_unet']
    
  
@opt_registry.add_to_registry('dds')
class DDSGuidance(BaseGuider):
    def __init__(self, is_source_guidance=False):
        self.is_source_guidance = is_source_guidance
    
    def grad_fn(self, data_dict):
        prompt_unet = data_dict['src_prompt_unet'] if self.is_source_guidance else data_dict['trg_prompt_unet']
        return data_dict['trg_prompt_unet'] - data_dict['src_prompt_unet']



@opt_registry.add_to_registry('latents_diff')
class LatentsDiffGuidance(BaseGuider):
    """
    \| z_t* - z_t \|^2_2
    """
    def grad_fn(self, data_dict):
        return 2 * (data_dict['latent'] - data_dict['inv_latent'])

@opt_registry.add_to_registry('latents_diff_grad')
class LatentsDiffGuidance(BaseGuider):
    """
    \| z_t* - z_t \|^2_2
    """
    def calc_energy(self, data_dict):
        return 2 * (data_dict['latent'] - data_dict['inv_latent']).mean()
    
    def single_output_clear(self):
        None
                
@opt_registry.add_to_registry('features_map_l2')
class FeaturesMapL2EnergyGuider(BaseGuider):
    def __init__(self, block='up'):
        assert block in ['down', 'up', 'mid', 'whole']
        self.block = block
        
    patched = True
    forward_hooks = ['cur_trg', 'inv_inv']
    def calc_energy(self, data_dict):
        return torch.mean(torch.pow(data_dict['features_map_l2_cur_trg'] - data_dict['features_map_l2_inv_inv'], 2))
        # return torch.mean(torch.abs(data_dict['features_map_l2_cur_trg'] - data_dict['features_map_l2_inv_inv']))
    
    def model_patch(self, model, self_attn_layers_num=None):
        def hook_fn(module, input, output):
            self.output = output 
        def mid_hook_fn(module, input, output):
            self.mid_hook = output
        if self.block == 'mid':
            model.unet.mid_block.register_forward_hook(hook_fn)
        elif self.block == 'up':
            model.unet.up_blocks[1].resnets[1].register_forward_hook(hook_fn)
        elif self.block == 'down':
            model.unet.down_blocks[1].resnets[1].register_forward_hook(hook_fn)
    
    def single_output_clear(self):
        None
    
    
    

@opt_registry.add_to_registry('trg_self_attn_map_l2')
class SelfAttnMapL2EnergyGuider(BaseGuider):
    patched = True
    forward_hooks = ['trg_inv', 'inv_inv']    
    def single_output_clear(self):
        return {
            "down_cross": [], "mid_cross": [], "up_cross": [],
            "down_self":  [], "mid_self":  [], "up_self":  []
        }
    
    def calc_energy(self, data_dict):
        result = 0.
        for unet_place, data in data_dict['self_attn_map_l2_cur_inv'].items():
            for elem_idx, elem in enumerate(data):
                result += torch.mean(
                    torch.pow(
                        elem - data_dict['self_attn_map_l2_inv_inv'][unet_place][elem_idx], 2
                    )
                )
        # print("self attn", result)
        self.single_output_clear()
        return result
    
    def model_patch(guider_self, model, self_attn_layers_num=None):
        def new_forward_info(self, place_unet):
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
                if is_self:
                    guider_self.output[f"{place_unet}_self"].append(attention_probs)
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
        
        def register_attn(module, place_in_unet, layers_num, cur_layers_num=0):
            if 'Attention' in module.__class__.__name__:
                if 2 * layers_num[0] <= cur_layers_num < 2 * layers_num[1]:
                    module.forward = new_forward_info(module, place_in_unet)
                return cur_layers_num + 1
            elif hasattr(module, 'children'):
                for module_ in module.children():
                    cur_layers_num = register_attn(module_, place_in_unet, layers_num, cur_layers_num)
                return cur_layers_num
        
        sub_nets = model.unet.named_children()
        for name, net in sub_nets:
            if "down" in name:
                register_attn(net, "down", self_attn_layers_num[0])
            if "mid" in name:
                register_attn(net, "mid", self_attn_layers_num[1])
            if "up" in name:
                register_attn(net, "up", self_attn_layers_num[2])


@opt_registry.add_to_registry('self_attn_map_l2')
class SelfAttnMapL2EnergyGuider(BaseGuider):
    patched = True
    forward_hooks = ['cur_inv', 'inv_inv']    
    def single_output_clear(self):
        return {
            "down_cross": [], "mid_cross": [], "up_cross": [],
            "down_self":  [], "mid_self":  [], "up_self":  []
        }
    
    def calc_energy(self, data_dict):
        result = 0.
        for unet_place, data in data_dict['self_attn_map_l2_cur_inv'].items():
            for elem_idx, elem in enumerate(data):
                result += torch.mean(
                    torch.pow(
                        elem - data_dict['self_attn_map_l2_inv_inv'][unet_place][elem_idx], 2
                    )
                )
        # print("self attn", result)
        self.single_output_clear()
        return result
    
    def model_patch(guider_self, model, self_attn_layers_num=None):
        def new_forward_info(self, place_unet):
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
                if is_self:
                    guider_self.output[f"{place_unet}_self"].append(attention_probs)
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
        
        def register_attn(module, place_in_unet, layers_num, cur_layers_num=0):
            if 'Attention' in module.__class__.__name__:
                if 2 * layers_num[0] <= cur_layers_num < 2 * layers_num[1]:
                    module.forward = new_forward_info(module, place_in_unet)
                return cur_layers_num + 1
            elif hasattr(module, 'children'):
                for module_ in module.children():
                    cur_layers_num = register_attn(module_, place_in_unet, layers_num, cur_layers_num)
                return cur_layers_num
        
        sub_nets = model.unet.named_children()
        for name, net in sub_nets:
            if "down" in name:
                register_attn(net, "down", self_attn_layers_num[0])
            if "mid" in name:
                register_attn(net, "mid", self_attn_layers_num[1])
            if "up" in name:
                register_attn(net, "up", self_attn_layers_num[2])
                
                
@opt_registry.add_to_registry('cross_attn_grep')
class CrossAttnGrep(BaseGuider):
    patched = True
    forward_hooks = ['cur_inv', 'inv_inv']    
    def single_output_clear(self):
        return {
            "down_cross": [], "mid_cross": [], "up_cross": [],
            "down_self":  [], "mid_self":  [], "up_self":  []
        }
    
    def calc_energy(self, data_dict):
        result = torch.tensor(0.)
        # print("self attn", result)
        self.single_output_clear()
        return result
    
    def model_patch(guider_self, model, self_attn_layers_num=None):
        def new_forward_info(self, place_unet):
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
                if not is_self:
                    guider_self.output[f"{place_unet}_cross"].append(attention_probs)
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
        
        def register_attn(module, place_in_unet, layers_num, cur_layers_num=0):
            if 'Attention' in module.__class__.__name__:
                if 2 * layers_num[0] <= cur_layers_num < 2 * layers_num[1]:
                    module.forward = new_forward_info(module, place_in_unet)
                return cur_layers_num + 1
            elif hasattr(module, 'children'):
                for module_ in module.children():
                    cur_layers_num = register_attn(module_, place_in_unet, layers_num, cur_layers_num)
                return cur_layers_num
        
        sub_nets = model.unet.named_children()
        for name, net in sub_nets:
            if "down" in name:
                register_attn(net, "down", self_attn_layers_num[0])
            if "mid" in name:
                register_attn(net, "mid", self_attn_layers_num[1])
            if "up" in name:
                register_attn(net, "up", self_attn_layers_num[2])

    
@opt_registry.add_to_registry('self_attn_map_l2_appearance')
class SelfAttnMapL2withAppearanceEnergyGuider(BaseGuider):
    patched = True
    forward_hooks = ['cur_inv', 'inv_inv']

    def __init__(
        self, self_attn_gs: float, app_gs: float, new_features: bool=False, 
        total_last_steps: Optional[int] = None, total_first_steps: Optional[int] = None
    ):
        super().__init__()
        
        self.new_features = new_features

        if total_last_steps is not None:
            self.app_gs = last_steps(app_gs, total_last_steps)
            self.self_attn_gs = last_steps(self_attn_gs, total_last_steps)
        elif total_first_steps is not None:
            self.app_gs = first_steps(app_gs, total_first_steps)
            self.self_attn_gs = first_steps(self_attn_gs, total_first_steps)
        else:
            self.app_gs = app_gs
            self.self_attn_gs = self_attn_gs

    def single_output_clear(self):
        return {
            "down_self":  [], 
            "mid_self":  [], 
            "up_self":  [],
            "features": None
        }
    
    def calc_energy(self, data_dict):
        self_attn_result = 0.
        unet_places = ['down_self', 'up_self', 'mid_self']
        for unet_place in unet_places:
            data = data_dict['self_attn_map_l2_appearance_cur_inv'][unet_place]
            for elem_idx, elem in enumerate(data):
                self_attn_result += torch.mean(
                    torch.pow(
                        elem - data_dict['self_attn_map_l2_appearance_inv_inv'][unet_place][elem_idx], 2
                    )
                )
        
        features_orig = data_dict['self_attn_map_l2_appearance_inv_inv']['features']
        features_cur = data_dict['self_attn_map_l2_appearance_cur_inv']['features']
        app_result = torch.mean(torch.abs(features_cur - features_orig))

        self.single_output_clear()

        if type(self.app_gs) == float:
            _app_gs = self.app_gs
        else:
            _app_gs = self.app_gs[data_dict['diff_iter']]

        if type(self.self_attn_gs) == float:
            _self_attn_gs = self.self_attn_gs
        else:
            _self_attn_gs = self.self_attn_gs[data_dict['diff_iter']]
        # print('diff: ', _self_attn_gs * self_attn_result + _app_gs * app_result)
        return _self_attn_gs * self_attn_result + _app_gs * app_result
    
    def model_patch(guider_self, model, self_attn_layers_num=None):
        def new_forward_info(self, place_unet):
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
                if is_self:
                    guider_self.output[f"{place_unet}_self"].append(attention_probs)
                
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
        
        def register_attn(module, place_in_unet, layers_num, cur_layers_num=0):
            if 'Attention' in module.__class__.__name__:
                if 2 * layers_num[0] <= cur_layers_num < 2 * layers_num[1]:
                    module.forward = new_forward_info(module, place_in_unet)
                return cur_layers_num + 1
            elif hasattr(module, 'children'):
                for module_ in module.children():
                    cur_layers_num = register_attn(module_, place_in_unet, layers_num, cur_layers_num)
                return cur_layers_num
        
        sub_nets = model.unet.named_children()
        for name, net in sub_nets:
            if "down" in name:
                register_attn(net, "down", self_attn_layers_num[0])
            if "mid" in name:
                register_attn(net, "mid", self_attn_layers_num[1])
            if "up" in name:
                register_attn(net, "up", self_attn_layers_num[2])
        
        def hook_fn(module, input, output):
            guider_self.output["features"] = output

        if guider_self.new_features:
            model.unet.up_blocks[-1].register_forward_hook(hook_fn)
        else:
            model.unet.conv_norm_out.register_forward_hook(hook_fn)
