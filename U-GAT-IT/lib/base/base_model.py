import torch
import torch.nn as nn

import numpy as np
import warnings
from collections import OrderedDict


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, **kwargs):
        raise NotImplementedError

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        trainable_params = sum([np.prod(p.size()) for p in model_parameters])
        all_params = sum(p.numel() for p in self.parameters())
        return super(BaseModel, self).__str__() + \
               f'\nNbr of parameters: {all_params}' +\
               f'\nNbr of trainable parameters: {trainable_params}'

    def load_pretrained_weights(self, state_dict):
        model_dict = self.state_dict()
        new_state_dict = OrderedDict()
        matched_layers, discarded_layers = [], []

        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]  # discard module.

            if k in model_dict and model_dict[k].size() == v.size():
                new_state_dict[k] = v
                matched_layers.append(k)
            else:
                discarded_layers.append(k)

        model_dict.update(new_state_dict)
        self.load_state_dict(model_dict)

        if len(matched_layers) == 0:
            warnings.warn(
                f'The pretrained weights cannot be loaded, '
                f'please check the key names manually '
                f'(** ignored and continue **)'
            )
        else:
            print(f'Successfully loaded pretrained weights')
            if len(discarded_layers) > 0:
                print(f'** The following layers are discarded due to unmatched keys or layer size: {discarded_layers}')

    def set_trainable_specified_layers(self, layers, is_trainable):
        if isinstance(layers, str):
            layers = [layers]

        for layer in layers:
            assert hasattr(self, layer), f'{layer} is not an attribute of the model, please provide the correct name'

        for name, module in self.named_children():
            if name in layers:
                module.train(is_trainable)
                for p in module.parameters():
                    p.requires_grad = is_trainable

    def set_trainable_all_layers(self, is_trainable):
        self.train(is_trainable)
        for p in self.parameters():
            p.requires_grad = is_trainable

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0001)
                m.bias.data.zero_()

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)