import importlib

import torch
import torch.nn.functional as F


def get_siamese_features(net, in_features, aggregator=None):
    """ Applies a network in a siamese way, to 'each' in_feature independently
    :param net: nn.Module, Feat-Dim to new-Feat-Dim
    :param in_features: B x  N-objects x Feat-Dim
    :param aggregator, (opt, None, torch.stack, or torch.cat)
    :return: B x N-objects x new-Feat-Dim
    """
    independent_dim = 1
    n_items = in_features.size(independent_dim)
    out_features = []
    for i in range(n_items):
        out_features.append(net(in_features[:, i]))
    if aggregator is not None:
        out_features = aggregator(out_features, dim=independent_dim)
    return out_features


def save_state_dicts(checkpoint_file, epoch=None, **kwargs):
    """Save torch items with a state_dict.
    """
    checkpoint = dict()

    if epoch is not None:
        checkpoint['epoch'] = epoch

    for key, value in kwargs.items():
        checkpoint[key] = value.state_dict()

    torch.save(checkpoint, checkpoint_file)


def load_state_dicts(checkpoint_file, map_location=None, **kwargs):
    """Load torch items from saved state_dictionaries.
    """
    if map_location is None:
        checkpoint = torch.load(checkpoint_file)
    else:
        checkpoint = torch.load(checkpoint_file, map_location=map_location)

    for key, value in kwargs.items():
        value.load_state_dict(checkpoint[key])

    epoch = checkpoint.get('epoch')
    if epoch:
        return epoch


def build_module(module_cfg, **replace_dict):
    module = module_cfg['module']
    module_cls = dynamic_import(module)
    args = module_cfg.get('args', [])
    kwargs = module_cfg.get('kwargs', {})
    kwargs = {k: v for k, v in kwargs.items()}
    for key in kwargs.keys():
        if key.startswith('$') and key[1:] in replace_dict.keys():
            value = replace_dict[key[1:]]
            print(f'setting {key} to {value}')
            kwargs[key[1:]] = value
            del kwargs[key]
    try:
        module = module_cls(*args, **kwargs)
    except Exception as e:
        print(f'bad arguments: \n{args}\n{kwargs}')
        raise e
    return module


def dynamic_import(module_path):
    module_spl = module_path.split('.')
    package, cls_name = '.'.join(module_spl[:-1]), module_spl[-1]
    try:
        module = importlib.import_module(package).__dict__[cls_name]
    except Exception:
        raise ValueError('unexpected network: ' + module_path)

    return module