import argparse
from collections import OrderedDict
from pathlib import Path
from typing import Union

"""
Méthodes pour préparer le input du argparser a être utilisé. 
Chaque méthode appelée par le argparser doivent n'avoir qu'un paramètre (comme un lambda)
"""


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def ckpt_parse(v):
    if v is None:
        return v
    elif v.lower() == "none":
        return None
    elif Path(v).is_file():
        return str(Path(v))
    elif isinstance(v, list):
        if Path(" ".join(v)).is_file():
            return " ".join(v)
    print(v)
    raise argparse.ArgumentTypeError('None or filepath expected')


"""def activation_fn_parser(v):
    from architectures import AVAILABLE_ACTIVATION_FN
    if v is torch.nn.Module:
        return v
    elif v in AVAILABLE_ACTIVATION_FN:
        return AVAILABLE_ACTIVATION_FN[v]
    else:
        raise argparse.ArgumentTypeError(f'Activation function must be known.\n '
                                         f"Current input: {v}\n"
                                         f'Available activation functions: {AVAILABLE_ACTIVATION_FN}')"""


def channels_2_tuple(channels):
    from datamodule import KNOWN_CHANNELS
    if isinstance(channels, tuple):
        return channels
    elif isinstance(channels, str):
        if channels in KNOWN_CHANNELS:
            return (channels,)  # to tuple
        else:
            raise argparse.ArgumentTypeError(f'Channels must be known.\n '
                                             f"Current input: {channels}\n"
                                             f'Known channels: {KNOWN_CHANNELS}')
    elif isinstance(channels, list):
        if all(channel in KNOWN_CHANNELS for channel in channels):
            return tuple(channels)
        else:
            raise argparse.ArgumentTypeError(f'Channels must be known.\n '
                                             f"Current input: {channels}\n"
                                             f'Known channels: {KNOWN_CHANNELS}')


def _transforms_str_2_ordereddict(v: Union[str, list], data_or_target_transforms: str):
    from datamodule import AVAILABLE_TARGET_TRANSFORMS, AVAILABLE_TRANSFORMS
    if data_or_target_transforms == "data":
        availables = AVAILABLE_TRANSFORMS
    elif data_or_target_transforms == "target":
        availables = AVAILABLE_TARGET_TRANSFORMS
    else:
        assert 0

    transforms_list = {"train": OrderedDict(), "val": OrderedDict()}

    if isinstance(v, str):
        v = [v]

    if any(transform == "None" for transform in v):
        transforms_list["train"] = None
        transforms_list["val"] = None
        return transforms_list

    if not all(tag in availables.keys() for tag in v):

        raise argparse.ArgumentTypeError(f"{data_or_target_transforms} transforms must be known "
                                         f"or \"None\". \n"
                                         f"Current input: \"{v}\"\n"
                                         f"Known {data_or_target_transforms} transforms: {availables}\n")

    for transform in v:
        transforms_list["train"][transform] = availables[transform]
        if not availables[transform].train_only:
            transforms_list["val"][transform] = availables[transform]
    return transforms_list


def data_t_2_odict(v: Union[str, list]):
    return _transforms_str_2_ordereddict(v, "data")


def target_t_2_odict(v: Union[str, list]):
    return _transforms_str_2_ordereddict(v, "target")


def smooth_eps_parser(v):
    if isinstance(v, float):
        pass
    if isinstance(v, str):
        try:
            v = float(v)
        except ValueError:
            raise argparse.ArgumentTypeError(f"La valeur donnée ne peux pas être transformée en nombre.")

    if 0 <= v <= 1:
        return v
    else:
        raise argparse.ArgumentTypeError(f"La valeur donnée n'est pas entre [0 et 1].")
