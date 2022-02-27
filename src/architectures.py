"""
Module contenant les différentes architectures disponibles à utiliser.
Pour en rajouter une:
- Faire une nouvelle classe qui hérite de ArchSkeleton.
- Définir une méthode forward acceptant un tuple de x_fluorescence et x_tabular.
- Définir les arguments optimaux pour le système utilisé.
- Updater AVAILABLE_ARCHS avec la nouvelle architecture.

Les classes ici ont un arbre de convolution pour ensuite concaténer les données tabulaire au niveau des couches denses.
"""

from argparse import ArgumentParser
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class ArchBaseClass(nn.Module):
    """
    La classe à utiliser pour hériter.
    """
    def __init__(self, num_classes: int = 0, num_channels: int = 0, tab_size: int = 0):
        super().__init__()
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.tab_size = tab_size
        self.activation = nn.ReLU
        self.scheduler = None


    @staticmethod
    def add_arch_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.set_defaults(max_epochs=300)  # already existing from trainer args, so modification only.
        parser.add_argument("--early_stop_patience", default=100, type=int,
                            help='Combien de steps le trainer devrait attendre avant de faire un early stop. '
                                 'Défaut=dépend de larchitecture')
        parser.add_argument("--scheduler_cooldown", default=0, type=int,
                            help='Combien de steps le trainer devrait attendre avant de faire un early stop. '
                                 'Défaut=dépend de larchitecture')
        parser.add_argument("--scheduler_patience", default=35, type=int,
                            help='Combien de steps le trainer devrait attendre avant de descendre le learning rate. '
                                 'Défaut=dépend de larchitecture')
        parser.add_argument('--lr', default=1E-3, type=float,
                            help='Learning rate de départ. Défaut=dépend de larchitecture')
        parser.add_argument('--bs', default=100, type=int,
                            help='Combien déchantillon par batch pour chaque step. '
                                 'Cest bien de prendre le plus gros bs possible qui rentre dans la mémoire de lordinateur. '
                                 'Est limité par la mémoire (cpu) ou VRAM (GPU).')
        return parser


# Everything VGG related here
# ----------------------------------------------------------------------------------

class VGG(ArchBaseClass):
    """
    Classe squelette pour les différents VGG possible. La tête du modèle est prédéfinie ici.
    """
    def __init__(self, num_classes: int, num_channels: int, tab_size: int, **kwargs):
        super().__init__(num_classes, num_channels, tab_size)

        self.tree = None
        self.head = nn.Sequential(OrderedDict([
            ("dense1", nn.Linear(14 * 512 + self.tab_size, 4096)),
            ("act1", self.activation(inplace=True)),
            ("dense2", nn.Linear(4096, 4096)),
            ("act2", self.activation(inplace=True)),
            ("dense3", nn.Linear(4096, num_classes)),
        ]))

    def forward(self, data_tuple: tuple):

        x = self.tree(data_tuple[0])
        # Concatenate tabular data to flattened first dense layer
        if self.tab_size > 0:
            x = torch.cat([x, data_tuple[1]], dim=1)
        x = self.head(x)
        return x

    @staticmethod
    def add_arch_specific_args(parent_parser):
        parser = ArchBaseClass.add_arch_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parser], add_help=False)
        parser.set_defaults(max_epochs=250,
                            scheduler_patience=35,
                            early_stop_patience=75,
                            lr=1E-3,
                            bs=4400)
        return parser


class VGG16(VGG):
    """
    La classe la plus facile à entrainer et optimiser.
    """
    def __init__(self, num_classes, num_channels, tab_size, **kwargs):
        super().__init__(num_classes, num_channels, tab_size)
        self.tree = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv1d(self.num_channels, 64, kernel_size=3, stride=1, padding=1)),
            ("act1", self.activation()),
            ("conv2m", nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1)),
            ("act2", self.activation()),
            ("conv3", nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)),
            ("act3", self.activation()),
            ("conv4m", nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1)),
            ("act4", self.activation()),
            ("conv5", nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)),
            ("act5", self.activation()),
            ("conv6", nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)),
            ("act6", self.activation()),
            ("conv7m", nn.Conv1d(256, 256, kernel_size=3, stride=2, padding=1)),
            ("act7", self.activation()),
            ("conv8", nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1)),
            ("act8", self.activation()),
            ("conv9", nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1)),
            ("act9", self.activation()),
            ("conv10m", nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1)),
            ("act10", self.activation()),
            ("conv11", nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1)),
            ("act11", self.activation()),
            ("conv12", nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1)),
            ("act12", self.activation()),
            ("conv13m", nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1)),
            ("act13", self.activation()),
            ("flatten", nn.Flatten()),
        ]))

    @staticmethod
    def add_arch_specific_args(parent_parser):
        parser = VGG.add_arch_specific_args(parent_parser)
        return parser


class VGG11(VGG):
    def __init__(self, num_classes, num_channels, tab_size, **kwargs):
        super().__init__(num_classes, num_channels, tab_size)
        self.tree = nn.Sequential(OrderedDict({
            "conv1m": nn.Conv1d(self.num_channels, 64, kernel_size=3, stride=2, padding=1),
            "act1": self.activation(),
            "conv2m": nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            "act2": self.activation(),
            "conv3": nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            "act3": self.activation(),
            "conv4m": nn.Conv1d(256, 256, kernel_size=3, stride=2, padding=1),
            "act4": self.activation(),
            "conv5": nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            "act5": self.activation(),
            "conv6m": nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            "act6": self.activation(),
            "conv7": nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            "act7": self.activation(),
            "conv8m": nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            "act8": self.activation(),
            "flatten": nn.Flatten(),
        }))

    @staticmethod
    def add_arch_specific_args(parent_parser):
        parser = VGG.add_arch_specific_args(parent_parser) #  not super() because not an instance (static)
        return parser


# Everything ResNet related here
# ----------------------------------------------------------------------------------


class StdConv1d(nn.Conv1d):
    """
    Remplace conv1d par une version standardisée (w = (w - m) / torch.sqrt(v + 1e-10))
    """
    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, unbiased=False)  # removed the dims specification. does not look to be needed (???)
        w = (w - m) / torch.sqrt(v + 1e-10)
        return F.conv1d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


def conv3x(cin, cout, stride=1, groups=1, bias=False):
    return StdConv1d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x(cin, cout, stride=1, bias=False):
    return StdConv1d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.
  Follows the implementation of "Identity Mappings in Deep Residual Networks":
  https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua
  Except it puts the stride on 3x3 conv when available.
  """

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4

        self.gn1 = nn.GroupNorm(32, cin)
        self.conv1 = conv1x(cin, cmid)
        self.gn2 = nn.GroupNorm(32, cmid)
        self.conv2 = conv3x(cmid, cmid, stride)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cmid)
        self.conv3 = conv1x(cmid, cout)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or cin != cout:
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x(cin, cout, stride)

    def forward(self, x):
        out = self.relu(self.gn1(x))

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(out)

        # Unit's branch
        out = self.conv1(out)
        out = self.conv2(self.relu(self.gn2(out)))
        out = self.conv3(self.relu(self.gn3(out)))

        return out + residual


class ResNetV2(ArchBaseClass):  # actual class to call
    """
    Version de ResnetV2 venant du github de https://github.com/google-research/big_transfer

    Voici les block_units size utilisé et le width_factor par Google:
    BiT-M-R50x1, [3, 4, 6, 3], 1
    BiT-M-R50x3, [3, 4, 6, 3], 3
    BiT-M-R101x1, [3, 4, 23, 3], 1
    BiT-M-R101x3, [3, 4, 23, 3], 3
    BiT-M-R152x2, [3, 8, 36, 3], 2
    BiT-M-R152x4, [3, 8, 36, 3], 4
    BiT-S-R50x1, [3, 4, 6, 3], 1
    BiT-S-R50x3, [3, 4, 6, 3], 3
    BiT-S-R101x1, [3, 4, 23, 3], 1
    BiT-S-R101x3, [3, 4, 23, 3], 3
    BiT-S-R152x2, [3, 8, 36, 3], 2
    BiT-S-R152x4, [3, 8, 36, 3], 4

    Beaucoup de ces tailles sont impossible sur un ordinateur simple.

    """
    def __init__(self, num_classes: int, num_channels: int, block_units: str, wf: int, tab_size: int, **kwargs):
        super().__init__(num_classes, num_channels, tab_size)
        self.wf = wf
        self.block_units = list(map(int, block_units.split("_")))  # todo block units parser

        self.scheduler = {"fn": torch.optim.lr_scheduler.StepLR,
                          "kwargs": dict(step_size=50, gamma=0.1, last_epoch=-1, verbose=True)}

        self.my_hparams = {"block_units": self.block_units, "wf": self.wf,
                        # only those two because those two are the only one unique to architecture
                        }

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv1d(self.num_channels, 64 * self.wf, kernel_size=7, stride=2, padding=3, bias=False)),
            ('pad', nn.ConstantPad1d(1, 0)),
            ('pool', nn.MaxPool1d(kernel_size=3, stride=2, padding=0)),
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit01', PreActBottleneck(cin=64 * self.wf, cout=256 * self.wf, cmid=64 * self.wf))] +
                [(f'unit{i:02d}', PreActBottleneck(cin=256 * self.wf, cout=256 * self.wf, cmid=64 * self.wf)) for i in
                 range(2, self.block_units[0] + 1)],
            ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit01', PreActBottleneck(cin=256 * self.wf, cout=512 * self.wf, cmid=128 * self.wf, stride=2))] +
                [(f'unit{i:02d}', PreActBottleneck(cin=512 * self.wf, cout=512 * self.wf, cmid=128 * self.wf)) for i in
                 range(2, self.block_units[1] + 1)],
            ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit01', PreActBottleneck(cin=512 * self.wf, cout=1024 * self.wf, cmid=256 * self.wf, stride=2))] +
                [(f'unit{i:02d}', PreActBottleneck(cin=1024 * self.wf, cout=1024 * self.wf, cmid=256 * self.wf)) for i
                 in
                 range(2, self.block_units[2] + 1)],
            ))),
            ('block4', nn.Sequential(OrderedDict(
                [('unit01', PreActBottleneck(cin=1024 * self.wf, cout=2048 * self.wf, cmid=512 * self.wf, stride=2))] +
                [(f'unit{i:02d}', PreActBottleneck(cin=2048 * self.wf, cout=2048 * self.wf, cmid=512 * self.wf)) for i
                 in
                 range(2, self.block_units[3] + 1)],
            ))),
        ]))

        self.head = nn.Sequential(OrderedDict([
            ('gn', nn.GroupNorm(32, 2048 * self.wf)),
            ('relu', nn.ReLU(inplace=True)),
            ('avg', nn.AdaptiveAvgPool1d(1)),
        ]))
        self.forehead = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(2048 * self.wf + self.tab_size, 2048 * self.wf, kernel_size=1, bias=True)),
            ('gn', nn.GroupNorm(32, 2048 * self.wf)),
            ('relu', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv1d(2048 * self.wf, self.num_classes, kernel_size=1, bias=True)),
            ('flatten', nn.Flatten())
        ]))

    def forward(self, data_tuple):
        x = self.head(self.body(self.root(data_tuple[0])))
        if self.tab_size > 0:
            x = torch.cat([x, data_tuple[1][:, :, np.newaxis]], dim=1)
        x = self.forehead(x)

        return x

    @staticmethod
    def add_arch_specific_args(parent_parser):
        parser = ArchBaseClass.add_arch_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parser], add_help=False)
        # already existing from trainer args, so modification only.
        parser.set_defaults(max_epochs=200,
                            early_stop_patience=55,
                            lr=3E-3,
                            bs=740,
                            )
        # those 3 options are optimized to max memory on a  NVIDIA GTX 1080 and floating point precision (fp) of 16
        # parser.add_argument('--bs', default=740, type=int)
        parser.add_argument("--block_units", default="3_4_6_3", type=str,
                            help="""
                            Spécifique à larchitecture ResNet. '
                            Voici les block_units size utilisé et le width_factor par Google:
                            BiT-M-R50x1, [3, 4, 6, 3], 1
                            BiT-M-R50x3, [3, 4, 6, 3], 3
                            BiT-M-R101x1, [3, 4, 23, 3], 1
                            BiT-M-R101x3, [3, 4, 23, 3], 3
                            BiT-M-R152x2, [3, 8, 36, 3], 2
                            BiT-M-R152x4, [3, 8, 36, 3], 4
                            BiT-S-R50x1, [3, 4, 6, 3], 1
                            BiT-S-R50x3, [3, 4, 6, 3], 3
                            BiT-S-R101x1, [3, 4, 23, 3], 1
                            BiT-S-R101x3, [3, 4, 23, 3], 3
                            BiT-S-R152x2, [3, 8, 36, 3], 2
                            BiT-S-R152x4, [3, 8, 36, 3], 4""")
        parser.add_argument("--wf", default=1, type=int, choices=[1, 2, 3], help='Voir block_units')
        return parser


class PrintLayer(nn.Module):
    """
    Layer à utiliser dans votre architecture pour débug.
    Print simplement la taille. Permet de détecter l'endroit ou les couches de neurones n'ont pas la même grandeur.
    """
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        print(x.size())
        return x

# Dictionnaire des architectures disponibles. Ceci est utilisé par le argparser et
# les nouvelles arch doivent y être ajoutés.
ARCHS = {"VGG16": VGG16, "VGG11": VGG11, "ResNetV2": ResNetV2}
