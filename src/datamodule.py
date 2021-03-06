from argparse import ArgumentParser
from pathlib import Path
from collections import OrderedDict
from typing import Union

from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
import numpy as np

from aceri_transforms import DataAugmenter, SimulateNewFluorimeter, FilterIntegrationOOB, \
    FilterByDate
from dataset import AceriDataset
from parser_sanitizers import data_t_2_odict, target_t_2_odict, channels_2_tuple

AVAILABLE_TRANSFORMS = {"da": DataAugmenter, "simulate": SimulateNewFluorimeter}
AVAILABLE_TARGET_TRANSFORMS = {}
# AVAILABLE_FILTERS = {"integration": filter_integration_oob}
KNOWN_CHANNELS = ["277", "380", "425", "lspr"]


class AceriDataModule(pl.LightningDataModule):

    def __init__(self, target_codes: Union[tuple, list],
                 channels: tuple,
                 bs: int,
                 seed: int,
                 transforms: OrderedDict,
                 target_transforms: OrderedDict,
                 current_split: int,
                 n_splits: int,
                 test_source: str,
                 train_source: str,
                 num_workers: int = 1,
                 shuffle: bool = True,
                 pin_memory: bool = True,
                 drop_last: bool = False,
                 **kwargs):
        super().__init__()
        # self.save_hyperparameters() i wish, self.my_hparams est un workaround pour enregistrer les hyperparametres du
        # datamodule dans le fichier tensorboard, rien d'important.

        self.target_codes = target_codes
        self.channels = channels
        self.bs = bs
        self.seed = seed
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.current_split = current_split
        self.n_splits = n_splits
        self.train_source = train_source
        self.test_source = test_source
        self.num_channels = len(channels)

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

        self.scalers = None

        self.num_classes = None
        self.labels = None
        self.tab_size = None

        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory

        self.my_hparams = {'target_codes': self.target_codes,
                           'bs': self.bs,
                           'seed': self.seed,
                           'transforms': list(self.transforms.keys()),
                           'n_splits': self.n_splits,
                           'current_split': self.current_split}

    def prepare_data(self):
        """
        M??thode appel??e sur un seul worker python, ne pas sauvegarder d'??tats de variables ici, ex:
         x = fonction(y)....
         Permet de t??l??charger la base de donn??e ou faire une requete SQL, ex:
         dataset.sql_grab(filepath=Path.cwd() / "dataset", date=[20180101, 20201220])
        """

    def setup(self, stage="fit"):
        """
        M??thode appel??e sur tous les worker, autant pour parral??liser le travail que
        pour qu'ils aient tous acc??s aux ??tats d??finis dans cette m??thode.
        Le dataset est travaill?? ici, le nombre de classe d??fini, la taille des donn??es....
        Ces informations sont ensuite utilis??es par le mod??le (lightningmodule) pour initier le r??seau neuronal
        de la bonne taille.

        Chaque portion du dataset (train, val, test) font ensuite appel au dataset entier.
        Le dataset est donc en m??moire deux fois pendant l'entrainement et une autre fois ?? la fin.
        La raison est que les transformations appliqu??es aux donn??es sont tous diff??rentes et une fois initi??,
        les dataloader ne peuvent pas communiquer entre eux.
        Si tu essai de communiquer plutot les transformations
        ?? faire pour sauver de la m??moire, il va avoir des fuite de VRAM.

        Si le dataset est trop gros pour ??tre charg?? en m??moire:
        Voir pd.read_csv(<filepath>, chunksize=<your_chunksize_here>)
        Voir la class IterativeDataset de pytorch
        Faire le train-test split d'avance

        """
        if stage == "fit" or stage is None:
            self.train_ds = self._get_dataset(self.train_source)

            self.labels = self.train_ds.labels
            self.num_classes = self.train_ds.num_classes
            self.tab_size = self.train_ds.tab_size

            train_i, val_i = self._split_dataset(self.train_ds)

            self.train_ds.scale_with(train_i)
            self.scalers = self.train_ds.scalers

            self.train_ds = Subset(self.train_ds, train_i)

            self.val_ds = self._get_dataset(self.train_source, is_train=False)
            self.val_ds.scale_with(self.scalers)
            self.val_ds = Subset(self.val_ds, val_i)

        if stage == "test" or stage is None:
            if Path(self.test_source).is_file():  # le test dataloader va utiliser self.test_ds au lieux d'un index
                self.test_ds = self._get_dataset(self.test_source, is_train=False)
                self.test_ds.scale_with(self.scalers)

            elif self.test_source == "is-val":
                self.test_ds = self._get_dataset(self.train_source, is_train=False)
                self.test_ds.scale_with(self.scalers)
                _, test_i = self._split_dataset(self.test_ds)
                self.test_ds = Subset(self.test_ds, test_i)

            else:
                raise NotImplementedError(f"--test_split must be either like-val, is-val or a file location. "
                                          f"{self.test_source} is neither, therefore not currently supported.")

    def _split_dataset(self, dataset: AceriDataset) -> tuple:
        """
        Splits the dataset into train and validation set
        """
        ds_size = len(dataset)
        skf = StratifiedKFold(n_splits=self.n_splits, random_state=self.seed, shuffle=True)
        kfold_iter = skf.split(np.arange(ds_size), dataset.full_truth)
        folds = list(kfold_iter)
        return folds[self.current_split]

    def _get_dataset(self, source_file_path, is_train: bool = True) -> AceriDataset:
        if is_train:
            transform_key = "train"
        else:
            transform_key = "val"
        return AceriDataset(source_file_path=source_file_path,
                            target_codes=self.target_codes,
                            transforms=self.transforms[transform_key],
                            target_transforms=self.target_transforms[transform_key],
                            filters=[FilterIntegrationOOB()],
                            channels=self.channels)

    def train_dataloader(self):
        return self._data_loader(self.train_ds, shuffle=self.shuffle)

    def val_dataloader(self):
        return self._data_loader(self.val_ds)

    def test_dataloader(self):
        return self._data_loader(self.test_ds)

    def _data_loader(self, dataset: AceriDataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.bs,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    @staticmethod
    def add_datamodule_specific_args(parent_parser: ArgumentParser):
        """
        Param??tres par d??fauts du dataloader.
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # dataset related
        parser.add_argument("--target_codes",
                            default="OVA OK".split(),
                            nargs=2,
                            metavar="[OVA, AVA] [cat??gorie]",
                            type=str,
                            help=f"Indique qulles sont les cat??gories ?? classifier et de quelle mani??re. "
                                 f"Voir le code de dataset.py pour celles pr??te ?? ??tre utilis??es. D??faut: OVA OK")
        parser.add_argument("--channels",
                            default=("277", "380", "425"),
                            metavar="[channel] [channel] ...",
                            nargs="?",
                            type=channels_2_tuple,
                            help=f"Les sources de signal ?? utiliser dans l'architecture. L'ordre est pris en compte. "
                                 f"Sources disponibles: {KNOWN_CHANNELS}. D??faut: 277 380 425")
        parser.add_argument("--transforms",
                            default="da",
                            metavar="[transform] [transform] ...",
                            nargs="?",
                            type=data_t_2_odict,
                            help=f"Les transformations de donn??es utilis??es. Peut ??tre None. L'ordre est important. "
                                 f"Transformation disponibles: {AVAILABLE_TRANSFORMS}. D??faut: da")
        parser.add_argument("--target_transforms",
                            default="None",
                            metavar="[transform] [transform] ...",
                            nargs="?",
                            type=target_t_2_odict,
                            help=f"Les transformations d'??tiquettes utilis??es. Peut ??tre None. L'ordre est important. "
                                 f"Transformation disponibles: {AVAILABLE_TARGET_TRANSFORMS}. D??faut: None")
        # dataloader related
        parser.add_argument("--n_splits",
                            default=5,
                            type=int,
                            metavar="[3,n]",
                            help="Le nombre de division du dataset ?? faire pour le train-val split "
                                 "(et test si d??sir?? en plus de la validation crois??e). D??faut: 5"),
        parser.add_argument("--current_split",
                            default=0, type=int,
                            metavar="[0,n-1]",
                            help="La division pr??sentement utilis??e pour cet entrainement. D??faut: 0"),
        parser.add_argument("--train_source",
                            default=None,
                            type=Path,
                            metavar="[chemin de fichier]",
                            help="Le fichier source du train dataset utilis??. D??faut: raw 30k.pkl")

        parser.add_argument("--test_source",
                            default="is-val",
                            type=str,
                            metavar="[is-val, like-val, chemin de fichier]",
                            help="Le type de division test utilis??e. "
                                 "is-val: Agis comme une validation crois??e en assumant qu'il n'y a pas de test."
                                 "like-val: Fait une division test aussi grosse que val (ex avec n_splits=5: 60:20:20)."
                                 " Un chemin de fichier peut aussi ??tre amen?? pour faire une comparaison avec un set de"
                                 " donn??e externe sp??cifique.")
        parser.add_argument("--num_workers",
                            default=1,
                            type=int,
                            metavar="[n]",
                            help="Le nombre de travailleur python qui vont "
                                 "travailler en parral??le sur le data processing")

        #  transforms related args
        for transform in (list(AVAILABLE_TRANSFORMS.values()) + list(AVAILABLE_TARGET_TRANSFORMS.values())):
            parser = transform.add_transform_specific_args(parser)

        return parser
