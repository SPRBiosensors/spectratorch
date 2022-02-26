from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
import torch

from typing import Union
from torch.utils.data import Dataset
from aceri_transforms import ChannelFilterMerger, IterativeMinMaxScaler
from src import aceri_transforms

PATH = Path(__file__).resolve().parents[1]


class AceriDataset(Dataset):
    """
    Wraps pytorch's Dataset class to take into account that you might want different classifcation targets.
    MinMaxScaler implemented direction here through IterativeMinMaxScaler
    """

    def __init__(self,
                 source_file_path: Union[str, Path] = None,
                 target_codes: Union[tuple, list] = ("OVA", "OK"),
                 channels: Union[tuple, list] = ("277", "380", "425"),
                 transforms=None,
                 target_transforms=None,
                 filters=None,
                 verbose=False,
                 **kwargs):

        super().__init__()
        self.verbose = verbose
        self.channels = channels
        self.num_classes = None
        self.labels = None
        self.targets = None
        self.scalers = None
        self.num_channels = len(self.channels)

        self.target_codes = target_codes
        self.filters = filters

        # file access logic
        if source_file_path is None:
            source_file_path = PATH / "datasets" / "30k_cleaned.pkl"
            data = pd.read_pickle(source_file_path)
        else:
            source_file_path = Path(source_file_path)
            assert source_file_path.is_file()
            if ".pkl" in source_file_path.suffix:
                data = pd.read_pickle(source_file_path)
            elif source_file_path.suffix == ".csv":
                data = pd.read_csv(source_file_path)
            else:
                raise NotImplementedError

        if self.filters is not None:
            for filtre in self.filters:
                data = filtre(data)

        # classification targets logic
        self.__set_targets(data)

        self.data_merger = ChannelFilterMerger(channels)
        # if dataset become iterative, will need to be done on the fly in __iterate__, not in __init__
        self.x = [self.data_merger(data), data[["greffon_transmittance"]].values]
        self.tab_size = 1  # todo, right now assume that only transmittance is useful

        # instanciate all transforms with custom params
        if transforms is not None:
            for transform in transforms:
                transforms[transform] = transforms[transform](**kwargs)
        if target_transforms is not None:
            for target_transform in target_transforms:
                target_transforms[target_transform] = target_transforms[target_transform](**kwargs)

        self.transforms = transforms
        self.target_transforms = target_transforms

    def __len__(self):
        return self.ids.shape[0]

    def __getitem__(self, idx):
        "Retourne x sous forme de liste detenseur et y sous forme de dictionnaire"
        x = [self.x[0][idx], self.x[1][idx]]
        y_dict = {"targets": np.array(self.targets[idx]),
             "ids": self.ids[idx]}
        if self.transforms is not None:
            for transform in self.transforms:
                x = self.transforms[transform](x)
        if self.target_transforms is not None:
            for target_transform in self.target_transforms:
                y_dict = self.target_transforms[target_transform](y_dict)

        # minmax scaling hardcoded
        x = self.scalers(x)
        # to tensor is absolutly needed, not a transform
        for key in range(len(x)):
            x[key] = torch.from_numpy(x[key]).float()
        y_dict["targets"] = torch.from_numpy(y_dict["targets"]).long()

        return x, y_dict

    def scale_with(self, this: Union[list, np.ndarray, type(IterativeMinMaxScaler)]):  # N, C, H
        if isinstance(this, list):
            if isinstance(this[0], int):
                x = [self.x[0][this], self.x[1][this]]
                self.scalers = IterativeMinMaxScaler(fit_with_this=x)
            elif isinstance(this[0], torch.Tensor):
                self.scalers = IterativeMinMaxScaler(fit_with_this=this)
            else:
                raise ValueError

        if isinstance(this, np.ndarray):
            x = [self.x[0][this], self.x[1][this]]
            self.scalers = IterativeMinMaxScaler(fit_with_this=x)

        elif isinstance(this, (aceri_transforms.IterativeMinMaxScaler, IterativeMinMaxScaler)):
            self.scalers = this

        else:
            raise ValueError

    def __set_targets(self, data): #todo redo this
        if self.verbose:
            print("Setting targets...")

        self.intensities = data["greffon_defaut_severite"].replace('OK - Inspecteur', "OK").replace('OK - SpectrAcer', "OK")
        assert all(self.intensities.loc[self.intensities.str.contains("OK")] == "OK")
        self.intensities = self.intensities.values

        self.types = data["greffon_defaut_type"].values.astype(str)
        self.full_truth = data["greffon_defaut_severite"]
        self.full_truth = self.full_truth.replace('OK - Inspecteur', "OK")
        self.full_truth = self.full_truth.replace('OK - SpectrAcer', "OK") + data["greffon_defaut_type"].astype(str)
        self.ids = data.index.values




        intensities_enc = LabelEncoder()
        types_enc = LabelEncoder()
        full_truth_enc = LabelEncoder()

        encoded_intensities = intensities_enc.fit_transform(self.intensities)
        encoded_full_truth = full_truth_enc.fit_transform(self.full_truth)
        encoded_types = types_enc.fit_transform(self.types)
        if self.verbose:
            print("\nIntensities found:")
            print(*intensities_enc.classes_, sep=", ")

            print("\nTypes found: ")
            print(*types_enc.classes_, sep=", ")

            print("\nCombo found:")
            print(*full_truth_enc.classes_, sep=", ")

        if self.target_codes[0] == "AVA":
            if self.target_codes[1] == "TY":
                if self.verbose:
                    print("Sorting by types")
                self.num_classes = len(types_enc.classes_)
                self.labels = types_enc.classes_
                self.targets = encoded_types

            elif self.target_codes[1] == "TYs":
                if self.verbose:
                    print("Sorting by simplified types")
                simp_types = self.types.astype("|S1").astype(str)  # shorten to remove second digit (11 -> 1)...
                encoded_types = types_enc.fit_transform(simp_types)
                self.num_classes = len(types_enc.classes_)
                self.labels = types_enc.classes_
                self.targets = encoded_types

            elif self.target_codes[1] == "IN":
                if self.verbose:
                    print("Sorting by intensities")
                self.labels = intensities_enc.classes_
                self.num_classes = len(intensities_enc.classes_)
                self.targets = encoded_intensities

            elif self.target_codes[1] == "INs":
                if self.verbose:
                    print("Sorting by simplified intensities")
                self.num_classes = 2
                self.labels = ["VR - NC", "OK - CROCHET"]
                squeezed = self.__squeeze_classes(encoded_intensities, intensities_enc.classes_,
                                                  [["VR", "NC"], ["OK", "CROCHET"]])
                self.targets = squeezed

            elif self.target_codes[1] == "INTY":
                if self.verbose:
                    print("Sorting by intensities and types, exploded together")
                self.labels = full_truth_enc.classes_
                self.num_classes = len(self.labels)
                self.targets = encoded_full_truth

            elif self.target_codes[1] == "INsTY":
                if self.verbose:
                    print("Sorting by simplified intensities and types, exploded together")
                raise NotImplementedError("INsTY not implemented, is the same as instys")

            elif self.target_codes[1] == "INTYs":
                if self.verbose:
                    print("Sorting by intensities and simplified types, exploded together")
                raise NotImplementedError("INTYs not implemented")

            elif self.target_codes[1] == "INsTYs":
                if self.verbose:
                    print("Sorting by simplified intensities and simplified types, exploded together")
                mask_OK = self.intensities == "OK"
                mask_CROCHET = self.intensities == "CROCHET"
                simp_types = self.types.astype("|S1").astype(str)
                simp_types[mask_CROCHET] = "CROCHET"
                simp_types[mask_OK] = "OK"
                encoded_types = types_enc.fit_transform(simp_types)
                self.num_classes = len(types_enc.classes_)
                self.labels = types_enc.classes_
                self.targets = encoded_types

            elif self.target_codes[1] == "OK+11+5":
                if self.verbose:
                    print("Sorting by OK, VR11, VR/NC5 and the rest")
                mask_OK = self.intensities == "OK"
                mask_11 = self.types == "11"
                mask_5 = self.types == "5"
                mask_rest = ~mask_5 & ~mask_11 & ~mask_OK
                self.targets = np.array([[mask_OK], [mask_11], [mask_5], [mask_rest]])
                self.labels = ["OK", "VR11", "VR/NC5", "The rest"]
                self.num_classes = len(self.labels)
                self.targets = encoded_full_truth

            else:
                raise ValueError("Target codes were not understood: target_codes={}".format(self.target_codes))

        elif self.target_codes[0] == "OVA":
            if self.target_codes[1] in [str(i) for i in types_enc.classes_]:
                if self.verbose:
                    print("Sorting {} against all the rest of the types".format(self.target_codes[1]))
                binarizer = LabelBinarizer()
                if self.target_codes[1] == "1":
                    self.types = self.types.astype("|S1").astype(str)
                bined = binarizer.fit_transform(self.types)
                self.num_classes = 2
                if self.verbose:
                    print(type(binarizer.classes_[0]))
                self.labels = ["The rest", self.target_codes[1]]
                self.targets = bined[:, list(binarizer.classes_).index(self.target_codes[1])]

            elif self.target_codes[1] in intensities_enc.classes_:
                if self.verbose:
                    print("Sorting {} against all the rest of the intensities".format(self.target_codes[1]))
                binarizer = LabelBinarizer()
                bined = binarizer.fit_transform(self.intensities)  # actual intensities not encoded
                self.num_classes = 2

                self.labels = ["The rest", self.target_codes[1]]
                self.targets = bined[:, list(binarizer.classes_).index(self.target_codes[1])]

            elif self.target_codes[1] in full_truth_enc.classes_:
                if self.verbose:
                    print("Sorting {} against all the rest of the combos".format(self.target_codes[1]))
                binarizer = LabelBinarizer()
                bined = binarizer.fit_transform(self.full_truth)
                self.num_classes = 2
                self.labels = ["The rest", self.target_codes[1]]
                self.targets = bined[:, list(binarizer.classes_).index(self.target_codes[1])]

            else:
                raise ValueError("Target codes were not understood: {}".format(self.target_codes))
        else:
            raise ValueError("Target codes were not understood: {}".format(self.target_codes))

    @staticmethod
    def __squeeze_classes(unsqueezed, current_labels, labels_goal):
        current_labels = current_labels.tolist()
        squeezed = np.empty_like(unsqueezed)
        for combination in labels_goal:
            for label in current_labels:
                if label in combination:
                    squeezed[unsqueezed == current_labels.index(label)] = labels_goal.index(combination)
        return squeezed
