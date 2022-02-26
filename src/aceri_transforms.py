"""
Module contenant les transformations possibles des données disponibles.
Contient autant les transformations obligatoires, telle que le minmax, mais aussi les augmentations dew données.

Il y a une section filtre qui peut être utilisé dans le module datamodule,
mais ceux-ci pourraient être simplement des commandes SQL lorsque cette fonctionnalité sera implémentée.

"""
from argparse import ArgumentParser
from typing import Union, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class DataAugmenter:
    """
    Classe permettant d'augmenter les données en faisant chaque augmentation dans le bon ordre.
    """
    train_only = True

    def __init__(self, laplace_scale=1.5,
                 rel_gaussian_scale=0.05,
                 std_dev_spectra_multiplier=0.1,
                 **kwargs):
        self.laplace_scale = laplace_scale
        self.rel_gaussian_scale = rel_gaussian_scale
        self.std_dev_spectra_multiplier = std_dev_spectra_multiplier

    def __call__(self, x: list):
        x[1] = self.add_relative_gaussian_noise(x[1])
        x[0] = self.multiply_spectra(x[0])
        x[0] = self.add_spectra_random_x_shift(x[0])
        x[0] = self.add_relative_gaussian_noise(x[0])
        return x

    def multiply_spectra(self, x: np.ndarray):
        """
        Facteur multiplicatif de chaque spectre de chaque échantillon.
        """
        x = x * np.random.normal(loc=1, scale=self.std_dev_spectra_multiplier, size=x.shape[0:-1])[..., np.newaxis]
        return x

    def add_relative_gaussian_noise(self, x: np.ndarray):
        """
        Bruit gaussien relatif à l'intensité du pixel.
        """
        x = x * np.random.normal(loc=1, scale=self.rel_gaussian_scale, size=x.shape)
        return x

    def add_spectra_random_x_shift(self, x: np.ndarray):
        """
        Shift sur l'axe des x selon une distribution Laplacienne.
        """
        if x.ndim == 3:
            x_shift_array = np.random.laplace(scale=self.laplace_scale, size=x.shape[0:2]).astype(int)
            for i in range(x.shape[0]):
                for j in range(x[i].shape[0]):
                    x_shift = x_shift_array[i, j]
                    if x_shift > 0:
                        x[i, j, 0:-x_shift] = x[i, j, x_shift:]
                        x[i, j, -x_shift:] = 0
                    elif x_shift < 0:
                        x[i, j, -x_shift:] = x[i, j, 0:x_shift]
                        x[i, j, 0:-x_shift] = 0
        elif x.ndim == 2:
            x_shift_array = np.random.laplace(scale=self.laplace_scale, size=x.shape[0]).astype(int)
            for i in range(x.shape[0]):
                x_shift = x_shift_array[i]
                if x_shift > 0:
                    x[i, 0:-x_shift] = x[i, x_shift:]
                    x[i, -x_shift:] = 0
                elif x_shift < 0:
                    x[i, -x_shift:] = x[i, 0:x_shift]
                    x[i, 0:-x_shift] = 0
        return x

    @staticmethod
    def add_transform_specific_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--da_xshift_scale", default=1.5, type=float, nargs=1,
                            help="Facteur de grandeur B de la distribution laplacienne utilisée pour le x shift. "
                                 "Défaut: 1.5")
        parser.add_argument("--da_rel_noise_scale", default=0.05, type=float, nargs=1,
                            help="Écart-type de la distribution gaussienne utilisée pour le bruit relatif. "
                                 "Défaut: 0.05")
        parser.add_argument("--da_multiplier_scale", default=0.1, type=float, nargs=1,
                            help="Écart-type de la distribution gaussienne utilisée pour la multiplication de "
                                 "l'intensité des données spectrales. Défaut: 0.1")
        return parser


class SimulateNewFluorimeter:
    """
    Classe utilisée lorsque Simon voulait tester les capacités du nouveaux fluorimètre.
    Cette classe transforme les données pour qu'elle est le même format que ce que le nouveau fluorimètre donnerait.
    """
    train_only = False

    def __init__(self, channels: tuple = ("277", "380", "425")):
        super().__init__()
        assert all(channels[i] in ["277", "380", "425", "lspr"] for i in range(4))
        assert any(channels[i] in ["277", "380", "425"] for i in range(3))
        self.num_channels = len(channels)
        self.channels = channels
        self.cut_from = {"277": 45, "380": 113, "425": 158, "lspr": None}

    def __call__(self, x):
        for i, channel in enumerate(self.channels):
            x[0][:, i, 0:self.cut_from[channel]] = 0
        return x

    @staticmethod
    def add_transform_specific_args(parent_parser: ArgumentParser):
        return parent_parser


class ChannelFilterMerger:
    """
    Classe qui prend le dataframe pour le transformer en un numpy avec les channels désirés dans le bon ordre.
    Peu être utilisé de façon itérative ici, donc par batch.
    """
    train_only = False

    def __init__(self, channels_to_keep: tuple = ("277", "380", "425")):
        super().__init__()
        self.channels_to_keep = channels_to_keep
        self.num_channels = len(channels_to_keep)
        assert all(channel in ["277", "380", "425", "lspr"] for channel in channels_to_keep)

    def __call__(self, data):
        spectrum = []
        lspr_index = None
        for channel in self.channels_to_keep:
            spectrum.append([col for col in data.columns if f'{channel}_' in col])

        fluo_axis = list(range(295, 741))
        spectra = np.zeros((data.shape[0], self.num_channels, len(fluo_axis)))

        if "lspr" in self.channels_to_keep:
            lspr_index = self.channels_to_keep.index("lspr")
            lspr_axis = list(range(450, 701))
            lspr_start = fluo_axis.index(lspr_axis[0])
            lspr_end = fluo_axis.index(lspr_axis[-1])
            spectra[:, lspr_index, lspr_start:(lspr_end + 1)] = data[spectrum[lspr_index]].values

        for i, spectr in enumerate(spectrum):
            if i == lspr_index:
                continue
            spectra[:, i, :] = data[spectr].values

        return spectra


class IterativeMinMaxScaler:
    """
    Classe qui prend le tuple x_spetra et x_tabular pour le minmax scale.
    Peux être fit complètement à l'initialisation.
    Peut aussi ne pas être initialisé et fitté partiellement par batch.
    Le résultat va être le même passé la 1e époch.
    """
    train_only = False

    def __init__(self, fit_with_this: Union[tuple, list, np.ndarray] = None):
        """
        if train_x or scalers_params are not given at initialization, will do a partial fit every train batch
        scalers_params take priority over train_x for fitting.
        :param fit_with_this:
        """
        self.was_fitted = False
        self.num_sources = None
        self.flatten_size = []
        self.scalers = []

        if fit_with_this is not None:
            if isinstance(fit_with_this, np.ndarray):
                fit_with_this = [fit_with_this]

            self.num_sources = len(fit_with_this)

            for data_source in fit_with_this:
                scaler = MinMaxScaler((-1, 1))
                flattened = self._flatten(data_source)
                self.flatten_size.append(flattened.shape[1])
                scaler.fit(flattened)
                self.scalers.append(scaler)

            self.was_fitted = True

    def __call__(self, x: list, train: bool = None, squeezed: bool = True):
        # looks like each transform is called once per sample, so dims are reduced by 1
        # if nothing was given at __init__
        if self.num_sources is None:
            self.num_sources = len(x)
            for _ in range(self.num_sources):
                self.scalers.append(MinMaxScaler((-1, 1)))

        for source in range(self.num_sources):
            if not self.was_fitted:
                flattened = self._flatten(x[source])
                self.flatten_size.append(flattened.shape[1])
            og_shape = x[source].shape
            x[source] = self._flatten_known(x[source], self.flatten_size[source])
            x[source] = self._flatten_known(x[source], self.flatten_size[source])
            if not self.was_fitted and train:
                # if no initial fit, iterative fit is done
                self.scalers[source].partial_fit(x[source])
            x[source] = self.scalers[source].transform(x[source])
            x[source] = x[source].reshape(og_shape)
        return x

    @staticmethod
    def _flatten(x):
        return x.reshape(x.shape[0], -1)

    @staticmethod
    def _flatten_known(x, flat_shape):
        return x.reshape(-1, flat_shape)

    @staticmethod
    def _flatten_single(x):
        return x.reshape(1, -1)


#  filters here
#  ---------------------------------------------------------------------------------------------------------------------


class FilterByTransmittance:
    """
    Permet de filtrer en fonction d'un cutoff de transmittance.
    Est une classe pour conserver les paramètre lors de l'entrainement.
    """
    def __init__(self, cutoff: int = 0, keep_above: bool = True):
        self.cutoff = cutoff
        self.keep_above = keep_above

    def __call__(self, df: pd.DataFrame):
        if self.keep_above:
            return df[df.loc["greffon_transmittance"] >= self.cutoff]
        else:
            return df[df.loc["greffon_transmittance"] <= self.cutoff]


class FilterNotVR1:
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = np.ones_like(df["greffon_defaut_type"] != 1)
        one_mask = np.array(df["greffon_defaut_type"] != 1)
        mask = mask & one_mask
        return df[mask]


class FilterIntegrationOOB:
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:

        if "integration_oob" in df.columns:
            return df.loc[df["integration_oob"] != 1]
        else:
            return df


class FilterByDate:

    def __init__(self, remove_before=None, remove_after=None):
        if remove_before is not None:
            self.remove_before = pd.Timestamp(remove_before)
        else:
            self.remove_before = None
        if remove_after is not None:
            self.remove_after = pd.Timestamp(remove_after)
        else:
            self.remove_after = None

    def __call__(self, df: pd.DataFrame):
        if "mesure_data" in df.columns:
            if self.remove_before is not None:
                df = df.loc[df["mesure_date"].astype("datetime64[ns]") > self.remove_before]
            if self.remove_after is not None:
                df = df.loc[df["mesure_date"].astype("datetime64[ns]") < self.remove_after]
        return df
