import numpy as np
import scipy.io as sio
import h5py
from typing import Dict
from pathlib import Path
from numbers import Number
from collections import namedtuple

import torch
import torch.nn as nn

from datasets.hsi_dataset import HSIDataset, CachedHSIDataset


Datasets = namedtuple('Datasets', ['train', 'val', 'test'])


class HSIDatasetBuilder:
    """
    Base Dataset Builder for Hyper-spectral Image (HSI) Classification
    Args:
        root (str or pathlib.Path): Root directory of the meta hsi dataset files.
        filenames (Dict): Dict of the data filename and gt filename.
        train (float, optional): The proportion of training samples.
            notice: the quantity of training samples, not a decimal meant proportion.
        val (float, optional): The proportion of valid samples.
            notice: the quantity of valid samples, not a decimal meant proportion.
        seed (int, optional): Seed for splitting the dataset, which is independent of the seed
            set in the main progress and default to be 0.
        window_size (int, optional): The size of the square window for each sample.
            notice: should be a singular number.
        use_edge (bool, optional): Whether to use samples at edge whose window will exceed the
            image edge.
        dataset_name (str, optional): The name of the HSI dataset.
        num_classes (int, optional): The number of categories in the HSI dataset.
            If the arg is not given, it will be obtained by the builder. Although it is given,
            the builder will take a check.
        num_bands (int, optional): The number of spectral bands of the samples in the HSI dataset.
            Same to the num_samples, the builder can obtain and check.
    Usage (take IndianBuilder as an example):
        1. Init a builder:
            builder = IndianBuilder(seed=1)  # require kwarg
        2. Generate datasets:
        The builder support two building modes
        generator mode (HSIDataset) or cache mode (CachedHSIDataset).
            generator_datasets = builder.get_datasets()  # default
            cache_datasets = builder.get_datasets(cache_mode=True)  # require kwarg
            cache_train_set = cache_datasets.train
            cache_train_set = builder.get_train_dataset(cache_mode=True)
    """

    def __init__(
            self,
            root,
            filenames: Dict,
            train: Number = 200,
            val: Number = 50,
            seed: int = 0,
            window_size: int = 27,
            use_edge: bool = False,
            dataset_name: str = 'hsi',
            num_classes: int = None,
            num_bands_HSI: int = None,
            num_bands_LiDAR: int = None
    ):

        self.root = root if isinstance(root, Path) else Path(root)
        self.split_settings = {'train': train, 'val': val,
                               'window_size': window_size, 'use_edge': use_edge}
        self.seed = seed
        self.dataset_name = dataset_name
        self._num_classes = num_classes
        self._num_bands_HSI = num_bands_HSI
        self._num_bands_LiDAR = num_bands_LiDAR
        self._cached = False

        self.meta_path = {}
        # assert 'data' in filenames.keys() and 'gt' in filenames.keys()
        for k, filename in filenames.items():
            filename = Path(filename)
            assert '.mat' in filename.suffix, NotImplementedError(
                f'`{Path(filename).suffix}` is not supported yet, please load `.mat` file')
            self.meta_path[k] = self.root / filename

        if self.num_classes is None or self.num_bands is None:
            self.load_meta_dataset()

        dataset_name, seed = self.dataset_name, self.seed
        train, val = self.get_train_val()

        self.cache_path = {
            f'{dataset_type}_{file_type}':
                self.root / f'{dataset_name}' /
                f'{dataset_name}_t{train}_v{val}_{seed}_{dataset_type}_{file_type}.npy'
            for dataset_type in ('train', 'val', 'test') for file_type in ('data', 'gt')}

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def num_bands(self):
        return self._num_bands_HSI, self._num_bands_LiDAR

    @property
    def cached(self):
        return self._cached

    def get_datasets(self, cache_mode: bool = False,
                     return_dict: bool = False, **kwargs):
        dataset_dict = {
            'train': self.get_train_dataset(cache_mode=cache_mode, **kwargs),
            'val': self.get_val_dataset(cache_mode=cache_mode, **kwargs),
            'test': self.get_test_dataset(cache_mode=cache_mode, **kwargs)}
        return dataset_dict if return_dict else Datasets(**dataset_dict)

    def load_splitted_dataset(self, dataset_type: str,
                              cache_mode: bool = False, **kwargs):
        if cache_mode:
            self.get_datasets_cache()
            data_path = self.cache_path[f'{dataset_type}_data']
            gt_path = self.cache_path[f'{dataset_type}_gt']
            print(f'Build {dataset_type} dataset of cache style.')
            return CachedHSIDataset(data_path, gt_path, **kwargs)
        else:
            data_path_HSI = self.meta_path['data_HSI']
            data_path_LiDAR = self.meta_path['data_LiDAR']
            gt_path = self.meta_path['gt']
            info_path = self.get_split_info(**self.split_settings)
            print(f'Build {dataset_type} dataset of generator style.')
            return HSIDataset(data_path_HSI,data_path_LiDAR, gt_path, info_path, dataset_type, **kwargs)

    def get_split_info(self, gt=None, train=200, val=50, window_size=27, use_edge=False):
        info_path = self.root / f'{self.dataset_name}' / \
                    f'{self.dataset_name}_t{train}_v{val}_{self.seed}_split_info.mat'

        if not info_path.exists():
            gt = self.load_meta_dataset()[2] if gt is None else gt
            # Get proposal coords
            coords = self.get_proposal_coords(gt, window_size, use_edge)
            # Generate rand permutation for splitting dataset
            num_samples = len(coords)
            # TODO: The same num_cache cannot get the same RandPerm
            # RandPerm = np.random.permutation(num_samples)
            RandPerm = torch.randperm(
                num_samples, dtype=torch.int,
                generator=torch.Generator().manual_seed(self.seed),
            ).numpy()
            info_path.parent.mkdir(exist_ok=True)

            sio.savemat(info_path, {'coords': coords, 'RandPerm': RandPerm,
                                    'window_size': window_size, 'use_edge': int(use_edge),
                                    'train': train, 'val': val})
            print(f'Build split_info file at {info_path}.')

        return info_path

    def get_datasets_cache(self):
        """prepare dataset of cache mode"""
        if not self.cached:
            dataset_name, seed = self.dataset_name, self.seed
            is_cache_exist = {k: path.exists() for k, path in self.cache_path.items()}
            if all(is_cache_exist.values()):
                print(f'Loading {dataset_name} dataset from the seed={seed} cache data.')
            elif not any(is_cache_exist.values()):
                print(f'Splitting with seed={self.seed} for {dataset_name} dataset.')
                self.split_hsi_dataset(**self.split_settings)
            else:
                raise FileExistsError(
                    f'The existence status of an exception for data of seed={seed}: \n'
                    f'{is_cache_exist}')
        self._cached = True

    def split_hsi_dataset(self, train=200, val=50, window_size=27, use_edge=False):
        # Load meta dataset
        data, gt = self.load_meta_dataset()
        data = 1 * ((data - np.min(data)) / (np.max(data) - np.min(data)))
        data = data.transpose([2, 0, 1])  # H, W, C --> C, H, W

        info_path = self.get_split_info(gt, train, val, window_size, use_edge)

        splitted_datasets = self.split_with_info(data, gt, info_path)
        self.save_splitted_cache(splitted_datasets)

    @staticmethod
    def split_with_info(data, gt, info_path):
        C, H, W = data.shape  # Channel, Height, Width
        info = sio.loadmat(info_path)
        coords = info['coords']
        RandPerm = info['RandPerm'].squeeze()
        window_size = info['window_size'].item()
        train = info['train'].item()
        val = info['val'].item()
        coords_x, coords_y = coords[:, 0], coords[:, 1]

        # window size
        assert window_size % 2 == 1, ValueError(
            f"The window_size {window_size} ought to be a singular number")
        half_window = window_size // 2

        # Calculate numbers of samples in splitted subsets
        num_samples = len(coords)
        trainval = train + val
        if trainval < 1:
            train = int(num_samples * train)
            val = int(num_samples * val)
        elif train % 1 == 0 and val % 1 == 0:
            assert trainval >= 1
        test = num_samples - trainval
        print(f"total: {num_samples} | "
              f"train: {train}({100. * train / num_samples:.2}%) | "
              f"val {val}({100. * val / num_samples:.2}%) | "
              f"test {test}({100. * test / num_samples:.2}%)")

        gt = gt - 1
        output = {}
        output['train_data'] = np.zeros([train, C, window_size, window_size], dtype=np.float32)
        # output['train_spectral_data'] = np.zeros([train, C_spectral, half_window_spectral, half_window_spectral], dtype=np.float32)
        output['train_gt'] = np.zeros([train], dtype=np.int64)
        output['val_data'] = np.zeros([val, C, window_size, window_size], dtype=np.float32)
        # output['val_spectral_data'] = np.zeros([val, C_spectral, half_window_spectral, half_window_spectral], dtype=np.float32)
        output['val_gt'] = np.zeros([val], dtype=np.int64)
        output['test_data'] = np.zeros([test, C, window_size, window_size], dtype=np.float32)
        # output['test_spectral_data'] = np.zeros([test, C_spectral, half_window_spectral, half_window_spectral], dtype=np.float32)
        output['test_gt'] = np.zeros([test], dtype=np.int64)
        for i in range(train):
            output['train_data'][i, :, :, :] = \
                data[:,
                coords_x[RandPerm[i]] - half_window: coords_x[RandPerm[i]] + half_window + 1,
                coords_y[RandPerm[i]] - half_window: coords_y[RandPerm[i]] + half_window + 1]

            # output['train_spectral_data'][i, :, :, :] = \
            #     data[:,
            #     coords_x[RandPerm[i]] - half_window: coords_x[RandPerm[i]] + half_window + 1,
            #     coords_y[RandPerm[i]] - half_window: coords_y[RandPerm[i]] + half_window + 1]


            output['train_gt'][i] = \
                gt[coords_x[RandPerm[i]], coords_y[RandPerm[i]]].astype(np.int64)

        for i in range(val):
            output['val_data'][i, :, :, :] = \
                data[:,
                coords_x[RandPerm[i + train]] - half_window: coords_x[RandPerm[i + train]] + half_window + 1,
                coords_y[RandPerm[i + train]] - half_window: coords_y[RandPerm[i + train]] + half_window + 1]

            # output['val_spectral_data'][i, :, :, :] = \
            #     data[:,
            #     coords_x[RandPerm[i + train]] - half_window: coords_x[RandPerm[i + train]] + half_window + 1,
            #     coords_y[RandPerm[i + train]] - half_window: coords_y[RandPerm[i + train]] + half_window + 1]

            output['val_gt'][i] = \
                gt[coords_x[RandPerm[i + train]], coords_y[RandPerm[i + train]]].astype(np.int64)

        for i in range(test):
            output['test_data'][i, :, :, :] = \
                data[:,
                coords_x[RandPerm[i + trainval]] - half_window: coords_x[RandPerm[i + trainval]] + half_window + 1,
                coords_y[RandPerm[i + trainval]] - half_window: coords_y[RandPerm[i + trainval]] + half_window + 1]

            # output['test_spectral_data'][i, :, :, :] = \
            #     data[:,
            #     coords_x[RandPerm[i + trainval]] - half_window: coords_x[RandPerm[i + trainval]] + half_window + 1,
            #     coords_y[RandPerm[i + trainval]] - half_window: coords_y[RandPerm[i + trainval]] + half_window + 1]

            output['test_gt'][i] = \
                gt[coords_x[RandPerm[i + trainval]], coords_y[RandPerm[i + trainval]]].astype(np.int64)

        print("Splitting successfully.")

        return output

    def save_splitted_cache(self, output):
        for dataset_type in ('train', 'val', 'test'):
            for file_type in ('data', 'gt'):
                self.cache_path[f'{dataset_type}_{file_type}'].parent.mkdir(exist_ok=True)
                np.save(self.cache_path[f'{dataset_type}_{file_type}'],
                        output[f'{dataset_type}_{file_type}'])
        print("Data cache saved.")

    def get_proposal_coords(self, gt, window_size=27, use_edge=False):
        H, W = gt.shape

        # window size
        assert window_size % 2 == 1, ValueError(
            f"The window_size {window_size} ought to be a singular number")
        half_window = window_size // 2

        # Edge processing
        # The edge samples are referred negative samples which are not used.
        if use_edge:
            raise NotImplementedError
        else:
            mask = np.zeros([H, W])  # 0:reject  1:accept
            mask[half_window + 1: -1 - half_window + 1, half_window + 1: -1 - half_window + 1] = 1
            gt = gt * mask
        coords_x, coords_y = np.nonzero(gt)
        coords = np.stack([coords_x, coords_y], -1)

        return coords

    def load_meta_dataset(self):
        data_HSI = {k: v for k, v in sio.loadmat(self.meta_path['data_HSI']).items()
                    if isinstance(v, np.ndarray)}
        data_LiDAR = {k: v for k, v in sio.loadmat(self.meta_path['data_LiDAR']).items()
                      if isinstance(v, np.ndarray)}
        gt = {k: v for k, v in sio.loadmat(self.meta_path['gt']).items()
              if isinstance(v, np.ndarray) and 'map' not in k}
        assert len(data_HSI) == 1 and len(gt) == 1, ValueError('Description Reading the MAT file conflicts.')
        data_HSI, data_LiDAR, gt = list(data_HSI.values())[0], list(data_LiDAR.values())[0], list(gt.values())[0]
        _, _, num_bands_HSI = data_HSI.shape  # Channel, Height, Width
        if len(data_LiDAR.shape)==2:
            data_LiDAR=np.expand_dims(data_LiDAR,axis=2)
            _, _, num_bands_LiDAR = data_LiDAR.shape
        else:
            _, _, num_bands_LiDAR = data_LiDAR.shape
        if self.num_bands is not None:
            assert self.num_bands[0] == num_bands_HSI
            assert self.num_bands[1] == num_bands_LiDAR
        else:
            self._num_bands_HSI = num_bands_HSI
            self._num_bands_LiDAR = num_bands_LiDAR
        num_classes = int(np.max(gt))
        if self.num_classes is not None:
            assert self.num_classes == num_classes
        else:
            self._num_classes = num_classes
        return data_HSI, data_LiDAR, gt

    def get_train_dataset(self, **kwargs):
        return self.load_splitted_dataset('train', **kwargs)

    def get_val_dataset(self, **kwargs):
        return self.load_splitted_dataset('val', **kwargs)

    def get_test_dataset(self, **kwargs):
        return self.load_splitted_dataset('test', **kwargs)

    def get_num_classes(self):
        return self.num_classes

    def get_num_bands(self):
        return self.num_bands

    def get_train_val(self):
        s = self.split_settings
        return s['train'], s['val']

    def __repr__(self) -> str:
        _repr_indent = 4
        head = self.__class__.__name__
        body = [f"Generating seed: {self.seed} ",
                f"Split settings: {self.split_settings.items()} ",
                f"Root location: {self.root} ",
                f"Dataset info: named {self.dataset_name}, "
                f"{self.num_bands} bands, "
                f"{self.num_classes} categories"]
        body += self.extra_repr().splitlines()
        lines = [head] + [" " * _repr_indent + line for line in body]
        return '\n'.join(lines)

    def extra_repr(self) -> str:
        return ""

class CASILiDARBiloder(HSIDatasetBuilder):
    def __init__(self, root, train: Number=200, val: Number=400,
                 seed: int=None, window_size: int=27,
                 use_edge: bool=False,):
        filenames = {'data_HSI': 'data_HS_LR.mat',
                     'data_LiDAR':'data_MS_HR.mat',
                     'gt': 'Houston_label.mat'}
        dataset_name = 'data_HS_LR_DMix'
        num_classes = 15
        num_bands_HSI=144
        num_bands_LiDAR = 1
        super(CASILiDARBiloder, self).__init__(
            root, filenames, train, val, seed, window_size,
            use_edge, dataset_name, num_classes, num_bands_HSI,num_bands_LiDAR)

class IndianBuilder(HSIDatasetBuilder):
    def __init__(self, root, train: Number=200, val: Number=50,
                 seed: int=None, window_size: int=27,
                 use_edge: bool=False,):
        filenames = {'data': 'Indian_pines_corrected.mat',
                     'gt': 'Indian_pines_gt.mat'}
        dataset_name = 'Indian_pines'
        num_classes = 16
        num_bands = 200
        super(IndianBuilder, self).__init__(
            root, filenames, train, val, seed, window_size,
            use_edge, dataset_name, num_classes, num_bands)


class PaviaBuilder(HSIDatasetBuilder):
    def __init__(self, root, train: Number=200, val: Number=50,
                 seed: int=None, window_size: int=27,
                 use_edge: bool=False,):
        filenames = {'data': 'Pavia.mat',
                     'gt': 'Pavia_groundtruth.mat'}
        dataset_name = 'Pavia'
        num_classes = 9
        num_bands = 103
        super(PaviaBuilder, self).__init__(
            root, filenames, train, val, seed, window_size,
            use_edge, dataset_name, num_classes, num_bands)


class KSCBuilder(HSIDatasetBuilder):
    def __init__(self, root, train: Number=150, val: Number=50,
                 seed: int=None, window_size: int=27,
                 use_edge: bool=False,):
        filenames = {'data': 'Kennedy_denoise.mat',
                     'gt': 'KSC_gt.mat'}
        dataset_name = 'KSC'
        num_classes = 13
        num_bands = 176
        super(KSCBuilder, self).__init__(
            root, filenames, train, val, seed, window_size,
            use_edge, dataset_name, num_classes, num_bands)


class SalinasBuilder(HSIDatasetBuilder):
    def __init__(self, root, train: Number=200, val: Number=50,
                 seed: int=None, window_size: int=27,
                 use_edge: bool=False,):
        filenames = {'data': 'Salinas_corrected.mat',
                     'gt': 'Salinas_gt.mat'}
        dataset_name = 'Salinas'
        num_classes = 16
        num_bands = 204
        super(SalinasBuilder, self).__init__(
            root, filenames, train, val, seed, window_size,
            use_edge, dataset_name, num_classes, num_bands)


class CASIBuilder(HSIDatasetBuilder):
    def __init__(self, root, train: Number=200, val: Number=50,
                 seed: int=None, window_size: int=33,
                 use_edge: bool=False,):
        filenames = {'data': 'CASI.mat',
                     'gt': 'CASI_gnd_flag.mat'}
        dataset_name = 'CASI'
        num_classes = 15
        num_bands = 144
        super(CASIBuilder, self).__init__(
            root, filenames, train, val, seed, window_size,
            use_edge, dataset_name, num_classes, num_bands)


class HS_SAR_Berlin_loder(HSIDatasetBuilder):
    def __init__(self, root, train: Number=200, val: Number=400,
                 seed: int=None, window_size: int=27,
                 use_edge: bool=False):
        filenames = {'data_HSI': 'data_HS_LR.mat',
                     'data_LiDAR':'data_SAR_HR.mat',
                     'gt': 'Berlin_groundtruth.mat'}
        dataset_name = 'data_HS_LR_Dmix'
        num_classes = 8
        num_bands_HSI=244
        num_bands_LiDAR = 4
        super(HS_SAR_Berlin_loder, self).__init__(
            root, filenames, train, val, seed, window_size,
            use_edge, dataset_name, num_classes, num_bands_HSI,num_bands_LiDAR)

class MuuFL_loder(HSIDatasetBuilder):
    def __init__(self, root, train: Number=200, val: Number=1000,
                 seed: int=None, window_size: int=27,
                 use_edge: bool=False):
        filenames = {'data_HSI': 'MuUFL_hsi.mat',
                     'data_LiDAR':'MUUFL_LiDAR.mat',
                     'gt': 'MuUFL_label.mat'}
        dataset_name = 'data_HS_LR_Dmix'
        num_classes = 11
        num_bands_HSI=64
        num_bands_LiDAR = 2
        super(MuuFL_loder, self).__init__(
            root, filenames, train, val, seed, window_size,
            use_edge, dataset_name, num_classes, num_bands_HSI,num_bands_LiDAR)


class Trento_loder(HSIDatasetBuilder):
    def __init__(self, root, train: Number=200, val: Number=400,
                 seed: int=None, window_size: int=33,
                 use_edge: bool=False,):
        filenames = {'data_HSI': 'HSI.mat',
                     'data_LiDAR':'LiDAR.mat',
                     'gt': 'Trento_label.mat'}
        dataset_name = 'data_HS_LR_HT_CNN'
        num_classes = 6
        num_bands_HSI=63
        num_bands_LiDAR = 1
        super(Trento_loder, self).__init__(
            root, filenames, train, val, seed, window_size,
            use_edge, dataset_name, num_classes, num_bands_HSI,num_bands_LiDAR)

class MS_SAR_Berlin_loder(HSIDatasetBuilder):
    def __init__(self, root, train: Number=200, val: Number=400,
                 seed: int=None, window_size: int=27, use_edge: bool=False):
        filenames = {'data_HSI': 'data_MSI.mat',
                     'data_LiDAR': 'data_SAR.mat',
                     'gt': 'label.mat'}
        dataset_name = 'data_HS_LR_Dmix'
        num_classes = 17
        num_bands_HSI = 10
        num_bands_LiDAR = 4
        super(MS_SAR_Berlin_loder, self).__init__(
            root, filenames, train, val, seed, window_size,
            use_edge, dataset_name, num_classes, num_bands_HSI, num_bands_LiDAR)

class MS_SAR_HK_loder(HSIDatasetBuilder):
    def __init__(self, root, train: Number=200, val: Number=0,
                 seed: int=None, window_size: int=27, use_edge: bool=False):
        filenames = {'data_HSI': 'data_MSI.mat',
                     'data_LiDAR': 'data_SAR.mat',
                     'gt': 'label.mat'}
        dataset_name = 'data_HS_LR_Dmix'
        num_classes = 17
        num_bands_HSI = 10
        num_bands_LiDAR = 4
        super(MS_SAR_HK_loder, self).__init__(
            root, filenames, train, val, seed, window_size,
            use_edge, dataset_name, num_classes, num_bands_HSI, num_bands_LiDAR)


class WuhanBuilder(HSIDatasetBuilder):
    def __init__(self, root, train: Number = 200, val: Number = 400,
                 seed: int = None, window_size: int = 27,
                 use_edge: bool = False):
        filenames = {
            'data_HSI': 'wuhan_hsi_2000.mat',
            'data_LiDAR': 'wuhan_msi_2000.mat',
            'gt': 'wuhan_label_2000_500k.mat'
        }

        dataset_name = 'data_HS_LR_Dmix'

        num_classes = 13
        num_bands_HSI = 116
        num_bands_LiDAR = 4

        super(WuhanBuilder, self).__init__(
            root, filenames, train, val, seed, window_size,
            use_edge, dataset_name, num_classes, num_bands_HSI, num_bands_LiDAR)

    # def load_meta_dataset(self):
    #     """
    #     加载 Wuhan 数据集：仅读取 HSI 和 SAR，并对 HSI 进行上采样
    #     """
    #     data_file = self.meta_path['data_HSI']
    #     label_file = self.meta_path['gt']
    #
    #     print(f"Loading Wuhan dataset from: {data_file}")
    #
    #     # 1. 使用 h5py 读取数据
    #     with h5py.File(data_file, 'r') as f:
    #         # 读取 HSI
    #         hsi_valid = np.array(f['HSI'])
    #
    #         # 仅读取 SAR，不读取 MSI
    #         sar_final = np.transpose(f['SAR'])
    #
    #         # 2. 读取标签
    #     with h5py.File(label_file, 'r') as f:
    #         label_valid = np.transpose(f['label'])  # (H, W)
    #
    #     # 3. HSI 上采样处理 (3倍)
    #     mm = nn.Upsample(scale_factor=3, mode='nearest', align_corners=None)
    #
    #     # 转换 Tensor 进行 Upsample
    #     # h5py 读取 v7.3 mat 通常是 (C, W, H) 或类似，需要调整维度适配 nn.Upsample
    #     hsi_tensor = torch.from_numpy(hsi_valid).float()
    #     hsi_tensor = hsi_tensor.unsqueeze(0)  # (1, C, W, H)
    #
    #     # 上采样 -> (C, W_new, H_new)
    #     hsi_upsampled = mm(hsi_tensor).squeeze(0).numpy()
    #
    #     # 转置为 (H, W, C)
    #     hsi_final = np.transpose(hsi_upsampled)
    #
    #     del hsi_valid, hsi_tensor, hsi_upsampled
    #
    #     # 4. 组装返回数据
    #     data_HSI = hsi_final.astype(np.float32)
    #
    #     # 将 SAR 直接作为辅助模态 (LiDAR slot)
    #     data_LiDAR = sar_final.astype(np.float32)
    #
    #     gt = label_valid.astype(np.int64)
    #
    #     # 5. 更新元信息
    #     self._num_bands_HSI = data_HSI.shape[2]
    #     self._num_bands_LiDAR = data_LiDAR.shape[2]
    #     self._num_classes = int(np.max(gt))
    #
    #     print(f"Loaded Wuhan Data (HSI+SAR): HSI {data_HSI.shape}, SAR {data_LiDAR.shape}, GT {gt.shape}")
    #
    #     return data_HSI, data_LiDAR, gt

# class WuhanBuilder(HSIDatasetBuilder):
#     def __init__(self, root, train: Number = 6000, val: Number = 1000,
#                  seed: int = None, window_size: int = 27,
#                  use_edge: bool = False):
#
#         filenames = {
#             'data_HSI': 'wuhan-002.mat',
#             'data_LiDAR': 'wuhan-002.mat',
#             'gt': 'wuhan_label_500k.mat'
#         }
#
#         dataset_name = 'Wuhan_HSI_MSI'
#
#         # HSI: 48, SAR: 2 , MSI: 4
#         num_classes = 13
#         num_bands_HSI = 116
#         num_bands_LiDAR = 4
#
#         # 【重要】Wuhan数据集 HSI 和 GT/SAR 的比例是 1:3
#         self.scale_factor = 3
#
#         super(WuhanBuilder, self).__init__(
#             root, filenames, train, val, seed, window_size,
#             use_edge, dataset_name, num_classes, num_bands_HSI, num_bands_LiDAR)
#
#     def load_meta_dataset(self):
#         """
#         仅加载原始分辨率数据，【绝不】在这里做全局上采样
#         """
#         data_file = self.meta_path['data_HSI']
#         label_file = self.meta_path['gt']
#
#         print(f"Loading Wuhan dataset (Raw Resolution) from: {data_file}")
#
#         with h5py.File(data_file, 'r') as f:
#             # 1. 读取原始低分辨率 HSI (C, W_lr, H_lr) -> 转置为 (H_lr, W_lr, C)
#             # 注意：不进行 Upsample，保留小尺寸以节省内存
#             hsi_raw = np.array(f['HSI'])
#             hsi_final = np.transpose(hsi_raw)  # (H_lr, W_lr, C)
#
#             # 2. 读取高分辨率 SAR (C, W_hr, H_hr) -> 转置为 (H_hr, W_hr, C)
#             # sar_final = np.transpose(f['SAR'])     # 换成MSI
#             sar_final = np.transpose(f['MSI'])
#
#         with h5py.File(label_file, 'r') as f:
#             # 3. 读取高分辨率标签 (H_hr, W_hr)
#             label_valid = np.transpose(f['label'])
#
#         # 数据类型转换
#         data_HSI = hsi_final.astype(np.float32)  # Low Resolution
#         data_LiDAR = sar_final.astype(np.float32)  # High Resolution
#         gt = label_valid.astype(np.int64)  # High Resolution
#
#         # 更新元信息
#         self._num_bands_HSI = data_HSI.shape[2]
#         self._num_bands_LiDAR = data_LiDAR.shape[2]
#         self._num_classes = int(np.max(gt))
#
#         print(f"Loaded Raw Data -> HSI_LR: {data_HSI.shape}, SAR_HR: {data_LiDAR.shape}, GT_HR: {gt.shape}")
#         return data_HSI, data_LiDAR, gt
#
#     def split_hsi_dataset(self, train=200, val=400, window_size=27, use_edge=False):
#         """
#         重写切分入口函数，处理多模态数据的归一化和传参
#         """
#         # 1. 获取数据 (HSI是低分，SAR和GT是高分)
#         hsi_lr, sar_hr, gt_hr = self.load_meta_dataset()
#
#         # 2. 归一化 (Min-Max Normalization)
#         # HSI 归一化
#         hsi_lr = 1.0 * ((hsi_lr - np.min(hsi_lr)) / (np.max(hsi_lr) - np.min(hsi_lr) + 1e-8))
#         hsi_lr = hsi_lr.transpose([2, 0, 1])  # (H, W, C) -> (C, H, W)
#
#         # SAR 归一化
#         sar_hr = 1.0 * ((sar_hr - np.min(sar_hr)) / (np.max(sar_hr) - np.min(sar_hr) + 1e-8))
#         sar_hr = sar_hr.transpose([2, 0, 1])  # (H, W, C) -> (C, H, W)
#
#         # 3. 获取切分信息 (基于高分辨率 GT)
#         info_path = self.get_split_info(gt_hr, train, val, window_size, use_edge)
#
#         # 4. 执行切分 (传入 LR HSI 和 HR SAR)
#         splitted_datasets = self.split_with_info_multiscale(hsi_lr, sar_hr, gt_hr, info_path)
#
#         # 5. 保存
#         self.save_splitted_cache(splitted_datasets)
#
#     def split_with_info_multiscale(self, hsi_lr, sar_hr, gt_hr, info_path):
#         """
#         核心逻辑：在切分 Patch 的瞬间进行局部上采样
#         hsi_lr: (C, H_lr, W_lr)
#         sar_hr: (C, H_hr, W_hr)
#         """
#         # 加载切分信息
#         info = sio.loadmat(info_path)
#         coords = info['coords']  # 这些坐标是基于 HR GT 的
#         RandPerm = info['RandPerm'].squeeze()
#         window_size = info['window_size'].item()  # 27
#         train = info['train'].item()
#         val = info['val'].item()
#
#         coords_x, coords_y = coords[:, 0], coords[:, 1]
#
#         # 窗口计算
#         assert window_size % 2 == 1
#         half_window = window_size // 2  # 13
#
#         # HSI (LR) 的窗口大小应该是 27 / 3 = 9
#         scale = self.scale_factor
#         window_size_lr = window_size // scale  # 9
#         half_window_lr = window_size_lr // 2  # 4
#
#         # 定义上采样层 (仅用于处理小 Patch)
#         upsample = nn.Upsample(size=(window_size, window_size), mode='nearest')
#
#         # 计算样本数量
#         num_samples = len(coords)
#         trainval = train + val
#         test = num_samples - trainval
#
#         print(f"Splitting: Total {num_samples} | Train {train} | Val {val} | Test {test}")
#
#         # 准备输出容器 (注意：现在 data_LiDAR 实际上存的是 SAR)
#         # 最终保存的数据，HSI 是上采样后的 (27x27)，SAR 也是 (27x27)
#         # 基类 HSIDataset 期望 data_HSI 和 data_LiDAR 分开存
#         # 我们可以把它们分别存到 output 字典里
#
#         def init_arr(n, c, w):
#             return np.zeros([n, c, w, w], dtype=np.float32)
#
#         C_hsi = hsi_lr.shape[0]
#         C_sar = sar_hr.shape[0]  # 这里是 SAR 的通道数
#
#         output = {}
#         for mode in ['train', 'val', 'test']:
#             size = train if mode == 'train' else (val if mode == 'val' else test)
#             output[f'{mode}_data'] = init_arr(size, C_hsi, window_size)  # 存 HSI
#             output[f'{mode}_lidar'] = init_arr(size, C_sar, window_size)  # 存 SAR (借用 lidar 名字)
#             output[f'{mode}_gt'] = np.zeros([size], dtype=np.int64)
#
#         # 辅助函数：根据索引处理单个样本
#         def process_sample(out_idx, global_idx, mode):
#             # 获取 HR 坐标
#             x_hr = coords_x[global_idx]
#             y_hr = coords_y[global_idx]
#
#             # 1. 提取 SAR Patch (直接在 HR 上切)
#             sar_patch = sar_hr[:,
#                         x_hr - half_window: x_hr + half_window + 1,
#                         y_hr - half_window: y_hr + half_window + 1]
#
#             # 2. 提取 HSI Patch (在 LR 上切，然后上采样)
#             # 计算 LR 坐标
#             x_lr = x_hr // scale
#             y_lr = y_hr // scale
#
#             # 提取 LR Patch (9x9)
#             # 注意：边界保护。如果整图 Padding 不够，这里可能会越界，但通常 dataset 中间部分都没问题
#             hsi_patch_lr = hsi_lr[:,
#                            x_lr - half_window_lr: x_lr + half_window_lr + 1,
#                            y_lr - half_window_lr: y_lr + half_window_lr + 1]
#
#             # 上采样 (9x9 -> 27x27)
#             # 转为 Tensor: (C, 9, 9) -> (1, C, 9, 9)
#             t_in = torch.from_numpy(hsi_patch_lr).unsqueeze(0)
#             # Upsample -> (1, C, 27, 27)
#             t_out = upsample(t_in)
#             hsi_patch_hr = t_out.squeeze(0).numpy()
#
#             # 存入 Output
#             output[f'{mode}_data'][out_idx] = hsi_patch_hr
#             output[f'{mode}_lidar'][out_idx] = sar_patch
#
#             # 存 GT (注意基类里 gt = gt - 1，这里根据你的数据看是否需要)
#             # 假设 wuhan_label.mat 已经是 0-indexed 或者 1-indexed，这里保持原样读取
#             label_val = gt_hr[x_hr, y_hr]
#             # 如果你的 label 是 1-15，需要减 1 变成 0-14；如果是 0-14 则不用
#             # 这里沿用常见习惯，减 1。如果跑出来 label 对不上请去掉 "-1"
#             output[f'{mode}_gt'][out_idx] = label_val - 1
#
#         # 执行循环
#         print("Processing Train set...")
#         for i in range(train):
#             process_sample(i, RandPerm[i], 'train')
#
#         print("Processing Val set...")
#         for i in range(val):
#             process_sample(i, RandPerm[i + train], 'val')
#
#         print("Processing Test set...")
#         for i in range(test):
#             process_sample(i, RandPerm[i + trainval], 'test')
#
#         # 兼容性处理：基类 HSIDataset 期望 'data' 字段是主模态，我们这里只存了 hsi 到 data
#         # 我们需要把 lidar 数据也保存下来。
#         # 上面的 save_splitted_cache 只保存 _data 和 _gt。
#         # 我们需要重写 save_splitted_cache 或者把 SAR 拼接到 data 里？
#         # 最好是把 SAR 拼接到 data 里 (C_hsi + C_sar)，或者修改 save 函数。
#
#         # 【方案】：为了不改动太多基类文件，我这里将 SAR 拼接到 data 的通道维度
#         # 但这样在 dataset 读取时需要拆分。
#         # 【更好的方案】：既然我们重写了 split_hsi_dataset，我们顺便手动保存一下吧。
#
#         return output
#
#     def save_splitted_cache(self, output):
#         """
#         重写保存逻辑，因为我们要保存 data(HSI), lidar(SAR) 和 gt
#         """
#         for dataset_type in ('train', 'val', 'test'):
#             # 1. 保存 HSI (主数据)
#             p_data = self.cache_path[f'{dataset_type}_data']
#             p_data.parent.mkdir(exist_ok=True, parents=True)
#             np.save(p_data, output[f'{dataset_type}_data'])
#
#             # 2. 保存 SAR (辅助数据)
#             # 我们需要造一个路径，因为基类只定义了 _data 和 _gt
#             # 这里假设基类的 load_splitted_dataset 会读取 meta_path['data_LiDAR']
#             # 但 Cache 模式下，基类只读 _data.npy。
#
#             # *为了兼容你的 HSIDataset 类（它可能只接受一个 data 路径），
#             # 我将 SAR 数据保存到一个单独的文件，但在 HSIDataset 里你可能需要改代码来读它*
#             #
#             # *权宜之计*：把 SAR 数据拼接到 HSI 数据后面保存！
#             # 这样不用改 Dataset 读取代码，只需要在 Model 里 split 即可。
#             # 或者，如果你的 HSIDataset 支持读取两个 cache 文件，那就分开存。
#
#             # 这里我采用：分开存，名字叫 _lidar.npy
#             p_lidar = str(p_data).replace('_data.npy', '_lidar.npy')
#             np.save(p_lidar, output[f'{dataset_type}_lidar'])
#
#             # 3. 保存 GT
#             p_gt = self.cache_path[f'{dataset_type}_gt']
#             np.save(p_gt, output[f'{dataset_type}_gt'])
#
#         print("Data cache saved (HSI + SAR + GT).")
#
#     # 注意：你还需要在 load_splitted_dataset 里支持读取这个 _lidar.npy
#     # 鉴于不能改基类，我们在这个子类里覆盖 load_splitted_dataset
#
#     def load_splitted_dataset(self, dataset_type: str, cache_mode: bool = False, **kwargs):
#         if cache_mode:
#             self.get_datasets_cache()
#             data_path = self.cache_path[f'{dataset_type}_data']
#             gt_path = self.cache_path[f'{dataset_type}_gt']
#             # 构造 lidar 路径
#             lidar_path = str(data_path).replace('_data.npy', '_lidar.npy')
#
#             print(f'Build {dataset_type} dataset of cache style (Wuhan HSI+SAR).')
#             # 假设 CachedHSIDataset 支持传入 extra_path，或者我们可以动态修改它
#             # 如果 CachedHSIDataset 不支持，我们只能在这里报错或者魔改
#
#             # 【最简单的兼容方法】：
#             # 返回一个标准的 HSIDataset，但是路径指向我们生成的 cache 文件
#             # 这里的逻辑稍微有点复杂，因为 HSIDataset 通常读取 .mat
#             # 而 CachedHSIDataset 读取 .npy
#
#             return CachedHSIDataset(data_path, gt_path, lidar_path=lidar_path, **kwargs)
#         else:
#             # Generator 模式 (实时切片) - 暂不推荐，因为太慢，我们主要用 Cache 模式
#             # 如果非要用，逻辑和 split_with_info 类似
#             return super().load_splitted_dataset(dataset_type, cache_mode, **kwargs)

# class WuhanBuilder(HSIDatasetBuilder):
#     def __init__(self, root, train: Number = 200, val: Number = 400,
#                  seed: int = None, window_size: int = 27,
#                  use_edge: bool = False):
#
#         filenames = {
#             'data_HSI': 'wuhan-002.mat',
#             'data_LiDAR': 'wuhan-002.mat',
#             'gt': 'wuhan_label_500k.mat'
#         }
#
#         dataset_name = 'Wuhan_HSI_MSI'
#
#         # HSI: 48, SAR: 4
#         num_classes = 13
#         num_bands_HSI = 116
#         num_bands_LiDAR = 4
#
#         self.scale_factor = 3
#
#         super(WuhanBuilder, self).__init__(
#             root, filenames, train, val, seed, window_size,
#             use_edge, dataset_name, num_classes, num_bands_HSI, num_bands_LiDAR)
#
#     def load_meta_dataset(self):
#         """仅加载原始分辨率数据"""
#         data_file = self.meta_path['data_HSI']
#         label_file = self.meta_path['gt']
#
#         print(f"Loading Wuhan dataset (Raw Resolution) from: {data_file}")
#
#         with h5py.File(data_file, 'r') as f:
#             hsi_raw = np.array(f['HSI'])
#             hsi_final = np.transpose(hsi_raw)  # (H_lr, W_lr, C)
#             sar_final = np.transpose(f['MSI'])  # 用MSI
#
#         with h5py.File(label_file, 'r') as f:
#             label_valid = np.transpose(f['label'])
#
#         data_HSI = hsi_final.astype(np.float32)
#         data_LiDAR = sar_final.astype(np.float32)
#         gt = label_valid.astype(np.int64)
#
#         self._num_bands_HSI = data_HSI.shape[2]
#         self._num_bands_LiDAR = data_LiDAR.shape[2]
#         self._num_classes = int(np.max(gt))
#
#         print(f"Loaded Raw Data -> HSI_LR: {data_HSI.shape}, SAR_HR: {data_LiDAR.shape}, GT_HR: {gt.shape}")
#         return data_HSI, data_LiDAR, gt
#
#     def split_hsi_dataset(self, train=200, val=50, window_size=27, use_edge=False):
#         """
#         重写切分入口函数：直接处理并写入磁盘，避免内存溢出
#         """
#         hsi_lr, sar_hr, gt_hr = self.load_meta_dataset()
#
#         # 归一化
#         print("Normalizing...")
#         hsi_lr = 1.0 * ((hsi_lr - np.min(hsi_lr)) / (np.max(hsi_lr) - np.min(hsi_lr) + 1e-8))
#         hsi_lr = hsi_lr.transpose([2, 0, 1])  # (C, H, W)
#
#         sar_hr = 1.0 * ((sar_hr - np.min(sar_hr)) / (np.max(sar_hr) - np.min(sar_hr) + 1e-8))
#         sar_hr = sar_hr.transpose([2, 0, 1])  # (C, H, W)
#
#         info_path = self.get_split_info(gt_hr, train, val, window_size, use_edge)
#
#         # 核心：使用 Memmap 写入
#         self.split_and_write_memmap(hsi_lr, sar_hr, gt_hr, info_path)
#
#     def split_and_write_memmap(self, hsi_lr, sar_hr, gt_hr, info_path):
#         """
#         使用 np.lib.format.open_memmap 直接在硬盘上创建 .npy 文件并写入
#         """
#         info = sio.loadmat(info_path)
#         coords = info['coords']
#         RandPerm = info['RandPerm'].squeeze()
#         window_size = info['window_size'].item()
#         train = info['train'].item()
#         val = info['val'].item()
#
#         coords_x, coords_y = coords[:, 0], coords[:, 1]
#
#         assert window_size % 2 == 1
#         half_window = window_size // 2
#
#         scale = self.scale_factor
#         window_size_lr = window_size // scale
#         half_window_lr = window_size_lr // 2
#
#         upsample = nn.Upsample(size=(window_size, window_size), mode='nearest')
#
#         num_samples = len(coords)
#         trainval = train + val
#         test = num_samples - trainval
#
#         print(f"Splitting Strategy: Train {train} | Val {val} | Test {test} (Total {num_samples})")
#
#         # 准备 Memmap 数组
#         C_hsi = hsi_lr.shape[0]
#         C_sar = sar_hr.shape[0]
#
#         # 确保目录存在
#         for p in self.cache_path.values():
#             p.parent.mkdir(exist_ok=True, parents=True)
#
#         # 定义文件映射
#         arrays = {}
#
#         for mode in ['train', 'val', 'test']:
#             size = train if mode == 'train' else (val if mode == 'val' else test)
#             if size == 0: continue
#
#             # HSI Path
#             p_data = self.cache_path[f'{mode}_data']
#             # SAR Path (手动构造)
#             p_lidar = Path(str(p_data).replace('_data.npy', '_lidar.npy'))
#             # GT Path
#             p_gt = self.cache_path[f'{mode}_gt']
#
#             print(f"Creating Memmap for {mode}: {size} samples...")
#             # 创建硬盘上的 .npy 文件 (模式 w+ 表示创建/覆盖)
#             arrays[f'{mode}_data'] = np.lib.format.open_memmap(
#                 p_data, mode='w+', dtype='float32', shape=(size, C_hsi, window_size, window_size))
#
#             arrays[f'{mode}_lidar'] = np.lib.format.open_memmap(
#                 p_lidar, mode='w+', dtype='float32', shape=(size, C_sar, window_size, window_size))
#
#             # GT 比较小，可以用普通数组最后一次性保存，也可以用 memmap
#             arrays[f'{mode}_gt'] = np.lib.format.open_memmap(
#                 p_gt, mode='w+', dtype='int64', shape=(size,))
#
#         # 处理函数
#         def process_sample(out_idx, global_idx, mode):
#             x_hr = coords_x[global_idx]
#             y_hr = coords_y[global_idx]
#
#             # SAR Patch
#             arrays[f'{mode}_lidar'][out_idx] = sar_hr[:,
#                                                x_hr - half_window: x_hr + half_window + 1,
#                                                y_hr - half_window: y_hr + half_window + 1]
#
#             # HSI Patch (Upsample)
#             x_lr = x_hr // scale
#             y_lr = y_hr // scale
#
#             hsi_patch_lr = hsi_lr[:,
#                            x_lr - half_window_lr: x_lr + half_window_lr + 1,
#                            y_lr - half_window_lr: y_lr + half_window_lr + 1]
#
#             t_in = torch.from_numpy(hsi_patch_lr).unsqueeze(0)
#             t_out = upsample(t_in)
#             arrays[f'{mode}_data'][out_idx] = t_out.squeeze(0).numpy()
#
#             # GT
#             label_val = gt_hr[x_hr, y_hr]
#             arrays[f'{mode}_gt'][out_idx] = label_val - 1  # 假设需要减1，如果不需要请删除-1
#
#         # 执行循环
#         print("Processing Train set...")
#         for i in range(train):
#             process_sample(i, RandPerm[i], 'train')
#
#         print("Processing Val set...")
#         for i in range(val):
#             process_sample(i, RandPerm[i + train], 'val')
#
#         print("Processing Test set (This will take a while due to large size)...")
#         for i in range(test):
#             process_sample(i, RandPerm[i + trainval], 'test')
#             if i % 10000 == 0:
#                 print(f"  Processed {i}/{test} samples...")
#
#         # 重要：将所有 memmap 刷入磁盘并关闭
#         # 删除对象会自动触发 flush 和 close
#         del arrays
#         print("All data successfully written to disk via Memmap.")
#
#     def load_splitted_dataset(self, dataset_type: str, cache_mode: bool = False, **kwargs):
#         # 必须强制使用 cache_mode，否则报错
#         if not cache_mode:
#             raise ValueError("Wuhan Dataset is too large! You MUST set cache_mode=True.")
#
#         self.get_datasets_cache()
#         data_path = self.cache_path[f'{dataset_type}_data']
#         gt_path = self.cache_path[f'{dataset_type}_gt']
#         lidar_path = str(data_path).replace('_data.npy', '_lidar.npy')
#
#         return CachedHSIDataset(data_path, gt_path, lidar_path=lidar_path, **kwargs)