import os
import torch
import numpy as np
import pickle
import logging
import random
from tqdm import tqdm
from torch_geometric.data import Dataset

class MMRBaseData(Dataset):
    """
    A base Dataset class for loading, processing, and splitting
    3D point data for action recognition tasks.
    """
    raw_data_path = 'data/raw'
    processed_data = 'data/processed/mmr_action/data.pkl'
    max_points = 22
    seed = 42
    partitions = (0.8, 0.1, 0.1)  # train, val, test (ratios)
    stacks = None
    zero_padding = 'per_data_point'
    zero_padding_styles = ['per_data_point', 'per_stack', 'data_point', 'stack']
    forced_rewrite = True

    def _parse_config(self, c):
        """
        Parse and set the configuration from the input dictionary 'c'.
        """
        c = {k: v for k, v in c.items() if v is not None}
        self.seed = c.get('seed', self.seed)
        self.processed_data = c.get('processed_data', self.processed_data)
        self.max_points = c.get('max_points', self.max_points)
        self.partitions = (
            c.get('train_split', self.partitions[0]),
            c.get('val_split', self.partitions[1]),
            c.get('test_split', self.partitions[2])
        )
        self.stacks = c.get('stacks', self.stacks)
        self.zero_padding = c.get('zero_padding', self.zero_padding)

        if self.zero_padding not in self.zero_padding_styles:
            raise ValueError(
                f'Zero padding style "{self.zero_padding}" is not supported.'
            )

        self.forced_rewrite = c.get('forced_rewrite', self.forced_rewrite)

    def __init__(
        self, 
        root, 
        partition, 
        transform=None, 
        pre_transform=None, 
        pre_filter=None,
        mmr_dataset_config=None
    ):
        """
        Initialize the dataset.

        Args:
            root (str): Root directory (often unused, required by PyG Dataset).
            partition (str): One of 'train', 'val', or 'test'.
            transform: (Unused by default in this script).
            pre_transform: (Unused by default).
            pre_filter: (Unused by default).
            mmr_dataset_config (dict): A configuration dict for dataset parameters.
        """
        super(MMRBaseData, self).__init__(root, transform, pre_transform, pre_filter)
        self._parse_config(mmr_dataset_config)

        # Load or process data
        if (not os.path.isfile(self.processed_data)) or self.forced_rewrite:
            self.data, _ = self._process()
            with open(self.processed_data, 'wb') as f:
                pickle.dump(self.data, f)
        else:
            with open(self.processed_data, 'rb') as f:
                self.data = pickle.load(f)

        # Extract the relevant partition
        total_samples = (len(self.data['train']) + 
                         len(self.data['val']) + 
                         len(self.data['test']))
        self.data = self.data[partition]
        self.num_samples = len(self.data)

        # Dataset info dictionary
        self.info = {
            'num_samples': self.num_samples,
            'max_points': self.max_points,
            'stacks': self.stacks,
            'partition': partition,
        }
        logging.info(
            f'Loaded {partition} data with {self.num_samples} samples; '
            f'overall total = {total_samples} samples.'
        )

    def len(self):
        return self.num_samples
    
    def get(self, idx):
        """
        Retrieve the sample (x, y) at index 'idx'.
        Returns:
            x (torch.Tensor): Shape [?, 3] or [stacks*max_points, 3], etc.
            y (torch.Tensor): Label.
        """
        data_point = self.data[idx]
        x = torch.tensor(data_point['new_x'], dtype=torch.float32)
        y = torch.tensor(data_point['y'], dtype=self.target_dtype)
        return x, y

    @property
    def raw_file_names(self):
        """
        A list of the raw file names to load from `raw_data_path`.
        """
        file_indices = list(range(19))
        return [f'{self.raw_data_path}/{idx}.pkl' for idx in file_indices]

    def _process(self):
        """
        Load, combine, and partition raw data into train/val/test sets.
        Return (data_map, num_samples).
        """
        data_list = []
        for fn in self.raw_file_names:
            logging.info(f'Loading {fn}')
            with open(fn, 'rb') as f:
                data_slice = pickle.load(f)
            data_list.extend(data_slice)

        num_samples = len(data_list)
        logging.info(f'Loaded {num_samples} data points from raw files.')
        data_list = self.stack_and_pad_frames(data_list)

        # Shuffle
        random.seed(self.seed)
        random.shuffle(data_list)

        # Partition
        train_end = int(self.partitions[0] * num_samples)
        val_end = train_end + int(self.partitions[1] * num_samples)
        train_data = data_list[:train_end]
        val_data = data_list[train_end:val_end]
        test_data = data_list[val_end:]

        data_map = {
            'train': train_data,
            'val': val_data,
            'test': test_data,
        }
        return data_map, num_samples


    def stack_and_padd_frames(self, data_list):
        """
        Stack multiple frames for each data point 
        and pad them up to max_points * stacks.
        """
        if self.stacks is None:
            return data_list
        # take multiple frames for each x
        xs = [d['x'] for d in data_list]
        stacked_xs = []
        padded_xs = []
        print("Stacking and padding frames...")
        pbar = tqdm(total=len(xs))

        if self.zero_padding in ['per_data_point', 'data_point']:
            for i in range(len(xs)):
                data_point = []
                for j in range(self.stacks):
                    if i - j >= 0:
                        mydata_slice = xs[i - j]
                        diff = self.max_points - mydata_slice.shape[0]
                        mydata_slice = np.pad(mydata_slice, ((0, max(diff, 0)), (0, 0)), 'constant')
                        mydata_slice = mydata_slice[np.random.choice(len(mydata_slice), self.max_points, replace=False)]  
                        data_point.append(mydata_slice)
                    else:
                        data_point.append(np.zeros((self.max_points, 3)))
                padded_xs.append(np.concatenate(data_point, axis=0))
                pbar.update(1)
        elif self.zero_padding in ['per_stack', 'stack']:
            for i in range(len(xs)):
                start = max(0, i - self.stacks)
                stacked_xs.append(np.concatenate(xs[start:i+1], axis=0))
                pbar.update(0.5)
            for x in stacked_xs:
                diff = self.max_points * self.stacks - x.shape[0]
                x = np.pad(x, ((0, max(diff, 0)), (0, 0)), 'constant')
                x = x[np.random.choice(len(x), self.max_points * self.stacks, replace=False)]  
                padded_xs.append(x)
                pbar.update(0.5)
        else:
            raise NotImplementedError()
        pbar.close()
        print("Stacking and padding frames done")
        # remap padded_xs to data_list
        new_data_list = [{**d, 'new_x': x} for d, x in zip(data_list, padded_xs)]
        return new_data_list
    
class MMRActionData(MMRBaseData):
    """
    Action Recognition Dataset class.
    Inherits from MMRBaseData, but adds action-label processing
    and filtering (e.g., removing samples with label == -1).
    """
    processed_data = 'data/processed/mmr_action/data.pkl'

    def __init__(self, *args, **kwargs):
        # Load action labels
        self.action_label = np.load('./data/raw/action_label.npy')
        super().__init__(*args, **kwargs)
        # Update number of classes in the info dictionary
        self.info['num_classes'] = len(np.unique(self.action_label)) - 1
        self.target_dtype = torch.int64

    def _process(self):
        """
        Load raw data, filter out invalid labels (-1),
        then stack/pad, shuffle, and split.
        """
        data_list = []
        for fn in self.raw_file_names:
            logging.info(f'Loading {fn}')
            with open(fn, 'rb') as f:
                data_slice = pickle.load(f)
            data_list.extend(data_slice)

        if len(data_list) != len(self.action_label):
            raise ValueError(
                f"Mismatch: data_list={len(data_list)} vs. action_label={len(self.action_label)}"
            )

        # Filter out any data with action_label == -1
        filtered_data_list = []
        for i, d in enumerate(data_list):
            label = self.action_label[i]
            if label != -1:
                d['y'] = label
                filtered_data_list.append(d)

        data_list = filtered_data_list
        num_samples = len(data_list)
        logging.info(f'After filtering, {num_samples} valid data points remain.')
        data_list = self.stack_and_pad_frames(data_list)
        random.seed(self.seed)
        random.shuffle(data_list)

        # Partition
        train_end = int(self.partitions[0] * num_samples)
        val_end = train_end + int(self.partitions[1] * num_samples)
        train_data = data_list[:train_end]
        val_data = data_list[train_end:val_end]
        test_data = data_list[val_end:]

        logging.info(
            f"Partition sizes: train={len(train_data)}, "
            f"val={len(val_data)}, test={len(test_data)}"
        )

        data_map = {
            'train': train_data,
            'val': val_data,
            'test': test_data,
        }
        return data_map, num_samples


if __name__ == "__main__":
    # Example usage/testing
    root_dir = ''  # Root directory or specify directory
    mmr_dataset_config = {
        'processed_data': 'data/processed/mmr_action/data.pkl',
        'stacks': 50,
        'max_points': 22,
        'zero_padding': 'per_data_point',
        'seed': 42
    }

    # Load partitions
    train_dataset = MMRActionData(root=root_dir, partition='train', mmr_dataset_config=mmr_dataset_config)
    val_dataset = MMRActionData(root=root_dir, partition='val', mmr_dataset_config=mmr_dataset_config)
    test_dataset = MMRActionData(root=root_dir, partition='test', mmr_dataset_config=mmr_dataset_config)
