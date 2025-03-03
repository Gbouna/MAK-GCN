import os
import torch
import numpy as np
import pickle
import logging
import random
from tqdm import tqdm
from torch_geometric.data import Dataset
from sklearn.model_selection import train_test_split

class MMRActionData(Dataset):
    raw_data_path = 'data/raw'
    processed_data = 'data/processed/mmr_action/data.pkl'
    max_points = 22
    seed = 42
    partitions = (0.8, 0.1, 0.1)
    stacks = None
    zero_padding = 'per_data_point'
    zero_padding_styles = ['per_data_point', 'per_stack', 'data_point', 'stack']
    num_keypoints = 9
    forced_rewrite = False

    def _parse_config(self, c):
        """Parse configuration dict and update class attributes."""
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
            raise ValueError(f'Zero padding style {self.zero_padding} not supported.')
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
        super(MMRActionData, self).__init__(root, transform, pre_transform, pre_filter)

        self.partition = partition 
        self._parse_config(mmr_dataset_config)

        # Either load existing processed data or create it anew.
        if (not os.path.isfile(self.processed_data)) or self.forced_rewrite:
            full_data, _ = self._process()  
            with open(self.processed_data, 'wb') as f:
                pickle.dump(full_data, f)
        else:
            with open(self.processed_data, 'rb') as f:
                full_data = pickle.load(f)

        all_labels = []
        for part_name in ['train', 'val', 'test']:
            all_labels.extend(d['y'] for d in full_data[part_name])
        unique_labels = set(all_labels)
        num_classes = len(unique_labels)

        total_samples = (
            len(full_data['train']) +
            len(full_data['val']) +
            len(full_data['test'])
        )
        self.data = full_data[self.partition]
        self.num_samples = len(self.data)
        self.target_dtype = torch.int64

        self.info = {
            'num_samples': self.num_samples,
            'num_classes': num_classes,
            'max_points': self.max_points,
            'stacks': self.stacks,
            'partition': self.partition,
        }

        logging.info(
            f'Loaded {self.partition} data with {self.num_samples} samples, '
            f'where the total number of samples (train+val+test) is {total_samples}. '
            f'Num classes: {num_classes}'
        )

    def len(self):
        """Return the number of samples in this partition."""
        return self.num_samples

    def get(self, idx):
        """Return (x, y) for the idx-th sample in the current partition."""
        data_point = self.data[idx]
        # If 'new_x' was created by stack_and_padd_frames(), use it; else use 'x'
        x = data_point['new_x'] if 'new_x' in data_point else data_point['x']
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(data_point['y'], dtype=self.target_dtype)
        return x, y

    def _process(self):
        """
        Loads data from the appropriate folder ('train' or 'test') based on self.partition.
        Then applies stacking/padding if needed.
        For 'train'/'val': we load from the train folder, do a stratified 80/20 split
                           to produce actual train vs val sets.
        For 'test': we load from the test folder, using all as test data.

        Returns:
           A dict: {
               'train': [...],
               'val': [...],
               'test': [...]
           }, num_samples
        """
        data_list = []
        file_names = self._get_partition_file_names()

        for fn in file_names:
            logging.info(f'Loading {fn}')
            with open(fn, 'rb') as f:
                data_slice = pickle.load(f)
            data_list.extend(data_slice)
        data_list = self.stack_and_padd_frames(data_list)

        if self.partition in ['train', 'val']:
            labels = [d['y'] for d in data_list]
            num_samples = len(data_list)
            train_data, val_data = train_test_split(
                data_list,
                test_size=0.2,
                random_state=self.seed,
                shuffle=True,
                stratify=labels
            )

            if self.partition == 'train':
                return {'train': train_data, 'val': [], 'test': []}, len(train_data)
            else:  
                return {'train': [], 'val': val_data, 'test': []}, len(val_data)

        else:
            num_samples = len(data_list)
            return {'train': [], 'val': [], 'test': data_list}, num_samples

    def _get_partition_file_names(self):
        """
        Decide which folder (train or test) to read from, based on whether
        self.partition is 'train'/'val' or 'test'.
        """
        if self.partition in ['train', 'val']:
            # We assume we have 5 pkl files in data/raw/train: 0.pkl..4.pkl generated using the process.py script
            file_nums = [0, 1, 2, 3, 4]
            return [os.path.join(self.raw_data_path, 'train', f'{i}.pkl') for i in file_nums]
        else:
            # We assume we have 5 pkl files in data/raw/test: 0.pkl..4.pkl generated using the process.py script
            file_nums = [0, 1, 2, 3, 4]
            return [os.path.join(self.raw_data_path, 'test', f'{i}.pkl') for i in file_nums]

    def stack_and_padd_frames(self, data_list):
        """
        If self.stacks is not None, zero-pad frames for each data point according
        to the chosen zero-padding style. If it's None, we just return data_list as is.

        Returns:
            new_data_list: same shape as data_list, but with 'new_x' arrays stacked/padded
        """
        if self.stacks is None:
            return data_list

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
                        # Pad up to max_points if needed
                        mydata_slice = np.pad(mydata_slice,
                                              ((0, max(diff, 0)), (0, 0)),
                                              mode='constant')
                        # Then randomly sample exactly max_points
                        mydata_slice = mydata_slice[
                            np.random.choice(len(mydata_slice), self.max_points, replace=False)
                        ]
                        data_point.append(mydata_slice)
                    else:
                        # If we don't have enough past frames, pad with zeros
                        data_point.append(np.zeros((self.max_points, 3)))
                # Concatenate all stacked frames for this index
                padded_xs.append(np.concatenate(data_point, axis=0))
                pbar.update(1)

        elif self.zero_padding in ['per_stack', 'stack']:
            for i in range(len(xs)):
                start = max(0, i - self.stacks)
                stacked_x = np.concatenate(xs[start:i+1], axis=0)
                stacked_xs.append(stacked_x)
                pbar.update(0.5)

            for x in stacked_xs:
                diff = self.max_points * self.stacks - x.shape[0]
                x = np.pad(x, ((0, max(diff, 0)), (0, 0)), 'constant')
                x = x[np.random.choice(len(x), self.max_points * self.stacks, replace=False)]
                padded_xs.append(x)
                pbar.update(0.5)

        else:
            raise NotImplementedError(f"Padding style '{self.zero_padding}' is not implemented")

        pbar.close()
        print("Stacking and padding frames done")
        new_data_list = [{**d, 'new_x': px} for d, px in zip(data_list, padded_xs)]
        return new_data_list


if __name__ == "__main__":
    # Example usage and testing
    root_dir = ''  # current directory or provide data path here
    mmr_dataset_config = {
        'processed_data': 'data/processed/mmr_action_distance/data.pkl',
        'stacks': 50,             
        'max_points': 22,
        'zero_padding': 'per_data_point',
        'seed': 42,
        'forced_rewrite': False
    }

    # Create train, val, and test datasets
    train_dataset = MMRActionData(root=root_dir, partition='train', mmr_dataset_config=mmr_dataset_config)
    val_dataset   = MMRActionData(root=root_dir, partition='val',   mmr_dataset_config=mmr_dataset_config)
    test_dataset  = MMRActionData(root=root_dir, partition='test',  mmr_dataset_config=mmr_dataset_config)


