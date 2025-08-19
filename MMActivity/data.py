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
        Modified to work with consolidated stacking
        """
        data_list = []
        file_names = self._get_partition_file_names()

        for fn in file_names:
            logging.info(f'Loading {fn}')
            with open(fn, 'rb') as f:
                data_slice = pickle.load(f)
            data_list.extend(data_slice)
        
        stacked_data = self.stack_and_padd_frames(data_list)
        
        if self.partition in ['train', 'val']:
            labels = [d['y'] for d in stacked_data]  
            train_data, val_data = train_test_split(
                stacked_data,  
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
            return {'train': [], 'val': [], 'test': stacked_data}, len(stacked_data)

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
        Stacking and padding function with consolidated mode:
        - Creates non-overlapping stacks with specified overlap
        - Returns fewer samples than input (stack_size/stride ratio)
        """
        if self.stacks is None:
            return data_list

        stack_size = self.stacks
        overlap = 10  
        stride = stack_size - overlap

        stacked_samples = []
        total_operations = sum([len(range(0, len(segment), stride)) 
                            for _, segment in self._segment_by_label(data_list)])
        
        print(f"Stacking and padding frames (consolidated mode)...")
        pbar = tqdm(total=total_operations)

        # Process each label segment separately
        for label, segment_data in self._segment_by_label(data_list):
            i = 0
            while i < len(segment_data):
                # Get current stack window
                end_idx = min(i + stack_size, len(segment_data))
                stack_data = segment_data[i:end_idx]
                
                # Handle padding by repeating last frame
                while len(stack_data) < stack_size:
                    stack_data.append(stack_data[-1] if stack_data else {
                        'x': np.zeros((self.max_points, 3)),
                        'y': label
                    })

                # Process each frame in the stack
                processed_frames = []
                for data in stack_data:
                    x = data['x']
                    # Handle undersized/oversized frames
                    if x.shape[0] < self.max_points:
                        x = np.pad(x, ((0, self.max_points - x.shape[0]), (0, 0)), 'constant')
                    elif x.shape[0] > self.max_points:
                        x = x[np.random.choice(x.shape[0], self.max_points, replace=False)]
                    processed_frames.append(x)

                # Create stacked sample
                stacked_x = np.concatenate(processed_frames, axis=0)
                new_sample = {
                    **stack_data[0],  
                    'new_x': stacked_x  
                }
                stacked_samples.append(new_sample)
                pbar.update(1)
                i += stride  

        pbar.close()
        print(f"Stacking complete. {len(data_list)} input → {len(stacked_samples)} output samples")
        return stacked_samples

    def _segment_by_label(self, data_list):
        """Helper to segment data by contiguous labels"""
        segments = []
        current_segment = []
        current_label = data_list[0]['y']
        
        for data in data_list:
            if data['y'] == current_label:
                current_segment.append(data)
            else:
                segments.append((current_label, current_segment))
                current_segment = [data]
                current_label = data['y']
        segments.append((current_label, current_segment))
        return segments
    
if __name__ == "__main__":
    # Example usage and testing
    root_dir = ''  # current directory or provide data path here
    mmr_dataset_config = {
        'processed_data': 'data/processed/mmr_action/data.pkl',
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