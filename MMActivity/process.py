import os
import glob
import pickle
import numpy as np

# Mapping from action-name to integer label
LABEL_MAP = {
    'boxing': 0,
    'jack': 1,
    'jump': 2,
    'squats': 3,
    'walk': 4
}

def parse_txt_file(file_path, label_idx):
    """
    Reads a single .txt file containing mmWave data, extracts frames,
    and returns a list of dictionaries. Each dictionary has:
        {
          'x': np.array of shape (N_points, 3) with columns [x, y, z],
          'y': label_idx  # an integer from LABEL_MAP
        }
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Keep track of all tokens in a single list for easier parsing.
    wordlist = []
    for line in lines:
        tokens = line.split()
        wordlist.extend(tokens)
    
    frame_num_count = -1
    # A dict where the key = frame index, and value = list of [x, y, z] points
    frames_data = {}
    
    length1 = len(wordlist)
    i = 0
    while i < length1:
        word = wordlist[i]
        
        # Detect new frame
        if word == "point_id:" and i+1 < length1:
            # If point_id == 0, increment the frame counter
            if wordlist[i+1] == "0":
                frame_num_count += 1
                if frame_num_count not in frames_data:
                    frames_data[frame_num_count] = []
            i += 2
            continue
        
        # Extract x, y, z
        elif word == "x:" and i+1 < length1:
            x_val = float(wordlist[i+1])
            i += 2
            
            # Next two tokens should be y: <val>, z: <val>
            if i+3 <= length1 and wordlist[i] == "y:" and wordlist[i+2] == "z:":
                y_val = float(wordlist[i+1])
                z_val = float(wordlist[i+3])
                i += 4 

                frames_data[frame_num_count].append([x_val, y_val, z_val])
            else:
                i += 1
            continue
        else:
            i += 1
    
    # Convert each frame’s list of [x, y, z] into a NumPy array
    frames_list = []
    for frame_idx in sorted(frames_data.keys()):
        arr = np.array(frames_data[frame_idx], dtype=float)  # shape (N_points, 3)
        # Create a dictionary for each frame
        frame_dict = {
            'x': arr,          # shape (N_points, 3)
            'y': label_idx     # integer label
        }
        frames_list.append(frame_dict)
    return frames_list

def process_data(root_dir='Data'):
    """
    This function loops through train/ and test/ in `root_dir`,
    then each action subfolder (boxing, jack, jump, squats, walk).
    For each action, it gathers all .txt files, parses them, and
    saves a pickle file containing a list of frame dictionaries.
    """
    # train and test subfolders containing data as downloaded from the github repo.
    partitions = ['train', 'test']  

    for partition in partitions:
        partition_path = os.path.join(root_dir, partition)
        
        # Loop over each action label in LABEL_MAP
        for action_name, label_idx in LABEL_MAP.items():
            action_path = os.path.join(partition_path, action_name)
            
            # Collect all .txt files for this action
            txt_files = glob.glob(os.path.join(action_path, '*.txt'))
            
            all_frames = []  # will hold a list of dictionaries (one per frame)
            
            for txt_file in txt_files:
                frames_from_file = parse_txt_file(txt_file, label_idx)
                all_frames.extend(frames_from_file)
            
            # Now save all frames for this action label in a pickle
            output_pickle = os.path.join(action_path, f'{action_name}_frames.pkl')
            
            with open(output_pickle, 'wb') as f:
                pickle.dump(all_frames, f)
            print(f'Saved {len(all_frames)} frames to {output_pickle}')


if __name__ == '__main__':
    # Data is the root directory where the downloaded data is saved
    process_data(root_dir='Data')
