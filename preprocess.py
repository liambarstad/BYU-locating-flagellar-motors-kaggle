import os
import random
import shutil
import argparse
from typing import Dict

parser = argparse.ArgumentParser(description='preprocess data for training')
parser.add_argument(
    '--data_dir',
    type=str,
    required=True,
    help='The absolute path to the data to be used and modified'
)
parser.add_argument(
    '--random_seed',
    type=int,
    help='Seed to preserve functionality across runs'
)
args = parser.parse_args()
data_dir = args.data_dir
random_seed = args.random_seed if args.random_seed else 123
random.seed(random_seed)
 

# suffles and indexes 
def train_val_split(split_ratio: float) -> Dict:
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    os.makedirs(val_dir, exist_ok=True)

    all_tomo_dirs = os.listdir(train_dir) + os.listdir(val_dir)
    random.shuffle(all_tomo_dirs)
    split_index = int(len(all_tomo_dirs)*split_ratio)

    # move to train
    for tomo_dir in all_tomo_dirs[:split_index]:
        if not os.path.exists(os.path.join(train_dir, tomo_dir)):
            shutil.move(os.path.join(val_dir, tomo_dir), os.path.join(train_dir, tomo_dir))

    # move to val
    for tomo_dir in all_tomo_dirs[split_index:]:
        if not os.path.exists(os.path.join(val_dir, tomo_dir)):
            shutil.move(os.path.join(train_dir, tomo_dir), os.path.join(val_dir, tomo_dir))


if __name__ == '__main__':
    train_val_split(0.8)