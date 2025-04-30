import os
import random
import shutil
from typing import Dict
 
# suffles and indexes 
def train_val_split(rand_seed: int, data_dir: str, split_ratio: float) -> Dict:
    random.seed(rand_seed)
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
    train_val_split(123, os.path.abspath('data/raw'), 0.8)