import os
import random
import shutil
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
from typing import Dict, Tuple

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
np.random.seed(random_seed)
 

TRAIN_DIR = os.path.join(data_dir, 'raw', 'train')
YOLO_IMAGES_TRAIN = os.path.join(data_dir, 'processed', 'images', 'train')
YOLO_LABELS_TRAIN = os.path.join(data_dir, 'processed', 'labels', 'train')
YOLO_IMAGES_VAL = os.path.join(data_dir, 'processed', 'images', 'val')
YOLO_LABELS_VAL = os.path.join(data_dir, 'processed', 'labels', 'val')

# Create directories
for dir_path in [YOLO_IMAGES_TRAIN, YOLO_LABELS_TRAIN, YOLO_IMAGES_VAL, YOLO_LABELS_VAL]:
    os.makedirs(dir_path, exist_ok=True)

TRUST = 4  # Number of slices above and below center slice (total 2*TRUST + 1 slices)
BOX_SIZE = 24  # Bounding box size for annotations (in pixels)
TRAIN_SPLIT = 0.8  # 80% for training, 20% for validation
LABELLED_SPLIT = 0.5 # percent of training data with labels/motors


# Image processing functions
def normalize_slice(slice_data: np.ndarray) -> np.ndarray:
    """
    Normalize slice data using 2nd and 98th percentiles
    """
    # Calculate percentiles
    p2 = np.percentile(slice_data, 2)
    p98 = np.percentile(slice_data, 98)
    
    # Clip the data to the percentile range
    clipped_data = np.clip(slice_data, p2, p98)
    
    # Normalize to [0, 255] range
    normalized = 255 * (clipped_data - p2) / (p98 - p2)
    
    return np.uint8(normalized)

    
# Function to process a set of tomograms that includes motors
def process_labelled_set(labels_df: pd.DataFrame, tomogram_ids: np.ndarray, images_dir: str, labels_dir: str, set_name: str) -> Tuple[int, int]:
    motor_counts = []
    for tomo_id in tomogram_ids:
        # Get all motors for this tomogram
        tomo_motors = labels_df[labels_df['tomo_id'] == tomo_id]
        for _, motor in tomo_motors.iterrows():
            if pd.isna(motor['Motor axis 0']):
                continue
            motor_counts.append(
                (tomo_id, 
                    int(motor['Motor axis 0']), 
                    int(motor['Motor axis 1']), 
                    int(motor['Motor axis 2']),
                    int(motor['Array shape (axis 0)']))
            )
    
    print(f'Will process approximately {len(motor_counts) * (2 * TRUST + 1)} labelled slices for {set_name}')
    
    # Process each motor
    processed_slices = 0
    
    for tomo_id, z_center, y_center, x_center, z_max in tqdm(motor_counts, desc=f'Processing {set_name} motors'):
        # Calculate range of slices to include
        z_min = max(0, z_center - TRUST)
        z_max = min(z_max - 1, z_center + TRUST)
        
        # Process each slice in the range
        for z in range(z_min, z_max + 1):
            # Create slice filename
            slice_filename = f'slice_{z:04d}.jpg'
            
            # Source path for the slice
            src_path = os.path.join(TRAIN_DIR, tomo_id, slice_filename)
            
            if not os.path.exists(src_path):
                print(f'Warning: {src_path} does not exist, skipping.')
                continue
            
            # Load and normalize the slice
            img = Image.open(src_path)
            img_array = np.array(img)
            
            # Normalize the image
            normalized_img = normalize_slice(img_array)
            
            # Create destination filename (with unique identifier)
            dest_filename = f'{tomo_id}_z{z:04d}_y{y_center:04d}_x{x_center:04d}.jpg'
            dest_path = os.path.join(images_dir, dest_filename)
            
            # Save the normalized image
            Image.fromarray(normalized_img).save(dest_path)
            
            # Get image dimensions
            img_width, img_height = img.size
            
            # Create YOLO format label
            # YOLO format: <class> <x_center> <y_center> <width> <height>
            # Values are normalized to [0, 1]
            x_center_norm = x_center / img_width
            y_center_norm = y_center / img_height
            box_width_norm = BOX_SIZE / img_width
            box_height_norm = BOX_SIZE / img_height
            
            # Write label file
            label_path = os.path.join(labels_dir, dest_filename.replace('.jpg', '.txt'))
            with open(label_path, 'w') as f:
                f.write(f'0 {x_center_norm} {y_center_norm} {box_width_norm} {box_height_norm}\n')
            
            processed_slices += 1
    
    return processed_slices, len(motor_counts)


def add_unlabelled_images(labels_df: pd.DataFrame) -> Tuple[int, int]:
    # get all tomos from labels that have motors
    labelled_tomos = labels_df[labels_df['Motor axis 0'] != -1.0]\
        [['tomo_id', 'Motor axis 0']]\
        .rename(columns={'Motor axis 0': 'z_index'})
    labelled_tomos.z_index = labelled_tomos.z_index.astype(int)

    labelled_tomos['z_indexes_to_exclude'] = labelled_tomos.z_index.apply(
        lambda z_center: [ 
            ind for ind in list(range(z_center - TRUST, z_center + TRUST + 1)) 
            if ind > -1
        ]
    )

    # get df of tomo ids and z_indexes that are +/- the TRUST value on the z axis
    tomos_to_exclude = labelled_tomos.explode('z_indexes_to_exclude')\
        .drop(columns=['z_index'])\
        .rename(columns={'z_indexes_to_exclude': 'z_index'})

    # put all filepaths in a df, with fpath, tomo_id and z_index
    raw_image_fpaths = []
    for dirpath, dirnames, filenames in os.walk(TRAIN_DIR):
        for filename in filenames:
            if filename.lower().endswith('jpg'):
                raw_image_fpaths.append(os.path.join(dirpath, filename))
    
    raw_images_df = pd.DataFrame({ 'fpath': raw_image_fpaths })

    raw_images_df['tomo_id'] = raw_images_df.fpath\
        .str.split('/train/').str[-1]\
        .str.split('/').str[0]

    raw_images_df['z_index'] = raw_images_df.fpath\
        .str.split('slice_').str[-1]\
        .str.split('.').str[0]\
        .astype(int)

    # get all items from raw_images_df that are not in the the tomos to exclude
    merged_df = pd.merge(
        raw_images_df,
        tomos_to_exclude,
        on=['tomo_id', 'z_index'],
        how='left',
        indicator=True
    )

    unlabelled_tomos = merged_df[merged_df._merge == 'left_only']\
        .drop(columns=['_merge'])\
        .fpath

    print(f'Number of tomo slides with labels: {len(tomos_to_exclude)}, tomos without labels: {len(unlabelled_tomos)}, total num: {len(raw_images_df)}')

    
    import ipdb; ipdb.sset_trace()


def prepare_yolo_dataset():
    '''
    Extract slices containing motors from tomograms and save to YOLO structure with annotations
    '''
    # Load the labels CSV
    labels_df = pd.read_csv(os.path.join(data_dir, 'raw', 'train_labels.csv'))
    
    # Count total number of motors
    total_motors = labels_df['Number of motors'].sum()
    print(f'Total number of motors in the dataset: {total_motors}')
    
    # Get unique tomograms that have motors
    tomo_df = labels_df[labels_df['Number of motors'] > 0].copy()
    unique_tomos = tomo_df['tomo_id'].unique()
    
    print(f'Found {len(unique_tomos)} unique tomograms with motors')
    
    # Perform the train-val split at the tomogram level (not motor level)
    # This ensures all slices from a single tomogram go to either train or val
    np.random.shuffle(unique_tomos)  # Shuffle the tomograms
    split_idx = int(len(unique_tomos) * TRAIN_SPLIT)
    train_tomos = unique_tomos[:split_idx]
    val_tomos = unique_tomos[split_idx:]
    
    print(f'Split: {len(train_tomos)} tomograms for training, {len(val_tomos)} tomograms for validation')

    # Process training tomograms
    #train_slices, train_motors = process_labelled_set(labels_df, train_tomos, YOLO_IMAGES_TRAIN, YOLO_LABELS_TRAIN, "training")
    
    # Process validation tomograms
    #val_slices, val_motors = process_labelled_set(labels_df, val_tomos, YOLO_IMAGES_VAL, YOLO_LABELS_VAL, "validation")

    unlabelled_train_slices_cnt, unlabelled_val_slices_cnt = add_unlabelled_images(labels_df)
    
    '''return {
        "dataset_dir": data_dir,
        "yaml_path": os.path.join(data_dir, 'yolo-config.yaml'),
        "train_tomograms": len(train_tomos),
        "val_tomograms": len(val_tomos),
        "train_motors": train_motors,
        "val_motors": val_motors,
        "train_slices": train_slices,
        "val_slices": val_slices
    }'''
    
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
    prepare_yolo_dataset()