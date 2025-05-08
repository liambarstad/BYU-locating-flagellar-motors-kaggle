import os
import shutil
import wandb
import argparse
import yaml
import pandas as pd
from ultralytics import YOLO
from ray import tune

parser = argparse.ArgumentParser(description='train yolo model')
parser.add_argument(
    '--data_dir',
    type=str,
    required=True,
    help='The absolute path to the data to be used and modified'
)
parser.add_argument(
    '--experiment_name',
    type=str,
    required=True,
    help='The unique name or identifier for this experiment run.'
)

args = parser.parse_args()

# optimizer
# cos_lr=False cos learning rate
# lr0=0.01, initial learning rate
# lrf=0.01, final learning rate as fraction of initial
# momentum=0.937
# weight_decay=0.0005 L2 regularization term
# warmup_epochs=3.0 learning rate warmup
# warmup_momentum=0.8 starting value to warm up from
# warmup_bias_lr=0.1 lr for bias parameters during warmup phase

config = {
    'use_ray': True,
    'epochs': 50,
    'iterations': 5,
    'batch': 64,
    'imgsz': 640,
    'cos_lr': True
}

tune_params = {
    "lr0": (1e-5, 1e-1),  # initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
    "lrf": (0.0001, 0.1),  # final OneCycleLR learning rate (lr0 * lrf)
    "momentum": (0.98, 0.3),  # SGD momentum/Adam beta1
    "weight_decay": (0.0, 0.001),  # optimizer weight decay 5e-4
    "warmup_epochs": (0.0, 5.0),  # warmup epochs (fractions ok)
    "warmup_momentum": (0.0, 0.95),  # warmup initial momentum
}


def log_metrics(run_dir: str):
    beta = 2
    results_file = os.path.join(run_dir, 'results.csv')
    if os.path.isfile(results_file):
        # is training job
        results = pd.read_csv(results_file)
    else:
        # is tuning job
        train_run_indexes = [ 
            name.split('train')[1] for name in os.listdir(os.path.dirname(run_dir)) if 'train' in name
        ]
        latest_train_dir = 'train' + max(train_run_indexes)
        results = pd.read_csv(os.path.join(os.path.dirname(run_dir), latest_train_dir, 'results.csv'))

        with open(os.path.join(latest_train_dir, 'args.yaml'), 'r') as best_args_f:
            best_args = yaml.safe_load(best_args_f)
            wandb.config.update({
                key: best_args[key]
                for key in tune_params
            }) 


    for _, row in results.iterrows():
        precision = row['metrics/precision(B)']
        recall = row['metrics/recall(B)']
        wandb.log({
            'epoch': row.epoch,
            'precision': precision,
            'recall': recall,
            'weighted_f1': (precision * recall) / (((beta**2)*precision) + recall)
            **{ col: row[col] for col in row.index }
        })


def train():
    yolo = YOLO('yolo12n.pt')

    run = wandb.init(
        project='BYU-locating-flagellar-motors',
        name=args.experiment_name,
        config={**config, **tune_params}
    )

    yolo_weights_dir = os.path.join(args.data_dir, 'output', 'yolo-weights')
    if os.path.exists(yolo_weights_dir):
        shutil.rmtree(yolo_weights_dir)

    results = yolo.tune(
        data=os.path.join(os.getcwd(), 'yolo-config.yaml'),
        project=yolo_weights_dir,
        name='motor_detector',
        val=True,
        verbose=True,
        device='cuda',
        single_cls=True,
        space={
            key: tune.uniform(*tune_params[key])
            for key in tune_params
        },
        **config
    )

    # Get the path to the run directory
    log_metrics(os.path.join(yolo_weights_dir, 'motor_detector'))
    
    run.finish()

if __name__ == '__main__':
    train()