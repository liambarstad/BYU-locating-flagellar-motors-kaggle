import os
import wandb
import argparse
from ultralytics import YOLO

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


def log_metrics(run_dir: str):
    pass

def train():
    yolo = YOLO('yolo12n.pt')

    config = {
        'epochs': 100,
        'batch_size': 64,
        'imgsz': 640,
        'patience': 5
    }

    wandb.init(
        project='BYU-locating-flagellar-motors',
        name=args.experiment_name,
        config=config
    )

    yolo_weights_dir = os.path.join(args.data_dir, 'output', 'yolo-weights')

    results = yolo.train(
        data=os.path.join(os.getcwd(), 'yolo-config.yaml'),
        project=yolo_weights_dir,
        name='motor_detector',
        exist_ok=True,
        save_period=5,
        val=True,
        verbose=True,
        device='cuda'
    )

    # Get the path to the run directory
    log_metrics(os.path.join(yolo_weights_dir, 'motor_detector'))
    

if __name__ == '__main__':
    train()