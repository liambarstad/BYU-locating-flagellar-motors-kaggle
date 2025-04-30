import os
import wandb
import argparse
from ultralytics import YOLO

parser = argparse.ArgumentParser(description='train yolo model')
parser.add_argument(
    '--experiment_name',
    type=str,
    required=True,
    help='The unique name or identifier for this experiment run.'
)

args = parser.parse_args()


def train():
    wandb.init(
        project='BYU-locating-flagellar-motors',
        name=args.experiment_name,
        config={}
    )

    yolo = YOLO('yolo12n.pt')


if __name__ == '__main__':
    train()