import click
from os.path import join, dirname, abspath
import subprocess
import torch
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
import yaml
from object_detection.dataset.field_data_module import FieldDataModule
from object_detection.models import faster_RCNN_FPN, deformable_DETR
# Set the default precision for matrix multiplication to float32
torch.set_float32_matmul_precision('medium')
# Argument parser for the training script
@click.command()
@click.option('--config', '-c', type=str, help='path to the config file (.yaml)', default=join(dirname(abspath(__file__)), 'config/config.yaml'))
@click.option('--weights', '-w', type=str, help='path to pretrained weights (.ckpt). Use this flag if you just want to load the weights from the checkpoint file without resuming training.', default=None)
@click.option('--checkpoint', '-ckpt', type=str, help='path to checkpoint file (.ckpt) to resume training.', default=None)
# Main function to train the model
def main(config, weights, checkpoint):
    cfg = yaml.safe_load(open(config))
    # Get the git commit version
    try:
        git_commit_version = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip().decode('utf-8')
    except subprocess.CalledProcessError:
        git_commit_version = 'unknown'
    cfg['git_commit_version'] = git_commit_version
    # Initialize the data module
    data = FieldDataModule(cfg)
    # Load the model type from the config file
    model_type = cfg['train']['model']
    # Load the model class based on the model type
    if model_type == 'faster_rcnn':
        model_class = faster_RCNN_FPN.MyFasterRCNN
    elif model_type == 'deformable_DETR':
        model_class = deformable_DETR.MyDeformableDETR
    # Load the model from the checkpoint file if provided
    if checkpoint is not None:
        model = model_class.load_from_checkpoint(checkpoint, hparams=cfg)
    elif weights is not None:
        model = model_class.load_from_checkpoint(weights, hparams=cfg)
    else:
        model = model_class(cfg)
    # Initialize a model checkpoint callback
    checkpoint_saver = ModelCheckpoint(monitor='train_loss', 
                                       save_top_k=5, 
                                       every_n_train_steps=10000, 
                                       filename=cfg['train']['experiment_id']+'_{epoch:02d}_{loss:.2f}',
                                       mode='min', 
                                       save_last=True)
    # Initialize a tensorboard logger (Hyperparameters metrics are not logged)
    tb_logger = pl_loggers.TensorBoardLogger('experiments/'+cfg['train']['experiment_id'], 
                                             default_hp_metric=False)
    # Initialize a trainer
    trainer = Trainer(devices=cfg['train']['n_gpus'], 
                      accelerator='gpu', 
                      logger=tb_logger, 
                      max_epochs=cfg['train']['max_epoch'], 
                      callbacks=[checkpoint_saver],
                      precision=cfg['train']['precision'],)
    # Train the model
    trainer.fit(model, data)
# Entry point of the script
if __name__ == "__main__":
    main()