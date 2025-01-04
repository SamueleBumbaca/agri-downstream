import click
from os.path import join, dirname, abspath
import subprocess
import torch
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
import yaml
import self_supervised_agrinet.datasets.datasets as datasets
import self_supervised_agrinet.models.models as models

torch.set_float32_matmul_precision('medium')

@click.command()
### Add your options here
@click.option('--config',
              '-c',
              type=str,
              help='path to the config file (.yaml)',
              default=join(dirname(abspath(__file__)),'config/config.yaml'))
@click.option('--weights',
              '-w',
              type=str,
              help='path to pretrained weights (.ckpt). Use this flag if you just want to load the weights from the checkpoint file without resuming training.',
              default=None)
@click.option('--checkpoint',
              '-ckpt',
              type=str,
              help='path to checkpoint file (.ckpt) to resume training.',
              default=None)

def main(config,weights,checkpoint):
    cfg = yaml.safe_load(open(config))

    # Handle Git commit version
    try:
        git_commit_version = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip().decode('utf-8')
    except subprocess.CalledProcessError:
        git_commit_version = 'unknown'
    # save the version of git we're using 
    cfg['git_commit_version'] = git_commit_version

    # Load data and model
    data = datasets.StatDataModule(cfg)    
    if checkpoint is not None:
        model = models.BarlowTwins.load_from_checkpoint(checkpoint, hparams=cfg)
    elif weights is not None:
        model = models.BarlowTwins.load_from_checkpoint(weights, hparams=cfg)
    else:
        model = models.BarlowTwins(cfg)
        
    # Add callbacks:
    #lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_saver = ModelCheckpoint(monitor='train:loss',
                                 save_top_k=5,
                                 every_n_train_steps = 10000,
                                 filename=cfg['experiment']['id']+'_{epoch:02d}_{loss:.2f}',
                                 mode='min',
                                 save_last=True)


    tb_logger = pl_loggers.TensorBoardLogger('experiments/'+cfg['experiment']['id'],
                                             default_hp_metric=False)

    # Setup trainer
    trainer = Trainer(devices=cfg['train']['n_gpus'],
                      accelerator='gpu',
                      logger=tb_logger,
                      max_epochs= cfg['train']['max_epoch'],
                      callbacks=[checkpoint_saver])
    # Train
    trainer.fit(model, data)

if __name__ == "__main__":
    main()
