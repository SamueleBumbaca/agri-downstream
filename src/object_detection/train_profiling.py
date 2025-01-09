import click
from os.path import join, dirname, abspath
import subprocess
import torch
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
    data.setup()
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

    # Move the model to GPU
    model = model.cuda()
    optimizer = model.configure_optimizers()[0][0]

    # Profiling
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        # Profile only a few batches
        for epoch in range(1):  # Profile only the first epoch
            for i, (batch_data, batch_labels) in enumerate(data.train_dataloader()):
                if i >= 10:  # Profile only the first 10 batches
                    break
                # Unpack the list elements
                batch_data = [d.cuda() for d in batch_data]
                batch_labels = [{k: v.cuda() for k, v in l.items()} for l in batch_labels]

                optimizer.zero_grad()
                output = model(batch_data,batch_labels)
                loss = sum(loss for loss in output.values())
                loss.backward()
                optimizer.step()
                prof.step()  # Step the profiler to record the current batch
    
    # Write profiling results to a log file
    profile_log_path = f"experiments/{cfg['train']['experiment_id']}/profile_log.txt"
    with open(profile_log_path, 'w') as f:
        f.write(prof.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=-1))

# Entry point of the script
if __name__ == "__main__":
    main()