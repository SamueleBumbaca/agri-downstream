import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from data_loader import ImageDataset
from data_augmentation.vae_model import VAE
from data_augmentation.config import create_config
import click
from datetime import datetime
import argparse


def train(model, dataloader, optimizer, device, writer=None):
    model.train()
    for batch_idx, data in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = output.loss
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Step {batch_idx}, Loss: {loss.item():.4f}')
            if writer:
                writer.add_scalar('Loss/Train', loss.item(), batch_idx)

def main(config_path):
    # Load configuration
    config = create_config(config_path)
    
    # Transformations
    transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
    ])
    
    # Dataset and DataLoader
    dataset = ImageDataset(config, transform=transform)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    # Model, Optimizer, and Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = 3 * config.image_size[0] * config.image_size[1]
    model = VAE(input_dim = input_dim, 
                hidden_dim = config.hidden_dim, 
                latent_dim = config.latent_dim,
                image_size = config.image_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), 
                                    lr=config.learning_rate, 
                                    weight_decay=config.weight_decay)
    
    # TensorBoard Writer
    writer = SummaryWriter(os.path.join(config.output_dir,
                                        f'vae_{datetime.now().strftime("%Y%m%d-%H%M%S")}')
                                    )
    
    # Training Loop
    for epoch in range(config.num_epochs):
        print(f'Epoch {epoch+1}/{config.num_epochs}')
        train(model, dataloader, optimizer, device, writer)
    
    # Save the model
    torch.save(model.state_dict(), os.path.join(config.output_dir, 'vae_model.pth'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a VAE model.')
    parser.add_argument('-c',
                        '--config', 
                        type=str, 
                        required=True, 
                        help='Path to the configuration file.')
    args = parser.parse_args()
    main(args.config)