import torch
from torch import nn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from transformers import DetrForObjectDetection, DetrConfig
import pytorch_lightning as pl
# Define the Deformable DETR model
class MyDeformableDETR(pl.LightningModule):
    def __init__(self, cfg):
        super(MyDeformableDETR, self).__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        num_classes = cfg['train']['num_classes']
        backbone_type = cfg['train']['backbone']
        backbone_weights = cfg['train']['backbone_weights']
        self.backbone = self.build_backbone(backbone_type, backbone_weights)
        if backbone_type == "custom" and backbone_weights:
            # Load custom pretrained backbone
            config = DetrConfig.from_pretrained("facebook/detr-resnet-50")
            self.detr = DetrForObjectDetection(config)
            self.load_backbone_weights(self.detr.model.backbone, backbone_weights)
        elif backbone_type == "default":
            # Load default pretrained backbone
            self.detr = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        else:
            raise ValueError(f"Invalid backbone type: {backbone_type}. Please check the config file. Backbone type must be either 'custom' or 'default'.")
        self.detr.class_labels_classifier = nn.Linear(self.detr.config.d_model, num_classes)
    # Forward pass
    def forward(self, pixel_values, pixel_mask):
        outputs = self.detr(pixel_values=pixel_values, pixel_mask=pixel_mask)
        return outputs
    # Training step
    def training_step(self, batch, batch_idx):
        pixel_values, pixel_mask, targets = batch
        outputs = self(pixel_values, pixel_mask)
        loss_dict = outputs.losses
        losses = sum(loss for loss in loss_dict.values())
        self.log('train_loss', losses, batch_size=len(pixel_values))
        return losses
    # Validation step
    def validation_step(self, batch, batch_idx):
        pixel_values, pixel_mask, targets = batch
        outputs = self(pixel_values, pixel_mask)
        loss_dict = outputs.losses
        losses = sum(loss for loss in loss_dict.values())
        self.log('val_loss', losses, batch_size=len(pixel_values))
        return losses
    # Test step
    def test_step(self, batch, batch_idx):
        pixel_values, pixel_mask, targets = batch
        self.detr.eval()
        with torch.no_grad():
            predictions = self(pixel_values, pixel_mask)
        return predictions
    # Configure the optimizer  
    def configure_optimizers(self):
        if self.cfg['train']['optimizer']['type'] == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), 
                                         lr=self.cfg['train']['optimizer']['lr'], 
                                         weight_decay=self.cfg['train']['optimizer']['weight_decay'])
        elif self.cfg['train']['optimizer']['type'] == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), 
                                        lr=self.cfg['train']['optimizer']['lr'], 
                                        momentum=self.cfg['train']['optimizer']['momentum'], 
                                        weight_decay=self.cfg['train']['weight_decay'])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                         milestones=self.cfg['train']['scheduler']['milestones'], 
                                                         gamma=self.cfg['train']['scheduler']['gamma'])
        return [optimizer], [scheduler]
    # Build the backbone
    def build_backbone(self, backbone_type, backbone_weights):
        if backbone_type == "custom" and backbone_weights:
            backbone = resnet_fpn_backbone('resnet50', pretrained=False)
            self.load_backbone_weights(backbone, backbone_weights)
        else:
            backbone = resnet_fpn_backbone('resnet50', pretrained=True)
        return backbone
    # Load backbone weights
    def load_backbone_weights(self, backbone, weights_path):
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=True)
        backbone.body.load_state_dict(checkpoint['state_dict'], strict=False)
    # Load the checkpoint
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        self.load_state_dict(checkpoint['state_dict'])