from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import ResNet50_Weights
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
import pytorch_lightning as pl
import torch

# Define the Faster RCNN model
class MyFasterRCNN(pl.LightningModule):
    def __init__(self, cfg):
        super(MyFasterRCNN, self).__init__()
        self.save_hyperparameters()
        # Load the configuration
        self.cfg = cfg
        num_classes = cfg['train']['num_classes']
        backbone_type = cfg['train']['backbone']
        backbone_weights = cfg['train']['backbone_weights']
        # Calculate the anchor size
        anchor_size = int(cfg['dataset']['min_dist_on_row'] / cfg['dataset']['resolution'])
        # Load the backbone
        if backbone_type == "custom" and backbone_weights:
            backbone = resnet_fpn_backbone(backbone_name='resnet50',
                                           weights=None)
            self.load_backbone_weights(backbone, backbone_weights)
        elif backbone_type == "default":
            backbone = resnet_fpn_backbone(backbone_name='resnet50',
                                            weights=ResNet50_Weights.DEFAULT)
        else:
            raise ValueError(f"Invalid backbone type: {backbone_type}. Please check the config file. Backbone type must be either 'custom' or 'default'.")
        
        # Define the anchor generator with the calculated size
        anchor_sizes = ((anchor_size*2.3,), 
                        (anchor_size*2.4,), 
                        (anchor_size*2.5,), 
                        (anchor_size*2.6,), 
                        (anchor_size*2.7,))
        aspect_ratios = ((1.0),) * len(anchor_sizes)
        rpn_anchor_generator = AnchorGenerator(sizes=anchor_sizes,
                                               aspect_ratios=aspect_ratios)
        
        # Define the ROI Pooler
        roi_pooler = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],  # Use only one level of the FPN
            output_size=7,  # Output size of the pooled feature maps
            sampling_ratio=2  # Sampling ratio for ROI pooling
        )
        
        # Initialize the model
        self.model = FasterRCNN(
            backbone, 
            num_classes=num_classes,
            rpn_anchor_generator=rpn_anchor_generator,
            box_roi_pool=roi_pooler
        )
    
    # Load the backbone weights
    def load_backbone_weights(self, backbone, weights_path):
        checkpoint = torch.load(weights_path, 
                                map_location='cpu', 
                                weights_only=True)
        backbone.body.load_state_dict(checkpoint['state_dict'], strict=False)
    # Load the checkpoint
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, 
                                map_location='cpu', 
                                weights_only=True)
        self.model.load_state_dict(checkpoint['state_dict'], strict=False)
    # Forward pass
    def forward(self, images, targets=None):
        return self.model(images, targets)
    # Training step
    def training_step(self, batch, batch_idx):
        images, targets = batch
        # Convert the batch of images to a list of 3D tensors
        images = [img.squeeze(0).cuda(non_blocking=True) for img in torch.split(images, 1, dim=0)]
        # Convert the batch of targets to a list of dictionaries
        targets = [{k: v.squeeze(0).cuda(non_blocking=True) for k, v in t.items()} for t in targets]
        # Ensure the model is in training mode to get the loss dictionary
        self.model.train()
        loss_dict = self.model(images, targets)
        if isinstance(loss_dict, dict):
            losses = sum(loss for loss in loss_dict.values())
        else:
            raise TypeError(f"Unexpected type for loss_dict: {type(loss_dict)}, expected dict. Please check the forward method of the model. Probably the model is not in training mode.")
        self.log('train_loss', losses, batch_size=len(images))
        return losses
    # Validation step
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        # Convert the batch of images to a list of 3D tensors
        images = [img.squeeze(0).cuda(non_blocking=True) for img in torch.split(images, 1, dim=0)]
        # Convert the batch of targets to a list of dictionaries
        targets = [{k: v.squeeze(0).cuda(non_blocking=True) for k, v in t.items()} for t in targets]
        # Ensure the model is in training mode to get the loss dictionary
        self.model.train()
        loss_dict = self.model(images, targets)
        # Ensure loss_dict is a dictionary and sum the losses
        if isinstance(loss_dict, dict):
            losses = sum(loss for loss in loss_dict.values())
        else:
            raise TypeError(f"Unexpected type for loss_dict: {type(loss_dict)}, expected dict. Please check the forward method of the model. Probably the model is not in training mode.")
        self.log('val_loss', losses, batch_size=len(images))
        return losses
    # Test step
    def test_step(self, batch, batch_idx):
        images, targets = batch
        # Convert the batch of images to a list of 3D tensors
        images = [img.squeeze(0).cuda(non_blocking=True) for img in torch.split(images, 1, dim=0)]
        # Convert the batch of targets to a list of dictionaries
        targets = [{k: v.squeeze(0).cuda(non_blocking=True) for k, v in t.items()} for t in targets]
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(images)
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
        # return [optimizer], [scheduler]
        return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',  # Call scheduler.step() after each epoch
                    'frequency': 1        # Adjust frequency if needed
                }
            }
                