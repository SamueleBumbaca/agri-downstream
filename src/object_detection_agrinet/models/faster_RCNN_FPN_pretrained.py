import pytorch_lightning as pl
import torch
import torchvision
from torch import nn

class MyFasterRCNN(pl.LightningModule):
    def __init__(self, num_classes, backbone_type='resnet50', backbone_weights=None):
        super(MyFasterRCNN, self).__init__()
        self.save_hyperparameters()
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        self.log('train_loss', losses)
        return losses

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        self.log('val_loss', losses)
        return losses

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
        return optimizer

class MyCustomBackboneFasterRCNN(pl.LightningModule):
    def __init__(self, num_classes, backbone_type='resnet50', backbone_weights=None):
        super(MyCustomBackboneFasterRCNN, self).__init__()
        self.save_hyperparameters()
        backbone = torchvision.models.resnet50(pretrained=False)
        if backbone_weights:
            backbone.load_state_dict(torch.load(backbone_weights))
        backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.model = torchvision.models.detection.FasterRCNN(backbone, num_classes=num_classes)

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        self.log('train_loss', losses)
        return losses

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        self.log('val_loss', losses)
        return losses

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
        return optimizer