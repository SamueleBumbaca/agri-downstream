import torch
from torch import nn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from transformers import DetrForObjectDetection, DetrConfig

class MyDeformableDETR(nn.Module):
    def __init__(self, num_classes, backbone_type="default", backbone_weights=None):
        super(MyDeformableDETR, self).__init__()
        self.backbone = build_backbone(backbone_type, backbone_weights)
        self.detr = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        self.detr.class_labels_classifier = nn.Linear(self.detr.config.d_model, num_classes)

    def forward(self, pixel_values, pixel_mask):
        outputs = self.detr(pixel_values=pixel_values, pixel_mask=pixel_mask)
        return outputs

def build_backbone(backbone_type, backbone_weights):
    if backbone_type == "custom" and backbone_weights:
        backbone = resnet_fpn_backbone('resnet50', pretrained=False)
        load_backbone_weights(backbone, backbone_weights)
    else:
        backbone = resnet_fpn_backbone('resnet50', pretrained=True)
    return backbone

def load_backbone_weights(backbone, weights_path):
    checkpoint = torch.load(weights_path, map_location='cpu')
    backbone.body.load_state_dict(checkpoint['state_dict'], strict=False)