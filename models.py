import monai
import torch
from torch import nn


class Resnet10(nn.Module):
    def __init__(self, n_classes: int = 7):
        super().__init__()
        self.backbone = monai.networks.nets.resnet10(spatial_dims=2, n_input_channels=3)
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=1024, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=n_classes, bias=True)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.backbone(x))


class Resnet50(nn.Module):
    def __init__(self, n_classes: int = 7):
        super().__init__()
        self.backbone = monai.networks.nets.resnet50(spatial_dims=2, n_input_channels=3)
        self.backbone.fc = nn.Linear(in_features=2048, out_features=n_classes, bias=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.backbone(x))


class EfficientNetB0(nn.Module):
    def __init__(self, n_classes: int = 7):
        super().__init__()
        self.backbone = monai.networks.nets.EfficientNetBN("efficientnet-b0",
                                                           spatial_dims=2,
                                                           in_channels=3,
                                                           pretrained=True,
                                                           num_classes=n_classes)
        # self.backbone.fc = nn.Sequential(
        #     nn.Linear(in_features=512, out_features=1024, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(in_features=1024, out_features=512, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(in_features=512, out_features=n_classes, bias=True)
        # )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.backbone(x))


class EfficientNetB1(nn.Module):
    def __init__(self, n_classes: int = 7):
        super().__init__()
        self.backbone = monai.networks.nets.EfficientNetBN("efficientnet-b1",
                                                           spatial_dims=2,
                                                           in_channels=3,
                                                           pretrained=True,
                                                           num_classes=n_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.backbone(x))


class DenseNet(nn.Module):
    def __init__(self, n_classes: int = 7):
        super().__init__()
        self.backbone = monai.networks.nets.DenseNet121(spatial_dims=2, in_channels=3, out_channels=n_classes,
                                                        pretrained=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.backbone(x))


class SeNet50(nn.Module):
    def __init__(self, n_classes: int = 7):
        super().__init__()
        self.backbone = monai.networks.nets.SEResNet50(spatial_dims=2, in_channels=3, num_classes=n_classes,
                                                       pretrained=True,dropout_prob=.7)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.backbone(x))


if __name__ == '__main__':
    model = Resnet50()
    print(model)
