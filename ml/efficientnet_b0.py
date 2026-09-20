"""
PyTorch implementation of EfficientNet-B0 and ResNet-50 Architectures for Crop Disease Detection.
Natively defined using PyTorch nn modules (MBConv, Squeeze-and-Excitation, SiLU activation).
"""

import math
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_OK = True
except Exception:
    _TORCH_OK = False
    class _FakeModule:
        pass
    class nn:
        Module = _FakeModule
        Sequential = _FakeModule

# PlantVillage 38 Class Names
PLANTVILLAGE_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Cotton___Bacterial_blight",
    "Cotton___Leaf_curl_virus",
    "Cotton___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Rice___Bacterial_leaf_blight",
    "Rice___Brown_spot",
    "Rice___Leaf_smut",
    "Rice___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
    "Wheat___Leaf_rust",
    "Wheat___Powdery_mildew",
    "Wheat___healthy"
]


class Swish(nn.Module):
    """SiLU / Swish Activation function: x * sigmoid(x)."""
    def forward(self, x):
        return x * torch.sigmoid(x)


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for channel-wise attention."""
    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_dim, kernel_size=1),
            Swish(),
            nn.Conv2d(reduced_dim, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Conv (MBConv) Block."""
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio=0.25):
        super(MBConvBlock, self).__init__()
        self.stride = stride
        self.use_residual = (self.stride == 1 and in_channels == out_channels)
        expanded_channels = in_channels * expand_ratio

        # Expansion phase
        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                Swish()
            )
        else:
            self.expand_conv = nn.Identity()

        # Depthwise Conv phase
        padding = (kernel_size - 1) // 2
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                expanded_channels, expanded_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, groups=expanded_channels, bias=False
            ),
            nn.BatchNorm2d(expanded_channels),
            Swish()
        )

        # Squeeze-and-Excitation
        reduced_dim = max(1, int(in_channels * se_ratio))
        self.se = SqueezeExcitation(expanded_channels, reduced_dim)

        # Output / Projection phase
        self.project_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        residual = x
        out = self.expand_conv(x)
        out = self.depthwise_conv(out)
        out = self.se(out)
        out = self.project_conv(out)

        if self.use_residual:
            out = out + residual
        return out


class EfficientNetB0(nn.Module):
    """EfficientNet-B0 Architecture for 38 PlantVillage Classes."""
    def __init__(self, num_classes=len(PLANTVILLAGE_CLASSES)):
        super(EfficientNetB0, self).__init__()
        self.model_name = "EfficientNet-B0"

        # Stem Conv
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            Swish()
        )

        # MBConv Stages Config: (in_c, out_c, k, stride, expand, repeats)
        stage_configs = [
            (32,  16, 3, 1, 1, 1),
            (16,  24, 3, 2, 6, 2),
            (24,  40, 5, 2, 6, 2),
            (40,  80, 3, 2, 6, 3),
            (80, 112, 5, 1, 6, 3),
            (112, 192, 5, 2, 6, 4),
            (192, 320, 3, 1, 6, 1)
        ]

        blocks = []
        for in_c, out_c, k, s, exp, repeats in stage_configs:
            for i in range(repeats):
                stride = s if i == 0 else 1
                input_c = in_c if i == 0 else out_c
                blocks.append(MBConvBlock(input_c, out_c, kernel_size=k, stride=stride, expand_ratio=exp))
        
        self.blocks = nn.Sequential(*blocks)

        # Head Conv & Classifier
        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            Swish(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class ResNet50(nn.Module):
    """ResNet-50 Architecture Option for Model Comparison."""
    def __init__(self, num_classes=len(PLANTVILLAGE_CLASSES)):
        super(ResNet50, self).__init__()
        self.model_name = "ResNet-50"
        
        # Initial Conv Layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Feature Extractor
        self.layer1 = self._make_layer(64, 64, 3, stride=1)
        self.layer2 = self._make_layer(256, 128, 4, stride=2)
        self.layer3 = self._make_layer(512, 256, 6, stride=2)
        self.layer4 = self._make_layer(1024, 512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def _make_layer(self, in_c, out_c, blocks, stride=1):
        layers = []
        # Downsample block
        layers.append(nn.Sequential(
            nn.Conv2d(in_c, out_c * 4, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_c * 4),
            nn.ReLU(inplace=True)
        ))
        for _ in range(1, blocks):
            layers.append(nn.Sequential(
                nn.Conv2d(out_c * 4, out_c * 4, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c * 4),
                nn.ReLU(inplace=True)
            ))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
