import torch
import torch.nn as nn


def mlp_block(in_channels: int, out_channels: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=1),
        nn.BatchNorm1d(out_channels),
        nn.ReLU()
    )


class PointNetSegmentation(nn.Module):

    def __init__(self, number_points: int, num_seg_classes: int, encode_onnx: bool = False):
        super(PointNetSegmentation, self).__init__()
        assert number_points % 8 == 0, f"Number of points must be divisible by 8: {number_points}"
        self.number_points = number_points
        self.num_seg_classes = num_seg_classes
        self.encode_onnx = encode_onnx

        # input-dimension: (batch_size, features (coordinates), number_points)

        # First MLP with weight sharing, implemented as 1d convolution
        self.point_features = nn.Sequential(
            mlp_block(in_channels=3, out_channels=64),
            mlp_block(in_channels=64, out_channels=128),
            #        mlp_block(in_channels=128, out_channels=128),
            mlp_block(in_channels=128, out_channels=256)
        )

        self.global_features = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=1),
            nn.BatchNorm1d(128)
        )

        # max-pooling across samples
        pooling = []
        remaining_dim = self.number_points
        while remaining_dim > 8:
            assert remaining_dim % 8 == 0, "number_points must be recursively divisible by 8"
            pooling.append(nn.MaxPool1d(kernel_size=8, stride=8))
            remaining_dim = remaining_dim // 8
        pooling.append(nn.MaxPool1d(kernel_size=remaining_dim, stride=remaining_dim))
        self.pooling = nn.Sequential(*pooling)

        self.classifier = nn.Sequential(
            mlp_block(576, 256),
            #        mlp_block(256, 256),
            mlp_block(256, 128),
            nn.Conv1d(128, self.num_seg_classes, kernel_size=1)
        )

    def forward(self, x):
        # input is in (batch x num_points x features), but we need (batch x features x num_points) for conv layers
        if not self.encode_onnx:
            if len(x.size()) == 2:
                x = torch.unsqueeze(x, 0)
            assert x.size(1) == self.number_points, f"Expect input of size (N x num_points x features), got {x.size()}"
            assert x.size(2) == 3, f"Expect input of size (N x num_points x features), got {x.size()}"
            x = torch.transpose(x, 2, 1)

        features = []
        for mlp in self.point_features:
            x = mlp(x)
            features.append(x)

        x = self.global_features(x)
        x = self.pooling(x)
        x = x.repeat(1, 1, self.number_points)
        features.append(x)

        x = torch.cat(features, dim=1)

        x = self.classifier(x)
        return x


class MiniPointNetSegmentation(nn.Module):

    def __init__(self, number_points: int, num_seg_classes: int, encode_onnx: bool = False):
        super(MiniPointNetSegmentation, self).__init__()
        assert number_points % 8 == 0, f"Number of points must be divisible by 8: {number_points}"
        self.number_points = number_points
        self.num_seg_classes = num_seg_classes
        self.encode_onnx = encode_onnx
        self.debug = False

        # input-dimension: (batch_size, features (coordinates), number_points)

        # First MLP with weight sharing, implemented as 1d convolution
        self.point_features = nn.Sequential(
            mlp_block(in_channels=3, out_channels=64),
            mlp_block(in_channels=64, out_channels=128)
        )

        self.global_features = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1),
            nn.BatchNorm1d(128)
        )

        # max-pooling across samples
        pooling = []
        remaining_dim = self.number_points
        while remaining_dim > 8:
            assert remaining_dim % 8 == 0, "number_points must be recursively divisible by 8"
            pooling.append(nn.MaxPool1d(kernel_size=8, stride=8))
            remaining_dim = remaining_dim // 8
        pooling.append(nn.MaxPool1d(kernel_size=remaining_dim, stride=remaining_dim))
        self.pooling = nn.Sequential(*pooling)

        self.classifier = nn.Sequential(
            mlp_block(320, 128),
            nn.Conv1d(128, self.num_seg_classes, kernel_size=1)
        )

    def forward(self, x):
        # input is in (batch x num_points x features), but we need (batch x features x num_points) for conv layers
        layer_results = []
        if not self.encode_onnx:
            if len(x.size()) == 2:
                x = torch.unsqueeze(x, 0)
            assert x.size(1) == self.number_points, f"Expect input of size (N x num_points x features), got {x.size()}"
            assert x.size(2) == 3, f"Expect input of size (N x num_points x features), got {x.size()}"
            x = torch.transpose(x, 2, 1)

        features = []
        for mlp in self.point_features:
            x = mlp(x)
            layer_results.append(x.detach().cpu().numpy())
            features.append(x)

        x = self.global_features(x)
        x = self.pooling(x)
        x = x.repeat(1, 1, self.number_points)
        features.append(x)

        x = torch.cat(features, dim=1)
        layer_results.append(x.detach().cpu().numpy())

        x = self.classifier(x)
        layer_results.append(x.detach().cpu().numpy())
        if self.debug:
            return x, layer_results
        else:
            return x
