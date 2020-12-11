import torch
import torch.nn as nn
import torch.nn.functional as F

import sys 
sys.path.append("..") 
sys.path.append("../..") 

from global_utils import get_backbone

# Channel Attention Module
class CABlock(nn.Module):
    def __init__(self, in_channels, resize_factor=4):
        super(CABlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        hid_channels = in_channels // resize_factor
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, hid_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid_channels, in_channels, kernel_size=1, bias=False),
        )

    def forward(self, image_features):
        avg_pool_weights = self.fc(self.avg_pool(image_features))
        max_pool_weights = self.fc(self.max_pool(image_features))
        weights = torch.sigmoid(avg_pool_weights + max_pool_weights)

        return image_features * weights, weights


# Semantic Channel Attention Block
class SCABlock(nn.Module):
    def __init__(self, in_channels, semantic_size, resize_factor=4):
        super(SCABlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        hid_channels = in_channels // resize_factor
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels+semantic_size, hid_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid_channels, in_channels, kernel_size=1, bias=False),
        )


    def forward(self, image_features, semantic_features):

        avg_pooled_image_features = self.avg_pool(image_features)
        max_pooled_image_features = self.max_pool(image_features)
        
        # concat in channel dimension
        avg_pooled_features = torch.cat((avg_pooled_image_features, semantic_features), 1)
        max_pooled_features = torch.cat((max_pooled_image_features, semantic_features), 1)
        avg_pool_weights = self.fc(avg_pooled_features)
        max_pool_weights = self.fc(max_pooled_features)
        weights = torch.sigmoid(avg_pool_weights + max_pool_weights)

        return image_features * weights, weights


# Spatial Attention Module
class SABlock(nn.Module):
    def __init__(self):
        super(SABlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, image_features):
        transpose_features = image_features.view(*image_features.shape[:2], -1).transpose(1, 2)
        avg_pooled_features = self.avg_pool(transpose_features)
        max_pooled_features = self.max_pool(transpose_features)
        pooled_features = torch.cat((avg_pooled_features, max_pooled_features), 2)
        pooled_features = pooled_features.transpose(1, 2).view(-1, 2, *image_features.shape[2:])
        weights = torch.sigmoid(self.conv(pooled_features))

        return image_features * weights, weights


# Semantic Spatial Attention Module
class SSABlock(nn.Module):
    def __init__(self):
        super(SSABlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, image_features, semantic_features):
        features = torch.cat((image_features, semantic_features.expand(*semantic_features.shape[:2], *image_features.shape[2:])), 1)    # broadcast along height and width dimension
        transpose_features = features.view(*features.shape[:2], -1).transpose(1, 2)
        avg_pooled_features = self.avg_pool(transpose_features)
        max_pooled_features = self.max_pool(transpose_features)
        pooled_features = torch.cat((avg_pooled_features, max_pooled_features), 2)
        pooled_features = pooled_features.transpose(1, 2).view(-1, 2, *image_features.shape[2:])
        weights = torch.sigmoid(self.conv(pooled_features))

        return image_features * weights, weights


class ProtoNetAGAM(nn.Module):
    def __init__(self, backbone, semantic_size, out_channels):
        super(ProtoNetAGAM, self).__init__()
        self.encoder = get_backbone(backbone)

        self.ca_block = CABlock(out_channels)
        self.sca_block = SCABlock(out_channels, semantic_size)
        self.sa_block = SABlock()
        self.ssa_block = SSABlock()

    def forward(self, inputs, semantics=None, output_weights=False):

        input_semantics = True if semantics is not None else False

        embeddings = self.encoder(inputs.view(-1, *inputs.shape[2:]))
        # attention after the last conv
        if input_semantics:    # attributes-guided

            semantics = semantics.float().view(-1, semantics.shape[2], 1, 1)

            ca_embeddings, ca_weights = self.ca_block(embeddings)
            sca_embeddings, sca_weights = self.sca_block(embeddings, semantics)
            embeddings = sca_embeddings

            sa_embeddings, sa_weights = self.sa_block(embeddings)
            ssa_embeddings, ssa_weights = self.ssa_block(embeddings, semantics)
            embeddings = ssa_embeddings

            if output_weights:
                return embeddings.view(*inputs.shape[:2], -1), ca_weights, sca_weights, sa_weights, ssa_weights

        else:    # self-guided
            ca_embeddings, ca_weights = self.ca_block(embeddings)
            embeddings = ca_embeddings
            sa_embeddings, sa_weights = self.sa_block(embeddings)
            embeddings = sa_embeddings

            if output_weights:
                return embeddings.view(*inputs.shape[:2], -1), ca_weights, sa_weights

        return embeddings.view(*inputs.shape[:2], -1)
