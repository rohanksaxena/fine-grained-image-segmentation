import torch
import torch.nn as nn
# from torch_geometric.nn import MessagePassing
from torchvision.models import vgg11_bn, VGG11_BN_Weights, resnet18, ResNet18_Weights
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights
from lib.ssn.ssn import ssn_iter, sparse_ssn_iter


def conv_bn_relu(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(True)
    )


class SSNModel(nn.Module):
    def __init__(self, feature_dim, nspix, training, n_iter=10):
        super().__init__()
        self.nspix = nspix
        self.n_iter = n_iter
        self.training = training

        self.scale1 = nn.Sequential(
            conv_bn_relu(5, 64),
            conv_bn_relu(64, 64)
        )
        self.scale2 = nn.Sequential(
            nn.MaxPool2d(3, 2, padding=1),
            conv_bn_relu(64, 64),
            conv_bn_relu(64, 64)
        )
        self.scale3 = nn.Sequential(
            nn.MaxPool2d(3, 2, padding=1),
            conv_bn_relu(64, 64),
            conv_bn_relu(64, 64)
        )

        self.output_conv = nn.Sequential(
            nn.Conv2d(64 * 3 + 5, feature_dim - 5, 3, padding=1),
            nn.ReLU(True)
        )
        # self.output_conv = nn.Sequential(
        #     conv_bn_relu(64 * 3 + 5, 100),
        #     conv_bn_relu(100, 50),
        #     conv_bn_relu(50, feature_dim - 5)
        #
        # )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        pixel_f = self.feature_extract(x)

        if self.training:
            # return *ssn_iter(pixel_f, self.nspix, self.n_iter)
            return ssn_iter(pixel_f, self.nspix, self.n_iter)
        else:
            # return *sparse_ssn_iter(pixel_f, self.nspix, self.n_iter)
            return sparse_ssn_iter(pixel_f, self.nspix, self.n_iter)

    def feature_extract(self, x):
        s1 = self.scale1(x)
        s2 = self.scale2(s1)
        s3 = self.scale3(s2)

        s2 = nn.functional.interpolate(s2, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        s3 = nn.functional.interpolate(s3, size=s1.shape[-2:], mode="bilinear", align_corners=False)

        cat_feat = torch.cat([x, s1, s2, s3], 1)
        feat = self.output_conv(cat_feat)

        return torch.cat([feat, x], 1)


class SSN_DINO(nn.Module):
    def __init__(self, feature_dim, nspix, training, n_iter=10):
        super().__init__()
        self.nspix = nspix
        self.n_iter = n_iter
        self.training = training
        self.backbone = torch.hub.load('facebookresearch/dino:main', "dino_vits16")
        self.output_conv = nn.Sequential(
            conv_bn_relu(389, 256),
            conv_bn_relu(256, 64),
            conv_bn_relu(64, feature_dim - 5)
        )

    def forward(self, x, coords):
        pixel_f, tokens = self.feature_extract(x, coords)
        if self.training:
            return *ssn_iter(pixel_f, self.nspix, self.n_iter), tokens
        else:
            return *sparse_ssn_iter(pixel_f, self.nspix, self.n_iter), tokens

    def feature_extract(self, x, coords):
        # print(f"coords: {coords.shape}, x: {x.shape}")
        height, width = x.shape[2:]

        # Define hook
        feat_out = {}

        def hook_dino_out(module, input, output):
            feat_out["qkv"] = output

        self.backbone._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(hook_dino_out)
        out = self.backbone(x)

        num_heads = self.backbone.blocks[0].attn.num_heads
        P = 16
        B, C, H, W = x.shape
        H_patch, W_patch = H // P, W // P
        H_pad, W_pad = H_patch * P, W_patch * P
        T = H_patch * W_patch + 1  # number of tokens, add 1 for [CLS]
        images = x[:, :, :H_pad, :W_pad]
        self.backbone.get_intermediate_layers(images)[0].squeeze(0)
        output_qkv = feat_out["qkv"].reshape(B, T, 3, num_heads, -1 // num_heads).permute(2, 0, 3, 1, 4)
        out = output_qkv[1].transpose(1, 2).reshape(B, T, -1)[:, 1:, :]
        features = out.reshape(B, 384, H_patch, W_patch)
        ######

        features = nn.functional.interpolate(features, size=(height, width), mode="bilinear", align_corners=False)
        features = torch.cat([x, coords, features], 1)
        features = self.output_conv(features)
        features = torch.cat([features, x, coords], 1)
        return features, out


class SSN_VGG(nn.Module):
    def __init__(self, feature_dim, nspix, training, n_iter=10):
        super().__init__()
        self.nspix = nspix
        self.n_iter = n_iter
        self.backbone = vgg11_bn(weights=VGG11_BN_Weights).features
        self.relu_layer_indices = ("2", "6", "10", "13", "17", "20", "24", "27")
        self.output_conv = nn.Sequential(
            conv_bn_relu(2757, 500),
            conv_bn_relu(500, 100),
            conv_bn_relu(100, feature_dim - 5)
        )

    def feature_extract(self, x, coords):
        height, width = x.shape[2:]
        input = x
        feat_out = {}

        def hook_vgg_2(module, input, output):
            feat_out["2"] = output

        def hook_vgg_6(module, input, output):
            feat_out["6"] = output

        def hook_vgg_10(module, input, output):
            feat_out["10"] = output

        def hook_vgg_13(module, input, output):
            feat_out["13"] = output

        def hook_vgg_17(module, input, output):
            feat_out["17"] = output

        def hook_vgg_20(module, input, output):
            feat_out["20"] = output

        def hook_vgg_24(module, input, output):
            feat_out["24"] = output

        def hook_vgg_27(module, input, output):
            feat_out["27"] = output

        self.backbone._modules["2"].register_forward_hook(hook_vgg_2)
        self.backbone._modules["6"].register_forward_hook(hook_vgg_6)
        self.backbone._modules["10"].register_forward_hook(hook_vgg_10)
        self.backbone._modules["13"].register_forward_hook(hook_vgg_13)
        self.backbone._modules["17"].register_forward_hook(hook_vgg_17)
        self.backbone._modules["20"].register_forward_hook(hook_vgg_20)
        self.backbone._modules["24"].register_forward_hook(hook_vgg_24)
        self.backbone._modules["27"].register_forward_hook(hook_vgg_27)
        out = self.backbone(x)

        s1 = nn.functional.interpolate(feat_out["2"], size=x.shape[-2:], mode="bilinear", align_corners=False)
        s2 = nn.functional.interpolate(feat_out["6"], size=x.shape[-2:], mode="bilinear", align_corners=False)
        s3 = nn.functional.interpolate(feat_out["10"], size=x.shape[-2:], mode="bilinear", align_corners=False)
        s4 = nn.functional.interpolate(feat_out["13"], size=x.shape[-2:], mode="bilinear", align_corners=False)
        s5 = nn.functional.interpolate(feat_out["17"], size=x.shape[-2:], mode="bilinear", align_corners=False)
        s6 = nn.functional.interpolate(feat_out["20"], size=x.shape[-2:], mode="bilinear", align_corners=False)
        s7 = nn.functional.interpolate(feat_out["24"], size=x.shape[-2:], mode="bilinear", align_corners=False)
        s8 = nn.functional.interpolate(feat_out["27"], size=x.shape[-2:], mode="bilinear", align_corners=False)

        features = torch.cat([x, coords, s1, s2, s3, s4, s5, s6, s7, s8], 1)
        features = self.output_conv(features)

        return torch.cat([features, x, coords], 1)

    def forward(self, x, coords):

        pixel_f = self.feature_extract(x, coords)
        if self.training:
            return ssn_iter(pixel_f, self.nspix, self.n_iter)
        else:
            return sparse_ssn_iter(pixel_f, self.nspix, self.n_iter)


class SSN_Resnet(nn.Module):

    def __init__(self, feature_dim, nspix, training, n_iter=10):
        super().__init__()
        self.nspix = nspix
        self.n_iter = n_iter
        self.backbone = resnet18(weights=ResNet18_Weights)
        self.layers = ("relu", "layer1", "layer2", "layer3", "layer4", "fc")
        self.output_conv = nn.Sequential(
            conv_bn_relu(1029, 500),
            conv_bn_relu(500, 100),
            conv_bn_relu(100, feature_dim - 5)
        )

    def forward(self, x, coords):

        pixel_f = self.feature_extract(x, coords)
        if self.training:
            return ssn_iter(pixel_f, self.nspix, self.n_iter)
        else:
            return sparse_ssn_iter(pixel_f, self.nspix, self.n_iter)

    def feature_extract(self, x, coords):
        height, width = x.shape[2:]
        input = x
        feat_out = {}

        def hook_resnet_relu(module, input, output):
            feat_out["relu"] = output

        def hook_resnet_l1(module, input, output):
            feat_out["layer1"] = output

        def hook_resnet_l2(module, input, output):
            feat_out["layer2"] = output

        def hook_resnet_l3(module, input, output):
            feat_out["layer3"] = output

        def hook_resnet_l4(module, input, output):
            feat_out["layer4"] = output

        self.backbone._modules["relu"].register_forward_hook(hook_resnet_relu)
        self.backbone._modules["layer1"].register_forward_hook(hook_resnet_l1)
        self.backbone._modules["layer2"].register_forward_hook(hook_resnet_l2)
        self.backbone._modules["layer3"].register_forward_hook(hook_resnet_l3)
        self.backbone._modules["layer4"].register_forward_hook(hook_resnet_l4)
        out = self.backbone(x)

        s1 = nn.functional.interpolate(feat_out["relu"], size=x.shape[-2:], mode="bilinear", align_corners=False)
        s2 = nn.functional.interpolate(feat_out["layer1"], size=x.shape[-2:], mode="bilinear", align_corners=False)
        s3 = nn.functional.interpolate(feat_out["layer2"], size=x.shape[-2:], mode="bilinear", align_corners=False)
        s4 = nn.functional.interpolate(feat_out["layer3"], size=x.shape[-2:], mode="bilinear", align_corners=False)
        s5 = nn.functional.interpolate(feat_out["layer4"], size=x.shape[-2:], mode="bilinear", align_corners=False)

        features = torch.cat([x, coords, s1, s2, s3, s4, s5], 1)
        features = self.output_conv(features)

        return torch.cat([features, x, coords], 1)


class SSN_DeepLab(nn.Module):
    def __init__(self, feature_dim, nspix, training, n_iter=10):
        super().__init__()
        self.nspix = nspix
        self.n_iter = n_iter
        self.backbone = deeplabv3_mobilenet_v3_large(weights=DeepLabV3_MobileNet_V3_Large_Weights)
        self.layers = ("relu", "layer1", "layer2", "layer3", "layer4", "fc")
        self.output_conv = nn.Sequential(
            conv_bn_relu(1029, 500),
            conv_bn_relu(500, 100),
            conv_bn_relu(100, feature_dim - 5)
        )

    def forward(self, x, coords):

        pixel_f = self.feature_extract(x, coords)
        if self.training:
            return ssn_iter(pixel_f, self.nspix, self.n_iter)
        else:
            return sparse_ssn_iter(pixel_f, self.nspix, self.n_iter)

# class MLP(MessagePassing):
#     def __init__(self, aggr='mean'):
#         super().__init__(aggr)
#         self.mlp = torch.nn.Sequential(torch.nn.Linear(40, 1), torch.nn.Sigmoid())
#         self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
#
#     def forward(self, x, edge_index):
#         return self.propagate(edge_index, x=x), self.probs, self.gts
#
#     def message(self, x_i, x_j):
#         concatenated = torch.cat([x_i, x_j], dim=1)
#         y = self.mlp(concatenated)
#         gts = self.cosine_similarity(x_i, x_j)
#         self.probs = y
#         self.gts = gts
#         return y
