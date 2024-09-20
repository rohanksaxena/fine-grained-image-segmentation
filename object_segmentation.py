import math
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy.sparse

from skimage.color import rgb2lab
from superpixel import Superpixel
from model import SSN_DINO
from scipy.sparse.linalg import eigsh
from PIL import Image
from accelerate import Accelerator
from tqdm import tqdm
from torchvision import transforms
from skimage.segmentation._slic import _enforce_label_connectivity_cython
from lib.dataset.SegmentationDataset import SegmentationDataset
from torch.utils.data import DataLoader


def compute_iou(pred, target):
    pred, target = pred.to(torch.bool), target.to(torch.bool)
    intersection = torch.sum(pred * (pred == target), dim=[-1, -2]).squeeze()
    union = torch.sum(pred + target, dim=[-1, -2]).squeeze()
    iou = (intersection.to(torch.float) / union).mean()
    iou = iou.item() if (iou == iou) else 0  # deal with nans, i.e. torch.nan_to_num(iou, nan=0.0)
    return iou


def get_transform(name: str):
    if any(x in name for x in ('dino', 'mocov3', 'convnext',)):
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        transform = transforms.Compose([transforms.ToTensor(), normalize])
    else:
        raise NotImplementedError()
    return transform


def get_model(name: str):
    if 'dino' in name:
        model = torch.hub.load('facebookresearch/dino:main', name)
        model.fc = nn.Identity()
        val_transform = get_transform(name)
        patch_size = model.patch_embed.patch_size
        num_heads = model.blocks[0].attn.num_heads
    else:
        raise ValueError(f'Cannot get model: {name}')
    model = model.eval()
    return model, val_transform, patch_size, num_heads


def get_diagonal(W: scipy.sparse.csr_matrix, threshold: float = 1e-12):
    """Gets the diagonal sum of a sparse matrix"""
    try:
        from pymatting.util.util import row_sum
    except:
        raise ImportError(
            'Please install pymatting to compute the diagonal sums:\n'
            'pip3 install pymatting'
        )

    D = row_sum(W)
    D[D < threshold] = 1.0  # Prevent division by zero.
    D = scipy.sparse.diags(D)
    return D


def extract_superpixels(reshaped_labels, num_spixels_width, height, width):
    spixel_list = []
    spixel_dict = {}
    for j in range(len(torch.unique(reshaped_labels))):
        pixel_indices_2d = torch.argwhere(reshaped_labels[0, :, :] == j)
        pixel_indices_1d = torch.argwhere(reshaped_labels[0, :, :].flatten() == j)
        spixel_dict[j] = Superpixel(index=j, features=None,
                                    pixel_indices_2d=pixel_indices_2d.double(),
                                    num_spixels_width=torch.tensor(num_spixels_width),
                                    image_width=width, num_spixels=torch.max(reshaped_labels),
                                    pixel_indices_1d=pixel_indices_1d.double(), height=height, width=width)
        spixel_list.append(spixel_dict[j])
    return spixel_list, spixel_dict


def inference(ssnmodel, image, nspix, n_iter, fdim=None, color_scale=0.26, pos_scale=2.5, weight=None,
              device="cpu"):
    ssnmodel.eval()
    height, width = image.shape[:2]

    # SSN Inference
    nspix_per_axis = int(math.sqrt(nspix))
    pos_scale = pos_scale * max(nspix_per_axis / height, nspix_per_axis / width)
    coords = torch.stack(torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device)), 0)
    coords = coords[None].float()
    image = rgb2lab(image)
    image = torch.from_numpy(image).permute(2, 0, 1)[None].to(device).float()
    Q, H, superpixel_features, num_spixels_width, num_spixels_height, tokens = ssnmodel(color_scale * image,
                                                                                        pos_scale * coords)

    labels = H.reshape(height, width).cpu().detach().numpy()
    return Q, H, superpixel_features, num_spixels_width, num_spixels_height, tokens, height, width


if __name__ == "__main__":
    import time
    import argparse
    import matplotlib.pyplot as plt
    from skimage.segmentation import mark_boundaries

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="/path/to/image")
    parser.add_argument("--weight", default=None, type=str, help="/path/to/pretrained_weight")
    parser.add_argument("--fdim", default=20, type=int, help="embedding dimension")
    parser.add_argument("--niter", default=10, type=int, help="number of iterations for differentiable SLIC")
    parser.add_argument("--nspix", default=100, type=int, help="number of superpixels")
    parser.add_argument("--color_scale", default=0.26, type=float)
    parser.add_argument("--pos_scale", default=2.5, type=float)
    parser.add_argument("--layer_number", default=3, type=int)
    parser.add_argument("--dataset", type=str, help="dataset")
    parser.add_argument("--wandb_key", type=str, help="Your wandb key")
    parser.add_argument("--wandb_entity", type=str, help="Entity name")
    parser.add_argument("--no_hard", action="store_true", help="in case of VOC_all setup")
    parser.add_argument("--patch_size", default=16, type=int, help="Patch resolution of the model.")
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ssnmodel = SSN_DINO(args.fdim, args.nspix, args.niter).to(device)
    ssnmodel.load_state_dict(torch.load(args.weight))

    model_name = "dino_vits16"
    model, val_transform, patch_size, num_heads = get_model(model_name)

    # Load dataset
    dataset = SegmentationDataset(args.dataset, mode="test")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    ious, lcc_ious = [], []
    progress_bar = tqdm(dataloader, desc="Processing images")
    for img, mask, img_path, mask_path, name in progress_bar:

        mask = torch.squeeze(mask)
        img_path = img_path[0]
        img = torch.squeeze(img)
        size_im = (
            img.shape[0],
            int(np.ceil(img.shape[1] / args.patch_size) * args.patch_size),
            int(np.ceil(img.shape[2] / args.patch_size) * args.patch_size),
        )
        padded_tensor = torch.zeros(size_im)
        padded_tensor[:, : img.shape[1], : img.shape[2]] = img
        org_img = plt.imread(img_path)
        if len(org_img.shape) == 2:
            org_img = np.expand_dims(org_img, axis=2)
        padded_mask = torch.zeros((size_im[1], size_im[2]))
        padded_mask[:mask.shape[0], :mask.shape[1]] = mask
        mask = padded_mask

        padded_tensor = torch.zeros(size_im)
        padded_tensor[:, : img.shape[1], : img.shape[2]] = img
        org_img = plt.imread(img_path)
        size_im = (
            int(np.ceil(img.shape[1] / args.patch_size) * args.patch_size),
            int(np.ceil(img.shape[2] / args.patch_size) * args.patch_size),
            img.shape[0]
        )
        padded = np.zeros(size_im).astype(np.uint8)
        if len(org_img.shape) == 2:
            org_img = np.stack([org_img] * 3, axis=-1)
        padded[: img.shape[1], : img.shape[2], :] = org_img
        org_img = padded

        Q, H_s, superpixel_features, num_spixels_width, num_spixels_height, tokens, height, width = inference(ssnmodel,
                                                                                                              org_img,
                                                                                                              args.nspix,
                                                                                                              args.niter,
                                                                                                              args.fdim,
                                                                                                              args.color_scale,
                                                                                                              args.pos_scale,
                                                                                                              args.weight,
                                                                                                              device)

        labels = H_s.reshape(height, width)
        reshaped_labels = labels.reshape(-1, height, width)
        labels = labels.to("cpu").detach().numpy()
        pp = plt.imread(img_path)

        # Enforce connectivity
        segment_size = height * width / args.nspix
        min_size = int(0.06 * segment_size)
        max_size = int(3.0 * segment_size)
        plabels = _enforce_label_connectivity_cython(labels[None], min_size, max_size)[0]
        plabels = torch.tensor(plabels)

        reshaped_labels = plabels.reshape(-1, height, width)
        spixel_list, superpixels = extract_superpixels(reshaped_labels, num_spixels_width, height, width)

        output_dict = {}
        P = args.patch_size
        B = 1
        C, H, W = padded_tensor.shape
        H_patch, W_patch = H // P, W // P
        H_pad, W_pad = H_patch * P, W_patch * P
        T = H_patch * W_patch + 1  # number of tokens, add 1 for [CLS]
        accelerator = Accelerator(cpu=False)

        # Add hook
        if 'dino' in model_name or 'mocov3' in model_name:
            feat_out = {}


            def hook_fn_forward_qkv(module, input, output):
                feat_out["qkv"] = output


            model._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(
                hook_fn_forward_qkv)
        else:
            raise ValueError(model_name)

        padded_tensor = torch.unsqueeze(padded_tensor, 0).to(device)
        model = model.to(accelerator.device)
        model.get_intermediate_layers(padded_tensor)[0].squeeze(0)

        output_qkv = feat_out["qkv"].reshape(B, T, 3, num_heads, -1)
        output_qkv_permuted = output_qkv.permute(2, 0, 3, 1, 4)
        output_dict['k'] = output_qkv_permuted[1].transpose(1, 2).reshape(B, T, -1)[:, 1:, :]
        feats = torch.squeeze(output_dict['k'])

        # Add centroid coordinates and node label to each superpixel
        num_patches_width = width // patch_size
        num_patches_height = height // patch_size
        W_feat = None
        for idx, spix in enumerate(spixel_list):
            spix.col = round(spix.centroid[1].item())
            spix.row = round(spix.centroid[0].item())
            patch_row = spix.row // patch_size
            patch_col = spix.col // patch_size
            spix.patch = patch_row * num_patches_width + patch_col
            if W_feat is None:
                W_feat = feats[spix.patch]
            else:
                W_feat = torch.vstack((W_feat, feats[spix.patch]))
        W_feat = (W_feat @ W_feat.T)
        W_feat = (W_feat * (W_feat > 0))
        W_feat = W_feat / W_feat.max()  # NOTE: If features are normalized, this naturally does nothing
        W_feat = W_feat.detach().cpu().numpy()

        W_comb = W_feat
        D_comb = np.array(get_diagonal(W_comb).todense())

        K = 2

        try:
            eigenvalues, eigenvectors = eigsh(D_comb - W_comb, k=K, sigma=0, which='LM', M=D_comb)
        except:
            eigenvalues, eigenvectors = eigsh(D_comb - W_comb, k=K, which='SM', M=D_comb)

        eigenvalues, eigenvectors = torch.from_numpy(eigenvalues), torch.from_numpy(eigenvectors.T).float()

        # Sign ambiguity
        for k in range(eigenvectors.shape[0]):
            if 0.5 < torch.mean((eigenvectors[k] > 0).float()).item() < 1.0:  # reverse segment
                eigenvectors[k] = 0 - eigenvectors[k]

        eigenvector = eigenvectors[1].numpy()
        segmap = eigenvector > 0.0
        y = segmap
        y = 1 * y

        # Instance seg logic
        for i in range(len(y)):
            plabels[plabels == i] = y[i]
        plabels_height, plabels_width = plabels.shape
        padded_mask = padded_mask[:plabels_height, :plabels_width]
        iou = compute_iou(plabels, padded_mask)
        ious.append(iou)

        plabels = plabels.to("cpu").detach().numpy()

        # Uncomment below line to save segmented images
        # plt.imsave(f"segmented/{args.dataset}/{name[0].split('/')[-1]}.jpg", mark_boundaries(padded, plabels))

        from skimage.measure import label

        temp_labels = label(plabels)
        largestCC = temp_labels == np.argmax(np.bincount(temp_labels.flat, weights=plabels.flat))
        lcc_iou = compute_iou(torch.tensor(largestCC), padded_mask)
        lcc_ious.append(lcc_iou)
        progress_bar.set_description(f"img: {name[0]}, iou: {iou:.4f}, lcc iou: {lcc_iou:.4f}")

    print(f"iou: {sum(ious) / len(ious)}")
    print(f"largest cc iou: {sum(lcc_ious) / len(lcc_ious)}")
