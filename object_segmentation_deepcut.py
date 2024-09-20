import os
import argparse
import math
import torch
import cv2
import util
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from skimage.color import rgb2lab
from skimage.segmentation import mark_boundaries
from skimage.segmentation._slic import _enforce_label_connectivity_cython
from torch.utils.data import DataLoader
from tqdm import tqdm
from extractor import ViTExtractor
from features_extract import deep_features
from lib.dataset.SegmentationDataset import SegmentationDataset
from lib.ssn.ssn import sparse_ssn_iter
from superpixel import Superpixel


def compute_iou(pred, target):
    pred, target = pred.to(torch.bool), target.to(torch.bool)
    intersection = torch.sum(pred * (pred == target), dim=[-1, -2]).squeeze()
    union = torch.sum(pred + target, dim=[-1, -2]).squeeze()
    iou = (intersection.to(torch.float) / union).mean()
    iou = iou.item() if (iou == iou) else 0  # deal with nans, i.e. torch.nan_to_num(iou, nan=0.0)
    return iou


def GNN_seg(image_path, plabels, num_spixels_width, num_spixels_height, spixel_list, mode, cut, alpha, epoch, K,
            pretrained_weights, save, cc, bs, log_bin, res,
            facet, layer,
            stride, device):
    """
    Segment entire dataset; Get bounding box (k==2 only) or segmentation maps
    bounding boxes will be in the following format: class, confidence, left, top , right, bottom
    (class and confidence not in use for now, set as '1')
    @param cut: chosen clustering functional: NCut==1, CC==0
    @param epoch: Number of epochs for every step in image
    @param K: Number of segments to search in each image
    @param pretrained_weights: Weights of pretrained images
    @param dir: Directory for chosen dataset
    @param out_dir: Output directory to save results
    @param cc: If k==2 chose the biggest component, and discard the rest (only available for k==2)
    @param b_box: If true will output bounding box (for k==2 only), else segmentation map
    @param log_bin: Apply log binning to the descriptors (correspond to smother image)
    @param device: Device to use ('cuda'/'cpu')
    """
    ##########################################################################################
    # Dino model init
    ##########################################################################################
    extractor = ViTExtractor('dino_vits8', stride, model_dir=pretrained_weights, device=device)
    feats_dim = 384
    # if two stage make first stage foreground detection with k == 2
    if mode == 1 or mode == 2:
        foreground_k = K
        K = 2

    if cut == 0:
        from gnn_pool import GNNpool
    else:
        from gnn_pool_cc import GNNpool

    model = GNNpool(feats_dim, 64, 32, K, device).to(device)
    torch.save(model.state_dict(), 'model.pt')
    model.train()
    if mode == 1 or mode == 2:
        model2 = GNNpool(feats_dim, 64, 32, foreground_k, device).to(device)
        torch.save(model2.state_dict(), 'model2.pt')
        model2.train()
    if mode == 2:
        model3 = GNNpool(feats_dim, 64, 32, 2, device).to(device)
        torch.save(model3.state_dict(), 'model3.pt')
        model3.train()
    image_tensor, image = util.load_data_img(image_path, res)

    F = deep_features(image_tensor, extractor, layer, facet, bin=log_bin, device=device)

    # Construct superpixel feature matrix (num_spix X feature_dim)
    spixel_F = []
    for spix in spixel_list:
        feature = F[spix.patch, :]
        spixel_F.append(feature)
    spixel_F = np.array(spixel_F)

    W = util.create_adj(spixel_F, cut, alpha)
    node_feats, edge_index, edge_weight = util.load_data(W, spixel_F)
    node_feats, edge_index, edge_weight = node_feats.to(device), edge_index.to(device), edge_weight.to(device)
    # re-init weights and optimizer for every image
    model.load_state_dict(torch.load('./model.pt', map_location=torch.device(device)))
    opt = optim.AdamW(model.parameters(), lr=0.001)

    ##########################################################################################
    # GNN pass
    ##########################################################################################
    for _ in range(epoch[0]):
        opt.zero_grad()
        A, S = model(node_feats, edge_index, edge_weight, torch.from_numpy(W).to(device))
        loss = model.loss(A, S)
        loss.backward()
        opt.step()

    # polled matrix (after softmax, before argmax)
    S = S.detach().cpu()
    S = torch.argmax(S, dim=-1)

    # Count the number of 0s and 1s
    stat_mode = torch.mode(S, -1)[0]

    if stat_mode != 0:
        S[S == 0] = stat_mode + 1
        S[S == stat_mode] = 0
        S[S != 0] = 1

    #########################################################################################
    # Post-processing Connected Component/bilateral solver
    ##########################################################################################
    labels = S.clone()

    if mode == 0:
        return labels

    ##########################################################################################
    # Second pass on foreground
    ##########################################################################################
    # extracting foreground sub-graph
    fg_spixels = torch.squeeze(torch.argwhere(S))

    F_2 = spixel_F[fg_spixels]

    W_2 = util.create_adj(F_2, cut, alpha)

    # Data to pytorch_geometric format
    node_feats, edge_index, edge_weight = util.load_data(W_2, F_2)
    node_feats, edge_index, edge_weight = node_feats.to(device), edge_index.to(device), edge_weight.to(device)

    # re-init weights and optimizer for every image
    model2.load_state_dict(torch.load('./model2.pt', map_location=torch.device(device)))
    opt = optim.AdamW(model2.parameters(), lr=0.001)

    ####################################################
    # GNN pass
    ####################################################
    for _ in range(epoch[1]):
        opt.zero_grad()
        A_2, S_2 = model2(node_feats, edge_index, edge_weight, torch.from_numpy(W_2).to(device))
        loss = model2.loss(A_2, S_2)
        loss.backward()
        opt.step()

    # fusing subgraph and original graph
    S_2 = S_2.detach().cpu()
    S_2 = torch.argmax(S_2, dim=-1)
    S[fg_spixels] = S_2 + 3

    labels = S.clone()

    if mode == 1:
        return labels

    ##########################################################################################
    # Second pass background
    ##########################################################################################
    # extracting background sub-graph
    F_3 = F[fg_spixels]
    W_3 = util.create_adj(F_3, cut, alpha)

    # Data to pytorch_geometric format
    node_feats, edge_index, edge_weight = util.load_data(W_3, F_3)
    node_feats, edge_index, edge_weight = node_feats.to(device), edge_index.to(device), edge_weight.to(device)

    # re-init weights and optimizer for every image
    model3.load_state_dict(torch.load('./model3.pt', map_location=torch.device(device)))
    opt = optim.AdamW(model3.parameters(), lr=0.001)
    for _ in range(epoch[2]):
        opt.zero_grad()
        A_3, S_3 = model3(node_feats, edge_index, edge_weight, torch.from_numpy(W_3).to(device))
        loss = model3.loss(A_3, S_3)
        loss.backward()
        opt.step()

    # fusing subgraph and original graph
    S_3 = S_3.detach().cpu()
    S_3 = torch.argmax(S_3, dim=-1)
    S[fg_spixels] = S_3 + foreground_k + 5

    labels = S.clone()
    if bs:
        mask_foreground = mask0
        mask_background = np.where(mask2 != foreground_k + 5, 0, 1)
        bs_foreground = bilateral_solver_output(image, mask_foreground)[1]
        bs_background = bilateral_solver_output(image, mask_background)[1]
        mask2 = bs_foreground + (bs_background * 2)

    return labels


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


@torch.no_grad()
def inference(res, img, nspix, n_iter, fdim=None, color_scale=0.26, pos_scale=2.5, weight=None,
              enforce_connectivity=True):
    """
    generate superpixels

    Args:
        image: numpy.ndarray
            An array of shape (h, w, c)
        nspix: int
            number of superpixels
        n_iter: int
            number of iterations
        fdim (optional): int
            feature dimension for supervised setting
        color_scale: float
            color channel factor
        pos_scale: float
            pixel coordinate factor
        weight: state_dict
            pretrained weight
        enforce_connectivity: bool
            if True, enforce superpixel connectivity in postprocessing

    Return:
        labels: numpy.ndarray
            An array of shape (h, w)
    """
    if weight is not None:
        from model import SSNModel, SSN_DINO
        model = SSN_DINO(fdim, nspix, n_iter).to("cuda")
        model.load_state_dict(torch.load(weight))
        model.eval()
    else:
        model = lambda data: sparse_ssn_iter(data, nspix, n_iter)

    # Inference using SSN model
    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, res)
    height, width = res
    nspix_per_axis = int(math.sqrt(nspix))
    pos_scale = pos_scale * max(nspix_per_axis / height, nspix_per_axis / width)
    coords = torch.stack(torch.meshgrid(torch.arange(height, device="cuda"), torch.arange(width, device="cuda")), 0)
    coords = coords[None].float()
    image = rgb2lab(image)
    image = torch.from_numpy(image).permute(2, 0, 1)[None].to("cuda").float()
    Q, H, feats, num_spixels_width, num_spixels_height, tokens = model(color_scale * image, pos_scale * coords)
    labels = H.reshape(height, width).to("cpu").detach().numpy()

    return Q, H, feats, num_spixels_width, num_spixels_height, labels


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, help="Path to dataset")
    parser.add_argument("--weight", default="", type=str, help="/path/to/pretrained_weight")
    parser.add_argument("--fdim", default=20, type=int, help="embedding dimension")
    parser.add_argument("--niter", default=10, type=int, help="number of iterations for differentiable SLIC")
    parser.add_argument("--nspix", default=100, type=int, help="number of superpixels")
    parser.add_argument("--color_scale", default=0.26, type=float)
    parser.add_argument("--pos_scale", default=2.5, type=float)
    parser.add_argument("--dataset", type=str, help="dataset")

    enforce_connectivity = True
    args = parser.parse_args()

    # Load dataset
    dataset = SegmentationDataset(args.dataset, mode="test")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    progress_bar = tqdm(dataloader, desc="Processing images")
    ious, lcc_ious = [], []
    for img, mask, img_path, mask_path, name in progress_bar:
        img_path = img_path[0]
        image = plt.imread(img_path)
        if len(image.shape) != 3:
            continue
        h, w, _ = image.shape
        res = (280, 280)
        Q, H, superpixel_features, num_spixels_width, num_spixels_height, labels = inference(res, img_path, args.nspix,
                                                                                             args.niter,
                                                                                             args.fdim,
                                                                                             args.color_scale,
                                                                                             args.pos_scale,
                                                                                             args.weight,
                                                                                             enforce_connectivity=False)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, res)
        mask = torch.squeeze(mask)
        mask = mask.cpu().numpy()

        mask = cv2.resize(mask, res)
        height, width = res
        segment_size = height * width / args.nspix
        min_size = int(0.06 * segment_size)
        max_size = int(3.0 * segment_size)
        labels = H.reshape(height, width)
        reshaped_labels = labels.reshape(-1, height, width)
        labels = labels.to("cpu").detach().numpy()
        plabels = _enforce_label_connectivity_cython(labels[None], min_size, max_size)[0]
        plabels = torch.tensor(plabels)
        reshaped_labels = plabels.reshape(-1, height, width)
        spixel_list, superpixels = extract_superpixels(reshaped_labels, num_spixels_width, height, width)

        spixel_list, superpixels = extract_superpixels(reshaped_labels, num_spixels_width, height, width)
        num_patches_width = 35
        patch_size = 8
        for idx, spix in enumerate(spixel_list):
            spix.col = round(spix.centroid[1].item())
            spix.row = round(spix.centroid[0].item())
            patch_row = spix.row // patch_size
            patch_col = spix.col // patch_size
            spix.patch = patch_row * num_patches_width + patch_col

        # Single Stage Segmentation
        mode = 0
        cut = 0
        alpha = 3
        # Numbers of epochs per stage [mode0,mode1,mode2]
        epochs = [10, 100, 10]
        step = 1
        K = 4
        # Show only largest component in segmentation map (for k == 2)
        cc = False
        # apply bilateral solver
        bs = False
        # Apply log binning to extracted descriptors (correspond to smoother segmentation maps)
        log_bin = False
        pretrained_weights = './dino_deitsmall8_pretrain_full_checkpoint.pth'
        stride = 8
        facet = 'key'
        layer = 11
        save = True

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # If Directory doesn't exist than download
        if not os.path.exists(pretrained_weights):
            url = 'https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain_full_checkpoint.pth'
            util.download_url(url, pretrained_weights)

        S = GNN_seg(img_path, plabels, num_spixels_width, num_spixels_height, spixel_list, mode, cut, alpha, epochs, K,
                    pretrained_weights, save, cc, bs, log_bin, res,
                    facet, layer, stride, device)

        for i in range(len(spixel_list)):
            plabels[plabels == i] = len(spixel_list) + S[i]

        plabels -= len(spixel_list)
        mask = torch.tensor(mask)
        iou = compute_iou(plabels, mask)
        ious.append(iou)

        plabels = plabels.to("cpu").detach().numpy()

        from skimage.measure import label

        temp_labels = label(plabels)
        largestCC = temp_labels == np.argmax(np.bincount(temp_labels.flat, weights=plabels.flat))
        lcc_iou = compute_iou(torch.tensor(largestCC), mask)
        lcc_ious.append(lcc_iou)

        progress_bar.set_description(f"img: {name[0]}, iou: {iou:.4f}, lcc iou: {lcc_iou:.4f}")

    print(f"iou: {sum(ious) / len(ious)}")
    print(f"largest cc iou: {sum(lcc_ious) / len(lcc_ious)}")
