import os
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
import argparse
import scipy.sparse

from scipy.sparse.linalg import eigsh
from torchvision import transforms
from skimage.color import rgb2lab
from skimage.segmentation import mark_boundaries
from skimage.segmentation._slic import _enforce_label_connectivity_cython
from accelerate import Accelerator
from tqdm import tqdm
from superpixel import Superpixel
from model import SSN_DINO
from lib.dataset.datasets import Dataset, bbox_iou
from matplotlib.patches import Rectangle


def get_largest_cc_box(mask: np.array):
    from skimage.measure import label as measure_label
    labels = measure_label(mask)  # get connected components
    largest_cc_index = np.argmax(np.bincount(labels.flat)[1:]) + 1
    mask = np.where(labels == largest_cc_index)
    ymin, ymax = min(mask[0]), max(mask[0]) + 1
    xmin, xmax = min(mask[1]), max(mask[1]) + 1
    return [xmin, ymin, xmax, ymax]


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


def inference(ssnmodel, path, image, nspix, n_iter, fdim=None, color_scale=0.26, pos_scale=2.5, weight=None,
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", default=None, type=str, help="/path/to/pretrained_weight")
    parser.add_argument("--fdim", default=20, type=int, help="embedding dimension")
    parser.add_argument("--niter", default=10, type=int, help="number of iterations for differentiable SLIC")
    parser.add_argument("--nspix", default=500, type=int, help="number of superpixels")
    parser.add_argument("--color_scale", default=0.26, type=float)
    parser.add_argument("--pos_scale", default=2.5, type=float)
    parser.add_argument("--dataset", type=str, help="dataset")
    parser.add_argument("--no_hard", action="store_true", help="in case of VOC_all setup")
    parser.add_argument("--patch_size", default=16, type=int, help="Patch resolution of the model.")
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ssnmodel = SSN_DINO(args.fdim, args.nspix, args.niter).to(device)
    ssnmodel.load_state_dict(torch.load(args.weight))

    model_name = "dino_vits16"
    model, val_transform, patch_size, num_heads = get_model(model_name)

    # Load dataset
    dataset = Dataset(args.dataset, "trainval", args.no_hard)
    pbar = tqdm(dataset.dataloader)
    cnt = 0
    corloc = np.zeros(len(dataset.dataloader))
    for im_id, inp in enumerate(pbar):
        im_name = dataset.get_image_name(inp[1])
        im_name_wo_ext = im_name[:-4]
        img = inp[0]
        init_image_size = img.shape

        # Padding the image with zeros to fit multiple of patch-size
        size_im = (
            img.shape[0],
            int(np.ceil(img.shape[1] / args.patch_size) * args.patch_size),
            int(np.ceil(img.shape[2] / args.patch_size) * args.patch_size),
        )
        padded_tensor = torch.zeros(size_im)
        padded_tensor[:, : img.shape[1], : img.shape[2]] = img
        gt_bbxs, gt_cls = dataset.extract_gt(inp[1], im_name)
        img_path = os.path.join(dataset.root_path, f"VOCdevkit/VOC{dataset.year}/JPEGImages", im_name)
        org_img = plt.imread(img_path)
        size_im = (
            int(np.ceil(img.shape[1] / args.patch_size) * args.patch_size),
            int(np.ceil(img.shape[2] / args.patch_size) * args.patch_size),
            img.shape[0]
        )
        padded = np.zeros(size_im).astype(np.uint8)
        padded[: img.shape[1], : img.shape[2], :] = org_img
        org_img = padded
        Q, H_s, superpixel_features, num_spixels_width, num_spixels_height, tokens, height, width = inference(ssnmodel,
                                                                                                              img_path,
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
        image_bgr = cv2.cvtColor(org_img, cv2.COLOR_RGB2BGR)
        # Enforce connectivity
        segment_size = height * width / args.nspix
        min_size = int(0.06 * segment_size)
        max_size = int(3.0 * segment_size)
        plabels = _enforce_label_connectivity_cython(labels[None], min_size, max_size)[0]

        # Uncomment below piece of code to visualize superpixel segmentations
        # plt.imshow(mark_boundaries(org_img, plabels))
        # plt.savefig(f"VOC{dataset.year}_results/{im_name_wo_ext}_spixel.jpg")

        plabels = torch.tensor(plabels)
        reshaped_labels = plabels.reshape(-1, height, width)
        spixel_list, superpixels = extract_superpixels(reshaped_labels, num_spixels_width, height, width)

        # Extract features from Dino
        output_dict = {}
        P = patch_size
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

        output_qkv = feat_out["qkv"].reshape(B, T, 3, num_heads, -1 // num_heads).permute(2, 0, 3, 1, 4)
        output_dict['k'] = output_qkv[1].transpose(1, 2).reshape(B, T, -1)[:, 1:, :]
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

        temp = plabels.numpy()

        # Uncomment the below lines to visualize segmentation maps
        # plt.imshow(temp)
        # plt.savefig(f"VOC{dataset.year}_results/{im_name_wo_ext}_segmap.jpg")

        if (torch.sum(plabels) == 0):
            bbox = [0, 0, width, height]
        else:
            bbox = get_largest_cc_box(plabels)

        # Uncomment below lines to visualize bounding boxes
        # plt.imshow(org_img)
        # plt.gca().add_patch(Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
        #                               edgecolor='red',
        #                               facecolor='none',
        #                               lw=4))
        # plt.savefig(f"VOC{dataset.year}_results/{im_name_wo_ext}_result.jpg")
        # plt.close()

        # Corloc
        ious = bbox_iou(torch.from_numpy(np.asarray(bbox)), torch.from_numpy(gt_bbxs))

        if torch.any(ious >= 0.5):
            corloc[im_id] = 1

        cnt += 1

    print(f"Corloc: {100 * np.sum(corloc) / cnt:.2f} ({int(np.sum(corloc))}/{cnt})")
