import math
import torch

from .pair_wise_distance import PairwiseDistFunction
from ..utils.sparse_utils import naive_sparse_bmm


def calc_init_centroid(images, num_spixels_width, num_spixels_height):
    """
    calculate initial superpixels

    Args:
        images: torch.Tensor
            A Tensor of shape (B, C, H, W)
        spixels_width: int
            initial superpixel width
        spixels_height: int
            initial superpixel height

    Return:
        centroids: torch.Tensor
            A Tensor of shape (B, C, H * W)
        init_label_map: torch.Tensor
            A Tensor of shape (B, H * W)
        num_spixels_width: int
            A number of superpixels in each column
        num_spixels_height: int
            A number of superpixels int each raw
    """
    batchsize, channels, height, width = images.shape
    device = images.device

    centroids = torch.nn.functional.adaptive_avg_pool2d(images, (num_spixels_height, num_spixels_width))

    with torch.no_grad():
        num_spixels = num_spixels_width * num_spixels_height
        labels = torch.arange(num_spixels, device=device).reshape(1, 1, *centroids.shape[-2:]).type_as(centroids)
        init_label_map = torch.nn.functional.interpolate(labels, size=(height, width), mode="nearest")
        init_label_map = init_label_map.repeat(batchsize, 1, 1, 1)

    init_label_map = init_label_map.reshape(batchsize, -1)
    centroids = centroids.reshape(batchsize, channels, -1)
    return centroids, init_label_map


@torch.no_grad()
def get_abs_indices(init_label_map, num_spixels_width):
    b, n_pixel = init_label_map.shape
    device = init_label_map.device
    r = torch.arange(-1, 2.0, device=device)
    relative_spix_indices = torch.cat([r - num_spixels_width, r, r + num_spixels_width], 0)
    abs_pix_indices = torch.arange(n_pixel, device=device)[None, None].repeat(b, 9, 1).reshape(-1).long()
    abs_spix_indices = (init_label_map[:, None] + relative_spix_indices[None, :, None]).reshape(-1).long()
    abs_batch_indices = torch.arange(b, device=device)[:, None, None].repeat(1, 9, n_pixel).reshape(-1).long()

    return torch.stack([abs_batch_indices, abs_spix_indices, abs_pix_indices], 0)


@torch.no_grad()
def get_hard_abs_labels(affinity_matrix, init_label_map, num_spixels_width):
    relative_label = affinity_matrix.max(1)[1]
    r = torch.arange(-1, 2.0, device=affinity_matrix.device)
    relative_spix_indices = torch.cat([r - num_spixels_width, r, r + num_spixels_width], 0)
    label = init_label_map + relative_spix_indices[relative_label]
    return label.long()


@torch.no_grad()
def sparse_ssn_iter(pixel_features, num_spixels, n_iter):
    """
    computing assignment iterations with sparse matrix
    detailed process is in Algorithm 1, line 2 - 6
    NOTE: this function does NOT guarantee the backward computation.

    Args:
        pixel_features: torch.Tensor
            A Tensor of shape (B, C, H, W)
        num_spixels: int
            A number of superpixels
        n_iter: int
            A number of iterations
        return_hard_label: bool
            return hard assignment or not
    """
    height, width = pixel_features.shape[-2:]
    num_spixels_width = int(math.sqrt(num_spixels * width / height))
    num_spixels_height = int(math.sqrt(num_spixels * height / width))
    # print(f'Num superpixels width: ', num_spixels_width)
    # print(f'Num superpixels height: ', num_spixels_height)
    # print('Number of superpixels: ', num_spixels_width * num_spixels_height)
    spixels_true_count = num_spixels_width * num_spixels_height

    spixel_features, init_label_map = \
        calc_init_centroid(pixel_features, num_spixels_width, num_spixels_height)
    # print('Initialized spixel features: ', spixel_features.shape)
    # print(f'Initialized label map: {init_label_map.shape}, values: {torch.unique(init_label_map)}')
    abs_indices = get_abs_indices(init_label_map, num_spixels_width)
    # print(f'abs_indices: {abs_indices.shape}, values: {torch.unique(abs_indices)}')
    pixel_features = pixel_features.reshape(*pixel_features.shape[:2], -1)
    permuted_pixel_features = pixel_features.permute(0, 2, 1)

    # print('pixel features: ', pixel_features.shape)
    # print('permuted pixel features: ', permuted_pixel_features.shape)

    for _ in range(n_iter):
        # print(f'Iteration {_}')
        dist_matrix = PairwiseDistFunction.apply(
            pixel_features, spixel_features, init_label_map, num_spixels_width, num_spixels_height)
        # print(f'distance matrix: {dist_matrix.shape}, values: {torch.unique(dist_matrix)}')
        affinity_matrix = (-dist_matrix).softmax(1)
        reshaped_affinity_matrix = affinity_matrix.reshape(-1)
        # print(f'affinity matrix: {affinity_matrix.shape}, values: {torch.unique(affinity_matrix)}')
        # print(f'reshaped affinity matrix: {reshaped_affinity_matrix.shape}, values: {torch.unique(reshaped_affinity_matrix)}')
        mask = (abs_indices[1] >= 0) * (abs_indices[1] < num_spixels)
        mask = (abs_indices[1] >= 0) * (abs_indices[1] < spixels_true_count)
        # print('abs_indices: ', mask[-100:])
        # print(f'mask: {mask.shape}, values: {torch.unique(mask)}')
        # print(f'sparse coo first: {abs_indices[:, mask].shape} | {torch.unique(abs_indices[:, mask])}')
        # print(f'sparse coo second: {reshaped_affinity_matrix[mask].shape} | {torch.unique(reshaped_affinity_matrix[mask])[:20]}')
        sparse_abs_affinity = torch.sparse_coo_tensor(abs_indices[:, mask], reshaped_affinity_matrix[mask])
        # print(f'sparse_abs_affinity: {sparse_abs_affinity.shape}, values: {sparse_abs_affinity}')
        spixel_features = naive_sparse_bmm(sparse_abs_affinity, permuted_pixel_features) \
            / (torch.sparse.sum(sparse_abs_affinity, 2).to_dense()[..., None] + 1e-16)

        spixel_features = spixel_features.permute(0, 2, 1)
        # print('spixel features', spixel_features.shape)

    hard_labels = get_hard_abs_labels(affinity_matrix, init_label_map, num_spixels_width)
    # print(f'hard labels: {hard_labels.shape}, values: {torch.unique(hard_labels)}')

    return sparse_abs_affinity, hard_labels, spixel_features, num_spixels_width, num_spixels_height


def ssn_iter(pixel_features, num_spixels, n_iter):
    """
    computing assignment iterations
    detailed process is in Algorithm 1, line 2 - 6

    Args:
        pixel_features: torch.Tensor
            A Tensor of shape (B, C, H, W)
        num_spixels: int
            A number of superpixels
        n_iter: int
            A number of iterations
        return_hard_label: bool
            return hard assignment or not
    """
    height, width = pixel_features.shape[-2:]
    num_spixels_width = int(math.sqrt(num_spixels * width / height))
    num_spixels_height = int(math.sqrt(num_spixels * height / width))

    spixel_features, init_label_map = \
        calc_init_centroid(pixel_features, num_spixels_width, num_spixels_height)
    abs_indices = get_abs_indices(init_label_map, num_spixels_width)

    pixel_features = pixel_features.reshape(*pixel_features.shape[:2], -1)
    permuted_pixel_features = pixel_features.permute(0, 2, 1).contiguous()

    for _ in range(n_iter):
        dist_matrix = PairwiseDistFunction.apply(
            pixel_features, spixel_features, init_label_map, num_spixels_width, num_spixels_height)

        affinity_matrix = (-dist_matrix).softmax(1)
        reshaped_affinity_matrix = affinity_matrix.reshape(-1)

        mask = (abs_indices[1] >= 0) * (abs_indices[1] < num_spixels)
        sparse_abs_affinity = torch.sparse_coo_tensor(abs_indices[:, mask], reshaped_affinity_matrix[mask])

        abs_affinity = sparse_abs_affinity.to_dense().contiguous()
        spixel_features = torch.bmm(abs_affinity, permuted_pixel_features) \
            / (abs_affinity.sum(2, keepdim=True) + 1e-16)

        spixel_features = spixel_features.permute(0, 2, 1).contiguous()


    hard_labels = get_hard_abs_labels(affinity_matrix, init_label_map, num_spixels_width)

    return abs_affinity, hard_labels, spixel_features, num_spixels_width, num_spixels_height