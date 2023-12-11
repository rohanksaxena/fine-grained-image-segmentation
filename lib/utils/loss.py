import torch
from .sparse_utils import naive_sparse_bmm, sparse_permute


def sparse_reconstruction(assignment, labels, hard_assignment=None):
    """
    reconstruction loss with the sparse matrix
    NOTE: this function doesn't use it in this project, because may not return correct gradients

    Args:
        assignment: torch.sparse_coo_tensor
            A Tensor of shape (B, n_spixels, n_pixels)
        labels: torch.Tensor
            A Tensor of shape (B, C, n_pixels)
        hard_assignment: torch.Tensor
            A Tensor of shape (B, n_pixels)
    """
    labels = labels.permute(0, 2, 1).contiguous()

    # matrix product between (n_spixels, n_pixels) and (n_pixels, channels)
    spixel_mean = naive_sparse_bmm(assignment, labels) / (torch.sparse.sum(assignment, 2).to_dense()[..., None] + 1e-16)
    if hard_assignment is None:
        # (B, n_spixels, n_pixels) -> (B, n_pixels, n_spixels)
        permuted_assignment = sparse_permute(assignment, (0, 2, 1))
        # matrix product between (n_pixels, n_spixels) and (n_spixels, channels)
        reconstructed_labels = naive_sparse_bmm(permuted_assignment, spixel_mean)
    else:
        # index sampling
        reconstructed_labels = torch.stack([sm[ha, :] for sm, ha in zip(spixel_mean, hard_assignment)], 0)
    return reconstructed_labels.permute(0, 2, 1).contiguous()


def reconstruction(assignment, labels, hard_assignment=None):
    """
    reconstruction

    Args:
        assignment: torch.Tensor
            A Tensor of shape (B, n_spixels, n_pixels)
        labels: torch.Tensor
            A Tensor of shape (B, C, n_pixels)
        hard_assignment: torch.Tensor
            A Tensor of shape (B, n_pixels)
    """
    # print('In reconstruction***')
    labels = labels.permute(0, 2, 1).contiguous()
    # print(f'labels: {labels.shape}')
    # matrix product between (n_spixels, n_pixels) and (n_pixels, channels)
    spixel_mean = torch.bmm(assignment, labels) / (assignment.sum(2, keepdim=True) + 1e-16)
    # print(f'spixel mean: {spixel_mean.shape}, vals: {torch.unique(spixel_mean)}')
    if hard_assignment is None:
        # (B, n_spixels, n_pixels) -> (B, n_pixels, n_spixels)
        permuted_assignment = assignment.permute(0, 2, 1).contiguous()
        # matrix product between (n_pixels, n_spixels) and (n_spixels, channels)
        reconstructed_labels = torch.bmm(permuted_assignment, spixel_mean)
    else:
        # index sampling
        reconstructed_labels = torch.stack([sm[ha, :] for sm, ha in zip(spixel_mean, hard_assignment)], 0)
    # print(f'Reconstructed labels: {reconstructed_labels.shape}')
    return reconstructed_labels.permute(0, 2, 1).contiguous()


def reconstruct_loss_with_cross_etnropy(assignment, labels, hard_assignment=None):
    """
    reconstruction loss with cross entropy

    Args:
        assignment: torch.Tensor
            A Tensor of shape (B, n_spixels, n_pixels)
        labels: torch.Tensor
            A Tensor of shape (B, C, n_pixels)
        hard_assignment: torch.Tensor
            A Tensor of shape (B, n_pixels)
    """
    # print('In reconstruction loss with cross entropy********')
    # print(f'assignment: {assignment.shape}')
    # print(f'labels: {labels.shape}')
    reconstracted_labels = reconstruction(assignment, labels, hard_assignment)
    # print(f'reconstructed labels: {reconstracted_labels.shape}')
    # print(f'values: min: {torch.min(reconstracted_labels)}, max: {torch.max(reconstracted_labels)}, unique: {torch.unique(reconstracted_labels)}')
    reconstracted_labels = reconstracted_labels / (1e-16 + reconstracted_labels.sum(1, keepdim=True))
    # print(f'normalized values: min: {torch.min(reconstracted_labels)}, max: {torch.max(reconstracted_labels)}, unique: {torch.unique(reconstracted_labels)}')
    mask = labels > 0
    # print(f'masked shape: {reconstracted_labels[mask].shape}, vals: {torch.unique(reconstracted_labels[mask])}')
    return -(reconstracted_labels[mask] + 1e-16).log().mean()


def reconstruct_loss_with_mse(assignment, labels, hard_assignment=None):
    """
    reconstruction loss with mse

    Args:
        assignment: torch.Tensor
            A Tensor of shape (B, n_spixels, n_pixels)
        labels: torch.Tensor
            A Tensor of shape (B, C, n_pixels)
        hard_assignment: torch.Tensor
            A Tensor of shape (B, n_pixels)
    """
    # print('In reconstruction loss with MSE')
    reconstracted_labels = reconstruction(assignment, labels, hard_assignment)
    return torch.nn.functional.mse_loss(reconstracted_labels, labels)