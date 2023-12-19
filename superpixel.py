import numpy as np
import torch
import torch.nn as nn


class Superpixel:
    def __init__(self, index, features, pixel_indices_2d, num_spixels_width, image_width, num_spixels, pixel_indices_1d,
                 height, width):
        self.index = index
        self.label = index
        self.pixel_indices_1d = pixel_indices_1d
        self.pixel_indices_2d = pixel_indices_2d
        self.num_spixels_width = num_spixels_width
        self.num_spixels = num_spixels
        self.centroid = torch.mean(self.pixel_indices_2d, axis=0)
        neighbor_spixels = []
        if self.index % num_spixels_width == 0:
            self.relative_spix_indices = torch.hstack(
                [-num_spixels_width, -num_spixels_width + 1, torch.ones(1), num_spixels_width, num_spixels_width + 1])
        elif self.index % num_spixels_width == num_spixels_width - 1:
            self.relative_spix_indices = torch.hstack(
                [-num_spixels_width - 1, -num_spixels_width, -1 * torch.ones(1), num_spixels_width - 1,
                 num_spixels_width])
        else:
            self.relative_spix_indices = torch.hstack(
                [-num_spixels_width - 1, -num_spixels_width, -num_spixels_width + 1, -1 * torch.ones(1), torch.ones(1),
                 num_spixels_width - 1,
                 num_spixels_width, num_spixels_width + 1])

        self.abs_spix_indices = index + self.relative_spix_indices
        for idx in self.abs_spix_indices:
            if int(idx) in np.arange(num_spixels.item() + 1):
                neighbor_spixels.append(idx)

        self.neighbor_spixels = torch.Tensor(neighbor_spixels)

    def compute_neighbor_weights(self, superpixel_list):
        self.neighbor_weights = {}
        for neighbor in self.neighbor_spixels:
            weight = np.dot(self.features.to('cpu').detach().numpy(),
                            superpixel_list[neighbor].features.to("cpu").detach().numpy().T)[0, 0]
            self.neighbor_weights[neighbor] = weight
        # Sort dictionary by values/weights
        self.neighbor_weights = dict(sorted(self.neighbor_weights.items(), key=lambda item: item[1], reverse=True))
        return

    def compute_neighbor_weights_sparse(self, superpixel_list):
        neighbor_weights = []
        max = 0
        min = 1
        for neighbor in self.neighbor_spixels:
            weight = np.dot(self.features.to("cpu").detach().numpy(),
                            superpixel_list[neighbor].features.to("cpu").detach().numpy().T)[0, 0]
            neighbor_weights.append(weight)
            neighbor_weights.append(weight)
            if weight > max:
                max = weight
                self.nn = neighbor
            if weight < min:
                min = weight
                self.fn = neighbor
        return np.asarray(neighbor_weights)

    def convert_spixel_index_to_coordinates(self):
        neighbor_indices = self.neighbor_spixels
        coords = []
        for idx in neighbor_indices:
            coord = [self.index, idx]
            rev_coord = [idx, self.index]
            coords.append(coord)
            coords.append(rev_coord)
        coords = np.asarray(coords)
        return coords.T
