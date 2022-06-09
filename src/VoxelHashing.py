import torch
import numpy as np
from .common import normalize_3d_coordinate
import torch.nn.functional as F


class VoxelHash(object):
    def __init__(self, shape, bound, mean=0.0, std=0.01):
        self.mean = mean
        self.std = std
        self._data = torch.zeros(shape).normal_(mean=mean, std=std)
        # self._allocated_idx = torch.tensor([0], dtype=torch.long)
        self._hash = {}
        self._hash_data = torch.zeros((shape[0], shape[1], 1)).normal_(mean=mean, std=std)
        self._hash_pos = torch.zeros((3, 1)).normal_(mean=mean, std=std)
        self.shape = shape
        self.bound = bound
        lx = bound[0, 1] - bound[0, 0]
        ly = bound[1, 1] - bound[1, 0]
        lz = bound[2, 1] - bound[2, 0]
        print("lx ", lx, "ly ", ly, "lz ", lz)
        self.dim = torch.tensor([lz / shape[2], ly / shape[3], lx / shape[4]])
        self.device = "cpu"
        print("shape ", self.shape)
        print("bound ", self.bound)
        print("dim ", self.dim)

    def get_data(self):
        return self._data
        # return self._hash_data

    @property
    def data(self):
        return self._hash_data

    def to(self, device):
        # print("voxel hash to")
        self._data = self._data.to(device)
        self._hash_data = self._hash_data.to(device)
        self.device = device
        return self

    def detach(self):
        # print("voxel hash detach")
        self._data = self._data.detach()
        self._hash_data = self._hash_data.detach()
        return self

    def share_memory_(self):
        # print("voxel hash share memory")
        self._data.share_memory_()
        self._hash_data.share_memory_()

    def get_idx(self, mask):
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        idx = mask.nonzero()  # nonzero returns all indices with mask == true
        coordinate = self.idx_to_coordinate(idx)
        linearized_idx = self._linearize_id(idx)
        return linearized_idx, coordinate

    def get_data_from_idx(self, linearized_idx):
        data_idx = [self._hash[int(idx)] for idx in linearized_idx]
        data = self._hash_data[:, :, data_idx]
        return data

    def set_data_from_idx(self, linearized_idx, value):
        data_idx = [self._hash[int(idx)] for idx in linearized_idx]
        self._hash_data[:, :, data_idx] = value

    def allocate_from_mask(self, mask):
        """Assume mask dimension is the same as shape"""
        linearized_idx, coordinate = self.get_idx(mask)
        for (idx, xyz) in zip(linearized_idx, coordinate):
            # print("idx in hash ", idx, int(idx) in self._hash)
            if not int(idx) in self._hash:
                # print("idx not in hash", idx, xyz)
                self._hash[int(idx)] = self._hash_data.shape[2]
                self._hash_data = torch.cat(
                    [
                        self._hash_data,
                        torch.zeros((self.shape[0], self.shape[1], 1))
                        .normal_(mean=self.mean, std=self.std)
                        .to(self._hash_data.device),
                    ],
                    dim=2,
                )
                self._hash_pos = torch.cat(
                    [
                        self._hash_pos,
                        xyz.reshape(3, 1).to(self._hash_pos.device),
                    ],
                    dim=1,
                )
            # else:
            #     print("idx in hash ", idx, self._hash)
        # print("idx ", linearized_idx)
        # print("hash ", self._hash)
        ####################test
        # print("mask ", mask.shape)
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        idx = mask.nonzero()  # nonzero returns all indices with mask == true
        # print("idx mask ", idx.shape)
        coordinate = self.idx_to_coordinate(idx)
        # print("coordinate ", coordinate)
        ####################test
        print("hash data", self._hash_data.shape)
        print("hash pos", self._hash_pos.shape)
        # print("data", self._data.shape)
        # print("shape ", self.shape[0] * self.shape[1] * self.shape[2] * self.shape[3] * self.shape[4])
        # isin = torch.isin(linearized_idx, self._allocated_idx)

    def apply_mask(self, mask):
        # print("apply mask", mask.shape)
        linearized_idx = self.get_idx(mask)
        # print("mask", mask)
        # print("before data ", self._data.shape)
        data = self._data[mask]
        # data = self.get_data_from_idx(linearized_idx)
        # print("after data ", data.shape)
        return data

    def set_mask_value(self, mask, value):
        # linearized_idx = self.get_idx(mask)
        # print("value ", value.shape, value.device)
        # self.set_data_from_idx(linearized_idx, value)
        # print("data ", self._data.shape, self._data.device)
        # print("mask ", mask.shape)
        # print("value ", value.shape, value.device)
        self._data[mask] = value

    def coordinate_to_idx(self, p, bound):
        """
        Normalize coordinate to [-1, 1], corresponds to the bounding box given.

        Args:
            p (tensor, N*3): coordinate.
            bound (tensor, 3*2): the scene bound.

        Returns:
            p (tensor, N*3): normalized coordinate.
        """
        p = p.reshape(-1, 3)
        p[:, 0] = ((p[:, 0] - bound[0, 0]) / (bound[0, 1] - bound[0, 0])) * 2 - 1.0
        p[:, 1] = ((p[:, 1] - bound[1, 0]) / (bound[1, 1] - bound[1, 0])) * 2 - 1.0
        p[:, 2] = ((p[:, 2] - bound[2, 0]) / (bound[2, 1] - bound[2, 0])) * 2 - 1.0
        p = (p + 1) / 2
        p[:, 0] *= self.shape[4]
        p[:, 1] *= self.shape[3]
        p[:, 2] *= self.shape[2]
        return p.type(torch.long)

    def idx_to_coordinate(self, idx):
        z = idx[:, 0]
        y = idx[:, 1]
        x = idx[:, 2]
        x = self.bound[0, 0] + x * self.dim[0]
        y = self.bound[1, 0] + y * self.dim[1]
        z = self.bound[2, 0] + z * self.dim[2]
        return torch.vstack([x, y, z]).T

    def sample(self, p, bound):
        # print("P ", p)
        # print("bound ", bound)
        # idx = self.coordinate_to_idx(p, bound)
        # # print("idx ", idx)
        # linearized_idx = self._linearize_id(idx)
        # # print("linearized_idx ", linearized_idx)
        # c = []
        # for idx in linearized_idx:
        #     # print("idx ", idx)
        #     if idx in self._hash:
        #         data = self._hash_data[:, :, self._hash[int(idx)]]
        #         print("data ", data.shape)
        #         c.append(data)
        #     if not idx in self._hash:
        #         c.append(torch.zeros((self.shape[0], self.shape[1])).to(self.device))
        # c = torch.cat(c, dim=0).T.unsqueeze(0)
        # print("c ", c.shape)
        # print("xyz ", xyz)
        # mask_x = (xyz[:, 0] < bound[0][1]) & (xyz[:, 0] > bound[0][0])
        # mask_y = (xyz[:, 1] < bound[1][1]) & (xyz[:, 1] > bound[1][0])
        # mask_z = (xyz[:, 2] < bound[2][1]) & (xyz[:, 2] > bound[2][0])
        # mask = mask_x & mask_y & mask_z
        # print("mask ", mask)
        p_nor = normalize_3d_coordinate(p.clone(), bound)
        p_nor = p_nor.unsqueeze(0)
        vgrid = p_nor[:, :, None, None].float()
        # print("sample grid feature ")
        # print("p_nor ", p_nor, p_nor.shape)
        # print("vgrid ", vgrid, vgrid.shape)
        # print("c ", c)
        # acutally trilinear interpolation if mode = 'bilinear'
        sample_mode = "bilinear"
        c = (
            F.grid_sample(self._data, vgrid, padding_mode="border", align_corners=True, mode=sample_mode)
            .squeeze(-1)
            .squeeze(-1)
        )
        # print("c ", c.shape)
        return c

    def _linearize_id(self, xyz: torch.Tensor):
        """
        :param xyz (N, 3) long id
        :return: (N, ) lineraized id to be accessed in self.indexer
        """
        return xyz[:, 2] + self.shape[-1] * xyz[:, 1] + (self.shape[-1] * self.shape[-2]) * xyz[:, 0]

    def interpolate_point(self, xyz, bound):
        mask_x = (xyz[:, 0] < bound[0][1]) & (xyz[:, 0] > bound[0][0])
        mask_y = (xyz[:, 1] < bound[1][1]) & (xyz[:, 1] > bound[1][0])
        mask_z = (xyz[:, 2] < bound[2][1]) & (xyz[:, 2] > bound[2][0])
        mask = mask_x & mask_y & mask_z

        out, voxel_mask = self.interpolate_bounded_point(xyz[mask])

        mask_combined = torch.zeros(xyz.shape[0], dtype=bool).to(mask.device)
        mask_combined[mask] = mask[mask] & voxel_mask
        return out, mask_combined

    def interpolate_bounded_point(self, xyz):

        xyz_normalized = (xyz - self.bound_min.unsqueeze(0)) / self.voxel_size
        xyz_normalized = xyz_normalized

        grid = self.cold_vars["latent_vecs"]

        low_ids = torch.floor(xyz_normalized - 0.5).int()
        d = xyz_normalized - low_ids

        offsets = (
            torch.tensor(
                [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1], [1, 0, 0]], dtype=int
            )
            .to(xyz.device)
            .unsqueeze(0)
        )

        corners = (torch.tile(low_ids.unsqueeze(1), (1, 8, 1)) + torch.tile(offsets, (low_ids.shape[0], 1, 1))).long()
        corners = torch.max(
            torch.min(
                corners,
                torch.tile(
                    (torch.tensor(self.n_xyz) - 1).reshape((1, 1, -1)), (corners.shape[0], corners.shape[1], 1)
                ).to(corners.device),
            ),
            torch.tile((torch.tensor([0, 0, 0])).reshape((1, 1, -1)), (corners.shape[0], corners.shape[1], 1)).to(
                corners.device
            ),
        )
        corners_linearized = self._linearize_id(corners.reshape(-1, 3)).reshape(-1, 8).long()
        corners_idx = self.cold_vars["indexer"][corners_linearized.reshape(-1)].reshape(-1, 8)

        mask = torch.sum((corners_idx == -1), dim=-1) == 0
        if corners_idx.shape[0] != mask.shape[0]:
            print(corners_idx.shape[0], mask.shape[0])
        # print(mask.shape)
        # print(mask)
        # print(corners_idx.shape)
        # print("")
        corners_idx = corners_idx[mask]
        d = d[mask]

        c00 = grid[corners_idx[:, 0], :] * (1 - d[:, 0]).unsqueeze(1) + grid[corners_idx[:, 7], :] * d[:, 0].unsqueeze(
            1
        )
        c01 = grid[corners_idx[:, 1], :] * (1 - d[:, 0]).unsqueeze(1) + grid[corners_idx[:, 6], :] * d[:, 0].unsqueeze(
            1
        )
        c10 = grid[corners_idx[:, 3], :] * (1 - d[:, 0]).unsqueeze(1) + grid[corners_idx[:, 4], :] * d[:, 0].unsqueeze(
            1
        )
        c11 = grid[corners_idx[:, 2], :] * (1 - d[:, 0]).unsqueeze(1) + grid[corners_idx[:, 5], :] * d[:, 0].unsqueeze(
            1
        )

        c0 = c00 * (1 - d[:, 1]).unsqueeze(1) + c10 * d[:, 1].unsqueeze(1)
        c1 = c01 * (1 - d[:, 1]).unsqueeze(1) + c11 * d[:, 1].unsqueeze(1)

        c = c0 * (1 - d[:, 2]).unsqueeze(1) + c1 * d[:, 2].unsqueeze(1)

        return c, mask

    # self.cfg = cfg
    # c = {}
    # c_dim = cfg["model"]["c_dim"]
    # self.bound = torch.from_numpy(np.array(cfg["mapping"]["bound"]) * self.scale)
    # bound_divisable = cfg["grid_len"]["bound_divisable"]
    # # enlarge the bound a bit to allow it divisable by bound_divisable
    # self.bound[:, 1] = (
    #     ((self.bound[:, 1] - self.bound[:, 0]) / bound_divisable).int() + 1
    # ) * bound_divisable + self.bound[:, 0]
    # xyz_len = self.bound[:, 1] - self.bound[:, 0]
    #
    # if self.cfg["coarse"]:
    #     coarse_key = "grid_coarse"
    #     coarse_val_shape = list(map(int, (xyz_len * self.coarse_bound_enlarge / coarse_grid_len).tolist()))
    #     coarse_val_shape[0], coarse_val_shape[2] = coarse_val_shape[2], coarse_val_shape[0]
    #     self.coarse_val_shape = coarse_val_shape
    #     val_shape = [1, c_dim, *coarse_val_shape]
    #     coarse_val = torch.zeros(val_shape).normal_(mean=0, std=0.01)
    #     c[coarse_key] = coarse_val
    #
    # middle_key = "grid_middle"
    # middle_val_shape = list(map(int, (xyz_len / middle_grid_len).tolist()))
    # middle_val_shape[0], middle_val_shape[2] = middle_val_shape[2], middle_val_shape[0]
    # self.middle_val_shape = middle_val_shape
    # val_shape = [1, c_dim, *middle_val_shape]
    # middle_val = torch.zeros(val_shape).normal_(mean=0, std=0.01)
    # c[middle_key] = middle_val
    #
    # fine_key = "grid_fine"
    # fine_val_shape = list(map(int, (xyz_len / fine_grid_len).tolist()))
    # fine_val_shape[0], fine_val_shape[2] = fine_val_shape[2], fine_val_shape[0]
    # self.fine_val_shape = fine_val_shape
    # val_shape = [1, c_dim, *fine_val_shape]
    # fine_val = torch.zeros(val_shape).normal_(mean=0, std=0.0001)
    # c[fine_key] = fine_val
    #
    # color_key = "grid_color"
    # color_val_shape = list(map(int, (xyz_len / color_grid_len).tolist()))
    # color_val_shape[0], color_val_shape[2] = color_val_shape[2], color_val_shape[0]
    # self.color_val_shape = color_val_shape
    # val_shape = [1, c_dim, *color_val_shape]
    # color_val = torch.zeros(val_shape).normal_(mean=0, std=0.01)
    # c[color_key] = color_val
    #
    # self.shared_c = c
