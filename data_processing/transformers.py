import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from util import rotation


class HDF5Loader(Dataset):

    # source: https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip
    def __init__(self, data_dir, train=True, num_points=1024, transform=None):
        train_files = [
            'ply_data_train0.h5',
            'ply_data_train1.h5',
            'ply_data_train2.h5',
            'ply_data_train3.h5',
            'ply_data_train4.h5',
        ]
        test_files = [
            'ply_data_test0.h5',
            'ply_data_test1.h5',
        ]

        data = []
        labels = []
        files = train_files if train else test_files
        for file in files:
            path = os.path.join(data_dir, file)
            h5file = h5py.File(path, mode='r')
            data.append(h5file['data'])
            labels.append(h5file['label'])
        self.data = np.concatenate(data, axis=0)[:, 0:num_points]
        self.data = np.swapaxes(self.data, 1, 2)
        self.labels = np.concatenate(labels, axis=0).astype('long')
        self.transform = transform
        self.num_classes = np.max(self.labels) + 1

        print(self.data.shape)
        print(self.labels.shape)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            data, label = self.transform((data, label))

        return data, label


class ConvertFromGeometric(object):

    def __call__(self, data):
        return data.pos.numpy(), None if data.face is None else data.face.numpy(), data.y.numpy()

    def __repr__(self):
        return self.__class__.__name__ + '()'


class NormalizeUnitSphere(object):

    def __call__(self, data):
        points, faces, label = data
        # center shape
        offset = np.mean(points, axis=0)
        points -= np.expand_dims(offset, axis=0)

        # scale to unit sphere
        distance = np.max(np.linalg.norm(points, ord=2, axis=1))
        points = points / distance
        assert np.max(np.linalg.norm(points, ord=2, axis=1)) <= 1.0001

        if faces is not None:
            faces = np.reshape(faces, (-1, 3))
            faces -= np.expand_dims(offset, axis=0)
            faces = faces / distance
            faces = np.reshape(faces, (-1, 3, 3))

        return points, faces, label

    def __repr__(self):
        return self.__class__.__name__ + '()'


class SelectPoints(object):

    def __init__(self, num_points):
        self.num_points = num_points

    def __call__(self, data):
        points, faces, label = data
        assert points.shape[0] >= self.num_points, "Cannot select more points than given in input data"
        return points[:self.num_points], None if faces is None else faces[:self.num_points], label[:self.num_points]

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomRotateZ(object):

    def __call__(self, data):
        points, faces, label = data
        theta = np.random.uniform(0, np.pi * 2)
        points = rotation.rotate_z(points, theta)
        if faces is not None:
            faces = rotation.rotate_z(faces, theta)
        return points, faces, label

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomRotateSO3(object):

    def __call__(self, data):
        points, faces, label = data
        theta = np.random.uniform(0, np.pi * 2, 3)
        points = rotation.rotate_so3(points, theta)
        if faces is not None:
            faces = rotation.rotate_so3(faces, theta)
        return points, faces, label

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Identity(object):

    def __call__(self, data):
        return data

    def __repr__(self):
        return self.__class__.__name__ + '()'


class GaussianNoise(object):

    def __call__(self, data):
        points, faces, label = data
        points += np.random.normal(0, 0.02, size=points.shape)
        return points, faces, label

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RemoveNones(object):

    def __call__(self, data):
        points, faces, label = data
        if faces is None:
            return points, label
        else:
            return points, faces, label

    def __repr__(self):
        return self.__class__.__name__ + '()'


class FarthestPoints(object):

    def __init__(self, num_points):
        self.num_points = num_points

    # compute euclidean distance matrix
    def euclidean_distance_matrix(self, x):
        r = np.sum(x * x, 1)
        r = r.reshape(-1, 1)
        distance_mat = r - 2 * np.dot(x, x.T) + r.T
        # return np.sqrt(distance_mat)
        return distance_mat

    # update distance matrix and select the farthest point from set S after a new point is selected
    def update_farthest_distance(self, far_mat, dist_mat, s):
        for i in range(far_mat.shape[0]):
            far_mat[i] = dist_mat[i, s] if far_mat[i] > dist_mat[i, s] else far_mat[i]
        return far_mat, np.argmax(far_mat)

    # initialize matrix to keep track of distance from set s
    def init_farthest_distance(self, far_mat, dist_mat, s):
        for i in range(far_mat.shape[0]):
            far_mat[i] = dist_mat[i, s]
        return far_mat

    def __call__(self, data):
        pos = data.pos.numpy()
        y = data.y if len(data.y) == len(pos) else None
        while pos.shape[0] < self.num_points:
            pos = np.concatenate([pos, pos], axis=0)
            y = np.concatenate([y, y], axis=0) if y is not None else None

        assert pos.shape[1] == 3 and pos.shape[0] >= self.num_points

        distance_matrix = self.euclidean_distance_matrix(pos)

        selected_points = []
        selected_faces = []
        selected_y = []
        s = np.random.randint(pos.shape[0])
        far_mat = self.init_farthest_distance(np.zeros((pos.shape[0])), distance_matrix, s)

        for i in range(self.num_points):
            selected_points.append(pos[s])
            if data.face is not None:
                selected_faces.append(data.face[s])
            if y is not None:
                selected_y.append(y[s])
            far_mat, s = self.update_farthest_distance(far_mat, distance_matrix, s)

        selected_points = torch.from_numpy(np.array(selected_points))
        data.pos = selected_points
        if data.face is not None:
            selected_faces = torch.stack(selected_faces)
            data.face = selected_faces
        if y is not None:
            selected_y = torch.tensor(selected_y)
            data.y = selected_y
        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.num_points)


# Adapted from torch geometric to preserve face information
class SamplePoints(object):
    r"""Uniformly samples :obj:`num` points on the mesh faces according to
    their face area.

    Args:
        num (int): The number of points to sample.
        remove_faces (bool, optional): If set to :obj:`False`, the face tensor
            will not be removed. (default: :obj:`True`)
        include_normals (bool, optional): If set to :obj:`True`, then compute
            normals for each sampled point. (default: :obj:`False`)
    """

    def __init__(self, num):
        self.num = num

    def __call__(self, data):
        print(data)
        pos, face = data.pos, data.face
        assert pos.size(1) == 3 and face.size(0) == 3

        pos_max = pos.max()
        pos = pos / pos_max

        area = (pos[face[1]] - pos[face[0]]).cross(pos[face[2]] - pos[face[0]])
        area = area.norm(p=2, dim=1).abs() / 2

        prob = area / area.sum()
        sample = torch.multinomial(prob, self.num, replacement=True)
        face = face[:, sample]

        frac = torch.rand(self.num, 2, device=pos.device)
        mask = frac.sum(dim=-1) > 1
        frac[mask] = 1 - frac[mask]

        vec1 = pos[face[1]] - pos[face[0]]
        vec2 = pos[face[2]] - pos[face[0]]

        pos_sampled = pos[face[0]]
        pos_sampled += frac[:, :1] * vec1
        pos_sampled += frac[:, 1:] * vec2

        pos_sampled = pos_sampled * pos_max
        data.pos = pos_sampled
        sampled_faces = torch.zeros(self.num, 3, 3)
        sampled_faces[:, 0, :] = pos[face[0]]
        sampled_faces[:, 1, :] = pos[face[1]]
        sampled_faces[:, 2, :] = pos[face[2]]

        sampled_faces = sampled_faces * pos_max
        data.face = sampled_faces

        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.num)


class TestDummy(object):

    def __call__(self, data):
        print(data)
        assert False

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
