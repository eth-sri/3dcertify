import numpy as np
import torch
from scipy.spatial.transform import Rotation


def rotation_matrix_z(theta: float) -> np.ndarray:
    rotation: Rotation = Rotation.from_euler('z', theta)
    return rotation.as_matrix()


def rotation_matrix_so3(theta: np.ndarray) -> np.ndarray:
    assert len(theta) == 3
    rotation: Rotation = Rotation.from_euler('xyz', theta)
    return rotation.as_matrix()


def rotate_so3(points: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return points.dot(rotation_matrix_so3(theta).T)


def rotate_z(points: np.ndarray, theta: float) -> np.ndarray:
    return points.dot(rotation_matrix_z(theta).T)


def random_rotate_so3(points: np.ndarray, theta_min: float = -np.pi, theta_max: float = np.pi) -> np.ndarray:
    return points.dot(rotation_matrix_so3(np.random.uniform(theta_min, theta_max, 3)).T)


def random_rotate_z(points: np.ndarray, theta_min: float = -np.pi, theta_max: float = np.pi) -> np.ndarray:
    return rotate_z(points, np.random.uniform(theta_min, theta_max))


def rotate_z_batch(points: torch.Tensor, theta: float) -> torch.Tensor:
    rotation_matrix = torch.from_numpy(rotation_matrix_z(theta).T).float().to(points.device)
    return torch.matmul(points, rotation_matrix)


def random_rotate_so3_batch(points: torch.Tensor, theta_min: float = -np.pi, theta_max: float = np.pi) -> torch.Tensor:
    np_points = points.cpu().numpy()
    rotated = np.array([random_rotate_so3(np_points[i], theta_min, theta_max) for i in range(np_points.shape[0])])
    return torch.from_numpy(rotated).float().to(points.device)


def random_rotate_z_batch(points: torch.Tensor, theta_min: float = -np.pi, theta_max: float = np.pi) -> torch.Tensor:
    np_points = points.cpu().numpy()
    rotated = np.array([random_rotate_z(np_points[i], theta_min, theta_max) for i in range(np_points.shape[0])])
    return torch.from_numpy(rotated).float().to(points.device)
