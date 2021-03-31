import os.path

from torch_geometric import datasets
from torchvision.transforms import transforms

from data_processing import transformers

__DATASET_ROOT = "./data/"


def modelnet40(num_points: int = 1024, split: str = 'train', rotate: str = 'z', add_noise: bool = None) -> datasets.modelnet.ModelNet:
    dataset_root = os.path.join(__DATASET_ROOT, "modelnet40fp")
    assert 1 <= num_points <= 1024, "num_points must be between 1 and 1024"
    assert split in ['train', 'test'], "split must either be 'train' or 'test'"
    assert rotate in ['none', 'z', 'so3'], "rotate must be one of 'none', 'z', 'so3'"
    train = split == 'train'

    pre_transforms = transforms.Compose([
        transformers.SamplePoints(num=4096),
        transformers.FarthestPoints(num_points=1024)
    ])

    if add_noise is None:
        add_noise = split == 'train'

    if rotate == 'none':
        random_rotate = transformers.Identity()
    elif rotate == 'z':
        random_rotate = transformers.RandomRotateZ()
    elif rotate == 'so3':
        random_rotate = transformers.RandomRotateSO3()
    else:
        raise Exception(f"Invalid rotation {rotate}")

    if train:
        post_transforms = transforms.Compose([
            transformers.ConvertFromGeometric(),
            transformers.NormalizeUnitSphere(),
            transformers.SelectPoints(num_points),
            random_rotate,
            transformers.RemoveNones()
        ])
    else:
        post_transforms = transforms.Compose([
            transformers.ConvertFromGeometric(),
            transformers.NormalizeUnitSphere(),
            transformers.SelectPoints(num_points),
            random_rotate,
            transformers.RemoveNones()
        ])
    if add_noise:
        post_transforms.transforms.append(transformers.GaussianNoise())

    return datasets.modelnet.ModelNet(root=dataset_root, name='40', train=train,
                                      pre_transform=pre_transforms, transform=post_transforms)


def shapenet(num_points: int = 1024, split: str = 'train', rotate: str = 'z') -> datasets.shapenet.ShapeNet:
    dataset_root = os.path.join(__DATASET_ROOT, "shapenet")
    assert 1 <= num_points <= 1024, "num_points must be between 1 and 1024"
    assert split in ['train', 'test'], "split must either be 'train' or 'test'"
    assert rotate in ['none', 'z', 'so3'], "rotate must be one of 'none', 'z', 'so3'"
    train = split == 'train'

    pre_transforms = transforms.Compose([
        transformers.FarthestPoints(num_points=1024)
    ])

    if rotate == 'none':
        random_rotate = transformers.Identity()
    elif rotate == 'z':
        random_rotate = transformers.RandomRotateZ()
    elif rotate == 'so3':
        random_rotate = transformers.RandomRotateSO3()
    else:
        raise Exception(f"Invalid rotation {rotate}")

    if train:
        split = 'trainval'
        post_transforms = transforms.Compose([
            transformers.ConvertFromGeometric(),
            transformers.NormalizeUnitSphere(),
            transformers.SelectPoints(num_points),
            random_rotate,
            transformers.GaussianNoise(),
            transformers.RemoveNones()
        ])
    else:
        split = 'test'
        post_transforms = transforms.Compose([
            transformers.ConvertFromGeometric(),
            transformers.NormalizeUnitSphere(),
            transformers.SelectPoints(num_points),
            random_rotate,
            transformers.RemoveNones()
        ])

    return datasets.shapenet.ShapeNet(root=dataset_root, include_normals=False, split=split,
                                      pre_transform=pre_transforms, transform=post_transforms)
