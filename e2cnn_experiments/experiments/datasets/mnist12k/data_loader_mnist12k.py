import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
import os.path
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from . import own_transforms

ROOT = "./datasets/mnist12k/"


class mnist_dataset(data.Dataset):
    """ flip-rotated MNIST dataset """
    
    def __init__(self, mode, transform=None, target_transform=None, reshuffle_seed=None):
        """
        :type  mode: string from ['train', 'valid', 'test']
        :param mode: determines which subset of the dataset is loaded and whether augmentation is used
        :type  transform: callable
        :param transform: transformation applied to PIL images, returning transformed version
        :type  target_transform: callable
        :param target_transform: transformation applied to labels
        :type  reshuffle_seed: int
        :param reshuffle_seed: seed to use to reshuffle train or valid sets. If None (default), they are not reshuffled
        """
        assert mode in ['train', 'valid', 'trainval', 'test']
        assert reshuffle_seed is None or (mode != "test" and mode != 'trainval')
        
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform
        
        # load the numpy arrays
        if mode in ["train", "valid", "trainval"]:
            filename = os.path.join(ROOT, 'mnist_trainval.npz')
            
            data = np.load(filename)

            num_train = len(data["labels"])
            indices = np.arange(0, num_train)
            
            if reshuffle_seed is not None:
                rng = np.random.RandomState(reshuffle_seed)
                rng.shuffle(indices)

            split = int(np.floor(num_train * 5/6))
            
            if mode == "train":
                data = {
                    "images": data["images"][indices[:split], :],
                    "labels": data["labels"][indices[:split]]
                }
            elif mode == "valid":
                data = {
                    "images": data["images"][indices[split:], :],
                    "labels": data["labels"][indices[split:]]
                }
            
        else:
            filename = os.path.join(ROOT, 'mnist_test.npz')
            data = np.load(filename)

        self.images = data['images'].astype(np.float32)
        self.labels = data['labels'].astype(np.int64)
        self.num_samples = len(self.labels)
    
    def __getitem__(self, index):
        """
        :type  index: int
        :param index: index of data
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        image, label = self.images[index], self.labels[index]
        # convert to PIL Image
        image = Image.fromarray(image)
        # transform images and labels
        if self.transform is not None:
            self.transform.update_randomization()
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label
    
    def __len__(self):
        return len(self.labels)


def build_mnist12k_loader(mode, batch_size, num_workers=8, rot_interpol_augmentation=False, interpolation=0,
                          reshuffle_seed=None, coords=False, test_rotation_angle=None, train_rotation_angle=None):
    """  """
    rng = np.random.RandomState(42)

    assert mode in ['train', 'valid', 'trainval', 'test']
    assert reshuffle_seed is None or (mode != "test" and mode != 'trainval')

    assert interpolation in [0, 2, 3]  # NEAREST, BILINEAR, BICUBIC
    
    transform = []
    if mode in ['valid', 'test']:
        
        shuffle = False
        drop_last = False
        ######## If a test_rotation_angle is provided, apply that rotation to test images
        if test_rotation_angle is not None:
            transform = [
                transforms.Lambda(lambda img: img.rotate(test_rotation_angle)),
                own_transforms.GrayToTensor()
            ]
        else:
            transform = [own_transforms.GrayToTensor()]
    elif mode in ['train', 'trainval']:
        shuffle = True
        drop_last = True
        if train_rotation_angle is not None:

            class RandomMultipleRotation:
                def __init__(self, angle_step):
                    self.angles = list(range(0, 360, int(round(angle_step))))


                def __call__(self, img):
                    angle = np.random.choice(self.angles)
                    return img.rotate(angle)
                    
            transform = [
                RandomMultipleRotation(train_rotation_angle),
                own_transforms.GrayToTensor()
            ]
        elif rot_interpol_augmentation:  ###
            interpolation_map = {
               0: InterpolationMode.NEAREST,
               2: InterpolationMode.BILINEAR,
               3: InterpolationMode.BICUBIC
            }
            interp_mode = interpolation_map[interpolation]

            
            transform = [
                transforms.RandomRotation(5, interpolation=interp_mode),    ##resample-> interpolation,
                own_transforms.GrayToTensor(),
            ]
        else:
            transform = [own_transforms.GrayToTensor()]
    else:
        raise ValueError('unknown mode for building mnist_rot loader')
    
    if coords:
        transform += [own_transforms.CoordinateField((28, 28))]

    transform = own_transforms.Compose(transform)
    
    dataset = mnist_dataset(mode, transform=transform, reshuffle_seed=reshuffle_seed)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        # sampler=torch.utils.data.sampler.RandomSampler(dataset),
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True
    )
    n_inputs = 1
    n_outputs = 10
    
    if coords:
        n_inputs += 2

    return loader, n_inputs, n_outputs

