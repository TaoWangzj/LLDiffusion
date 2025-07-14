import os
import random
from functools import lru_cache

import numpy as np
import PIL
import torch
from torch.nn.functional import pad
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms.functional import center_crop


def augment(x, y):
    aug = torch.randint(8, [1]).item()
    if aug == 1:
        x = x.flip(1)
        y = y.flip(1)
    elif aug == 2:
        x = x.flip(2)
        y = y.flip(2)
    elif aug == 3:
        x = torch.rot90(x, dims=(1, 2))
        y = torch.rot90(y, dims=(1, 2))
    elif aug == 4:
        x = torch.rot90(x, dims=(1, 2), k=2)
        y = torch.rot90(y, dims=(1, 2), k=2)
    elif aug == 5:
        x = torch.rot90(x, dims=(1, 2), k=3)
        y = torch.rot90(y, dims=(1, 2), k=3)
    elif aug == 6:
        x = torch.rot90(x.flip(1), dims=(1, 2))
        y = torch.rot90(y.flip(1), dims=(1, 2))
    elif aug == 7:
        x = torch.rot90(x.flip(2), dims=(1, 2))
        y = torch.rot90(y.flip(2), dims=(1, 2))
    return x, y


class LOLDataset(Dataset):
    def __init__(self, dir, patch_size, n, transforms, filelist=None, parse_patches=True, augment=None):
        super().__init__()

        if filelist is None:
            LOL_dir = dir
            input_names, gt_names = [], []

            # low train filelist
            lr_root = os.path.join(LOL_dir, 'low')
            gt_root = os.path.join(LOL_dir, 'high')
            images = sorted(os.listdir(lr_root))
            images = [image for image in images if image.endswith(".png")]

            input_names += [os.path.join(lr_root, i) for i in images]
            gt_names += [os.path.join(gt_root, i) for i in images]

            x = list(enumerate(input_names))
            indices, input_names = zip(*x)
            gt_names = [gt_names[idx] for idx in indices]
            self.dir = None
        else:
            self.dir = dir
            train_list = os.path.join(dir, filelist)
            with open(train_list) as f:
                contents = f.readlines()
                input_names = [i.strip() for i in contents]
                gt_names = [i.strip().replace('input', 'gt')
                            for i in input_names]

        self.input_names = input_names
        self.gt_names = gt_names
        self.patch_size = patch_size
        self.transforms = transforms
        self.n = n
        self.parse_patches = parse_patches
        self.augment = augment

    @staticmethod
    def collate_fn(batch):
        lows, highs, labels = [], [], []
        for (low, high), y in batch:
            lows.append(low)
            highs.append(high)
            labels.append(y)
        lows = torch.stack(lows)
        highs = torch.stack(highs)
        labels = torch.LongTensor(labels)
        if lows.ndim == 5:
            n_patches = lows.size(1)
            lows = lows.flatten(0, 1)
            highs = highs.flatten(0, 1)
            labels = labels.repeat_interleave(n_patches)
        return (lows, highs), labels

    @staticmethod
    def get_params(img, output_size, n):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i_list = [random.randint(0, h - th) for _ in range(n)]
        j_list = [random.randint(0, w - tw) for _ in range(n)]
        return i_list, j_list, th, tw

    @staticmethod
    def n_random_crops(img, x, y, h, w):
        crops = []
        for i in range(len(x)):
            new_crop = img.crop((y[i], x[i], y[i] + w, x[i] + h))
            crops.append(new_crop)
        return tuple(crops)

    @lru_cache()
    def get_image(self, index):
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]

        basename = os.path.basename(input_name)
        stem, ext = os.path.splitext(basename)
        img_id = int(stem)

        input_img = PIL.Image.open(os.path.join(
            self.dir, input_name)) if self.dir else PIL.Image.open(input_name)
        try:
            gt_img = PIL.Image.open(os.path.join(
                self.dir, gt_name)) if self.dir else PIL.Image.open(gt_name)
        except:
            gt_img = PIL.Image.open(os.path.join(self.dir, gt_name)).convert('RGB') if self.dir else \
                PIL.Image.open(gt_name).convert('RGB')
        return (input_img, gt_img), img_id

    def __getitem__(self, index):
        (input_img, gt_img), img_id = self.get_image(index)

        if self.parse_patches:
            i, j, h, w = self.get_params(
                input_img, (self.patch_size, self.patch_size), self.n)
            input_patches = self.n_random_crops(input_img, i, j, h, w)
            gt_patches = self.n_random_crops(gt_img, i, j, h, w)

            input_patches = [self.transforms(_) for _ in input_patches]
            gt_patches = [self.transforms(_) for _ in gt_patches]

            if self.augment is not None:
                for i in range(self.n):
                    a, b = input_patches[i], gt_patches[i]
                    a, b = self.augment(a, b)
                    input_patches[i], gt_patches[i] = a, b

            input_patches = torch.stack(input_patches)
            gt_patches = torch.stack(gt_patches)
            return (input_patches, gt_patches), img_id
        else:
            assert self.augment is None
            # Resizing images to multiples of 16 for whole-image restoration
            w, h = input_img.size
            wd_new, ht_new = input_img.size
            if ht_new > wd_new and ht_new > 1024:
                wd_new = int(np.ceil(wd_new * 1024 / ht_new))
                ht_new = 1024
            elif ht_new <= wd_new and wd_new > 1024:
                ht_new = int(np.ceil(ht_new * 1024 / wd_new))
                wd_new = 1024
            wd_new = int(16 * np.ceil(wd_new / 16.0))
            ht_new = int(16 * np.ceil(ht_new / 16.0))

            input_img = self.transforms(input_img)
            gt_img = self.transforms(gt_img)

            # padding
            pad_size = (0, wd_new - w, 0, ht_new - h)
            input_img = pad(input_img[None], pad_size, mode='reflect')[0]
            if self.patch_size is not None:
                input_img = center_crop(input_img, self.patch_size)
                gt_img = center_crop(gt_img, self.patch_size)
            return (input_img, gt_img), img_id

    def __len__(self):
        return len(self.input_names)


class LOL:
    def __init__(self, config):
        self.config = config
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

    def train_loader(self, training=True):
        root = self.config.data.root
        train_dataset = LOLDataset(
            dir=os.path.join(root, 'train'),
            n=self.config.training.patch_n,
            patch_size=self.config.data.image_size,
            transforms=self.transforms,
            parse_patches=True,
            # augment=augment if training else None,
        )

        if not training:
            self.config.training.batch_size = 1
            self.config.sampling.batch_size = 1

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            persistent_workers=True,
            pin_memory=True,
            collate_fn=LOLDataset.collate_fn,
        )
        return train_loader

    def val_loader(self, training=True):
        root = self.config.data.root
        val_dataset = LOLDataset(
            dir=os.path.join(root, 'test'),
            n=self.config.training.patch_n,
            patch_size=None,  # 256 for validation
            transforms=self.transforms,
            parse_patches=False,
        )

        if not training:
            self.config.training.batch_size = 1
            self.config.sampling.batch_size = 1

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.sampling.batch_size,
            shuffle=False,
            pin_memory=True,
            collate_fn=LOLDataset.collate_fn,
        )
        return val_loader

    def test_loader(self, training=False):
        root = self.config.data.root
        test_dataset = LOLDataset(
            dir=os.path.join(root, 'test'),
            n=self.config.training.patch_n,
            patch_size=None,
            transforms=self.transforms,
            parse_patches=False,
        )

        if not training:
            self.config.training.batch_size = 1
            self.config.sampling.batch_size = 1

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.sampling.batch_size,
            shuffle=False,
            pin_memory=True,
            collate_fn=LOLDataset.collate_fn,
        )
        return test_loader