# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL
import random
import math
from einops.einops import rearrange
import numpy as np
from torchvision import datasets, transforms
import torchvision
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def build_dataset(is_train, args):
    transform = MaskTransform(is_train, args)

    #root = os.path.join(args.data_path, 'train' if is_train else 'val')
    #dataset = datasets.ImageFolder(root, transform=transform)
    if args.dataset == "c10":
         dataset = torchvision.datasets.CIFAR10(
        root='./data', train=is_train, download=True, transform=transform)
    elif args.dataset == "c100":   
        dataset = torchvision.datasets.CIFAR100(
            root='./data', train=is_train, download=True, transform=transform)
    print(dataset)
    return dataset


def build_transform(is_train, args):
    if args.dataset == "c10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif args.dataset == "c100":
        mean = (0.5071 ,0.4865 ,0.4409)
        std = (0.2009 ,0.1984 ,0.2023)
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)




class RandomMaskingGenerator:
    def __init__(self, input_size, mask_ratio, regular=False):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2

        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_mask = int(mask_ratio * self.num_patches)
        self.regular = regular

        if regular:
            assert mask_ratio == 0.75
        
            candidate_list = []
            while True: # add more
                for j in range(4):
                    candidate = np.ones(4)
                    candidate[j] = 0
                    candidate_list.append(candidate)
                if len(candidate_list) * 4 >= self.num_patches * 2:
                    break
            self.mask_candidate = np.vstack(candidate_list) 
            print('using regular, mask_candidate shape = ', 
                  self.mask_candidate.shape)

    def __repr__(self):
        repr_str = "Mask: total patches {}, mask patches {}, regular {}".format(
            self.num_patches, self.num_mask, self.regular
        )
        return repr_str

    def __call__(self):
        if not self.regular:
            mask = np.hstack([
                np.zeros(self.num_patches - self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
        else:
            mask = self.mask_candidate.copy()
            np.random.shuffle(mask)
            mask = rearrange(mask[:self.num_patches//4], '(h w) (p1 p2) -> (h p1) (w p2)', 
                             h=self.height//2, w=self.width//2, p1=2, p2=2)
            mask = mask.flatten()

        return mask 


class MaskTransform(object):
    def __init__(self, is_train , args):
        self.is_train = is_train
        self.transform = build_transform(is_train , args)

        if not hasattr(args, 'mask_regular'):
            args.mask_regular = False

        self.masked_position_generator = RandomMaskingGenerator(
            args.token_size, args.mask_ratio, args.mask_regular
        )

    def __call__(self, image):
        if self.is_train==True:
            return self.transform(image), self.masked_position_generator()
        else:
            return self.transform(image)

    def __repr__(self):
        repr = "(MaskTransform,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr