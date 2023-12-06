from collections import Counter
import numpy as np
from torchvision.datasets import CIFAR100, FGVCAircraft, Food101
import torchvision as tv

from ..transforms import get_transforms
from ...utils import logging
logger = logging.get_logger("MOSA")

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


class CIFAR100Dataset(CIFAR100):
    def __init__(self, cfg, split):
        root = cfg.DATA.DATAPATH
        train = True if split == "train" else False
        transform = get_transforms(split, cfg.DATA.CROPSIZE)
        super(CIFAR100Dataset, self).__init__(root=root, train=train, transform=transform)
        self.raw_ds = CIFAR100(root=root, train=train)
        logger.info("Constructing {} dataset {}...".format(
            cfg.DATA.NAME, split))
        
        self.cfg = cfg
        self._split = split
        self.name = cfg.DATA.NAME
        self._construct_imdb()
    
    def _construct_imdb(self):
        logger.info("Number of images: {}".format(len(self.data)))
        logger.info("Number of classes: {}".format(len(self.class_to_idx)))
    
    def get_info(self):
        num_imgs = len(self.data)
        return num_imgs, self.get_class_num()

    def get_class_num(self):
        return self.cfg.DATA.NUMBER_CLASSES
        # return len(self._class_ids)

    def get_class_weights(self, weight_type):
        """get a list of class weight, return a list float"""
        if "train" not in self._split:
            raise ValueError(
                "only getting training class distribution, " + \
                "got split {} instead".format(self._split)
            )

        cls_num = self.get_class_num()
        if weight_type == "none":
            return [1.0] * cls_num
        
        id2counts = Counter(self.classes)
        assert len(id2counts) == cls_num
        num_per_cls = np.array([id2counts[i] for i in self.classes])

        if weight_type == 'inv':
            mu = -1.0
        elif weight_type == 'inv_sqrt':
            mu = -0.5
        weight_list = num_per_cls ** mu
        weight_list = np.divide(
            weight_list, np.linalg.norm(weight_list, 1)) * cls_num
        return weight_list.tolist()
    
    def __getitem__(self, index):
        img, target = super(CIFAR100Dataset, self).__getitem__(index)
        raw_transform = tv.transforms.Compose(
            [
                tv.transforms.Resize(256),
                tv.transforms.CenterCrop(224),
                tv.transforms.ToTensor(),
            ]
        )
        raw_img, test_target = self.raw_ds.__getitem__(index)
        assert(test_target == target)

        sample = {
            "image": img,
            "label": target,
            "raw": raw_transform(raw_img)
        }
        return sample


class AircraftDataset(FGVCAircraft):
    def __init__(self, cfg, split):
        root = cfg.DATA.DATAPATH
        if split == "train":
            split = "trainval"
        if split == "val":
            split = "test"
        transform = get_transforms(split, cfg.DATA.CROPSIZE)
        super(AircraftDataset, self).__init__(root=root, split=split, transform=transform)
        logger.info("Constructing {} dataset {}...".format(
            cfg.DATA.NAME, split))
        
        self.cfg = cfg
        self._split = split
        self.name = cfg.DATA.NAME
        self._construct_imdb()
    
    def _construct_imdb(self):
        logger.info("Number of images: {}".format(len(self._image_files)))
        logger.info("Number of classes: {}".format(len(self.class_to_idx)))
    
    def get_info(self):
        num_imgs = len(self._image_files)
        return num_imgs, self.get_class_num()

    def get_class_num(self):
        return self.cfg.DATA.NUMBER_CLASSES
        # return len(self._class_ids)

    def get_class_weights(self, weight_type):
        """get a list of class weight, return a list float"""
        if "train" not in self._split:
            raise ValueError(
                "only getting training class distribution, " + \
                "got split {} instead".format(self._split)
            )

        cls_num = self.get_class_num()
        if weight_type == "none":
            return [1.0] * cls_num
        
        id2counts = Counter(self._labels)
        assert len(id2counts) == cls_num
        num_per_cls = np.array([id2counts[i] for i in self._labels])

        if weight_type == 'inv':
            mu = -1.0
        elif weight_type == 'inv_sqrt':
            mu = -0.5
        weight_list = num_per_cls ** mu
        weight_list = np.divide(
            weight_list, np.linalg.norm(weight_list, 1)) * cls_num
        return weight_list.tolist()
    
    def __getitem__(self, index):
        img, target = super(AircraftDataset, self).__getitem__(index)
        sample = {
            "image": img,
            "label": target,
        }
        return sample


class Food101Dataset(Food101):
    def __init__(self, cfg, split):
        root = cfg.DATA.DATAPATH
        if split == "val":
            split = "test"
        transform = get_transforms(split, cfg.DATA.CROPSIZE)
        super(Food101Dataset, self).__init__(root=root, split=split, transform=transform)
        logger.info("Constructing {} dataset {}...".format(
            cfg.DATA.NAME, split))
        
        self.cfg = cfg
        self._split = split
        self.name = cfg.DATA.NAME
        self._construct_imdb()
    
    def _construct_imdb(self):
        logger.info("Number of images: {}".format(len(self._image_files)))
        logger.info("Number of classes: {}".format(len(self.class_to_idx)))
    
    def get_info(self):
        num_imgs = len(self._image_files)
        return num_imgs, self.get_class_num()

    def get_class_num(self):
        return self.cfg.DATA.NUMBER_CLASSES
        # return len(self._class_ids)

    def get_class_weights(self, weight_type):
        """get a list of class weight, return a list float"""
        if "train" not in self._split:
            raise ValueError(
                "only getting training class distribution, " + \
                "got split {} instead".format(self._split)
            )

        cls_num = self.get_class_num()
        if weight_type == "none":
            return [1.0] * cls_num
        
        id2counts = Counter(self._labels)
        assert len(id2counts) == cls_num
        num_per_cls = np.array([id2counts[i] for i in self._labels])

        if weight_type == 'inv':
            mu = -1.0
        elif weight_type == 'inv_sqrt':
            mu = -0.5
        weight_list = num_per_cls ** mu
        weight_list = np.divide(
            weight_list, np.linalg.norm(weight_list, 1)) * cls_num
        return weight_list.tolist()
    
    def __getitem__(self, index):
        img, target = super(Food101Dataset, self).__getitem__(index)
        sample = {
            "image": img,
            "label": target,
        }
        return sample
