import torch.utils.data as data

from PIL import Image
import os
import os.path
from glob import glob
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import torch
import re
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]



def ucf_collate(batch):
    label = torch.zeros(len(batch))
    input = torch.zeros(len(batch), 16, 3, 112, 112)
    for i in range(len(batch)):
        input_label = batch[i]
        label[i] = input_label[1]
        input_list = input_label[0]
        for j in range(len(input_list)):
            input[i][j] = input_list[j]

    input = input.permute(0, 2, 1, 3, 4)

    return (input, label)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    item = []
    for target in os.listdir(dir):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, nfolder, _ in sorted(os.walk(d)):
            for folder in nfolder:
                for _, subfolders, _ in sorted(os.walk(d + '/' + folder)):
                    for subfolder in subfolders:
                        if '16' in subfolder:
                            for _, _, frames in sorted(os.walk(d + '/' + folder + '/' + subfolder)):
                                item = []
                                for fname in sorted(frames):
                                    if is_image_file(fname):
                                        path = os.path.join(root, folder, subfolder, fname)
                                        item.append(path)
                                images.append((item, class_to_idx[target]))

    return images


def default_loader(path):
    changed = []
    for i in range(len(path)):
        changed.append(Image.open(path[i]).convert('RGB').resize((128, 171)))         # , collate_fn=ucf_collate
    return changed              #Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
             changed_img = []
             for i in range(len(img)):
                 changed_img.append(self.transform(img[i]))
            #img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return changed_img, target

    def __len__(self):
        return len(self.imgs)
