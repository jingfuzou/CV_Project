import numpy as np
import os
import torch
import PIL.Image as Image
import torchvision.transforms as tfs
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn

voc_root = '/home/jeffzou/dataset/VOC2012'


def read_images(root=voc_root, train=True):
    txt_fname = root + '/ImageSets/Segmentation/' + ('train.txt' if train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    data = [os.path.join(root, 'JPEGImages', i + '.jpg') for i in images]
    label = [os.path.join(root, 'SegmentationClass', i + '.png') for i in images]
    return data, label


def rand_crop(data, label, height, width):
    """
    :param data: PIL.Image object
    :param label:  PIL.Image object
    :param height:
    :param width:
    :return:
    """
    data, rect = tfs.RandomCrop((height, width))(data)
    label = tfs.FixedCrop(*rect)(label)
    return data, label


def image2label(im):
    data = np.array(im, dtype='int32')
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    return np.array(cm2lbl[idx], dtype='int64')  # 根据索引得到 label 矩阵


def img_transforms(im, label, crop_size):
    im, label = rand_crop(im, label, *crop_size)
    im_tfs = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    im = im_tfs(im)
    label = image2label(label)
    label = torch.from_numpy(label)
    return im, label


class VOCSegDataset(Dataset):
    '''
    voc dataset
    '''

    def __init__(self, train, crop_size, transforms):
        self.crop_size = crop_size
        self.transforms = transforms
        data_list, label_list = read_images(train=train)
        self.data_list = self._filter(data_list)
        self.label_list = self._filter(label_list)
        print('Read ' + str(len(self.data_list)) + ' images')

    def _filter(self, images):  # 过滤掉图片大小小于 crop 大小的图片
        return [im for im in images if (Image.open(im).size[1] >= self.crop_size[0] and
                                        Image.open(im).size[0] >= self.crop_size[1])]

    def __getitem__(self, idx):
        img = self.data_list[idx]
        label = self.label_list[idx]
        img = Image.open(img)
        label = Image.open(label).convert('RGB')
        img, label = self.transforms(img, label, self.crop_size)
        return img, label

    def __len__(self):
        return len(self.data_list)


if __name__ == '__main__':

    classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
               'dog', 'horse', 'motorbike', 'person', 'potted plant',
               'sheep', 'sofa', 'train', 'tv/monitor']

    # RGB color for each class
    colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
                [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
                [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
                [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
                [0, 192, 0], [128, 192, 0], [0, 64, 128]]

    len(classes), len(colormap)

    cm2lbl = np.zeros(256 ** 3)  # 每个像素点有 0 ~ 255 的选择，RGB 三个通道
    for i, cm in enumerate(colormap):
        cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i  # 建立索引,将一个类别的RGB值对应到一个整数上

    label_im = Image.open(voc_root + '/SegmentationClass/2007_000033.png').convert('RGB')
    label = image2label(label_im)
    print(label[150:160, 240:250])

    # 实例化数据集
    input_shape = (320, 480)
    voc_train = VOCSegDataset(True, input_shape, img_transforms)
    voc_test = VOCSegDataset(False, input_shape, img_transforms)

    train_data = DataLoader(voc_train, 64, shuffle=True, num_workers=4)
    valid_data = DataLoader(voc_test, 128, num_workers=4)

    # 转置卷积
    x = torch.randn(1, 3, 120, 120)
    print(x)
    conv_trans = nn.ConvTranspose2d(3, 10, 4, stride=2, padding=1)
    y = conv_trans(x)
    print(y.shape)
