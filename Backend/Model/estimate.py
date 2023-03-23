"""
@author: hao
"""

from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
import cv2
import random
from Model.utils import *
from Model.hlnet import *
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
import os
import argparse
from time import time
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import glob
plt.switch_backend('agg')


def read_image(x):
    img_arr = np.array(Image.open(x))
    if len(img_arr.shape) == 2:  # grayscale
        img_arr = np.tile(img_arr, [3, 1, 1]).transpose(1, 2, 0)
    return img_arr


class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):

        image, target, gtcount = sample['image'], sample['target'], sample['gtcount']
        h, w = image.shape[:2]

        if isinstance(self.output_size, tuple):
            new_h = min(self.output_size[0], h)
            new_w = min(self.output_size[1], w)
            assert (new_h, new_w) == self.output_size
        else:
            crop_size = min(self.output_size, h, w)
            assert crop_size == self.output_size
            new_h = new_w = crop_size
        if gtcount > 0:
            mask = target > 0
            ch, cw = int(np.ceil(new_h / 2)), int(np.ceil(new_w / 2))
            mask_center = np.zeros((h, w), dtype=np.uint8)
            mask_center[ch:h-ch+1, cw:w-cw+1] = 1
            mask = (mask & mask_center)
            idh, idw = np.where(mask == 1)
            if len(idh) != 0:
                ids = random.choice(range(len(idh)))
                hc, wc = idh[ids], idw[ids]
                top, left = hc-ch, wc-cw
            else:
                top = np.random.randint(0, h-new_h+1)
                left = np.random.randint(0, w-new_w+1)
        else:
            top = np.random.randint(0, h-new_h+1)
            left = np.random.randint(0, w-new_w+1)

        image = image[top:top+new_h, left:left+new_w, :]
        target = target[top:top+new_h, left:left+new_w]

        return {'image': image, 'target': target, 'gtcount': gtcount}


class RandomFlip(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        image, target, gtcount = sample['image'], sample['target'], sample['gtcount']
        do_mirror = np.random.randint(2)
        if do_mirror:
            image = cv2.flip(image, 1)
            target = cv2.flip(target, 1)
        return {'image': image, 'target': target, 'gtcount': gtcount}


class Normalize(object):

    def __init__(self, scale, mean, std):
        self.scale = scale
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        # image, target, gtcount = sample['image'], sample['target'], sample['gtcount']
        image = sample['image']
        #image, target = image.astype('float32'), target.astype('float32')
        image = image.astype('float32')

        # pixel normalization
        image = (self.scale * image - self.mean) / self.std

        #image, target = image.astype('float32'), target.astype('float32')
        image = image.astype('float32')

        # return {'image': image, 'target': target, 'gtcount': gtcount}
        return {'image': image}


class ZeroPadding(object):
    def __init__(self, psize=32):
        self.psize = psize

    def __call__(self, sample):
        psize = self.psize

        #image, target, gtcount = sample['image'], sample['target'], sample['gtcount']
        image = sample['image']

        h, w = image.size()[-2:]
        ph, pw = (psize-h % psize), (psize-w % psize)
        # print(ph,pw)

        (pl, pr) = (pw//2, pw-pw//2) if pw != psize else (0, 0)
        (pt, pb) = (ph//2, ph-ph//2) if ph != psize else (0, 0)
        if (ph != psize) or (pw != psize):
            tmp_pad = [pl, pr, pt, pb]
            # print(tmp_pad)
            image = F.pad(image, tmp_pad)
            #target = F.pad(target, tmp_pad)

        return {'image': image}
        # return {'image': image, 'target': target, 'gtcount': gtcount}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self):
        pass

    def __call__(self, sample):
        # swap color axis
        # numpy image: H x W x C
        # torch image: C X H X W
        #image, target, gtcount = sample['image'], sample['target'], sample['gtcount']
        image = sample['image']

        image = image.transpose((2, 0, 1))
        # target = np.expand_dims(target, axis=2)
        # target = target.transpose((2, 0, 1))
        #image, target = torch.from_numpy(image), torch.from_numpy(target)
        image = torch.from_numpy(image)

        # return {'image': image, 'target': target, 'gtcount': gtcount}
        return {'image': image}


class UAVDataset(Dataset):
    def __init__(self, images_input, ratio, train=True, transform=None):
        self.images_input = images_input
        self.ratio = ratio
        self.train = train
        self.transform = transform

        # store images and generate ground truths
        self.images = {}
        self.targets = {}
        self.gtcounts = {}
        self.dotimages = {}

    def __len__(self):
        return len(self.images_input)

    def __getitem__(self, idx):
        image = self.images_input[idx]

        h, w = image.shape[:2]
        nh = int(np.ceil(h * self.ratio))
        nw = int(np.ceil(w * self.ratio))
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
        # target = np.zeros((nh, nw), dtype=np.float32)
        # dotimage = image.copy()

        self.images.update({idx: image})
        # self.targets.update({file_name[0]: target})
        # self.gtcounts.update({file_name[0]: gtcount})
        # self.dotimages.update({file_name[0]: dotimage})

        sample = {
            'image': self.images[idx]
            # 'target': self.targets[file_name[0]],
            # 'gtcount': self.gtcounts[file_name[0]]
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


# prevent dataloader deadlock, uncomment if deadlock occurs
# cv.setNumThreads(0)
cudnn.enabled = True

# constant
IMG_SCALE = 1./255
IMG_MEAN = [0.3252, 0.5357, 0.4468]
IMG_STD = [0.1797, 0.2015, 0.1787]
SCALES = [0.7, 1, 1.3]
SHORTER_SIDE = 224

# system-related parameters
DATA_DIR = './Model/data/maize_tassels_counting_uav_dataset'
DATASET = 'uav'
EXP = 'tasselnetv2plus_rf110_i64o8_r0125_crop256_lr-2_bs9_epoch500'
DATA_LIST = './Model/data/maize_tassels_counting_uav_dataset/train.txt'
DATA_VAL_LIST = './Model/data/maize_tassels_counting_uav_dataset/input.txt'

RESTORE_FROM = 'model_best.pth.tar'
SNAPSHOT_DIR = './Model/estimate'
RESULT_DIR = './Model/estimate'

# model-related parameters
INPUT_SIZE = 64
OUTPUT_STRIDE = 8
MODEL = 'tasselnetv2plus'
RESIZE_RATIO = 0.125

# training-related parameters
OPTIMIZER = 'sgd'  # choice in ['sgd', 'adam']
BATCH_SIZE = 9
CROP_SIZE = (256, 256)
LEARNING_RATE = 1e-2
MILESTONES = [200, 400]
MOMENTUM = 0.95
MULT = 1
NUM_EPOCHS = 500
NUM_CPU_WORKERS = 0
PRINT_EVERY = 1
RANDOM_SEED = 2020
WEIGHT_DECAY = 5e-4
VAL_EVERY = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# add a new entry here if creating a new data loader
# dataset_list = {
#    'uav': UAVDataset
# }


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Object Counting Framework")
    # constant
    parser.add_argument("--image-scale", type=float, default=IMG_SCALE,
                        help="Scale factor used in normalization.")
    parser.add_argument("--image-mean", nargs='+', type=float,
                        default=IMG_MEAN, help="Mean used in normalization.")
    parser.add_argument("--image-std", nargs='+', type=float,
                        default=IMG_STD, help="Std used in normalization.")
    parser.add_argument("--scales", type=int,
                        default=SCALES, help="Scales of crop.")
    parser.add_argument("--shorter-side", type=int,
                        default=SHORTER_SIDE, help="Shorter side of the image.")
    # system-related parameters
    parser.add_argument("--data-dir", type=str, default=DATA_DIR,
                        help="Path to the directory containing the dataset.")
    parser.add_argument("--dataset", type=str,
                        default=DATASET, help="Dataset type.")
    parser.add_argument("--exp", type=str, default=EXP,
                        help="Experiment path.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--data-val-list", type=str, default=DATA_VAL_LIST,
                        help="Path to the file listing the images in the val dataset.")
    parser.add_argument("--restore-from", type=str,
                        default=RESTORE_FROM, help="Name of restored model.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--result-dir", type=str, default=RESULT_DIR,
                        help="Where to save inferred results.")
    parser.add_argument("--save-output", action="store_true",
                        help="Whether to save the output.")
    # model-related parameters
    parser.add_argument("--input-size", type=int, default=INPUT_SIZE,
                        help="the minimum input size of the model.")
    parser.add_argument("--output-stride", type=int,
                        default=OUTPUT_STRIDE, help="Output stride of the model.")
    parser.add_argument("--resize-ratio", type=float,
                        default=RESIZE_RATIO, help="Resizing ratio.")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="model to be chosen.")
    parser.add_argument("--use-pretrained", action="store_true",
                        help="Whether to use pretrained model.")
    parser.add_argument("--freeze-bn", action="store_true",
                        help="Whether to freeze encoder bnorm layers.")
    parser.add_argument("--sync-bn", action="store_true",
                        help="Whether to apply synchronized batch normalization.")
    # training-related parameters
    parser.add_argument("--optimizer", type=str, default=OPTIMIZER,
                        choices=['sgd', 'adam'], help="Choose optimizer.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--milestones", nargs='+', type=int,
                        default=MILESTONES, help="Multistep policy.")
    parser.add_argument("--crop-size", nargs='+', type=int,
                        default=CROP_SIZE, help="Size of crop.")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Whether to perform evaluation.")
    parser.add_argument("--learning-rate", type=float,
                        default=LEARNING_RATE, help="Base learning rate for training.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimizer.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--mult", type=float, default=MULT,
                        help="LR multiplier for pretrained layers.")
    parser.add_argument("--num-epochs", type=int,
                        default=NUM_EPOCHS, help="Number of training epochs.")
    parser.add_argument("--num-workers", type=int,
                        default=NUM_CPU_WORKERS, help="Number of CPU cores used.")
    parser.add_argument("--print-every", type=int,
                        default=PRINT_EVERY, help="Print information every often.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--val-every", type=int, default=VAL_EVERY,
                        help="How often performing validation.")
    return parser.parse_args()


def save_checkpoint(state, snapshot_dir, filename='model_ckpt.pth.tar'):
    torch.save(state, '{}/{}'.format(snapshot_dir, filename))


def validate(net, val_loader, args):
    # switch to 'eval' mode
    net.eval()
    cudnn.benchmark = False
    total_time = 0

    pd_counts = []
    files = glob.glob('./Heatmaps/*')
    for f in files:
        os.remove(f)

    with torch.no_grad():
        for i, sample in enumerate(val_loader): 
            torch.cuda.synchronize()
            start = time()

            # image, gtcount = sample['image'], sample['gtcount']
            image = sample['image']

            # inference
            output = net(image.cuda(), is_normalize=not args.save_output)
            output_save = output

            # normalization
            output = Normalizer.gpu_normalizer(output, image.size()[2], image.size()[
                                                   3], args.input_size, args.output_stride)
            # postprocessing
            output = np.clip(output, 0, None)

            pdcount = output.sum()
            torch.cuda.synchronize()
            end = time()

            total_time += (end - start)
            # gtcount = float(gtcount.numpy())
            # gtcount = -1

            # print('manual count=%4.2f, inferred count=%4.2f' % (-1, pdcount))
            # Start Graph
            torch.cuda.synchronize()
            start = time()
            image_name = str(i)
            cmap = plt.cm.get_cmap('nipy_spectral')
            output_save = np.clip(
                output_save.squeeze().cpu().numpy(), 0, None)
            output_save = recover_countmap(
                output_save, image, args.input_size, args.output_stride)
            output_save = output_save / (output_save.max() + 1e-12)
            output_save = cmap(output_save) * 255.

            # overlay heatmap on image
            image = image.squeeze().cpu().numpy()
            image = image.transpose(1, 2, 0)
            image = image * np.array([0.229, 0.224, 0.225]) + \
                np.array([0.485, 0.456, 0.406])
            image = image * 255.
            image = image.astype(np.uint8)
            output_save = 0.5 * image + 0.5 * output_save[:, :, 0:3]
            output_save = output_save.astype(np.uint8)

            sizes = np.shape(output_save)
            fig = plt.figure()
            fig.set_size_inches(1. * sizes[1] / sizes[0], 1, forward=False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()

            fig.add_axes(ax)
            ax.imshow(output_save)
            fig.suptitle('inferred count=%4.2f' %
                             (pdcount), fontsize=4, horizontalalignment='left', verticalalignment='top', x=0.01, y=0.99, color='white')
            
            plt.savefig(os.path.join("./Heatmaps", image_name.replace(
                '.jpg', '.png')), bbox_inches='tight', pad_inches = 0,  dpi=500)
            plt.close()
            # End graph

            pd_counts.append(pdcount)
            torch.cuda.synchronize()
            end = time()
            heatmap_time = (end - start)

    return pd_counts, total_time, heatmap_time


# list of np.array, returns list of ints
def RunOnImages(images: list) -> list:
    args = get_arguments()

    args.evaluate_only = True
    args.save_output = True

    args.image_mean = np.array(args.image_mean).reshape((1, 1, 3))
    args.image_std = np.array(args.image_std).reshape((1, 1, 3))

    args.crop_size = tuple(args.crop_size) if len(
        args.crop_size) > 1 else args.crop_size

    # seeding for reproducbility
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    # instantiate dataset
    # dataset = dataset_list[args.dataset]
    dataset = UAVDataset

    args.snapshot_dir = os.path.join(
        args.snapshot_dir, args.dataset.lower(), args.exp)
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    args.result_dir = os.path.join(
        args.result_dir, args.dataset.lower(), args.exp)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    args.restore_from = os.path.join(args.snapshot_dir, args.restore_from)

    arguments = vars(args)
    for item in arguments:
        print(item, ':\t', arguments[item])

    # instantiate network
    net = CountingModels(
        arc=args.model,
        input_size=args.input_size,
        output_stride=args.output_stride
    )

    net = nn.DataParallel(net)
    net.cuda()

    # filter parameters
    learning_params = [p[1] for p in net.named_parameters()]
    pretrained_params = []

    # define loss function and optimizer
    criterion = nn.L1Loss(reduction='mean').cuda()

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            [
                {'params': learning_params},
                {'params': pretrained_params, 'lr': args.learning_rate / args.mult},
            ],
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            [
                {'params': learning_params},
                {'params': pretrained_params, 'lr': args.learning_rate / args.mult},
            ],
            lr=args.learning_rate
        )
    else:
        raise NotImplementedError

    # restore parameters
    start_epoch = 0
    net.train_loss = {
        'running_loss': [],
        'epoch_loss': []
    }
    net.val_loss = {
        'running_loss': [],
        'epoch_loss': []
    }
    net.measure = {
        'mae': [],
        'mse': [],
        'rmae': [],
        'rmse': [],
        'r2': []
    }
    if args.restore_from is not None:
        if os.path.isfile(args.restore_from):
            checkpoint = torch.load(args.restore_from)
            net.load_state_dict(checkpoint['state_dict'])
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch']
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'train_loss' in checkpoint:
                net.train_loss = checkpoint['train_loss']
            if 'val_loss' in checkpoint:
                net.val_loss = checkpoint['val_loss']
            if 'measure' in checkpoint:
                net.measure['mae'] = checkpoint['measure']['mae'] if 'mae' in checkpoint['measure'] else [
                ]
                net.measure['mse'] = checkpoint['measure']['mse'] if 'mse' in checkpoint['measure'] else [
                ]
                net.measure['rmae'] = checkpoint['measure']['rmae'] if 'rmae' in checkpoint['measure'] else [
                ]
                net.measure['rmse'] = checkpoint['measure']['rmse'] if 'rmse' in checkpoint['measure'] else [
                ]
                net.measure['r2'] = checkpoint['measure']['r2'] if 'r2' in checkpoint['measure'] else [
                ]
            print("==> load checkpoint '{}' (epoch {})"
                  .format(args.restore_from, start_epoch))
        else:
            with open(os.path.join(args.snapshot_dir, args.exp+'.txt'), 'a') as f:
                for item in arguments:
                    print(item, ':\t', arguments[item], file=f)
            print("==> no checkpoint found at '{}'".format(args.restore_from))

    # define transform
    transform_val = [
        Normalize(
            args.image_scale,
            args.image_mean,
            args.image_std
        ),
        ToTensor(),
        ZeroPadding(args.output_stride)
    ]
    composed_transform_val = transforms.Compose(transform_val)

    # define dataset loader
    valset = dataset(
        images_input=images,
        ratio=args.resize_ratio,
        train=False,
        transform=composed_transform_val
    )
    val_loader = DataLoader(
        valset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    res = validate(net, val_loader, args)
    return res


if __name__ == "__main__":
    img_arr = []
    for i in [
            "./data/maize_tassels_counting_uav_dataset/images/DJI_0952 (2).JPG",
            "./data/maize_tassels_counting_uav_dataset/images/DJI_0393 (2).JPG",
            "./data/maize_tassels_counting_uav_dataset/images/DJI_0397.JPG"]:
        img_arr.append(read_image(i))

    print(RunOnImages(img_arr))

    # size_arr = []
    # for i in [
    #         "./data/maize_tassels_counting_uav_dataset/labels/DJI_0952 (2).csv",
    #         "./data/maize_tassels_counting_uav_dataset/labels/DJI_0393 (2).csv",
    #         "./data/maize_tassels_counting_uav_dataset/labels/DJI_0397.csv"]:
    #     size_arr.append(sum(1 for line in open(i)) - 2)

    # print(size_arr)
