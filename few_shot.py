import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
import numpy as np
from PIL import Image
from PIL import ImageFile
from torchvision import transforms
from tensorboardX import SummaryWriter
from torchvision.utils import save_image
import time
from PIL import Image, ImageOps, ImageEnhance, ImageFile
from glow_wb import calc_mean_std

import net
from sampler import InfiniteSamplerWrapper
from dataset_camera import BasicDataset_
from dataset_msa import BasicDataset
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F

from math import log, sqrt, pi

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated


class mae_loss():

    @staticmethod
    def compute(output, target):
        loss = torch.sum(torch.abs(output - target)) / output.size(0)
        return loss


def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


def val_camera(glow, test_loader, name, camera_name, i, tf_writer):
    glow.eval()
    loss = 0.
    count = 0
    pred = []

    iter_num = len(test_loader)
    for batch in test_loader:
        # for i, (x_t, y_t, camera_t) in enumerate(test_loader):
        x_t = batch['image']
        y_t = batch['gt-AWB']
        camera_t = batch['label']
        x_t = x_t.float().to(device)
        y_t = y_t.float().to(device)
        camera_t = camera_t.float().to(device)
        loss_this, pred_this = forward_aux_loss_camera(glow, x_t, y_t,camera_t)
        loss = loss_this.cpu().numpy() + loss

        if (count + 1) % 5 == 0:
            # mapname = camera_name + '_' + str(i)
            tf_writer.add_images(f'{name}/{camera_name}_{str(count)}/', pred_this.cpu().permute(0, 2, 3, 1).numpy()[0],
                                 i, dataformats="HWC")

        count = count + 1
    loss_ = loss / count
    # print(f'{camera_name}---- %d Loss = %.6f ---' % (iter_num, loss_))
    tf_writer.add_scalar(f'{name}/{camera_name}', loss_.item(), i)
    return loss_


def forward_aux_loss_camera(glow, x, y,cam):
    ".unsqueeze(0)"
    x = x.to(device)
    y = y.to(device)
    cam = cam.to(device)

    z_c = glow(x,cam, forward=True)
    pred = glow(z_c,cam, forward=False)

    loss = mae_loss.compute(pred, y)
    return loss, pred


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save(glow, optimizer, filename, total_it):
    state_dict = glow.module.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].to(torch.device('cpu'))

    state = {'iter': total_it, 'state_dict': state_dict, 'optimizer': optimizer.state_dict()}
    torch.save(state, filename)


if __name__ == '__main__':
    "9:IMG,10:8D5U, 0:Canon1DsMKIII, 1: Caonon600D, 2:FujifilmXM1, 3:NikonD40,4:NikonD5200 "

    "不同相机对应的子数据集"

    "检查大数据集的数据是不是从1-100里选的"


    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default='./logs',
                        help='Directory to save the log')
    parser.add_argument('--lr', type=float, default=1e-4)  # 1e-5
    parser.add_argument('--lr_decay', type=float, default=5e-5)  # 5e-5
    "160000, 320000"
    parser.add_argument('--max_episode', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=1)

    parser.add_argument('--mse_weight', type=float, default=0)
    parser.add_argument('--style_weight', type=float, default=1)
    parser.add_argument('--content_weight', type=float, default=0.1)

    # save options
    parser.add_argument('--n_threads', type=int, default=8)
    "200"
    parser.add_argument('--print_interval', type=int, default=200)
    "5000"
    parser.add_argument('--save_model_interval', type=int, default=10000)
    parser.add_argument('--start_iter', type=int, default=0, help='starting iteration')
    "glow_clp.pth"
    "model/GLOW_WB_CAMTRANS_GCONS/140000.tar"
    "model/GLOW_WB_CAMTRANS_CONS_REVER/20000.tar"
    parser.add_argument('--resume', default='/home/lcx/artflow/output/model/GLOW_WB_RE/best_model.tar', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    # glow parameters
    parser.add_argument('--n_flow', default=8, type=int, help='number of flows in each block')  # 32
    parser.add_argument('--n_block', default=2, type=int, help='number of blocks')  # 4
    parser.add_argument('--no_lu', action='store_true', help='use plain convolution instead of LU decomposed version')
    parser.add_argument('--affine', default=False, type=bool, help='use affine coupling instead of additive')
    parser.add_argument('--operator', type=str, default='wb',
                        help='style feature transfer operator')
    parser.add_argument('--name', default='GLOW_fewshot_5', type=str, help='')
    parser.add_argument('--save_dir', default='./output', type=str, help='')
    "Sam 6893.855 ✅"
    "PanasonicGX1 6458.826 ✅"
    "SonyA57 6123.828 ✅"
    "OlympusEPL6 7023.258 ✅"
    parser.add_argument('--cam', default='OlympusEPL6', type=str, help='')
    parser.add_argument('--K', default=5, type=int, help='')

    args = parser.parse_args()

    device = torch.device('cuda:1')

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    # args.resume = os.path.join(args.save_dir, args.resume)

    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    print(f"--------Training--{args.name}----------")

    # l1 loss
    model_dir = '%s/model/%s' % (args.save_dir, args.name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    tf_dir = '%s/log/%s' % (args.save_dir, args.name)
    if not os.path.exists(tf_dir):
        os.makedirs(tf_dir)

    tf_writer = SummaryWriter(log_dir=tf_dir)
    # mseloss = nn.MSELoss()

    # ---------------------------glow------------------------------
    from glow_wb import Glow, Camera_Glow, Camera_Glow_norev, Camera_Glow_norev_re, Camera_Glow_wbwithcam
    glow_single = Camera_Glow_norev_re(3, 12, args.n_flow, args.n_block, affine=args.affine,
                                       conv_lu=not args.no_lu)

    # -----------------------resume training------------------------
    if args.resume:
        if os.path.isfile(args.resume):
            print("--------loading checkpoint----------")
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_iter = checkpoint['iter']
            glow_single.load_state_dict(checkpoint['state_dict'])
        else:
            print("--------no checkpoint found---------")

    state_dict = glow_single.state_dict()
    pretrained_dict1 = {k: v for k, v in state_dict.items() if 'wb' in k or 'cam' in k}
    state = {'state_dict': pretrained_dict1}
    torch.save(state, '/home/lcx/artflow/output/model/GLOW_WB_RE/few_shot.tar')


    "固定参数"
    print('--------fixed feature projection--------')
    ## ,glow_single.wb
    frozen_layers = [glow_single.blocks]
    for layer in frozen_layers:
        for name, value in layer.named_parameters():
            value.requires_grad = False
    for k, v in glow_single.named_parameters():
        print('{}: {}'.format(k, v.requires_grad))

    glow_single = glow_single.to(device)
    glow = nn.DataParallel(glow_single, device_ids=[1])
    glow.train()

    patch_size = 256

    "optimizer"
    optimizer = torch.optim.Adam(glow.module.parameters(), lr=args.lr)

    val_sam_dataset = BasicDataset(name=args.cam+'_ct_5', patch_size=patch_size, patch_num_per_image=1, type='val')
    val_dataset = DataLoader(val_sam_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    max_acc = 10000000
    acc_list = []
    for episode in range(args.max_episode):
        print(f'--------episode:-{episode}------------')
        "训练数据,sample出来K个数据"
        # glow.train()
        train_dataset = BasicDataset_(name=args.cam, patch_size=patch_size, K=args.K+1, type='train')
        train_iter = iter(DataLoader(
            train_dataset, batch_size=args.batch_size,
            sampler=InfiniteSamplerWrapper(train_dataset),
            num_workers=args.n_threads))

        ".to(device)"

        for ii in range(args.K):
            adjust_learning_rate(optimizer, iteration_count=ii)
            glow.train()

            train_images = next(train_iter)
            input_img = train_images['image'].float().to(device)
            gt_img = train_images['gt-AWB'].float().to(device)
            camidx = train_images['label'].float().to(device)

            # if episode == 0:
            #     with torch.no_grad():
            #         # _ = glow.module(input_img, forward=True)
            #         _ = glow.module(input_img, camidx, forward=True)
            #         # _,_ = glow.module(input_img,camidx, forward=True)
            #         continue

            z_c = glow(input_img,camidx, forward=True)
            correct_img = glow(z_c,camidx, forward=False)

            "image loss"
            loss = mae_loss.compute(correct_img, gt_img)

            # optimizer update
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(glow.module.parameters(), 5)
            optimizer.step()

            print(f'{args.name}---- %d -- %d-- Loss_mae = %.6f ' % (episode,ii, loss))
            # tf_writer.add_scalar(f'{args.name}', loss.item(), episode)

            "测试"
            if ii == (args.K-1):
                glow.eval()
                kk = episode
                with torch.no_grad():
                    # val_images = next(val_iter)
                    acc = val_camera(glow, val_dataset, args.name, args.cam, kk, tf_writer)
                    print(f'{args.name}-{args.cam}- %d Loss = %.3f ---' % (episode, acc))

                    acc_list.append(acc)

                    if acc < max_acc:
                        print("best model! save...")
                        max_acc = acc
                        outfile = model_dir + '/' + args.cam+'_best_model.tar'
                        # outfile = os.path.join(model_dir, 'best_model.tar')
                        save(glow, optimizer, outfile,episode)
                    else:
                        print('GG!! best accuracy {:f}'.format(max_acc))

                glow.train()

    acc_list = np.array(acc_list)
    avg_acc = np.average(acc_list)
    print('AVG_LOSS:')
    print(avg_acc)
    np.save(model_dir+'/'+args.cam+'_'+'acc.npy',acc_list)




