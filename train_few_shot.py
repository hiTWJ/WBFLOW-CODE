#-------------------------------------
# Project: Learning to Compare: Relation Network for Few-Shot Learning
# Date: 2017.9.21
# Author: Flood Sung
# All Rights Reserved
#-------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import task_generator_test as tg
import os
import math
import argparse
import scipy as sp
import scipy.stats
from dataset_camera import BasicDataset
from torch.utils.data import DataLoader, random_split

from glow_wb import Glow

parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-f","--feature_dim",type = int, default = 48)
parser.add_argument("-r","--relation_dim",type = int, default = 8)
parser.add_argument("-w","--class_num",type = int, default = 8)
# parser.add_argument("-w","--batch_size",type = int, default = 10)
parser.add_argument("-s","--sample_num_per_class",type = int, default = 5)
parser.add_argument("-b","--batch_num_per_class",type = int, default = 10)
parser.add_argument("-e","--episode",type = int, default= 500000)
parser.add_argument("-t","--test_episode", type = int, default = 600)
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-u","--hidden_unit",type=int,default=10)
# glow parameters
parser.add_argument('--n_flow', default=8, type=int, help='number of flows in each block')# 32
parser.add_argument('--n_block', default=2, type=int, help='number of blocks')# 4
parser.add_argument('--no_lu', action='store_true', help='use plain convolution instead of LU decomposed version')
parser.add_argument('--affine', default=False, type=bool, help='use affine coupling instead of additive')
parser.add_argument('--operator', type=str, default='wb',
                    help='style feature transfer operator')

parser.add_argument('--resume', default='/home/lcx/artflow/output/model/GLOW_WB_noconv_feat/best_model.tar', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

args = parser.parse_args()


# Hyper Parameters
FEATURE_DIM = args.feature_dim
RELATION_DIM = args.relation_dim
CLASS_NUM = args.class_num
SAMPLE_NUM_PER_CLASS = args.sample_num_per_class
BATCH_NUM_PER_CLASS = args.batch_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m,h



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

def main():
    # Step 1: init data folders
    print("init data folders")
    # init character folders for dataset construction
    metatrain_folders,metatest_folders = tg.mini_camera_folders()

    # Step 2: init neural networks
    print("init neural networks")

    glow_single = Glow(3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu)

    # feature_encoder = CNNEncoder()
    # relation_network = RelationNetwork(FEATURE_DIM,RELATION_DIM)

    # feature_encoder.apply(weights_init)
    # relation_network.apply(weights_init)

    glow_single.cuda(GPU)
    # relation_network.cuda(GPU)

    feature_encoder_optim = torch.optim.Adam(glow_single.parameters(),lr=LEARNING_RATE)
    feature_encoder_scheduler = StepLR(feature_encoder_optim,step_size=100000,gamma=0.5)

    # relation_network_optim = torch.optim.Adam(relation_network.parameters(),lr=LEARNING_RATE)
    # relation_network_scheduler = StepLR(relation_network_optim,step_size=100000,gamma=0.5)

    print("--------loading checkpoint----------")
    print("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    # args.start_iter = checkpoint['iter']
    glow_single.load_state_dict(checkpoint['state_dict'])
    glow_single = glow_single.to(GPU)


    # for k, v in glow_single.named_parameters():
    #     print('{}: {}'.format(k, v.requires_grad))

    "冻结层"
    ## 冻参数
    print('--------fixed feature projection--------')
    frozen_layers = [glow_single.blocks]

    for layer in frozen_layers:
        for name, value in layer.named_parameters():
            value.requires_grad = False

    for k, v in glow_single.named_parameters():
        print('{}: {}'.format(k, v.requires_grad))


    glow = nn.DataParallel(glow_single, device_ids=[1])
    glow.train()

    # if os.path.exists(str("./models/miniimagenet_feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")):
    #     feature_encoder.load_state_dict(torch.load(str("./models/miniimagenet_feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")))
    #     print("load feature encoder success")
    # if os.path.exists(str("./models/miniimagenet_relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")):
    #     relation_network.load_state_dict(torch.load(str("./models/miniimagenet_relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")))
    #     print("load relation network success")

    # Step 3: build graph
    print("--------Training--------")

    last_accuracy = 1000000
    patch_size = 256

    for episode in range(EPISODE):

        "每一个test_episode都随机选择10个训练数据和10个测试数据？"

        feature_encoder_scheduler.step(episode)
        # relation_network_scheduler.step(episode)


        # train_ = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        # init dataset
        # sample_dataloader is to obtain previous samples for compare
        # batch_dataloader is to batch samples for training
        task = tg.MiniImagenetTask(metatrain_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,BATCH_NUM_PER_CLASS)
        "sample -> train"
        sample_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False)
        "batch -> test"
        batch_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=BATCH_NUM_PER_CLASS,split="test",shuffle=True)

        # sample datas
        samples,sample_labels = sample_dataloader.__iter__().next() #25*3*84*84
        batches,batch_labels = batch_dataloader.__iter__().next()

        # calculate features
        sample_features = feature_encoder(Variable(samples).cuda(GPU)) # 25*64*19*19
        sample_features = sample_features.view(CLASS_NUM,SAMPLE_NUM_PER_CLASS,FEATURE_DIM,19,19)
        "test"
        sample_features = torch.sum(sample_features,1).squeeze(1)
        batch_features = feature_encoder(Variable(batches).cuda(GPU)) # 20x64*5*5

        # calculate relations
        # each batch sample link to every samples to calculate relations
        # to form a 100x128 matrix for relation network
        sample_features_ext = sample_features.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS*CLASS_NUM,1,1,1,1)
        batch_features_ext = batch_features.unsqueeze(0).repeat(CLASS_NUM,1,1,1,1)
        batch_features_ext = torch.transpose(batch_features_ext,0,1)
        relation_pairs = torch.cat((sample_features_ext,batch_features_ext),2).view(-1,FEATURE_DIM*2,19,19)
        relations = relation_network(relation_pairs).view(-1,CLASS_NUM)

        mse = nn.MSELoss().cuda(GPU)
        one_hot_labels = Variable(torch.zeros(BATCH_NUM_PER_CLASS*CLASS_NUM, CLASS_NUM).scatter_(1, batch_labels.view(-1,1), 1).cuda(GPU))
        loss = mse(relations,one_hot_labels)


        # training

        feature_encoder.zero_grad()
        relation_network.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm(feature_encoder.parameters(),0.5)
        torch.nn.utils.clip_grad_norm(relation_network.parameters(),0.5)

        feature_encoder_optim.step()
        relation_network_optim.step()


        if (episode+1)%100 == 0:
                print("episode:",episode+1,"loss",loss.data[0])

        if episode%5000 == 0:

            # test
            print("Testing...")
            accuracies = []
            for i in range(TEST_EPISODE):
                "每一个test_episode都随机选择10个训练数据和10个测试数据？"
                total_rewards = 0
                task = tg.MiniImagenetTask(metatest_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,15)
                sample_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False)
                num_per_class = 5
                test_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=num_per_class,split="test",shuffle=False)

                sample_images,sample_labels = sample_dataloader.__iter__().next()
                for test_images,test_labels in test_dataloader:
                    batch_size = test_labels.shape[0]
                    # calculate features
                    sample_features = feature_encoder(Variable(sample_images).cuda(GPU)) # 5x64
                    sample_features = sample_features.view(CLASS_NUM,SAMPLE_NUM_PER_CLASS,FEATURE_DIM,19,19)
                    sample_features = torch.sum(sample_features,1).squeeze(1)
                    test_features = feature_encoder(Variable(test_images).cuda(GPU)) # 20x64

                    # calculate relations
                    # each batch sample link to every samples to calculate relations
                    # to form a 100x128 matrix for relation network
                    sample_features_ext = sample_features.unsqueeze(0).repeat(batch_size,1,1,1,1)

                    test_features_ext = test_features.unsqueeze(0).repeat(1*CLASS_NUM,1,1,1,1)
                    test_features_ext = torch.transpose(test_features_ext,0,1)
                    relation_pairs = torch.cat((sample_features_ext,test_features_ext),2).view(-1,FEATURE_DIM*2,19,19)
                    relations = relation_network(relation_pairs).view(-1,CLASS_NUM)

                    _,predict_labels = torch.max(relations.data,1)

                    rewards = [1 if predict_labels[j]==test_labels[j] else 0 for j in range(batch_size)]

                    total_rewards += np.sum(rewards)


                accuracy = total_rewards/1.0/CLASS_NUM/15
                accuracies.append(accuracy)


            test_accuracy,h = mean_confidence_interval(accuracies)

            print("test accuracy:",test_accuracy,"h:",h)

            if test_accuracy > last_accuracy:

                # save networks
                torch.save(feature_encoder.state_dict(),str("./models/miniimagenet_feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"))
                torch.save(relation_network.state_dict(),str("./models/miniimagenet_relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"))

                print("save networks for episode:",episode)

                last_accuracy = test_accuracy





if __name__ == '__main__':

    "0:Canon1D, 2:FujiflimXM1, 4:Nikon, 5:Olympus, 6:PansonicGX1, 7:SamsungNX2000, 8: SonyA57"

    main()