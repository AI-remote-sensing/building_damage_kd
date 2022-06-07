import os
import argparse
os.environ["MKL_NUM_THREADS"] = "2" 
os.environ["NUMEXPR_NUM_THREADS"] = "2" 
os.environ["OMP_NUM_THREADS"] = "2" 

eps = 1e-6
from os import path, makedirs, listdir
import sys
import numpy as np
np.random.seed(1)
import random
random.seed(1)
import time

import torch
from torch import nn
import torch.nn.functional as F
from torch.backends import cudnn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

from apex import amp

from adamw import AdamW
from losses import dice_round, ComboLoss

import pandas as pd
from tqdm import tqdm
import timeit
import cv2

from zoo.models import SeResNext50_Unet_Double,SeResNext50_Unet_Double_KD, SeResNext50_Unet_Loc

from imgaug import augmenters as iaa

from utils import *

from skimage.morphology import square, dilation

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

import gc
from emailbox import EmailBot




from mongo_logger import Logger

DB = "building_damage_kd"
COLLECTION = "v3_cls"
logger = Logger(DB,COLLECTION)

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="T-S", choices=["onlyT", "onlyS", "T-S","TwoTeacher"])
parser.add_argument("--transfer",default=0,choices=[0, 1],type=int)
parser.add_argument("--LWF",default=0,choices=[0, 1],type=int)
parser.add_argument("--locLWF",default=0,choices=[0, 1],type=int)
parser.add_argument("--LFL",default=0,choices=[0, 1],type=int)
parser.add_argument("--locLFL",default=0,choices=[0, 1],type=int)
parser.add_argument("--KL",default=0,choices=[0, 1],type=int)
parser.add_argument("--locKL",default=0,choices=[0, 1],type=int)
parser.add_argument(
    "--dataset", default="/data1/su/app/xview2/building_damage_kd/"
)
parser.add_argument("--checkpoint_path", default="weights")
parser.add_argument("--seed", default=1, type=int)
parser.add_argument("--vis_dev", default=1, type=int)
parser.add_argument("--loc_folder", default='pred_loc_val', type=str)
parser.add_argument("--batch_size", default=5, type=int)
parser.add_argument("--val_batch_size", default=4, type=int)
parser.add_argument("--lr", default=0.002, type=float)
parser.add_argument("--weight_decay", default=1e-6, type=float)
parser.add_argument("--theta", default=1.0, type=float)
parser.add_argument("--alpha", default=1.0, type=float)
parser.add_argument("--alpha_loc", default=1.0, type=float)
parser.add_argument("--beta", default=1.0, type=float)
parser.add_argument("--beta_loc", default=1.0, type=float)
parser.add_argument("--m", default=0.2, type=float)

args = parser.parse_args()
logger.add_attr("LWF",args.LWF,'info')
logger.add_attr("locLWF",args.locLWF,'info')
logger.add_attr("LFL",args.LFL,'info')
logger.add_attr("locLFL",args.locLFL,'info')
logger.add_attr("KL",args.KL,'info')
logger.add_attr("locKL",args.locKL,'info')
logger.add_attr("mode",args.mode,'info')
logger.add_attr("transfer",args.mode,'info')
logger.add_attr("lr",args.lr,'info')
logger.add_attr("theta",args.theta,'info')
logger.add_attr("alpha",args.alpha,'info')
logger.add_attr("alpha_loc",args.alpha_loc,'info')
logger.add_attr("beta",args.beta,'info')
logger.add_attr("beta_loc",args.beta_loc,'info')
logger.add_attr("m",args.m,'info')
logger.add_attr("weight_decay",args.weight_decay,'info')
logger.insert_into_db("info")

emailbot = EmailBot('settings.json')
emailbot.sendOne({'title':'显卡%s训练任务开始训练cls'%args.vis_dev,'content':'mode=%s,transfer=%s,LWF=%s,KL=%s,LFL=%s,locLWF=%s,locLFL=%s,locKL=%s'%(args.mode,args.transfer,args.LWF,args.KL,args.LFL,args.locLWF,args.locLFL,args.locKL)})
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

train_dirs = ['train', 'tier3']

models_folder = args.checkpoint_path

loc_folder = args.loc_folder

input_shape = (512, 512)


all_files = []
for d in train_dirs:
    for f in sorted(listdir(path.join(args.dataset + d, 'images'))):
        if '_pre_disaster.png' in f:
            all_files.append(path.join(args.dataset + d, 'images', f))


class TrainData(Dataset):
    def __init__(self, train_idxs):
        super().__init__()
        self.train_idxs = train_idxs
        self.elastic = iaa.ElasticTransformation(alpha=(0.25, 1.2), sigma=0.2)

    def __len__(self):
        return len(self.train_idxs)

    def __getitem__(self, idx):
        _idx = self.train_idxs[idx]

        fn = all_files[_idx]

        img = cv2.imread(fn, cv2.IMREAD_COLOR)
        img2 = cv2.imread(fn.replace('_pre_disaster', '_post_disaster'), cv2.IMREAD_COLOR)

        msk0 = cv2.imread(fn.replace('/images/', '/masks/'), cv2.IMREAD_UNCHANGED)
        lbl_msk1 = cv2.imread(fn.replace('/images/', '/masks/').replace('_pre_disaster', '_post_disaster'), cv2.IMREAD_UNCHANGED)
        msk1 = np.zeros_like(lbl_msk1)
        msk2 = np.zeros_like(lbl_msk1)
        msk3 = np.zeros_like(lbl_msk1)
        msk4 = np.zeros_like(lbl_msk1)
        msk2[lbl_msk1 == 2] = 255
        msk3[lbl_msk1 == 3] = 255
        msk4[lbl_msk1 == 4] = 255
        msk1[lbl_msk1 == 1] = 255

        if random.random() > 0.5:
            img = img[::-1, ...]
            img2 = img2[::-1, ...]
            msk0 = msk0[::-1, ...]
            msk1 = msk1[::-1, ...]
            msk2 = msk2[::-1, ...]
            msk3 = msk3[::-1, ...]
            msk4 = msk4[::-1, ...]

        if random.random() > 0.05:
            rot = random.randrange(4)
            if rot > 0:
                img = np.rot90(img, k=rot)
                img2 = np.rot90(img2, k=rot)
                msk0 = np.rot90(msk0, k=rot)
                msk1 = np.rot90(msk1, k=rot)
                msk2 = np.rot90(msk2, k=rot)
                msk3 = np.rot90(msk3, k=rot)
                msk4 = np.rot90(msk4, k=rot)
                    
        if random.random() > 0.8:
            shift_pnt = (random.randint(-320, 320), random.randint(-320, 320))
            img = shift_image(img, shift_pnt)
            img2 = shift_image(img2, shift_pnt)
            msk0 = shift_image(msk0, shift_pnt)
            msk1 = shift_image(msk1, shift_pnt)
            msk2 = shift_image(msk2, shift_pnt)
            msk3 = shift_image(msk3, shift_pnt)
            msk4 = shift_image(msk4, shift_pnt)
            
        if random.random() > 0.2:
            rot_pnt =  (img.shape[0] // 2 + random.randint(-320, 320), img.shape[1] // 2 + random.randint(-320, 320))
            scale = 0.9 + random.random() * 0.2
            angle = random.randint(0, 20) - 10
            if (angle != 0) or (scale != 1):
                img = rotate_image(img, angle, scale, rot_pnt)
                img2 = rotate_image(img2, angle, scale, rot_pnt)
                msk0 = rotate_image(msk0, angle, scale, rot_pnt)
                msk1 = rotate_image(msk1, angle, scale, rot_pnt)
                msk2 = rotate_image(msk2, angle, scale, rot_pnt)
                msk3 = rotate_image(msk3, angle, scale, rot_pnt)
                msk4 = rotate_image(msk4, angle, scale, rot_pnt)

        crop_size = input_shape[0]
        if random.random() > 0.1:
            crop_size = random.randint(int(input_shape[0] / 1.15), int(input_shape[0] / 0.85))

        bst_x0 = random.randint(0, img.shape[1] - crop_size)
        bst_y0 = random.randint(0, img.shape[0] - crop_size)
        bst_sc = -1
        try_cnt = random.randint(1, 10)
        for i in range(try_cnt):
            x0 = random.randint(0, img.shape[1] - crop_size)
            y0 = random.randint(0, img.shape[0] - crop_size)
            _sc = msk2[y0:y0+crop_size, x0:x0+crop_size].sum() * 5 + msk3[y0:y0+crop_size, x0:x0+crop_size].sum() * 5 + msk4[y0:y0+crop_size, x0:x0+crop_size].sum() * 2 + msk1[y0:y0+crop_size, x0:x0+crop_size].sum()
            if _sc > bst_sc:
                bst_sc = _sc
                bst_x0 = x0
                bst_y0 = y0
        x0 = bst_x0
        y0 = bst_y0
        img = img[y0:y0+crop_size, x0:x0+crop_size, :]
        img2 = img2[y0:y0+crop_size, x0:x0+crop_size, :]
        msk0 = msk0[y0:y0+crop_size, x0:x0+crop_size]
        msk1 = msk1[y0:y0+crop_size, x0:x0+crop_size]
        msk2 = msk2[y0:y0+crop_size, x0:x0+crop_size]
        msk3 = msk3[y0:y0+crop_size, x0:x0+crop_size]
        msk4 = msk4[y0:y0+crop_size, x0:x0+crop_size]
        
        if crop_size != input_shape[0]:
            img = cv2.resize(img, input_shape, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, input_shape, interpolation=cv2.INTER_LINEAR)
            msk0 = cv2.resize(msk0, input_shape, interpolation=cv2.INTER_LINEAR)
            msk1 = cv2.resize(msk1, input_shape, interpolation=cv2.INTER_LINEAR)
            msk2 = cv2.resize(msk2, input_shape, interpolation=cv2.INTER_LINEAR)
            msk3 = cv2.resize(msk3, input_shape, interpolation=cv2.INTER_LINEAR)
            msk4 = cv2.resize(msk4, input_shape, interpolation=cv2.INTER_LINEAR)
            

        if random.random() > 0.96:
            img = shift_channels(img, random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5))
        elif random.random() > 0.96:
            img2 = shift_channels(img2, random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5))

        if random.random() > 0.96:
            img = change_hsv(img, random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5))
        elif random.random() > 0.96:
            img2 = change_hsv(img2, random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5))

        if random.random() > 0.9:
            if random.random() > 0.96:
                img = clahe(img)
            elif random.random() > 0.96:
                img = gauss_noise(img)
            elif random.random() > 0.96:
                img = cv2.blur(img, (3, 3))
        elif random.random() > 0.9:
            if random.random() > 0.96:
                img = saturation(img, 0.9 + random.random() * 0.2)
            elif random.random() > 0.96:
                img = brightness(img, 0.9 + random.random() * 0.2)
            elif random.random() > 0.96:
                img = contrast(img, 0.9 + random.random() * 0.2)

        if random.random() > 0.9:
            if random.random() > 0.96:
                img2 = clahe(img2)
            elif random.random() > 0.96:
                img2 = gauss_noise(img2)
            elif random.random() > 0.96:
                img2 = cv2.blur(img2, (3, 3))
        elif random.random() > 0.9:
            if random.random() > 0.96:
                img2 = saturation(img2, 0.9 + random.random() * 0.2)
            elif random.random() > 0.96:
                img2 = brightness(img2, 0.9 + random.random() * 0.2)
            elif random.random() > 0.96:
                img2 = contrast(img2, 0.9 + random.random() * 0.2)

                
        if random.random() > 0.96:
            el_det = self.elastic.to_deterministic()
            img = el_det.augment_image(img)

        if random.random() > 0.96:
            el_det = self.elastic.to_deterministic()
            img2 = el_det.augment_image(img2)

        msk0 = msk0[..., np.newaxis]
        msk1 = msk1[..., np.newaxis]
        msk2 = msk2[..., np.newaxis]
        msk3 = msk3[..., np.newaxis]
        msk4 = msk4[..., np.newaxis]

        msk = np.concatenate([msk0, msk1, msk2, msk3, msk4], axis=2)
        msk = (msk > 127)

        msk[..., 0] = True
        msk[..., 1] = dilation(msk[..., 1], square(5))
        msk[..., 2] = dilation(msk[..., 2], square(5))
        msk[..., 3] = dilation(msk[..., 3], square(5))
        msk[..., 4] = dilation(msk[..., 4], square(5))
        msk[..., 1][msk[..., 2:].max(axis=2)] = False
        msk[..., 3][msk[..., 2]] = False
        msk[..., 4][msk[..., 2]] = False
        msk[..., 4][msk[..., 3]] = False
        msk[..., 0][msk[..., 1:].max(axis=2)] = False
        msk = msk * 1

        lbl_msk = msk.argmax(axis=2)

        img = np.concatenate([img, img2], axis=2)
        img = preprocess_inputs(img)

        img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        msk = torch.from_numpy(msk.transpose((2, 0, 1))).long()

        sample = {'img': img, 'msk': msk, 'lbl_msk': lbl_msk, 'fn': fn}
        return sample


class ValData(Dataset):
    def __init__(self, image_idxs):
        super().__init__()
        self.image_idxs = image_idxs

    def __len__(self):
        return len(self.image_idxs)

    def __getitem__(self, idx):
        _idx = self.image_idxs[idx]

        fn = all_files[_idx]

        img = cv2.imread(fn, cv2.IMREAD_COLOR)
        img2 = cv2.imread(fn.replace('_pre_disaster', '_post_disaster'), cv2.IMREAD_COLOR)

        msk_loc = cv2.imread(path.join(loc_folder, '{0}.png'.format(fn.split('/')[-1].replace('.png', '_part1.png'))), cv2.IMREAD_UNCHANGED) > (0.3*255)

        msk0 = cv2.imread(fn.replace('/images/', '/masks/'), cv2.IMREAD_UNCHANGED)
        lbl_msk1 = cv2.imread(fn.replace('/images/', '/masks/').replace('_pre_disaster', '_post_disaster'), cv2.IMREAD_UNCHANGED)
        msk1 = np.zeros_like(lbl_msk1)
        msk2 = np.zeros_like(lbl_msk1)
        msk3 = np.zeros_like(lbl_msk1)
        msk4 = np.zeros_like(lbl_msk1)
        msk1[lbl_msk1 == 1] = 255
        msk2[lbl_msk1 == 2] = 255
        msk3[lbl_msk1 == 3] = 255
        msk4[lbl_msk1 == 4] = 255

        msk0 = msk0[..., np.newaxis]
        msk1 = msk1[..., np.newaxis]
        msk2 = msk2[..., np.newaxis]
        msk3 = msk3[..., np.newaxis]
        msk4 = msk4[..., np.newaxis]

        msk = np.concatenate([msk0, msk1, msk2, msk3, msk4], axis=2)
        msk = (msk > 127)

        msk = msk * 1

        lbl_msk = msk[..., 1:].argmax(axis=2)
        
        img = np.concatenate([img, img2], axis=2)
        img = preprocess_inputs(img)

        img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        msk = torch.from_numpy(msk.transpose((2, 0, 1))).long()

        sample = {'img': img, 'msk': msk, 'lbl_msk': lbl_msk, 'fn': fn, 'msk_loc': msk_loc}
        return sample


def validate(model, data_loader):
    global logger
    dices0 = []

    tp = np.zeros((5,))
    fp = np.zeros((5,))
    fn = np.zeros((5,))

    _thr = 0.3

    with torch.no_grad():
        for i, sample in enumerate(tqdm(data_loader)):
            msks = sample["msk"].numpy()
            lbl_msk = sample["lbl_msk"].numpy()
            imgs = sample["img"].cuda(non_blocking=True)
            msk_loc = sample["msk_loc"].numpy() * 1
            t1 = time.time()
            out = model(imgs)
            t2 = time.time()
            logger.add_attr("batch_%s" % i,t2-t1, "time_difference")

            msk_pred = msk_loc
            msk_damage_pred = torch.softmax(out, dim=1).cpu().numpy()[:, 1:, ...]
            
            for j in range(msks.shape[0]):
                tp[4] += np.logical_and(msks[j, 0] > 0, msk_pred[j] > 0).sum()
                fn[4] += np.logical_and(msks[j, 0] < 1, msk_pred[j] > 0).sum()
                fp[4] += np.logical_and(msks[j, 0] > 0, msk_pred[j] < 1).sum()


                targ = lbl_msk[j][msks[j, 0] > 0]
                pred = msk_damage_pred[j].argmax(axis=0)
                pred = pred * (msk_pred[j] > _thr)
                pred = pred[msks[j, 0] > 0]
                for c in range(4):
                    tp[c] += np.logical_and(pred == c, targ == c).sum()
                    fn[c] += np.logical_and(pred != c, targ == c).sum()
                    fp[c] += np.logical_and(pred == c, targ != c).sum()

    logger.insert_into_db("time_difference")
    d0 = 2 * tp[4] / (2 * tp[4] + fp[4] + fn[4]) # loc segmentation f1 two class

    f1_sc = np.zeros((4,))
    for c in range(4):
        f1_sc[c] = 2 * tp[c] / (2 * tp[c] + fp[c] + fn[c])  # mutli-class f1 for four class

    f1 = 4 / np.sum(1.0 / (f1_sc + 1e-6))

    sc = 0.3 * d0 + 0.7 * f1
    logger.add_attr("score",sc)
    logger.add_attr("d0",d0)
    logger.add_attr("F1",f1)
    logger.add_attr("F1_0",f1_sc[0])
    logger.add_attr("F1_1",f1_sc[1])
    logger.add_attr("F1_2",f1_sc[2])
    logger.add_attr("F1_3",f1_sc[3])
    print("Val Score: {}, Dice: {}, F1: {}, F1_0: {}, F1_1: {}, F1_2: {}, F1_3: {}".format(sc, d0, f1, f1_sc[0], f1_sc[1], f1_sc[2], f1_sc[3]))
    return sc


def evaluate_val_kd(args, data_val, best_score, model, snapshot_name, current_epoch):
    global logger
    model.eval()
    d = validate(model, data_loader=data_val)
    logger.add_attr("epoch",epoch)

    if d > best_score:
        torch.save({
            'epoch': current_epoch + 1,
            'state_dict': model.state_dict(),
            'best_score': d,
        }, path.join(models_folder, snapshot_name + '_best'))
        best_score = d

    emailbot = EmailBot('settings.json')
    emailbot.sendOne({'title':'显卡%s训练任务进行epoch=%s的测试'%(args.vis_dev,current_epoch),'content':'测试分数%s'%d})
    print("score: {}\tscore_best: {}".format(d, best_score))
    return best_score


def train_epoch_kd(args, current_epoch, seg_loss, ce_loss, models, optimizer, scheduler, train_data_loader):
    model_s, model_t, model_t_loc = models
    theta = args.theta
    alpha = args.alpha
    beta = args.beta
    global logger
    losses = AverageMeter()
    losses1 = AverageMeter()

    dices = AverageMeter()

    iterator = tqdm(train_data_loader)

    if args.mode == "onlyT":
        model_t.train(mode=True)
    elif args.mode == "onlyS":
        model_s.train(mode=True)
    else:
        model_s.train(mode=True)
        model_t.eval()
        if args.mode == "TwoTeacher":
            model_t_loc.eval()

    for i, sample in enumerate(iterator):
        imgs = sample["img"].cuda(non_blocking=True)
        msks = sample["msk"].cuda(non_blocking=True)
        lbl_msk = sample["lbl_msk"].cuda(non_blocking=True)
        
        # with torch.no_grad():
        if args.mode != "onlyS":
            out_t = model_t(imgs)
            # soft_out_t = torch.exp(soft_out_t)
            feature_tmp = model_t.conv1(imgs[:,:3,:,:])
            feature_tmp = model_t.conv2(feature_tmp)
            feature_tmp = model_t.conv3(feature_tmp)
            feature_tmp = model_t.conv4(feature_tmp)
            feature_tmp = model_t.conv5(feature_tmp)
            feature_t = model_t.conv1(imgs[:,3:,:,:])
            feature_t = model_t.conv2(feature_t)
            feature_t = model_t.conv3(feature_t)
            feature_t = model_t.conv4(feature_t)
            feature_t = model_t.conv5(feature_t)
            feature_t = torch.cat([feature_tmp,feature_t],1)
        if args.mode != "onlyT":
            out_s = model_s(imgs)
            # soft_out_s = torch.exp(soft_out_s)
            feature_tmp = model_s.conv1(imgs[:,:3,:,:])
            feature_tmp = model_s.conv2(feature_tmp)
            feature_tmp = model_s.conv3(feature_tmp)
            feature_tmp = model_s.conv4(feature_tmp)
            feature_tmp = model_s.conv5(feature_tmp)
            feature_s = model_s.conv1(imgs[:,3:,:,:])
            feature_s = model_s.conv2(feature_s)
            feature_s = model_s.conv3(feature_s)
            feature_s = model_s.conv4(feature_s)
            feature_s = model_s.conv5(feature_s)
            feature_s = torch.cat([feature_tmp,feature_s],1)
        if args.mode == 'TwoTeacher':
            out_t_loc = model_t_loc(imgs[:,:3,:,:])[:,0,...]
            
            feature_t_loc = model_t_loc.conv1(imgs[:,:3,:,:])
            feature_t_loc = model_t_loc.conv2(feature_t_loc)
            feature_t_loc = model_t_loc.conv3(feature_t_loc)
            feature_t_loc = model_t_loc.conv4(feature_t_loc)
            feature_t_loc = model_t_loc.conv5(feature_t_loc)

        lbl_msk_04 =torch.cat(((lbl_msk==0).unsqueeze(1),
                     (lbl_msk==1).unsqueeze(1),
                     (lbl_msk==2).unsqueeze(1),
                     (lbl_msk==3).unsqueeze(1),
                     (lbl_msk==4).unsqueeze(1),
                    ), dim=1)
        lbl_msk_01 = torch.cat(((lbl_msk==0).unsqueeze(1),
                       (lbl_msk==1).unsqueeze(1)),dim=1)
        if args.mode in ["T-S","TwoTeacher"]:
            loss_cls = - (F.log_softmax(out_s,dim=1)*lbl_msk_04).mean()
            loss = theta * loss_cls 
            if args.LWF:
                loss_ko = - (F.softmax(out_t,dim=1) * F.log_softmax(out_s,dim=1) * lbl_msk_04).mean()
                loss += beta * loss_ko
            if args.LFL:
                loss_kf = torch.norm(feature_s-feature_t,p=2,dim=0).mean()
                loss += alpha * loss_kf
            if args.KL:
                loss_kl = ((F.log_softmax(out_s,dim=1) - F.log_softmax(out_t,dim=1)) * F.softmax(out_s,dim=1)).mean()
                loss += loss_kl
            if args.mode == 'TwoTeacher':
                loss_loc = theta * loss_cls
                if args.locLWF:
                    soft_out_s = channel_five2two(torch.exp(out_s / 2.0))
                    soft_out_s = soft_out_s[:,1,...] / torch.sum(soft_out_s,dim=1)
                    soft_out_t_loc = torch.exp(out_t_loc / 2.0)
                    soft_out_t_loc = soft_out_t_loc / (1+ soft_out_t_loc)
                    loss_ko_loc = -(
                        (soft_out_t_loc * msks + (1 - soft_out_t_loc) * (1 - msks))
                        * torch.log(1e-9+ soft_out_s * msks + (1 - soft_out_s) * (1 - msks))
                    ).mean() / 2.0
                    loss_loc += args.beta_loc * loss_ko_loc
                if args.locLFL:
                    loss_kf_loc = torch.norm(feature_s[:,:2048,...]-feature_t_loc,p=2,dim=0).mean()
                    loss_loc += args.alpha_loc * loss_kf_loc
                if args.locKL:
                    out_t_loc = F.sigmoid(out_t_loc)
                    soft_out_t_loc = torch.cat(((1- out_t_loc).unsqueeze(1),out_t_loc.unsqueeze(1)) ,dim = 1)
                    soft_out_s = channel_five2two(torch.exp(out_s))
                    soft_out_s = soft_out_s / torch.sum(soft_out_s,dim=1).unsqueeze(1)
                    loss_kl_loc = ((torch.log(1e-9+soft_out_s) - torch.log(1e-9+soft_out_t_loc)) * soft_out_s).mean()
                    loss_loc += loss_kl_loc
                
                loss = (1 - args.m) * loss + args.m * loss_loc

            loss0 = seg_loss(out_s[:, 0, ...], msks[:, 0, ...])
            loss1 = seg_loss(out_s[:, 1, ...], msks[:, 1, ...])
            loss2 = seg_loss(out_s[:, 2, ...], msks[:, 2, ...])
            loss3 = seg_loss(out_s[:, 3, ...], msks[:, 3, ...])
            loss4 = seg_loss(out_s[:, 4, ...], msks[:, 4, ...])

            loss5 = ce_loss(out_s, lbl_msk)

            loss_extra = 0.1 * loss0 + 0.1 * loss1 + 0.3 * loss2 + 0.3 * loss3 + 0.2 * loss4 + loss5 * 11
            loss += loss_extra

            with torch.no_grad():
                _probs = 1 - torch.sigmoid(out_s[:, 0, ...])
                dice_sc = 1 - dice_round(_probs, 1 - msks[:, 0, ...])
        elif args.mode == "onlyT":
            loss0 = seg_loss(out_t[:, 0, ...], msks[:, 0, ...])
            loss1 = seg_loss(out_t[:, 1, ...], msks[:, 1, ...])
            loss2 = seg_loss(out_t[:, 2, ...], msks[:, 2, ...])
            loss3 = seg_loss(out_t[:, 3, ...], msks[:, 3, ...])
            loss4 = seg_loss(out_t[:, 4, ...], msks[:, 4, ...])

            loss5 = ce_loss(out_t, lbl_msk)
            loss = 0.1 * loss0 + 0.1 * loss1 + 0.3 * loss2 + 0.3 * loss3 + 0.2 * loss4 + loss5 * 11
            with torch.no_grad():
                _probs = 1 - torch.sigmoid(out_t[:, 0, ...])
                dice_sc = 1 - dice_round(_probs, 1 - msks[:, 0, ...])
        else:
            loss0 = seg_loss(out_s[:, 0, ...], msks[:, 0, ...])
            loss1 = seg_loss(out_s[:, 1, ...], msks[:, 1, ...])
            loss2 = seg_loss(out_s[:, 2, ...], msks[:, 2, ...])
            loss3 = seg_loss(out_s[:, 3, ...], msks[:, 3, ...])
            loss4 = seg_loss(out_s[:, 4, ...], msks[:, 4, ...])

            loss5 = ce_loss(out_s, lbl_msk)
            loss = 0.1 * loss0 + 0.1 * loss1 + 0.3 * loss2 + 0.3 * loss3 + 0.2 * loss4 + loss5 * 11
            with torch.no_grad():
                _probs = 1 - torch.sigmoid(out_s[:, 0, ...])
                dice_sc = 1 - dice_round(_probs, 1 - msks[:, 0, ...])


        losses.update(loss.item(), imgs.size(0))
        losses1.update(loss5.item(), imgs.size(0))

        dices.update(dice_sc.item(), imgs.size(0))

        if not args.LWF:
            loss_ko = torch.tensor(0)
        if not args.LFL:
            loss_kf = torch.tensor(0)
        if not args.KL:
            loss_kl = torch.tensor(0)
        if args.mode == "T-S":
            iterator.set_description(
                "epoch: {}; lr {:.7f}; Loss {loss.val:.4f} ({loss.avg:.4f}),Loss_cls {loss_cls:.4f},Loss_kf {loss_kf:.4f},Loss_ko {loss_ko:.4f},Loss_kl {loss_kl:.4f}; cce_loss {loss1.val:.4f} ({loss1.avg:.4f}); Dice {dice.val:.4f} ({dice.avg:.4f})".format(
                    current_epoch, scheduler.get_lr()[-1], loss=losses,loss_cls=theta * loss_cls.item(),loss_kf=alpha*loss_kf.item(),loss_ko=beta*loss_ko.item(), loss_kl = loss_kl.item(),loss1=losses1,dice=dices))
        elif args.mode == "TwoTeacher":
            if not args.locLWF:
                loss_ko_loc = torch.tensor(0)
            if not args.locLFL:
                loss_kf_loc = torch.tensor(0)
            if not args.locKL:
                loss_kl_loc = torch.tensor(0)
            iterator.set_description(
                "epoch: {}; lr {:.7f}; Loss {loss.val:.4f} ({loss.avg:.4f}),Loss_cls {loss_cls:.4f},Loss_kf {loss_kf:.4f},Loss_ko {loss_ko:.4f},Loss_kl {loss_kl:.4f},Loss_kf_loc {loss_kf_loc:.4f},Loss_ko_loc {loss_ko_loc:.4f},Loss_kl_loc {loss_kl_loc:.4f}; cce_loss {loss1.val:.4f} ({loss1.avg:.4f}); Dice {dice.val:.4f} ({dice.avg:.4f})".format(
                    current_epoch, scheduler.get_lr()[-1], loss=losses,loss_cls=theta * loss_cls.item(),loss_kf=alpha*loss_kf.item(),loss_ko=beta*loss_ko.item(),loss_kl = loss_kl.item(),loss_kf_loc=alpha*loss_kf_loc.item(),loss_ko_loc=beta*loss_ko_loc.item(), loss_kl_loc=loss_kl_loc.item(),loss1=losses1,dice=dices))
        else:
            iterator.set_description(
                "epoch: {}; lr {:.7f}; Loss {loss.val:.4f}; cce_loss {loss1.val:.4f} ({loss1.avg:.4f}); Dice {dice.val:.4f} ({dice.avg:.4f})".format(
                    current_epoch, scheduler.get_lr()[-1], loss=losses, loss1=losses1, dice=dices
                )
            )

        optimizer.zero_grad()
        # loss.backward()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 0.999)
        optimizer.step()

    scheduler.step(current_epoch)

    logger.add_attr('lr',scheduler.get_last_lr()[-1])
    print("epoch: {}; lr {:.7f}; Loss {loss.avg:.4f}; CCE_loss {loss1.avg:.4f}; Dice {dice.avg:.4f}".format(
            current_epoch, scheduler.get_lr()[-1], loss=losses, loss1=losses1, dice=dices))



if __name__ == '__main__':
    t0 = timeit.default_timer()

    makedirs(models_folder, exist_ok=True)
    
    seed = args.seed
    vis_dev = args.vis_dev

    os.environ["CUDA_VISIBLE_DEVICES"] = str(vis_dev)

    cudnn.benchmark = True

    batch_size = args.batch_size
    val_batch_size = args.val_batch_size

    snapshot_name = 'cls_KD_{}_best'.format(logger.log_id)

    file_classes = []
    for fn in tqdm(all_files):
        fl = np.zeros((4,), dtype=bool)
        msk1 = cv2.imread(fn.replace('/images/', '/masks/').replace('_pre_disaster', '_post_disaster'), cv2.IMREAD_UNCHANGED)
        for c in range(1, 5):
            fl[c-1] = c in msk1
        file_classes.append(fl)
    file_classes = np.asarray(file_classes)

    train_idxs0, val_idxs = train_test_split(np.arange(len(all_files)), test_size=0.1, random_state=seed)

    np.random.seed(seed + 1234)
    random.seed(seed + 1234)

    train_idxs = []
    for i in train_idxs0:
        train_idxs.append(i)
        if file_classes[i, 1:].max():
            train_idxs.append(i)
        if file_classes[i, 1:3].max():
            train_idxs.append(i)
    train_idxs = np.asarray(train_idxs)

    steps_per_epoch = len(train_idxs) // batch_size
    validation_steps = len(val_idxs) // val_batch_size

    print('steps_per_epoch', steps_per_epoch, 'validation_steps', validation_steps)

    data_train = TrainData(train_idxs)
    val_train = ValData(val_idxs)

    train_data_loader = DataLoader(data_train, batch_size=batch_size, num_workers=6, shuffle=True, pin_memory=False, drop_last=True)
    val_data_loader = DataLoader(val_train, batch_size=val_batch_size, num_workers=6, shuffle=False, pin_memory=False)

    if args.mode == "onlyT":
        model_t = SeResNext50_Unet_Double().cuda()
    elif args.mode == "onlyS":
        model_s = SeResNext50_Unet_Double_KD().cuda()
    else:
        model_t = SeResNext50_Unet_Double().cuda()
        model_s = SeResNext50_Unet_Double_KD().cuda()
        if args.mode == "TwoTeacher":
            model_t_loc = SeResNext50_Unet_Loc().cuda()
            checkpoint = torch.load("weights/res50_loc_0_tuned_best", map_location="cpu")
            print("loaded checkpoint '{}' (epoch {}, best_score {})"
                    .format("weights/res50_loc_0_tuned_best", checkpoint['epoch'], checkpoint['best_score']))

            loaded_dict = checkpoint["state_dict"]
            sd = model_t_loc.state_dict()
            for k in model_t_loc.state_dict():
                if k in loaded_dict and sd[k].size() == loaded_dict[k].size():
                    sd[k] = loaded_dict[k]
            loaded_dict = sd
            model_t_loc.load_state_dict(loaded_dict)
            # named_parameters()包含网络模块名称 key为模型模块名称 value为模型模块值，可以通过判断模块名称进行对应模块冻结
            for key, value in model_t_loc.named_parameters():
                value.requires_grad = False
            del loaded_dict
            del sd
            del checkpoint
        

    if args.mode != "onlyT":
        params = model_s.parameters()
        optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
        model_s, optimizer = amp.initialize(model_s, optimizer, opt_level="O0")
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[5, 11, 17, 23, 29, 33, 47, 50, 60, 70, 90, 110, 130, 150, 170, 180, 190], gamma=0.5)
    else:
        params = model_t.parameters()
        optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
        model_t, optimizer = amp.initialize(model_t, optimizer, opt_level="O0")
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[5, 11, 17, 23, 29, 33, 47, 50, 60, 70, 90, 110, 130, 150, 170, 180, 190], gamma=0.5)

    if args.transfer and args.mode !='onlyT':
        snap_to_load = 'res50_loc_{}_KD_best'.format(seed)
        print("=> loading checkpoint '{}'".format(snap_to_load))
        checkpoint = torch.load(path.join(models_folder, snap_to_load), map_location='cpu')
        loaded_dict = checkpoint['state_dict']
        sd = model_s.state_dict()
        for k in model_s.state_dict():
            if k in loaded_dict and sd[k].size() == loaded_dict[k].size():
                sd[k] = loaded_dict[k]
        loaded_dict = sd
        model_s.load_state_dict(loaded_dict)

    if args.mode in ["T-S", "TwoTeacher"]:

    
        snap_to_load = 'weights/res50_cls_cce_1_tuned_best'
        checkpoint = torch.load(snap_to_load,map_location='cpu')
        loaded_dict = checkpoint['state_dict']
        print("loaded checkpoint '{}' (epoch {}, best_score {})"
                .format(snap_to_load, checkpoint['epoch'], checkpoint['best_score']))
        sd = model_t.state_dict()
        for k in model_t.state_dict():
            if k in loaded_dict and sd[k].size() == loaded_dict[k].size():
                sd[k] = loaded_dict[k]
        loaded_dict = sd
        model_t.load_state_dict(loaded_dict)
        for key, value in model_t.named_parameters():# named_parameters()包含网络模块名称 key为模型模块名称 value为模型模块值，可以通过判断模块名称进行对应模块冻结
            value.requires_grad = False
        del loaded_dict
        del sd
        del checkpoint
        
    
    # model_s = nn.DataParallel(model_s).cuda()
    # model_t = nn.DataParallel(model_t).cuda()
    gc.collect()
    torch.cuda.empty_cache()


    seg_loss = ComboLoss({'dice': 0.5, 'focal': 2.0}, per_image=False).cuda()
    ce_loss = nn.CrossEntropyLoss().cuda()

    best_score = 0
    torch.cuda.empty_cache()

    if args.mode == "onlyT":
        model_train = model_t
        models = (None, model_t, None)
    else:
        model_train = model_s
        if args.mode == "onlyS":
            models = (model_s, None, None)
        elif args.mode == "T-S":
            models = (model_s, model_t, None)
        else:
            models = (model_s, model_t, model_t_loc)
    

    for epoch in range(30):
        train_epoch_kd(args, epoch, seg_loss, ce_loss, models, optimizer, scheduler, train_data_loader)
        if epoch % 2 == 0:
            torch.cuda.empty_cache()
            best_score = evaluate_val_kd(args , val_data_loader, best_score, model_train, snapshot_name, epoch)
            logger.insert_into_db()

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))
    emailbot = EmailBot('settings.json')
    emailbot.sendOne({'title':'显卡%s训练任务完成'%args.vis_dev,'content':'最佳分数%s'%best_score})