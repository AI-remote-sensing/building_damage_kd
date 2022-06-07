import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from os import path, makedirs, listdir
import sys
import numpy as np

np.random.seed(1)
import random

random.seed(1)

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import time

eps = 1e-6
from apex import amp

from util.adamw import AdamW
from util.losses import dice_round, ComboLoss

import pandas as pd
from tqdm import tqdm
import timeit
import cv2

from zoo.models import (
    SeResNext50_Unet_Loc,
    SeResNext50_Unet_Loc_KD,
    SeResNext50_Unet_Double,
)

from imgaug import augmenters as iaa

from util.utils import *

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

import gc
from util.emailbox import EmailBot

from apex import amp

import argparse
from util.mongo_logger import Logger

DB = "building_damage_kd"
COLLECTION = "v3_loc"
logger = Logger(DB, COLLECTION)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode", default="T-S", choices=["onlyT", "onlyS", "T-S", "TwoTeacher"]
)
parser.add_argument("--LWF", default=0, choices=[0, 1], type=int)
parser.add_argument("--LFL", default=0, choices=[0, 1], type=int)
parser.add_argument("--clsLFL", default=0, choices=[0, 1], type=int)
parser.add_argument("--KL", default=0, choices=[0, 1], type=int)
parser.add_argument("--dataset", default="../data")
parser.add_argument("--checkpoint_path", default="../weights")
parser.add_argument("--seed", default=1, type=int)
parser.add_argument("--vis_dev", default=0, type=int)
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--val_batch_size", default=4, type=int)
parser.add_argument("--lr", default=0.002, type=float)
parser.add_argument("--weight_decay", default=1e-6, type=float)
parser.add_argument("--theta", default=1.0, type=float)
parser.add_argument("--alpha", default=1.0, type=float)
parser.add_argument("--alpha_cls", default=1.0, type=float)
parser.add_argument("--beta", default=1.0, type=float)
parser.add_argument("--m", default=0.2, type=float)

args = parser.parse_args()
logger.add_attr("LWF", args.LWF, "info")
logger.add_attr("LFL", args.LFL, "info")
logger.add_attr("clsLFL", args.clsLFL, "info")
logger.add_attr("KL", args.KL, "info")
logger.add_attr("mode", args.mode, "info")
logger.add_attr("lr", args.lr, "info")
logger.add_attr("theta", args.theta, "info")
logger.add_attr("alpha", args.alpha, "info")
logger.add_attr("alpha_cls", args.alpha_cls, "info")
logger.add_attr("beta", args.beta, "info")
logger.add_attr("m", args.m, "info")
logger.add_attr("weight_decay", args.weight_decay, "info")
logger.insert_into_db("info")

emailbot = EmailBot("../settings.json")
emailbot.sendOne(
    {
        "title": "显卡%s训练任务开始训练loc" % args.vis_dev,
        "content": "mode=%s,LWF=%s,KL=%s,LFL=%s,clsLFL=%s"
        % (args.mode, args.LWF, args.KL, args.LFL, args.clsLFL),
    }
)
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

train_dirs = ["train", "tier3"]

models_folder = args.checkpoint_path

input_shape = (512, 512)


all_files = []
for d in train_dirs:
    for f in sorted(listdir(path.join(args.dataset + d, "images"))):
        if "_pre_disaster.png" in f:
            all_files.append(path.join(args.dataset + d, "images", f))


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

        if random.random() > 0.985:
            img = cv2.imread(
                fn.replace("_pre_disaster", "_post_disaster"), cv2.IMREAD_COLOR
            )

        msk0 = cv2.imread(fn.replace("/images/", "/masks/"), cv2.IMREAD_UNCHANGED)

        if random.random() > 0.5:
            img = img[::-1, ...]
            msk0 = msk0[::-1, ...]

        if random.random() > 0.05:
            rot = random.randrange(4)
            if rot > 0:
                img = np.rot90(img, k=rot)
                msk0 = np.rot90(msk0, k=rot)

        if random.random() > 0.9:
            shift_pnt = (random.randint(-320, 320), random.randint(-320, 320))
            img = shift_image(img, shift_pnt)
            msk0 = shift_image(msk0, shift_pnt)

        if random.random() > 0.9:
            rot_pnt = (
                img.shape[0] // 2 + random.randint(-320, 320),
                img.shape[1] // 2 + random.randint(-320, 320),
            )
            scale = 0.9 + random.random() * 0.2
            angle = random.randint(0, 20) - 10
            if (angle != 0) or (scale != 1):
                img = rotate_image(img, angle, scale, rot_pnt)
                msk0 = rotate_image(msk0, angle, scale, rot_pnt)

        crop_size = input_shape[0]
        if random.random() > 0.3:
            crop_size = random.randint(
                int(input_shape[0] / 1.1), int(input_shape[0] / 0.9)
            )

        bst_x0 = random.randint(0, img.shape[1] - crop_size)
        bst_y0 = random.randint(0, img.shape[0] - crop_size)
        bst_sc = -1
        try_cnt = random.randint(1, 5)
        for i in range(try_cnt):
            x0 = random.randint(0, img.shape[1] - crop_size)
            y0 = random.randint(0, img.shape[0] - crop_size)
            _sc = msk0[y0 : y0 + crop_size, x0 : x0 + crop_size].sum()
            if _sc > bst_sc:
                bst_sc = _sc
                bst_x0 = x0
                bst_y0 = y0
        x0 = bst_x0
        y0 = bst_y0
        img = img[y0 : y0 + crop_size, x0 : x0 + crop_size, :]
        msk0 = msk0[y0 : y0 + crop_size, x0 : x0 + crop_size]

        if crop_size != input_shape[0]:
            img = cv2.resize(img, input_shape, interpolation=cv2.INTER_LINEAR)
            msk0 = cv2.resize(msk0, input_shape, interpolation=cv2.INTER_LINEAR)

        if random.random() > 0.99:
            img = shift_channels(
                img, random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5)
            )

        if random.random() > 0.99:
            img = change_hsv(
                img, random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5)
            )

        if random.random() > 0.99:
            if random.random() > 0.99:
                img = clahe(img)
            elif random.random() > 0.99:
                img = gauss_noise(img)
            elif random.random() > 0.99:
                img = cv2.blur(img, (3, 3))
        elif random.random() > 0.99:
            if random.random() > 0.99:
                img = saturation(img, 0.9 + random.random() * 0.2)
            elif random.random() > 0.99:
                img = brightness(img, 0.9 + random.random() * 0.2)
            elif random.random() > 0.99:
                img = contrast(img, 0.9 + random.random() * 0.2)

        if random.random() > 0.999:
            el_det = self.elastic.to_deterministic()
            img = el_det.augment_image(img)

        msk = msk0[..., np.newaxis]

        msk = (msk > 127) * 1

        img = preprocess_inputs(img)

        img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        msk = torch.from_numpy(msk.transpose((2, 0, 1))).long()

        sample = {"img": img, "msk": msk, "fn": fn}
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

        msk0 = cv2.imread(fn.replace("/images/", "/masks/"), cv2.IMREAD_UNCHANGED)

        msk = msk0[..., np.newaxis]

        msk = (msk > 127) * 1

        img = preprocess_inputs(img)

        img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        msk = torch.from_numpy(msk.transpose((2, 0, 1))).long()

        sample = {"img": img, "msk": msk, "fn": fn}
        return sample


def validate(model, data_loader):
    global logger
    dices0 = []

    _thr = 0.5

    with torch.no_grad():
        for i, sample in enumerate(tqdm(data_loader)):
            msks = sample["msk"].numpy()
            imgs = sample["img"].cuda(non_blocking=True)
            t1 = time.time()
            out = model(imgs)
            t2 = time.time()
            logger.add_attr("batch_%s" % i, t2 - t1, "time_difference")
            msk_pred = torch.sigmoid(out[:, 0, ...]).cpu().numpy()

            for j in range(msks.shape[0]):
                dices0.append(dice(msks[j, 0], msk_pred[j] > _thr))

    logger.insert_into_db("time_difference")
    d0 = np.mean(dices0)
    logger.add_attr("d0", d0)

    print("Val Dice: {}".format(d0))
    return d0


def evaluate_val_kd(args, data_val, best_score, model, snapshot_name, current_epoch):
    global logger
    model.eval()
    d = validate(model, data_loader=data_val)
    logger.add_attr("epoch", epoch)

    if d > best_score:
        torch.save(
            {
                "epoch": current_epoch + 1,
                "state_dict": model.state_dict(),
                "best_score": d,
            },
            path.join(models_folder, snapshot_name + "_best"),
        )
        best_score = d

    emailbot = EmailBot("../settings.json")
    emailbot.sendOne(
        {
            "title": "显卡%s训练任务进行epoch=%s的测试" % (args.vis_dev, current_epoch),
            "content": "测试分数%s" % d,
        }
    )
    print("score: {}\tscore_best: {}".format(d, best_score))
    return best_score


def train_epoch_kd(
    args,
    current_epoch,
    seg_loss,
    models,
    optimizer,
    scheduler,
    train_data_loader,
):
    model_s, model_t, model_t_cls = models
    theta = args.theta
    alpha = args.alpha
    beta = args.beta
    global logger
    losses = AverageMeter()

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
            model_t_cls.eval()

    for i, sample in enumerate(iterator):
        imgs = sample["img"].cuda(non_blocking=True)
        msks = sample["msk"].cuda(non_blocking=True)

        if args.mode != "onlyS":
            out_t = model_t(imgs)[:, 0, ...]
            soft_out_t = torch.sigmoid(out_t / 2)
            feature_t = model_t.conv1(imgs)
            feature_t = model_t.conv2(feature_t)
            feature_t = model_t.conv3(feature_t)
            feature_t = model_t.conv4(feature_t)
            feature_t = model_t.conv5(feature_t)
        if args.mode != "onlyT":
            out_s = model_s(imgs)[:, 0, ...]
            soft_out_s = torch.sigmoid(out_s / 2)
            feature_s = model_s.conv1(imgs)
            feature_s = model_s.conv2(feature_s)
            feature_s = model_s.conv3(feature_s)
            feature_s = model_s.conv4(feature_s)
            feature_s = model_s.conv5(feature_s)
        if args.mode == "TwoTeacher":
            # out_t_cls = model_t_cls(imgs)
            # soft_out_t_cls = channel_five2two(F.softmax(out_t_cls, dim=1))[:, 1, ...]
            feature_tmp = model_t_cls.conv1(imgs)
            feature_tmp = model_t_cls.conv2(feature_tmp)
            feature_tmp = model_t_cls.conv3(feature_tmp)
            feature_tmp = model_t_cls.conv4(feature_tmp)
            feature_tmp = model_t_cls.conv5(feature_tmp)
            feature_t_cls = model_t_cls.conv1(imgs)
            feature_t_cls = model_t_cls.conv2(feature_t_cls)
            feature_t_cls = model_t_cls.conv3(feature_t_cls)
            feature_t_cls = model_t_cls.conv4(feature_t_cls)
            feature_t_cls = model_t_cls.conv5(feature_t_cls)
            feature_t_cls = torch.cat([feature_tmp, feature_t_cls], 1)

        # parser.add_argument('--loss',default='onlyCls',choices = ['onlyCls','Cls+LWF','Cls+LFL','Cls+LWF+LFL','TwoTeacher'])
        if args.mode in ["T-S", "TwoTeacher"]:
            loss_seg = seg_loss(soft_out_s, msks)
            loss_cls = -torch.log(
                1e-9 + soft_out_s * msks + (1 - soft_out_s) * (1 - msks)
            ).mean()
            loss = theta * loss_cls + loss_seg

            if args.LWF:
                loss_ko = -(
                    (soft_out_t * msks + (1 - soft_out_t) * (1 - msks))
                    * torch.log(
                        1e-9 + soft_out_s * msks + (1 - soft_out_s) * (1 - msks)
                    )
                ).mean()
                loss += loss_ko * beta
            if args.LFL:
                loss_kf = torch.norm(feature_t - feature_s, p=2, dim=0).mean()
                loss += loss_kf * alpha
            if args.KL:
                soft_out_s = torch.sigmoid(out_s)
                softmax_s = torch.cat(
                    ((1 - soft_out_s).unsqueeze(1), soft_out_s.unsqueeze(1)), dim=1
                )
                soft_out_t = torch.sigmoid(out_t)
                softmax_t = torch.cat(
                    ((1 - soft_out_t).unsqueeze(1), soft_out_t.unsqueeze(1)), dim=1
                )
                loss_kl = (
                    (torch.log(1e-9 + softmax_s) - torch.log(1e-9 + softmax_t))
                    * softmax_s
                ).mean()
                loss += loss_kl
            if args.mode == "TwoTeacher":
                loss_t_cls = theta * loss_cls
                # if args.LWF:
                #     loss_ko_cls = (
                #         -(
                #             (soft_out_t * msks + (1 - soft_out_t) * (1 - msks))
                #             * torch.log(
                #                 soft_out_s * msks + (1 - soft_out_s) * (1 - msks)
                #             )
                #         ).mean()
                #         / 2.0
                #     )
                #     loss_t_cls += beta * loss_ko_cls
                if args.clsLFL:
                    loss_kf_cls = torch.norm(
                        feature_s - feature_t_cls[:, :2048, ...], p=2, dim=0
                    ).mean()
                    loss_t_cls += args.alpha_cls * loss_kf_cls
                # if args.KL:
                #     loss_kl_cls = (
                #         (torch.log(softmax_s) - F.log_softmax(out_t_cls, dim=1))
                #         * softmax_s
                #     ).mean()
                #     loss_t_cls += loss_kl_cls
                loss = (1 - args.m) * loss + args.m * loss_t_cls
            with torch.no_grad():
                dice_sc = 1 - dice_round(soft_out_s, msks[:, 0, ...])
        elif args.mode == "onlyT":
            loss_seg = seg_loss(soft_out_t, msks)
            loss_cls = -torch.log(
                1e-9 + soft_out_t * msks + (1 - soft_out_t) * (1 - msks)
            ).mean()
            loss = theta * loss_cls + loss_seg

            with torch.no_grad():
                dice_sc = 1 - dice_round(soft_out_t, msks[:, 0, ...])
        else:
            loss_seg = seg_loss(soft_out_s, msks)
            loss_cls = -torch.log(
                1e-9 + soft_out_s * msks + (1 - soft_out_s) * (1 - msks)
            ).mean()
            loss = theta * loss_cls + loss_seg

            with torch.no_grad():
                dice_sc = 1 - dice_round(soft_out_s, msks[:, 0, ...])

        losses.update(loss.item(), imgs.size(0))

        dices.update(dice_sc, imgs.size(0))
        if not args.LWF:
            loss_ko = torch.tensor(0)
        if not args.LFL:
            loss_kf = torch.tensor(0)
        if not args.KL:
            loss_kl = torch.tensor(0)
        if args.mode == "T-S":
            iterator.set_description(
                "epoch: {}; lr {:.7f}; Loss {loss.val:.4f} ({loss.avg:.4f}),Loss_cls {loss_cls:.4f},Loss_kf {loss_kf:.4f},Loss_ko {loss_ko:.4f},Loss_kl {loss_kl:.4f},Loss_seg {loss_seg:.4f}; Dice {dice.val:.4f} ({dice.avg:.4f})".format(
                    current_epoch,
                    scheduler.get_lr()[-1],
                    loss=losses,
                    loss_cls=theta * loss_cls.item(),
                    loss_kf=alpha * loss_kf.item(),
                    loss_ko=beta * loss_ko.item(),
                    loss_kl=loss_kl.item(),
                    loss_seg=loss_seg.item(),
                    dice=dices,
                )
            )
        elif args.mode == "TwoTeacher":

            loss_ko_cls = torch.tensor(0)
            if not args.clsLFL:
                loss_kf_cls = torch.tensor(0)

            loss_kl_cls = torch.tensor(0)
            iterator.set_description(
                "epoch: {}; lr {:.7f}; Loss {loss.val:.4f} ({loss.avg:.4f}),Loss_cls {loss_cls:.4f},Loss_kf {loss_kf:.4f},Loss_ko {loss_ko:.4f},Loss_kl {loss_kl:.4f},Loss_kf_cls {loss_kf_cls:.4f},Loss_ko_cls {loss_ko_cls:.4f},Loss_kl_cls {loss_kl_cls:.4f},Loss_seg {loss_seg:.4f}; Dice {dice.val:.4f} ({dice.avg:.4f})".format(
                    current_epoch,
                    scheduler.get_lr()[-1],
                    loss=losses,
                    loss_cls=theta * loss_cls.item(),
                    loss_kf=alpha * loss_kf.item(),
                    loss_ko=beta * loss_ko.item(),
                    loss_kl=loss_kl.item(),
                    loss_kf_cls=alpha * loss_kf_cls.item(),
                    loss_ko_cls=beta * loss_ko_cls.item(),
                    loss_kl_cls=loss_kl_cls.item(),
                    loss_seg=loss_seg.item(),
                    dice=dices,
                )
            )

        else:
            iterator.set_description(
                "epoch: {}; lr {:.7f}; Loss {loss.val:.4f}; Dice {dice.val:.4f} ({dice.avg:.4f})".format(
                    current_epoch, scheduler.get_lr()[-1], loss=losses, dice=dices
                )
            )

        optimizer.zero_grad()

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        # loss.backward()
        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1.1)
        optimizer.step()

    scheduler.step(current_epoch)

    print(
        "epoch: {}; lr {:.7f}; Loss {loss.avg:.4f}".format(
            current_epoch, scheduler.get_lr()[-1], loss=losses
        )
    )


if __name__ == "__main__":
    t0 = timeit.default_timer()

    makedirs(models_folder, exist_ok=True)

    seed = args.seed
    vis_dev = args.vis_dev

    os.environ["CUDA_VISIBLE_DEVICES"] = str(vis_dev)

    cudnn.benchmark = True

    batch_size = args.batch_size
    val_batch_size = args.val_batch_size

    snapshot_name = "loc_KD_{}_best".format(logger.log_id)

    train_idxs, val_idxs = train_test_split(
        np.arange(len(all_files)), test_size=0.1, random_state=seed
    )

    np.random.seed(seed + 123)
    random.seed(seed + 123)

    steps_per_epoch = len(train_idxs) // batch_size
    validation_steps = len(val_idxs) // val_batch_size

    print("steps_per_epoch", steps_per_epoch, "validation_steps", validation_steps)

    data_train = TrainData(train_idxs)
    val_train = ValData(val_idxs)

    train_data_loader = DataLoader(
        data_train,
        batch_size=batch_size,
        num_workers=5,
        shuffle=True,
        pin_memory=False,
        drop_last=True,
    )
    val_data_loader = DataLoader(
        val_train,
        batch_size=val_batch_size,
        num_workers=5,
        shuffle=False,
        pin_memory=False,
    )

    if args.mode == "onlyT":
        model_t = SeResNext50_Unet_Loc().cuda()
    elif args.mode == "onlyS":
        model_s = SeResNext50_Unet_Loc_KD().cuda()
    else:
        model_s = SeResNext50_Unet_Loc_KD().cuda()
        model_t = SeResNext50_Unet_Loc().cuda()
        if args.mode == "TwoTeacher":
            model_t_cls = SeResNext50_Unet_Double().cuda()
            checkpoint = torch.load(
                "weights/res50_cls_cce_1_tuned_best", map_location="cpu"
            )
            loaded_dict = checkpoint["state_dict"]
            sd = model_t_cls.state_dict()
            for k in model_t_cls.state_dict():
                if k in loaded_dict and sd[k].size() == loaded_dict[k].size():
                    sd[k] = loaded_dict[k]
            loaded_dict = sd
            model_t_cls.load_state_dict(loaded_dict)
            for (
                key,
                value,
            ) in (
                model_t_cls.named_parameters()
            ):  # named_parameters()包含网络模块名称 key为模型模块名称 value为模型模块值，可以通过判断模块名称进行对应模块冻结
                value.requires_grad = False

    if args.mode != "onlyT":
        params = model_s.parameters()
        optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
        model_s, optimizer = amp.initialize(model_s, optimizer, opt_level="O0")
        scheduler = lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[4, 6, 8, 10, 12, 14, 15, 16, 17, 18, 19, 20],
            gamma=0.5,
        )
    else:
        params = model_t.parameters()
        optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
        model_t, optimizer = amp.initialize(model_t, optimizer, opt_level="O0")
        scheduler = lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[4, 6, 8, 10, 12, 14, 15, 16, 17, 18, 19, 20],
            gamma=0.5,
        )

    seg_loss = ComboLoss({"dice": 3.0, "focal": 10.0}, per_image=False).cuda()

    if args.mode in ["T-S", "TwoTeacher"]:
        snap_to_load = "weights/res50_loc_0_tuned_best"
        checkpoint = torch.load(snap_to_load, map_location="cpu")
        loaded_dict = checkpoint["state_dict"]
        sd = model_t.state_dict()
        for k in model_t.state_dict():
            if k in loaded_dict and sd[k].size() == loaded_dict[k].size():
                sd[k] = loaded_dict[k]
        loaded_dict = sd
        model_t.load_state_dict(loaded_dict)
        print(
            "loaded checkpoint '{}' (epoch {}, best_score {})".format(
                snap_to_load, checkpoint["epoch"], checkpoint["best_score"]
            )
        )
        # named_parameters()包含网络模块名称 key为模型模块名称 value为模型模块值，可以通过判断模块名称进行对应模块冻结
        for key, value in model_t.named_parameters():
            value.requires_grad = False
        del loaded_dict
        del sd
        del checkpoint

    best_score = 0
    _cnt = -1
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
            models = (model_s, model_t, model_t_cls)

    for epoch in range(30):
        train_epoch_kd(
            args,
            epoch,
            seg_loss,
            models,
            optimizer,
            scheduler,
            train_data_loader,
        )
        if epoch % 2 == 0:
            _cnt += 1
            torch.cuda.empty_cache()
            best_score = evaluate_val_kd(
                args, val_data_loader, best_score, model_train, snapshot_name, epoch
            )

    elapsed = timeit.default_timer() - t0
    print("Time: {:.3f} min".format(elapsed / 60))
    emailbot = EmailBot("../settings.json")
    emailbot.sendOne(
        {"title": "显卡%s训练任务完成" % args.vis_dev, "content": "最佳分数%s" % best_score}
    )
