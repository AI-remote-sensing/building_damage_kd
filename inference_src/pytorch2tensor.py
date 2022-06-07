import os

from os import path, makedirs, listdir
import sys
import numpy as np

np.random.seed(1)
import random

random.seed(1)

import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable
import pandas as pd
from tqdm import tqdm
import timeit
import cv2

from torch2trt import torch2trt


from zoo.models import SeResNext50_Unet_Double_KD

from utils import *


cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

test_dir = "../test/images"
models_folder = "../weights"


def valid():
    # if __name__ == "__main__":
    t0 = timeit.default_timer()

    # vis_dev = sys.argv[2]

    # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # os.environ["CUDA_VISIBLE_DEVICES"] = vis_dev

    pred_folder = "cls_KD_333_best_best"
    makedirs(pred_folder, exist_ok=True)

    # cudnn.benchmark = True

    models = []

    # for seed in [1]:
    snap_to_load = "cls_KD_1610592762_best_best"

    model = SeResNext50_Unet_Double_KD().cuda()
    # model = nn.DataParallel(model).cuda()
    # TODO(sujinhua): change the model mode into kd mode

    print("=> loading checkpoint '{}'".format(snap_to_load))
    checkpoint = torch.load(path.join(models_folder, snap_to_load), map_location="cpu")
    loaded_dict = checkpoint["state_dict"]
    sd = model.state_dict()
    for k in model.state_dict():
        if k in loaded_dict and sd[k].size() == loaded_dict[k].size():
            sd[k] = loaded_dict[k]
    loaded_dict = sd
    model.load_state_dict(loaded_dict)
    print(
        "loaded checkpoint '{}' (epoch {}, best_score {})".format(
            snap_to_load, checkpoint["epoch"], checkpoint["best_score"]
        )
    )

    inp = torch.ones(1, 6, 1024, 1024).cuda()
    print(inp.shape)
    model_trt = torch2trt(model, [inp])

    model_trt.eval()
    models.append(model_trt)

    with torch.no_grad():
        for f in tqdm(sorted(listdir(test_dir))):
            if "_pre_" in f:
                fn = path.join(test_dir, f)

                img = cv2.imread(fn, cv2.IMREAD_COLOR)
                img2 = cv2.imread(fn.replace("_pre_", "_post_"), cv2.IMREAD_COLOR)

                img = np.concatenate([img, img2], axis=2)
                img = preprocess_inputs(img)

                inp = []
                inp.append(img)
                inp.append(img[::-1, ...])
                inp.append(img[:, ::-1, ...])
                inp.append(img[::-1, ::-1, ...])
                # TODO:(sujinhua) there is a trick using the transpose the picy
                inp = np.asarray(inp, dtype="float")
                inp = torch.from_numpy(inp.transpose((0, 3, 1, 2))).float()
                # inp = Variable(inp).cuda()
                inp = inp.cuda()
                print(inp.shape)
                # print(inp)
                # 4, 6, 1024, 1024
                # print('len model {}'.format(len(models)))
                pred = []
                for model in models:

                    msk = model(inp)
                    msk = torch.softmax(msk[:, :, ...], dim=1)
                    msk = msk.cpu().numpy()

                    msk[:, 0, ...] = 1 - msk[:, 0, ...]

                    pred.append(msk[0, ...])
                    pred.append(msk[1, :, ::-1, :])
                    pred.append(msk[2, :, :, ::-1])
                    pred.append(msk[3, :, ::-1, ::-1])

                pred_full = np.asarray(pred).mean(axis=0)

                msk = pred_full * 255
                msk = msk.astype("uint8").transpose(1, 2, 0)
                cv2.imwrite(
                    path.join(
                        pred_folder, "{0}.png".format(f.replace(".png", "_part1.png"))
                    ),
                    msk[..., :3],
                    [cv2.IMWRITE_PNG_COMPRESSION, 9],
                )
                cv2.imwrite(
                    path.join(
                        pred_folder, "{0}.png".format(f.replace(".png", "_part2.png"))
                    ),
                    msk[..., 2:],
                    [cv2.IMWRITE_PNG_COMPRESSION, 9],
                )

    elapsed = timeit.default_timer() - t0
    print("Time: {:.3f} min".format(elapsed / 60))


def trt_valid():
    t0 = timeit.default_timer()

    # vis_dev = sys.argv[2]

    # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # os.environ["CUDA_VISIBLE_DEVICES"] = vis_dev

    pred_folder = "cls_KD_2222_best_best"
    makedirs(pred_folder, exist_ok=True)
    from torch2trt import TRTModule

    import onnx

    onnx_model = onnx.load("SeResNext50.onnx")
    onnx.checker.check_model(onnx_model)

    import onnxruntime

    ort_session = onnxruntime.InferenceSession("SeResNext50.onnx")

    def to_numpy(tensor):
        return (
            tensor.detach().cpu().numpy()
            if tensor.requires_grad
            else tensor.cpu().numpy()
        )

    # models = []
    # models.append(model_trt)

    with torch.no_grad():
        for f in tqdm(sorted(listdir(test_dir))):
            if "_pre_" in f:
                fn = path.join(test_dir, f)

                img = cv2.imread(fn, cv2.IMREAD_COLOR)
                img2 = cv2.imread(fn.replace("_pre_", "_post_"), cv2.IMREAD_COLOR)

                img = np.concatenate([img, img2], axis=2)
                img = preprocess_inputs(img)

                inp = []
                inp.append(img)
                inp.append(img[::-1, ...])
                inp.append(img[:, ::-1, ...])
                inp.append(img[::-1, ::-1, ...])
                # TODO:(sujinhua) there is a trick using the transpose the picy
                inp = np.asarray(inp, dtype="float")
                inp = torch.from_numpy(inp.transpose((0, 3, 1, 2))).float()
                inp = Variable(inp).cuda()
                print(inp.shape)
                print("1")
                ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(inp)}
                print("2")
                msk = ort_session.run(None, ort_inputs)
                print("3")
                # print(inp)
                # 4, 6, 1024, 1024
                # print('len model {}'.format(len(models)))
                # print(msk.shape)
                msk = torch.tensor(msk[0])
                print(msk.shape)
                pred = []
                msk = torch.softmax(msk[:, :, ...], dim=1)
                msk = msk.cpu().numpy()

                msk[:, 0, ...] = 1 - msk[:, 0, ...]

                pred.append(msk[0, ...])
                pred.append(msk[1, :, ::-1, :])
                pred.append(msk[2, :, :, ::-1])
                pred.append(msk[3, :, ::-1, ::-1])

                pred_full = np.asarray(pred).mean(axis=0)

                msk = pred_full * 255
                msk = msk.astype("uint8").transpose(1, 2, 0)
                cv2.imwrite(
                    path.join(
                        pred_folder, "{0}.png".format(f.replace(".png", "_part1.png"))
                    ),
                    msk[..., :3],
                    [cv2.IMWRITE_PNG_COMPRESSION, 9],
                )
                cv2.imwrite(
                    path.join(
                        pred_folder, "{0}.png".format(f.replace(".png", "_part2.png"))
                    ),
                    msk[..., 2:],
                    [cv2.IMWRITE_PNG_COMPRESSION, 9],
                )

    elapsed = timeit.default_timer() - t0
    print("Time: {:.3f} min".format(elapsed / 60))


def torch_2_trt():
    #
    t0 = timeit.default_timer()

    # vis_dev = sys.argv[2]

    # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # os.environ["CUDA_VISIBLE_DEVICES"] = vis_dev

    pred_folder = "cls_KD_1610592762_best_best"
    makedirs(pred_folder, exist_ok=True)

    # cudnn.benchmark = True

    models = []

    # for seed in [1]:
    snap_to_load = "cls_KD_1610592762_best_best"

    model = SeResNext50_Unet_Double_KD().cuda()
    # model = nn.DataParallel(model).cuda()

    print("=> loading checkpoint '{}'".format(snap_to_load))
    checkpoint = torch.load(path.join(models_folder, snap_to_load), map_location="cpu")
    loaded_dict = checkpoint["state_dict"]
    sd = model.state_dict()
    for k in model.state_dict():
        if k in loaded_dict and sd[k].size() == loaded_dict[k].size():
            sd[k] = loaded_dict[k]
    loaded_dict = sd
    model.load_state_dict(loaded_dict)
    print(
        "loaded checkpoint '{}' (epoch {}, best_score {})".format(
            snap_to_load, checkpoint["epoch"], checkpoint["best_score"]
        )
    )

    model.eval()

    # inp = torch.ones(4, 6, 1024, 1024).cuda()
    # print(inp.shape)
    # # print(inp)
    # # 4, 6, 1024, 1024
    # model_trt = torch2trt(model, [inp])

    # torch.save(model_trt.state_dict(), "model_trt.pth")

    # import torchvision
    # import torch
    from torch.autograd import Variable
    import onnx

    # print(torch.__version__)

    input_name = ["input"]
    output_name = ["output"]
    input = Variable(torch.randn(1, 6, 1024, 1024)).cuda()

    # model = torchvision.models.resnet50(pretrained=True).cuda()
    torch.onnx.export(
        model,
        input,
        "SeResNext50.onnx",
        input_names=input_name,
        output_names=output_name,
        dynamic_axes={
            "input": {0: "batch_size"},  # variable length axes
            "output": {0: "batch_size"},
        },
    )

    test = onnx.load("SeResNext50.onnx")
    onnx.checker.check_model(test)
    print("==> Passed")


if __name__ == "__main__":
    # torch_2_trt()
    trt_valid()
    # valid()
