from django.http import HttpResponse,JsonResponse
import json
import uuid
import torch
import random
import os 
import cv2

# loc import

import argparse
import os
import shutil
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import torch.optim
from torch.utils.data import Dataset, DataLoader

import xviewapi.loc.net4 as dn
from xviewapi.loc.dataset2 import LabeledImageDataset,_read_image_as_array,_normalize
import xviewapi.loc.util
import numpy as np
import matplotlib
import scipy
import copy

from xviewapi.loc.loss import *
from xviewapi.loc.metrics import runningScore
from xviewapi.loc.test2 import inference

from .util import error,error102,success

import matplotlib
import matplotlib.image
matplotlib.rcParams['image.cmap'] = 'gray'
 
from uuid import uuid4
import json
from imantics import Polygons, Mask
from simplification.cutil import simplify_coords_vwp

# process_loc_inference

from xviewapi.tools.process_data_inference import process_img_poly

# cls import
import torchvision.datasets as datasets
import xviewapi.cls.senet as senet
import xviewapi.cls.loaddata as loaddata

import shapely.wkt
import shapely
from shapely.geometry import Polygon
from collections import defaultdict, OrderedDict

import pandas as pd
import numpy as np

from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelBinarizer

# deal with cls output
from xviewapi.tools.combine_jsons import combine_output
from xviewapi.tools.inference_image_output import create_inference_image, create_prediction_image

# async
from threading import Thread
import base64

MAX_WIDTH = 102400
MAX_HEIGHT = 102400
RAW_IMG_PATH = "api_data/"
CUT_IMG_PATH = "results/sub_picture/"
CROP_SIZE = 1024
GPU_LIST = [0]

# 储存之前的两个model的path
MODEL_LOC_PATH = "models/_200checkpoint.pth.tar"
MODEL_CLS_PATH = "models/model_best.pth.tar"

# end2end模型path
MODEL_E2E_PATH = "models/end2end_model_best.pth.tar"

LOC_RESULT = "results/loc_result/"

POLYGON_CSV_PATH = "results/polygon_csv/"
POLYGON_IMG_PATH = "results/polygon_img"
POLYGON_CLS_PATH = "results/polygon_cls/"
COMBINE_JSON_PATH = "results/combine_json/"
CLS_IMG_PATH = "results/cls_img/"
MAX_REQUEST=5
global REQUEST_COUNT
REQUEST_COUNT=0

MEDIA_PATH = 'media/'

def test_json(request):
    if request.method == "POST":
        req = json.loads(request.body)
        key_flag = req.get("title") and req.get("content") and len(req)==2
        if key_flag:
            title = req["title"]
            content = req["content"]
            return JsonResponse({"status":"BS.200","msg":"publish article sucess."})
        else:
            return  JsonResponse({"status":"BS.102","message":"please check param."})
    else:
        return  JsonResponse({"status":"BS.101","message":"not a post","tensor_num":float(x.cpu())})

# api 1
def get_sub_picture(request):
    global REQUEST_COUNT
    if request.method == "POST":
        req = json.loads(request.body)
        key_flag = req.get("id") and req.get("x") and req.get("y") and req.get("width") and req.get("height") and len(req) == 5
        if key_flag:
            id_ = int(req['id'])
            x = int(req['x'])
            y = int(req['y'])

            width = int(req['width'])
            height = int(req['height'])
            if id_ <= 100 and x >= 0 and y >= 0 and width > 0 and height > 0 and x+width <= MAX_WIDTH and y+height <= MAX_HEIGHT:
                if REQUEST_COUNT < MAX_REQUEST:
                    files = os.listdir(RAW_IMG_PATH + str(id_))
                    if 'post' in files[0]:
                        post_file = files[0]
                    else:
                        post_file = files[1]
                    pre_file = post_file.replace('post', 'pre')
                    img1 = cv2.imread(RAW_IMG_PATH + str(id_)+"/"+pre_file)
                    img2 = cv2.imread(RAW_IMG_PATH + str(id_)+"/" +post_file)
                    uid = uuid.uuid1()
                    path1 = CUT_IMG_PATH+str(uid)+'_pre.png'
                    path2 = CUT_IMG_PATH+str(uid)+'_post.png'
                    cv2.imwrite(path1, img1[y:y+height,x:x+width,:])
                    cv2.imwrite(path2, img2[y:y+height,x:x+width,:])

                    img_str_pre = getByte(path1)
                    img_str_post = getByte(path2)

                    thr = Thread(target=run_cls,args=(str(uid),))
                    thr.start()
                    REQUEST_COUNT += 1
                    return JsonResponse({"status":"BS.200","id":uid,'pre_cut':img_str_pre,'post_cut':img_str_post})
                else:
                    return JsonResponse({"status":"BS.104","message":"run %s processes at most at the same time."%MAX_REQUEST})
            else:
                return JsonResponse({"status":"BS.102","message":"x, y, width or height is not proper."})
        else:
            return JsonResponse({"status":"BS.102","message":"please check param."})
    else:
        return JsonResponse({"status":"BS.101","message":"POST required."})

def getByte(path):
    with open(path, 'rb') as f:
        img_byte = base64.b64encode(f.read())
    img_str = img_byte.decode('ascii')
    return img_str
 
# api 2
def get_cls_picture(request):
    if request.method == "GET":
        id_ = request.GET.get("id",default=None)
        if id_:
            file_path = CLS_IMG_PATH+id_+'.png'
            if os.path.exists(file_path):
                img_str = getByte(file_path)
                return JsonResponse({"status":"BS.200","message":"return cls img str",'img':img_str})
            else:
                return JsonResponse({"status":"BS.104","message":"please wait a few seconds."})
        
        else:
            return JsonResponse({"status":"BS.102","message":"please check param."})
    else:
        return JsonResponse({"status":"BS.101","message":"GET required"})

# loading model first

""" # loc model
original_model = dn.drn_d_105(pretrained=True)
original_model2 = dn.drn_d_105(pretrained=True)
G_loc = dn.DRNDepthLabel2_4(original_model,original_model2)       
# load model in GPUs
G_loc = torch.nn.DataParallel(G_loc,device_ids = GPU_LIST).cuda()
cudnn.benchmark = True
checkpoint = torch.load(MODEL_LOC_PATH)
G_loc.load_state_dict(checkpoint['state_dict'])
G_loc.eval() """

""" # cls model
original_model3 = senet.senet154(pretrained='imagenet')
G_cls = senet.Model(original_model3)
G_cls = torch.nn.DataParallel(G_cls,device_ids = GPU_LIST).cuda()
cudnn.benchmark = True
checkpoint = torch.load(MODEL_CLS_PATH)
G_cls.load_state_dict(checkpoint['state_dict'])
G_cls.eval()  """

original_model = dn.drn_d_105(pretrained=True)
original_model2 = dn.drn_d_105(pretrained=True)

G_e2e = dn.DRNDepthLabelE2E(original_model, original_model2).cuda()
G_e2e = torch.nn.DataParallel(G_e2e, device_ids = GPU_LIST).cuda()

cudnn.benchmark = True
checkpoint = torch.load(MODEL_E2E_PATH)

# create new state_dict
new_state_dict = OrderedDict()
for k, v in checkpoint['state_dict'].items():
    name = "module." + k # add 'module.' of DataParallel
    new_state_dict[name]=v

G_e2e.load_state_dict(new_state_dict)
G_e2e.eval()

# process_img  预处理图片的函数保留
def process_img_for_loc(id_,data_source='demo',picture_format ='png'):
    if data_source == 'demo':
        img_pre = _read_image_as_array(CUT_IMG_PATH+id_+'_pre.png',np.float32)
        img_post = _read_image_as_array(CUT_IMG_PATH+id_+'_post.png',np.float32)
    else:
        img_pre = _read_image_as_array(MEDIA_PATH+id_+'_pre.%s'%picture_format,np.float32)
        img_post = _read_image_as_array(MEDIA_PATH+id_+'_post.%s'%picture_format,np.float32)
        
    h, w, _ = img_pre.shape
    
    img_pre = img_pre.transpose(2, 0, 1)/255
    img_pre = torch.from_numpy(img_pre.copy())

    img_post = img_post.transpose(2, 0, 1)/255
    img_post = torch.from_numpy(img_post.copy())


    img_pre = _normalize(img_pre)
    img_post = _normalize(img_post)
    
    img_pre = img_pre.unsqueeze(0)
    img_post = img_post.unsqueeze(0)
    
    return img_pre,img_post,w,h    
    
def run_cls(id_,data_source='demo',picture_format='png'):
    global REQUEST_COUNT
    if data_source == 'demo':
        img_pre,img_post,w,h = process_img_for_loc(id_)
    else:
        img_pre,img_post,w,h = process_img_for_loc(id_,data_source='upload',picture_format=picture_format)
        
    with torch.no_grad():
        img_pre = img_pre.cuda()
        img_post = img_post.cuda()
        output = G_e2e(img_pre, img_post)  # the output 
        
        prediction_proba = output.squeeze().data.cpu().numpy()
        prediction = prediction_proba.argmax(axis=0)

        # print(output.shape)
    torch.cuda.empty_cache()
    
    if data_source == 'demo':
        create_prediction_image(prediction,CLS_IMG_PATH+id_+'.png',w,h)
    else:
        create_prediction_image(prediction,MEDIA_PATH+id_+'_result.jpg',w,h)

    REQUEST_COUNT -= 1

    '''
    # loc inference
    
    # with torch.no_grad():
    #     output = G_loc(img_pre,img_post)
    # # pred = output.data.max(1)[1].cpu().numpy()
    #     inference(img_pre,output.squeeze().data.cpu().numpy(),LOC_RESULT+id_+'.json')
    # # deal with loc inference output
    # if data_source == 'demo':
    #     process_img_poly(CUT_IMG_PATH+id_+'_pre.png',LOC_RESULT+id_+'.json',POLYGON_IMG_PATH,POLYGON_CSV_PATH+id_+'.csv')
    # else:
    #     process_img_poly(MEDIA_PATH+id_+'_pre.%s'%picture_format,LOC_RESULT+id_+'.json',POLYGON_IMG_PATH,POLYGON_CSV_PATH+id_+'.csv')
    # torch.cuda.empty_cache()
    
    # cls inference
    # cls_data_loader = loaddata.getTestingData2(POLYGON_IMG_PATH,POLYGON_CSV_PATH+id_+'.csv',batch_size = 1)
    damage_intensity_encoding = dict() 
    damage_intensity_encoding[4] = 'destroyed' 
    damage_intensity_encoding[3] = 'major-damage'
    damage_intensity_encoding[2] = 'minor-damage'
    damage_intensity_encoding[1] = 'no-damage'
    damage_intensity_encoding[0] = 'un-classified'
    # df = pd.read_csv(POLYGON_CSV_PATH+id_+'.csv')
    predictions_json = dict()

    with torch.no_grad():
        # run cls on every different parts of the image (based on loc results)
        for i, sample_batched in enumerate(cls_data_loader):
            images, target = sample_batched['image'], sample_batched['label']

            images = images.cuda()
            target = target.cuda()

            predictions = G_cls(images)
            predictions = predictions.squeeze().data.cpu().numpy()

          
            weights = np.array([1,4,15,50])
#             print('-------------')
#             print((predictions-predictions.min()+1)/(predictions.max()-predictions.min())*weights)
            predicted_indices = np.argmax((predictions-predictions.min()+1)/(predictions.max()-predictions.min())*weights)
#             predicted_indices = np.argmax(predictions)
            image_name = df.iloc[i, 1]
            predictions_json[str(image_name)] = damage_intensity_encoding[predicted_indices]
            torch.cuda.empty_cache()
            
    with open(POLYGON_CLS_PATH+id_+'.json', 'w') as outfile:
        json.dump(predictions_json, outfile)
        
    # deal with cls inference output
    combine_output(LOC_RESULT+id_+'.json',POLYGON_CLS_PATH+id_+'.json',COMBINE_JSON_PATH+id_+'.json')
    if data_source == 'demo':
        create_inference_image(COMBINE_JSON_PATH+id_+'.json',CLS_IMG_PATH+id_+'.png',w,h)
    else:
        create_inference_image(COMBINE_JSON_PATH+id_+'.json',MEDIA_PATH+id_+'_result.jpg',w,h)'''
    

# api 5
def cls_for_upload(request):
    global REQUEST_COUNT
    if request.method == "POST":
        req = json.loads(request.body)
        key_flag = req.get("preName") and req.get("postName")
        if key_flag:
            pre_file_name,post_file_name = req.get("preName"), req.get("postName")
            if os.path.exists(MEDIA_PATH+pre_file_name) and  os.path.exists(MEDIA_PATH+post_file_name) and pre_file_name.split('.')[-1] in ['png','jpg'] and post_file_name.split('.')[-1] in ['png','jpg']:
                img_pre = _read_image_as_array(MEDIA_PATH+pre_file_name,np.float32)
                img_post = _read_image_as_array(MEDIA_PATH+post_file_name,np.float32)

                if img_pre.shape[0] == img_post.shape[0] and img_pre.shape[1] == img_post.shape[1] and img_pre.shape[2] == img_post.shape[2] and pre_file_name.split('.')[-1] == post_file_name.split('.')[-1]:

                    if REQUEST_COUNT < MAX_REQUEST:
                        pictiure_format = pre_file_name.split('.')[-1]
                        id_ = pre_file_name.split('_pre')[0]
                        torch.cuda.empty_cache()
                        os.system('rm -rf {0}{1}'.format(MEDIA_PATH,pre_file_name.replace('pre','result')))
                        thr = Thread(target=run_cls,args=(str(id_),'upload',pictiure_format,))
                        thr.start()
                        REQUEST_COUNT += 1
                        return success({'fileName':pre_file_name.replace('pre','result')})
                    else:
                        return error(104,"run %s processes at most at the same time."%MAX_REQUEST)
                else:
                    return error(102,"pre_img and post_img's width or height or channel or picture format are not the same.")
            else:
                return error(103,"files not found or picture format is not png or jpg.")
        else:
            return error(102,"please check param.")
    else:
        return error(101,"POST required.")



# api 6
def check_download_result(request):
    if request.method == "GET":
        fileName = request.GET.get("fileName",default=None)
        if fileName:
            file_path = MEDIA_PATH+fileName
            if os.path.exists(file_path):
                return success({'isFinish':True})
            else:
                return success({'isFinish':False})
        else:
            return error102()
    else:
        return error(101,'Request type error！')
         

