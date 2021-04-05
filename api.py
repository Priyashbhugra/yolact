from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import eval
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
import torch
from yolact import Yolact
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, Request, Form
from argparse import Namespace
from inf_eval import evaluate
from data import config
import torch.backends.cudnn as cudnn
import argparse
from data import COCODetection, get_label_map, MEANS, COLORS
from yolact import Yolact
from utils.augmentations import BaseTransform, FastBaseTransform, Resize
from utils.functions import MovingAverage, ProgressBar
from layers.box_utils import jaccard, center_size, mask_iou
from utils import timer
from utils.functions import SavePath
from layers.output_utils import postprocess, undo_image_transformation
import pycocotools
from data import cfg, set_cfg, set_dataset
import os


# Declaring our FastAPI instance
app = FastAPI()

args = Namespace(display_scores=True,display_best_bboxes_only=True,display_bboxes=True,
                 display_text=True,display_fps=False,display_best_masks_only=True,
                 display_masks=True,crop=True, display_lincomb = False, cuda =True,
                 mask_proto_debug=False ,fast_nms= True, cross_class_nms = True,
                 trained_model = "weights/yolact_im700_54_800000.pth",
                 score_threshold = 0.15, top_k = 1, display_only_car = True,
                 image = "crowd2.jpg")

       
with torch.no_grad():
        if not os.path.exists('results'):
            os.makedirs('results')

        if args.cuda:
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')
      

        print('Loading model...')
        net = Yolact(args)
        net.load_weights(args.trained_model)
        net.eval()
        print(' Done.')


        if args.cuda:
            net = net.cuda()

        evaluate(net, "None", args)
        print("Inference completed")
