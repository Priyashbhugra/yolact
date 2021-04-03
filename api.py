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
from inference import evaluate
from data import config
import torch.backends.cudnn as cudnn

# Declaring our FastAPI instance
app = FastAPI()

args = Namespace(display_scores=True,display_best_bboxes_only=False,display_bboxes=True,
                 display_text=True,display_fps=False,display_best_masks_only=True,
                 display_masks=True,crop=True, display_lincomb = False, cuda =True,
                 mask_proto_debug=False ,fast_nms= True, cross_class_nms = True,
                 trained_model = "weights/yolact_im700_54_800000.pth",
                 score_threshold = 0.15, top_k = 100, display_only_car = False,
                 image = "cars_with_pedestrian.jpg")

with torch.no_grad():

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.cuda:
        cudnn.fastest = True
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    net = Yolact(args)
    net.load_weights(args.trained_model) 
    net.eval()
    print("done")

    evaluate(net, "None", args)

    print("Evaluation completed")




