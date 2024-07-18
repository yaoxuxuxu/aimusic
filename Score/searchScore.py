import torch
import cv2
import os
import numpy as np
from utils.torch_utils import select_device
from utils.augmentations import letterbox
from utils.general import non_max_suppression 

fn_outimg = "./out.jpg"

fn_img = "./score.jpg"
input_size = 640
fn_model = "./models/region.torchscript.pt"


img1=cv2.imread(fn_img)
img1 = letterbox(img1, input_size, auto=False)[0]
img1 = img1.transpose((2, 0, 1))[::-1]
#img1 = img1 / 255.0 
img1 = np.array([img1],dtype=np.float32)
img1 = torch.from_numpy(img1)
region_model = torch.jit.load(fn_model)
region_model.eval()


pred = region_model(img1)
pred = pred[0].cpu()
print(pred[0].shape)
exit(0)



fn_img = "./region.jpg"
input_size = 320
fn_model = "./models/score_note_320.torchscript.pt"
    
img_orig = cv2.imread(fn_img)

width_orig = img_orig.shape[1]
height_orig = img_orig.shape[0]

img = letterbox(img_orig, input_size, auto=False)
dw = img[2][0]
dh = img[2][1]


img1 = img[0]
img1 = img1.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
img1 = np.ascontiguousarray(img1)

img1 = torch.from_numpy(img1)
img1 = img1.float()
img1 = img1 / 255.0
if len(img1.shape) == 3:
    img1 = img1[None]


width = img1.shape[3]
height = img1.shape[2]

region_model = torch.jit.load(fn_model)
region_model.eval()

conf_thres = 0.7
iou_thres = 0.45
max_det = 100

pred = region_model(img1)[0]
pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=max_det)
pred = pred[0].cpu()
print(pred)

for item in pred:
    item[0] = (item[0] - dw)/(width - 2*dw)
    item[2] = (item[2] - dw)/(width - 2*dw)
    item[1] = (item[1] - dh)/(height - 2*dh)
    item[3] = (item[3] - dh)/(height - 2*dh)

    x1 = int(item[0] * width_orig)
    y1 = int(item[1] * height_orig)
    x2 = int(item[2] * width_orig)
    y2 = int(item[3] * height_orig)
    conf = item[4]
    classno = int(item[5])
    color = (0, 255, 0)
    cv2.rectangle(img_orig, (x1, y1), (x2, y2), color)
    cv2.putText(img_orig, str(classno), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

cv2.imwrite(fn_outimg, img_orig)
cv2.imshow("out", img_orig)
cv2.waitKey(0)