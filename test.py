import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

def test(model, device, data):
    model.eval()
    with torch.no_grad():
        blur = nn.Conv2d(5, 5, 9, padding=4, bias=False, groups=5).to(device)
        nn.init.constant_(blur.weight, 1 / 81)
        img = data.to(device)  # data = torch.Size([1, 3, 600, 800]) 1 image
        predictmaps, label = model(img)
        predictmaps = F.softmax(predictmaps, dim=1)
        predictmap = predictmaps[0,...]
        predictmap = predictmap * 255

        predictmap = blur(predictmap.unsqueeze(0)).squeeze()
        predictmap = np.array(predictmap.cpu())  # predictmap.shape = <class 'tuple'>: (5, 600, 800)
        predict = predictmap[1]
        print(time.time())
        # predict = np.array(predict).transpose(1,0)
        print(predict.shape,type(predict))
        cv2.imwrite('./pred_6ms_leff/'+str(time.time())+'.jpg',predict)

        return predict
