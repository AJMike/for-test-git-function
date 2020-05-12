import argparse
import os
from collections import OrderedDict
import torch
from PIL import Image
import matplotlib.pyplot as plt

from model import *
from test import *
from ranging import *

def scnn(img_data):
    parser = argparse.ArgumentParser(description='PyTorch SCNN Model')
    parser.add_argument('--gpu', metavar='N', type=int, nargs='+', default=[0],
                        help='GPU ids')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--weights', metavar='DIR', default='./neocam_epoch_600.pth',
                        help='use finetuned model')
    args = parser.parse_args()

    # random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # cuda and seed
    use_cuda = args.gpu[0] >= 0 and torch.cuda.is_available()
    device = torch.device('cuda:{0}'.format(args.gpu[0]) if use_cuda else 'cpu')
    torch.manual_seed(args.seed)
    if use_cuda:
        print('Use Device: GPU', args.gpu)
    else:
        print('Use Device: CPU')

    # model
    model = SCNN().to(device)
    if len(args.gpu) > 1:
        model = torch.nn.DataParallel(model, device_ids=args.gpu)

    if args.weights is not None:
        assert os.path.isfile(args.weights)
        print('Start loading weights')
        model_dict = model.state_dict()
        ############## debug: modify module ###################
        '''
        for k, v in model_dict.items():
            print('original model', k, v.shape)
        '''
        weights = torch.load(args.weights)
        # for k, v in weights.items():
        #     print(k,v)
        weights = weights["model_state_dict"]
        weights1 = OrderedDict()
        for k in weights.items():  # modify the key-name in weights
            mk = k[0]
            mv = weights[mk]
            # print(mk, type(mk), type(mv))
            mk1 = mk.replace('module.', '')  # delete module. prefix in key-name
            # print(mk1)
            # if mk in model.state_dict():
            weights1[mk1] = mv

        # diff = ['fc9.weight',]
        # weights = {k: v for k, v in weights.items() if k not in diff}
        # weights = {k: v for k, v in weights.items() if k in model.state_dict()}
        '''
        for k, v in weights1.items():
            print('load weights from weights1', k, v.shape)
        '''
        model_dict.update(weights1)
        # for k,v in model_dict.items():
        #     print('After load', k, v.shape)
        model.load_state_dict(model_dict)
        print('Loading weights done.')

    mean = [0.3598, 0.3653, 0.3662]
    std = [0.2573, 0.2663, 0.2756]

    # path = './image/cone.png'
    # image = Image.open(path) # T (1024,768)
    image = Image.fromarray(cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB))
    image = image.resize((480,270), Image.BICUBIC)  # T (800,600)
    image = np.array(image).transpose((2, 0, 1)) / 255  # T 1440000
    data = torch.from_numpy(image).to(torch.float)  # T torch.Size([3, 600, 800])

    for t, m, s in zip(data, mean, std):  # T torch.Size([600, 800])
        t.sub_(m).div_(s)
    image_data = torch.unsqueeze(data, 0)

    test(model, device, image_data)  # type <class 'numpy.ndarray'>
    path = 'next.png'
    info = ranging(path)

    return info
