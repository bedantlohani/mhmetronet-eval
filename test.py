from matplotlib import pyplot as plt

import matplotlib
import os
import random
import torch
from torch.autograd import Variable
import torchvision.transforms as standard_transforms
import misc.transforms as own_transforms
import pandas as pd
import h5py
from models.CC import CrowdCounter
from config import cfg
from misc.utils import *
import scipy.io as sio
from PIL import Image, ImageOps
import cv2

torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True

exp_name = '/home/bedant/mh-metronet/results'
if not os.path.exists(exp_name):
    os.mkdir(exp_name)

if not os.path.exists(exp_name + '/pred'):
    os.mkdir(exp_name + '/pred')

if not os.path.exists(exp_name + '/gt'):
    os.mkdir(exp_name + '/gt')

mean_std = ([0.452016860247, 0.447249650955, 0.431981861591], [0.23242045939, 0.224925786257, 0.221840232611])
img_transform = standard_transforms.Compose([
    standard_transforms.ToTensor(),
    standard_transforms.Normalize(*mean_std)
])
restore = standard_transforms.Compose([
    own_transforms.DeNormalize(*mean_std),
    standard_transforms.ToPILImage()
])
pil_to_tensor = standard_transforms.ToTensor()

dataRoot = '/home/bedant/mh-metronet/test'

model_path = '/home/bedant/mh-metronet/mcnn_shtechB_110.h5'


def main():
    file_list = [filename for root, dirs, filename in os.walk(dataRoot + '/img/')]

    test(file_list[0], model_path)


def test(file_list, model_path):
    net = CrowdCounter(cfg.GPU_ID, cfg.NET)
    net.load_state_dict(torch.load(model_path))
    net.cuda()
    net.eval()

    f1 = plt.figure(1)

    gts = []
    preds = []
    mae = 0
    mse = 0

    for filename in file_list:
        print(filename)
        imgname = dataRoot + '/img/' + filename
        filename_no_ext = filename.split('.')[0]
        denname = dataRoot + '/den/' + filename_no_ext + '.h5'
        den = h5py.File(denname,'r')
        den = np.asarray(den['density'])
        #den = pd.read_csv(denname, sep=',',header=None).values
        #den = h5py.File(denname, 'r')
        #den = np.asarray(den['density'])
        #den = den['density']
        den = den.astype(np.float32)

        img = Image.open(imgname)

        if img.mode == 'L':
            img = img.convert('RGB')

        img = img_transform(img)

        gt = np.sum(den)
        with torch.no_grad():
            img = Variable(img[None, :, :, :]).cuda()
            pred_map = net.test_forward(img)

        # sio.savemat(exp_name+'/pred/'+filename_no_ext+'.mat',{'data':pred_map.squeeze().cpu().numpy()/100.})
        # sio.savemat(exp_name+'/gt/'+filename_no_ext+'.mat',{'data':den})

        pred_map = pred_map.cpu().data.numpy()[0, 0, :, :]


        pred = np.sum(pred_map) / 100.0
        pred_map = pred_map / np.max(pred_map + 1e-20)
        mae += abs(gt - pred)
        mse += (gt - pred)**2

        den = den / np.max(den + 1e-20)

        den_frame = plt.gca()
        plt.imshow(den, 'jet')
        den_frame.axes.get_yaxis().set_visible(False)
        den_frame.axes.get_xaxis().set_visible(False)
        den_frame.spines['top'].set_visible(False)
        den_frame.spines['bottom'].set_visible(False)
        den_frame.spines['left'].set_visible(False)
        den_frame.spines['right'].set_visible(False)
        plt.savefig(exp_name+'/'+filename_no_ext+'_gt_'+str(int(gt))+'.png',\
            bbox_inches='tight',pad_inches=0,dpi=150)

        plt.close()

        #sio.savemat(exp_name+'/'+filename_no_ext+'_gt_'+str(int(gt))+'.mat',{'data':den})

        pred_frame = plt.gca()
        plt.imshow(pred_map, 'jet')
        pred_frame.axes.get_yaxis().set_visible(False)
        pred_frame.axes.get_xaxis().set_visible(False)
        pred_frame.spines['top'].set_visible(False)
        pred_frame.spines['bottom'].set_visible(False)
        pred_frame.spines['left'].set_visible(False)
        pred_frame.spines['right'].set_visible(False)
        plt.savefig(exp_name+'/'+filename_no_ext+'_pred_'+str(float(pred))+'.png',\
            bbox_inches='tight',pad_inches=0,dpi=150)

        plt.close()

        #sio.savemat(exp_name+'/'+filename_no_ext+'_pred_'+str(float(pred))+'.mat',{'data':pred_map})
        #den = cv2.resize(den,(int(den.shape[1]/8),int(den.shape[0]/8)),interpolation = cv2.INTER_CUBIC)*64
        diff = den-pred_map

        diff_frame = plt.gca()
        plt.imshow(diff, 'jet')
        plt.colorbar()
        diff_frame.axes.get_yaxis().set_visible(False)
        diff_frame.axes.get_xaxis().set_visible(False)
        diff_frame.spines['top'].set_visible(False)
        diff_frame.spines['bottom'].set_visible(False)
        diff_frame.spines['left'].set_visible(False)
        diff_frame.spines['right'].set_visible(False)
        plt.savefig(exp_name+'/'+filename_no_ext+'_diff.png',\
            bbox_inches='tight',pad_inches=0,dpi=150)

        plt.close()

        #sio.savemat(exp_name+'/'+filename_no_ext+'_diff.mat',{'data':diff})

    print(mae / len(file_list))
    print(mse / len(file_list))

if __name__ == '__main__':
    main()


def validate_V1(self):  # validate_V1 for SHHA, SHHB, UCF-QNRF, UCF50

    self.net.eval()

    losses = AverageMeter()
    maes = AverageMeter()
    mses = AverageMeter()

    for vi, data in enumerate(self.val_loader, 0):
        img, gt_map = data

        with torch.no_grad():
            img = Variable(img).cuda()
            gt_map = Variable(gt_map).cuda()

            pred_map = self.net.forward(img, gt_map)

            pred_map = pred_map.data.cpu().numpy()
            gt_map = gt_map.data.cpu().numpy()

            for i_img in range(pred_map.shape[0]):
                pred_cnt = np.sum(pred_map[i_img]) / self.cfg_data.LOG_PARA
                gt_count = np.sum(gt_map[i_img]) / self.cfg_data.LOG_PARA

                losses.update(self.net.loss.item())
                maes.update(abs(gt_count - pred_cnt))
                mses.update((gt_count - pred_cnt) * (gt_count - pred_cnt))
            if vi == 0:
                vis_results(self.exp_name, self.epoch, self.writer, self.restore_transform, img, pred_map, gt_map)

    mae = maes.avg
    mse = np.sqrt(mses.avg)
    loss = losses.avg
