from self_attention_cv.transunet import TransUnet
from torch.nn.parallel import DataParallel
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from args import ARGS
from utils.get_dataset import get_dataset
import time
from torch.utils.data import DataLoader
from utils.crop_prediction import get_test_patches, recompone_overlap

from PIL import Image
import os
import scipy.misc
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import numpy as np


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i+1  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return class_wise_dice

class CalculateMetricProcess:
    def __init__(self):
        self.net = TransUnet(in_channels=3, img_dim=128, vit_blocks=1, vit_dim_linear_mhsa_block=512, classes=9)
        if (ARGS['gpu']):
            self.net = DataParallel(module=self.net.cuda())
        
        self.net.load_state_dict(torch.load(ARGS['weight']))

        self.metric_dataset = get_dataset(dataset_name=ARGS['dataset'], part='metric')
        self.dice_loss = DiceLoss(9)
    def predict(self):

        start = time.time()
        self.net.eval()
        metric_dataloader = DataLoader(self.metric_dataset, batch_size=1) # only support batch size = 1
        os.makedirs(ARGS['prediction_save_folder'], exist_ok=True)
        cut = 0
        sum_loss = 0
        for items in metric_dataloader:
            images, labels, mask = items['image'], items['label'], items['mask']
            filename = items['filename']
            images = images.float()

            image_patches, big_h, big_w = get_test_patches(images, ARGS['crop_size'], ARGS['stride_size'])
            test_patch_dataloader = DataLoader(image_patches, batch_size=ARGS['batch_size'], shuffle=False, drop_last=False)
            test_results = []

            for patches in test_patch_dataloader:                
                if ARGS['gpu']:
                    patches = patches.cuda()
                with torch.no_grad():
                    result_patches = self.net(patches)
                
                test_results.append(result_patches.cpu())           
            
            test_results = torch.cat(test_results, dim=0)
            # merge
            test_results = recompone_overlap(test_results, ARGS['crop_size'], ARGS['stride_size'], big_h, big_w)
            test_results = test_results[:, :, :images.size(2), :images.size(3)]
            cut += 1
            print(cut)
            loss = self.dice_loss(test_results, labels, softmax=True)
            print(loss)
            sum_loss += np.array(loss)
            res = test_results[0].cpu().numpy()
            np.save(os.path.join(ARGS['prediction_save_folder'], filename[0]), test_results)
        print(sum_loss)
        print(sum_loss/cut)
        finish = time.time()

        print('Calculating metric time consumed: {:.2f}s'.format(finish - start))

if __name__ == "__main__":
    cmp = CalculateMetricProcess()
    cmp.predict()
