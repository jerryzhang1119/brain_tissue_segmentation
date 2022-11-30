from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DataParallel
from self_attention_cv.transunet import TransUnet
from args import ARGS
import time
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from utils.get_dataset import get_dataset
import os
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"


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
        return loss / self.n_classes

class TrainValProcess():
    def __init__(self):
        self.net = TransUnet(in_channels=3, img_dim=128, vit_blocks=1, vit_dim_linear_mhsa_block=512, classes=9)
        if (ARGS['gpu']):
            self.net = DataParallel(module=self.net.cuda())
        self.train_dataset = get_dataset(dataset_name=ARGS['dataset'], part='train')
        self.val_dataset = get_dataset(dataset_name=ARGS['dataset'], part='val')

        self.optimizer = Adam(self.net.parameters(), lr=ARGS['lr'])
        # Use / to get an approximate result, // to get an accurate result
        total_iters = len(self.train_dataset) // ARGS['batch_size'] * ARGS['num_epochs']
        self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda=lambda iter: (1 - iter / total_iters) ** ARGS['scheduler_power'])
        self.dice_loss = DiceLoss(9)

    def train(self, epoch):

        start = time.time()
        self.net.train()
        train_dataloader = DataLoader(self.train_dataset, batch_size=ARGS['batch_size'], shuffle=False)
        epoch_loss = 0.
        iou_sum = 0.
        for batch_index, items in enumerate(train_dataloader):
            images, labels = items['image'], items['label']
            images = images.float()
            labels = labels.long()


            if ARGS['gpu']:
                labels = labels.cuda()
                images = images.cuda()


            self.optimizer.zero_grad()
            outputs = self.net(images)
           

            loss = self.dice_loss(outputs, labels, softmax=True)
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            n_iter = (epoch - 1) * len(train_dataloader) + batch_index + 1
            epoch_loss += loss.item()

        epoch_loss /= len(train_dataloader)
        finish = time.time()

        print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))


    def train_val(self):
        print('Begin training and validating:')
        for epoch in range(ARGS['num_epochs']):
            self.train(epoch)
            self.net.state_dict()
            print(f'Finish training and validating epoch #{epoch+1}')
            if (epoch + 1) % ARGS['epoch_save'] == 0:
                os.makedirs(ARGS['weight_save_folder'], exist_ok=True)
                torch.save(self.net.state_dict(), os.path.join(ARGS['weight_save_folder'], f'epoch_{epoch+1}.pth'))
                print(f'Model saved for epoch #{epoch+1}.')
        print('Finish training and validating.')

if __name__ == "__main__":
    tv = TrainValProcess()
    tv.train_val()
