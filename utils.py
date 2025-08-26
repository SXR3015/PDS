import csv
import os
import torch
import torchvision
import torch.nn as nn
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import numpy as np
class AverageMeter(object):
    '''computes and stores the average and current value'''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count =0

    def update(self,val,n=1):
        self.val = val
        self.sum += val*n
        self.count +=n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t') #\t为一个tab

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value

def vgg():
    vgg = torchvision.models.vgg16(pretrained=True)
    vgg.classifier = nn.Sequential(*vgg.classifier[:-2])
    _vgg = vgg.cuda()
    return _vgg

def calculate_accuracy(outputs, labels):
    batch_size = labels.size(0)
    _, pred = outputs.topk(k=1, dim=1, largest=True)
    _, labels_ = labels.topk(k=1, dim=1, largest=True)
    # pred = pred.t()
    correct = pred.squeeze().eq(labels_.squeeze().cuda())
    n_correct_elems = correct.float().sum().data

    return n_correct_elems / batch_size

def calculate_recall(outputs, labels, opt):
    _, pred = outputs.topk(k=1, dim=1, largest=True)
    _, labels_ = labels.topk(k=1, dim=1, largest=True)
    # pred = pred.t()  # 转置成行
    labels = labels_.squeeze().cuda()
    pred = pred.squeeze()
    if opt.n_classes == 3:
        TP = (((pred.data == 1) & (labels.data == 1)) | (
                    (pred.data == 2) & (labels.data == 2))).cpu().float().sum().data
        # TN = ((pred.data == 0) & (labels.data == 0)).cpu().float().sum().data
        FP = (((pred.data == 1) & (labels.data == 0)) | (
                    (pred.data == 2) & (labels.data == 0))|((pred.data == 2) & (labels.data == 1))).cpu().float().sum().data
        TN = ((pred.data == 0) & (labels.data == 0)).cpu().float().sum().data
        FN = (((pred.data == 0) & (labels.data == 1)) | (
                    (pred.data == 0) & (labels.data == 2))| ((pred.data == 1) & (labels.data == 2)) ).cpu().float().sum().data
        if TP + FN == 0:
            r = 0
        else:
            r = TP / (TP + FN)  # recall
            # r = torch.tensor(r).float()
        if TP + FP == 0:
            p = torch.tensor(0).float()
        else:
            p = TP / (TP + FP)
            # p = torch.tensor(p).float()
        if r + p == 0:
            f1 = torch.tensor(0).float()
        else:
            f1 = 2 * r * p / (r + p)
            # f1 = torch.tensor(f1).float()
        if TP + FN == 0:
            sen = torch.tensor(0).float()
        else:
            sen = TP / (TP + FN)
            # sen = torch.tensor(sen).float()
        if TN + FP == 0:
            sp = torch.tensor(0).float()
        else:
            sp = TN / (TN + FP)
    
        return r, p, f1, sen, sp
    else:
        TP = ((pred.data == 1) & (labels.data == 1)).cpu().float().sum().data
        FP = ((pred.data == 1) & (labels.data == 0)).cpu().float().sum().data
        TN = ((pred.data == 0) & (labels.data == 0)).cpu().float().sum().data
        # TN = ((pred.data == 0) & (labels.data == 0)).cpu().float().sum().data
        FN = (((pred.data == 0) & (labels.data == 1))).cpu().float().sum().data
        if TP + FN == 0:
            r = 0
        else:
            r = TP / (TP + FN)  # recall
            # r = torch.tensor(r).float()
        if TP + FP == 0:
            p = torch.tensor(0).float()
        else:
            p = TP / (TP + FP)
            # p = torch.tensor(p).float()
        if r + p == 0:
            f1 = torch.tensor(0).float()
        else:
            f1 = 2 * r * p / (r + p)
            # f1 = torch.tensor(f1).float()
        if TP + FN == 0:
            sen = torch.tensor(0).float()
        else:
            sen = TP / (TP + FN)
            # sen = torch.tensor(sen).float()
        if TN + FP == 0:
            sp = torch.tensor(0).float()
        else:
            sp = TN / (TN + FP)
        return r, p, f1, sen, sp
def OsJoin(*args):
    p = os.path.join(*args)
    p = p.replace('\\', '/')
    return p


def generate_target_label(labels,opt):
    if opt.n_classes == 2:
        labels[labels == 1] = 2
        labels[labels == 0] = 1
        labels[labels == 2] = 0
    elif opt.n_classes == 3:
        for i in range(labels.shape[0]):
            _, label_i = torch.transpose(labels[i].unsqueeze(1), 1, 0).topk(k=1, dim=1, largest=True)
            if label_i == 0:
                labels[i] = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
            elif label_i == 1:
                labels[i] = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
            elif label_i == 2:
                labels[i] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)

    return labels
def save_mat_png(array,opt, epoch, mode, mode_array='degeneration',category='HC'):
    result_path = OsJoin(opt.root_path, opt.result_path)
    save_path = OsJoin(result_path, opt.data_type, opt.mode_net, opt.category,'gen images')
    array_dict = {"%s_%s"%(category,mode_array): array.detach().cpu().numpy(), "label": "%s_%s"%(category,mode_array)}
    sio.savemat(os.path.join(save_path, '%s_%s_%s_epoch%s.mat' % (mode,category,mode_array,epoch)), array_dict)
    save_name = OsJoin(save_path, '%s_epoch%d_%s_%s.png' % (mode, epoch, category,mode_array))
    plt.imshow(array.detach().cpu().numpy())
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(save_name)
    plt.close()
def generate_neurodegeneration(gen,gen_target,labels,opt,epoch, mode='train'):
    _, labels_ = labels.topk(k=1, dim=1, largest=True)
    subtract_label = gen_target - gen
    result_path = OsJoin(opt.root_path, opt.result_path)
    save_path = OsJoin(result_path, opt.data_type, opt.mode_net, opt.category,'gen images')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if opt.n_classes == 3:
        index_HC =torch.where(labels_==0)[0]
        index_SCD = torch.where(labels_ == 1)[0]
        index_MCI = torch.where(labels_ == 2)[0]
        HC_degeneration = torch.sum(subtract_label[index_HC,...],dim=0)/len(index_HC)
        SCD_degeneration = torch.sum(subtract_label[index_SCD,...], dim=0)/len(index_SCD)
        MCI_degeneration = torch.sum(subtract_label[index_MCI,...], dim=0)/len(index_MCI)
        HC_gen = torch.sum(gen[index_HC,...],dim=0)/len(index_HC)
        SCD_gen = torch.sum(gen[index_SCD,...], dim=0)/len(index_SCD)
        MCI_gen = torch.sum(gen[index_MCI,...], dim=0)/len(index_MCI)
        HC_target = torch.sum(gen_target[index_HC,...],dim=0)/len(index_HC)
        SCD_target= torch.sum(gen_target[index_SCD,...], dim=0)/len(index_SCD)
        MCI_target= torch.sum(gen_target[index_MCI,...], dim=0)/len(index_MCI)
        save_mat_png(HC_degeneration, opt, epoch, mode, mode_array='degeneration', category='HC')
        save_mat_png(MCI_degeneration, opt, epoch, mode, mode_array='degeneration', category='MCI')
        save_mat_png(SCD_degeneration, opt, epoch, mode, mode_array='degeneration', category='SCD')
        save_mat_png(HC_gen, opt, epoch, mode, mode_array='gen', category='HC')
        save_mat_png(MCI_gen, opt, epoch, mode, mode_array='gen', category='MCI')
        save_mat_png(SCD_gen, opt, epoch, mode, mode_array='gen', category='SCD')
        save_mat_png(HC_target, opt, epoch, mode, mode_array='target', category='HC')
        save_mat_png(MCI_target, opt, epoch, mode, mode_array='target', category='MCI')
        save_mat_png(SCD_target, opt, epoch, mode, mode_array='target', category='SCD')
    elif opt.n_classes == 2 and opt.category =='HC_SCD':
        index_HC =torch.where(labels_==0)[0]
        index_SCD = torch.where(labels_ == 1)[0]
        HC_degeneration = torch.sum(subtract_label[index_HC,...],dim=0)/len(index_HC)
        SCD_degeneration = torch.sum(subtract_label[index_SCD,...], dim=0)/len(index_SCD)
        HC_gen = torch.sum(gen[index_HC,...],dim=0)/len(index_HC)
        SCD_gen = torch.sum(gen[index_SCD,...], dim=0)/len(index_SCD)
        HC_target = torch.sum(gen_target[index_HC,...],dim=0)/len(index_HC)
        SCD_target= torch.sum(gen_target[index_SCD,...], dim=0)/len(index_SCD)
        save_mat_png(HC_degeneration, opt, epoch, mode, mode_array='degeneration', category='HC')
        save_mat_png(SCD_degeneration, opt, epoch, mode, mode_array='degeneration', category='SCD')
        save_mat_png(HC_gen, opt, epoch, mode, mode_array='gen', category='HC')
        save_mat_png(SCD_gen, opt, epoch, mode, mode_array='gen', category='SCD')
        save_mat_png(HC_target, opt, epoch, mode, mode_array='target', category='HC')
        save_mat_png(SCD_target, opt, epoch, mode, mode_array='target', category='SCD')
    elif opt.n_classes == 2 and opt.category == 'HC_MCI':
        index_HC = torch.where(labels_ == 0)[0]
        index_MCI = torch.where(labels_ == 1)[0]
        HC_degeneration = torch.sum(subtract_label[index_HC, ...], dim=0)/len(index_HC)
        MCI_degeneration = torch.sum(subtract_label[index_MCI, ...], dim=0)/len(index_MCI)
        HC_gen = torch.sum(gen[index_HC,...],dim=0)/len(index_HC)
        MCI_gen = torch.sum(gen[index_MCI,...], dim=0)/len(index_MCI)
        HC_target = torch.sum(gen_target[index_HC,...],dim=0)/len(index_HC)
        MCI_target= torch.sum(gen_target[index_MCI,...], dim=0)/len(index_MCI)
        save_mat_png(HC_degeneration, opt, epoch, mode, mode_array='degeneration', category='HC')
        save_mat_png(MCI_degeneration, opt, epoch, mode, mode_array='degeneration', category='MCI')
        save_mat_png(HC_gen, opt, epoch, mode, mode_array='gen', category='HC')
        save_mat_png(MCI_gen, opt, epoch, mode, mode_array='gen', category='MCI')
        save_mat_png(HC_target, opt, epoch, mode, mode_array='target', category='HC')
        save_mat_png(MCI_target, opt, epoch, mode, mode_array='target', category='MCI')

    elif opt.n_classes == 2 and opt.category == 'MCI_SCD':
        index_SCD = torch.where(labels_ == 0)[0]
        index_MCI = torch.where(labels_ == 1)[0]
        SCD_degeneration = torch.sum(subtract_label[index_SCD, ...], dim=0)/len(index_SCD)
        MCI_degeneration = torch.sum(subtract_label[index_MCI, ...], dim=0)/len(index_MCI)
        SCD_gen = torch.sum(gen[index_SCD,...], dim=0)/len(index_SCD)
        MCI_gen = torch.sum(gen[index_MCI,...], dim=0)/len(index_MCI)
        SCD_target= torch.sum(gen_target[index_SCD,...], dim=0)/len(index_SCD)
        MCI_target= torch.sum(gen_target[index_MCI,...], dim=0)/len(index_MCI)
        save_mat_png(MCI_degeneration, opt, epoch, mode, mode_array='degeneration', category='MCI')
        save_mat_png(SCD_degeneration, opt, epoch, mode, mode_array='degeneration', category='SCD')
        save_mat_png(MCI_gen, opt, epoch, mode, mode_array='gen', category='MCI')
        save_mat_png(SCD_gen, opt, epoch, mode, mode_array='gen', category='SCD')
        save_mat_png(MCI_target, opt, epoch, mode, mode_array='target', category='MCI')
        save_mat_png(SCD_target, opt, epoch, mode, mode_array='target', category='SCD')
