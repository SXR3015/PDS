import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from opts import parse_opts
from model import generate_model
from dataset import TrainSet, ValidSet
from utils import Logger, OsJoin
from train import train_epoch
from train_refine import  train_epoch_refine
from validation import val_epoch
# from test import test_epoch
import random
import numpy as np
from dataset import TestSet
from diffusers.optimization import get_scheduler
from models.scheduler import DDIMScheduler
from tensorboardX import SummaryWriter
from models import ema
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
#os.environ["CUDA_VISIBLE_DEVICES"]= '0,1,3,4,5,6,7'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:6144"
torch.distributed.init_process_group(backend="nccl")
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
torch.autograd.set_detect_anomaly(True)
def run(fold_id, opt):
    if opt.root_path != '':
        result_path = OsJoin(opt.root_path, opt.result_path)
        event_path = OsJoin(opt.root_path, opt.event_path)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
    opt.arch ='{}-{}'.format(opt.model_name,opt.model_depth)
    #print(opt)

    print('-'*50, 'RUN FOLD %s'%str(fold_id), '-'*50)

    model, parameters = generate_model(opt)

    model.to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[local_rank],
                                                      output_device=local_rank,
                                                      find_unused_parameters=False
                                                      )
    # print(model)
    # if opt.pretrain == True:
    #     # try:
    #         checkpoint = torch.load(opt.resume_path, weights_only=True)
    #         opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
    #         assert opt.arch == checkpoint['arch']
    #         model.load_state_dict(checkpoint['state_dict'])
    #         print('Load Model successfully')

    criterion = nn.CrossEntropyLoss()

    if not opt.no_cuda:
        criterion = criterion.cuda()

    if not opt.no_train:
        training_data = TrainSet(fold_id=fold_id)
        train_loader = DataLoader(training_data, batch_size=opt.batch_size,
                                  num_workers=8, pin_memory=True, sampler=DistributedSampler(training_data))
        if opt.pretrain_path:
            log_path = OsJoin(result_path, opt.data_type, opt.model_name + '_' + str(opt.model_depth) + '_pretrain',
                              'logs_fold%s' % str(fold_id))
            event_path = OsJoin(event_path, opt.data_type, opt.model_name + '_' + str(opt.model_depth) + '_pretrain',
                                'logs_fold%s' % str(fold_id))
        elif not opt.pretrain_path:
            if opt.mode_net == 'image_generator':
                log_path = OsJoin(result_path, opt.data_type,opt.mode_net,opt.model_name + '_' + str(opt.model_depth),
                                  'logs_%s_fold%s_%s_epoch%d' % (opt.category, str(fold_id), opt.features, opt.n_epochs))
                event_path = OsJoin(event_path, opt.data_type, opt.mode_net,
                                     'logs_%s_fold%s_%s_epoch%d' % (opt.category, str(fold_id), opt.features, opt.n_epochs))
            elif opt.mode_net == 'region-specific':
                log_path = OsJoin(result_path, opt.data_type,opt.mode_net,opt.model_name + '_' + str(opt.model_depth),
                                  'logs_%s_fold%s_%s_epoch%d' % (opt.category, str(fold_id), opt.features, opt.n_epochs))
                event_path = OsJoin(event_path, opt.data_type, opt.mode_net,
                                     'logs_%s_fold%s_%s_epoch%d' % (opt.category, str(fold_id), opt.features, opt.n_epochs))
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        if opt.mode_net == 'image_generator':
            train_logger = Logger(
                OsJoin(log_path,'train.log'),
                # ['epoch','loss','acc','lr',])
            ['epoch', 'loss','lr'])

            if opt.refine == True:
                train_batch_logger = Logger(
                    OsJoin(log_path, 'train_batch.log'),
                    ['epoch','batch','iter','loss_avg', 'loss','lr'])
            else:
                train_batch_logger = Logger(
                    OsJoin(log_path, 'train_batch.log'),
                    ['epoch', 'batch', 'iter', 'loss_avg', 'loss', 'lr', 'step', 'gamma'])
        if opt.mode_net == 'pretrained classifier' or opt.mode_net == 'region-specific':
            parameters = model.parameters()
        elif opt.mode_net == 'image_generator':
            try:

                parameters =model.module.parameters()
                # print('load module ')
            except:
                parameters = model.parameters()


        if opt.refine == True:
            optimizer = torch.optim.Adam(parameters, lr=opt.learning_rate_refine)
        else:
            optimizer = torch.optim.Adam(parameters, lr=opt.learning_rate)
        # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min',
        #                                            factor=opt.lr_decay_factor, patience =opt.lr_patience)
    if not opt.no_val:
        validation_data = ValidSet(fold_id=fold_id)
        val_loader = DataLoader(validation_data, batch_size = opt.batch_size,
                                                    num_workers = 8, pin_memory=True, sampler=DistributedSampler(validation_data))
        if opt.mode_net == 'image_generator':
            # val_logger = Logger(OsJoin(log_path, 'val.log'),
            #                     ['epoch', 'loss_G','loss_D', 'acc'])
            val_logger = Logger(OsJoin(log_path, 'val.log'),
                        ['epoch', 'loss','lr'])
    writer = SummaryWriter(logdir=event_path)
    if opt.refine == False:
        global_step_ = 0
        noise_scheduler = DDIMScheduler(num_train_timesteps=opt.n_timesteps,
                                        beta_schedule="cosine")
        # noise_scheduler_dti_fmri= DDIMScheduler(num_train_timesteps=opt.n_timesteps,
        #                                 beta_schedule="cosine")
        gamma_ = opt.gamma
        # gamma_d2f = opt.gamma
        steps_per_epcoch = len(train_loader)
        total_num_steps = (steps_per_epcoch * opt.num_epochs) // opt.gradient_accumulation_steps
        total_num_steps += int(total_num_steps * 10 / 100)
        ema_ = ema.EMA(model, gamma_, total_num_steps)
        # ema_d2f = ema.EMA(model, gamma_d2f, total_num_steps)
        lr_scheduler = get_scheduler(
            opt.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=opt.lr_warmup_steps,
            num_training_steps=total_num_steps,
        )
        scaler = GradScaler(enabled=opt.fp16_precision)
        SEED = 42
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
    for i in range(1, opt.n_epochs+1):
        
        torch.cuda.empty_cache()
        if i < 74:
            continue
        if not opt.no_train and opt.refine == False:
            train_epoch(i, fold_id, train_loader, model, criterion, opt,
                        train_logger, train_batch_logger, writer,optimizer, global_step=global_step_,\
                        noise_scheduler_ =noise_scheduler, \
                        scaler_=scaler, lr_scheduler_=lr_scheduler,\
                        Ema_=ema_, gamma_=gamma_)
        elif not opt.no_train and opt.refine == True:
            train_epoch_refine(i, fold_id, train_loader, model, criterion, opt,
                        train_logger, train_batch_logger, writer, optimizer)
       # if not opt.no_val:
       #     validation_loss = val_epoch(i,val_loader, model, criterion, opt, val_logger, writer,optimizer,\
       #                                 global_step=global_step_,
       #                 noise_scheduler_ =noise_schedul, \
       #                                 scaler_=scaler, lr_scheduler_=lr_scheduler,\
        #                Ema_=ema_, gamma_=gamma_)
        if not opt.no_train and not opt.no_val:
            lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar('lr', lr, i)
        # global_step_ = global_step_ +1
    writer.close()
    # test_data = TestSet()
    # test_loader = torch.utils.data.DataLoader(test_data, batch_size = opt.batch_size, shuffle=False,
    #                                                         num_workers = 0, pin_memory=True)
    # if opt.mode_net == 'pretrained classifier' or opt.mode_net == 'region-specific':
    #     test_logger = Logger(OsJoin(log_path, 'test.log'),
    #                         ['epoch', 'loss', 'acc', 'recall', 'precision', 'f1', 'sensitivity', 'specificity'])
    # elif opt.mode_net == 'image_generator':
    #     test_logger = Logger(OsJoin(log_path, 'test.log'),
    #                         ['epoch', 'loss_G', 'loss_D', 'acc', 'recall', 'precision', 'f1', 'sensitivity',
    #                          'specificity'])
    # test_epoch(1, test_loader, model, writer, fold_id, criterion, opt, test_logger)
    print('-'*47, 'FOLD %s FINISHED'%str(fold_id), '-'*48)


if __name__ == '__main__':
    opt = parse_opts()
    # 交叉验证
    for fold_id in range(1, opt.n_fold + 1):
        run(fold_id, opt)
        if fold_id > 1:
             break