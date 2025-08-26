import torch
# from torch.autograd import Variable
import os

import torchvision
import torch.nn as nn
from einops import rearrange, repeat
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import nibabel as nib
from diffusers.optimization import get_scheduler
import nibabel as nib
from torch.nn.functional import interpolate
from models.scheduler import DDIMScheduler
from models.model import unet
import time
# import io
# from PIL import Image
import numpy as np
from utils import OsJoin, vgg
from torch.nn.functional import interpolate
import time
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
# from models.my_model  import DM_MRI
from utils import AverageMeter,calculate_accuracy,generate_target_label,generate_neurodegeneration
# from models.scheduler import DDIMScheduler
# from models import ema
n_inference_timesteps = 10
n_timesteps = 1000
'''
best performance at 83 epoch
'''

def train_epoch_refine(epoch, fold_id, data_loader, model, criterion,\
                opt, epoch_logger, batch_logger, writer,optimizer):
    print('train at epoch {}'.format(epoch))
    '''
    multi-process generation will lead to noise results output.
    '''
    '''
    clone() method lead to revise input tensor
    '''
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses= AverageMeter()
    losses_pe = AverageMeter()
    losses_gen = AverageMeter()
    losses_mse = AverageMeter()
    losses_corr = AverageMeter()
    accuracies = AverageMeter()
    writer = writer
    losses_log = 0
    end_time = time.time()
    writer_index = np.random.randint(1,len(data_loader),size=1)
    index_train = np.random.randint(1,len(data_loader),size=10)
    atlas_arr = nib.load(opt.mask_path).get_fdata()

    atlas_arr[atlas_arr>87]=0
    atlas_arr[atlas_arr>0]=1
    atlas = torch.from_numpy(np.array(atlas_arr)).type(torch.FloatTensor)
    noise_scheduler_ = DDIMScheduler(num_train_timesteps=n_timesteps,
                                     beta_schedule="cosine")
    model_pretrain = unet.UNet(1, hidden_dims=[64, 128, 256, 512],
                               use_flash_attn=True)
    model_pretrain = model_pretrain.cuda()
    model_pretrain.eval()
    with torch.no_grad():
        checkpoint = torch.load(opt.pretrain_diffusion_path, map_location='cuda:0')['state_dict']
        print('loading pretrained model {}'.format(opt.pretrain_diffusion_path))
        for key in list(checkpoint.keys()):
            if 'module.' in key:
                checkpoint[key.replace('module.', '')] = checkpoint[key]
                del checkpoint[key]
        model_pretrain.load_state_dict(checkpoint)

        generator = torch.manual_seed(0)
        generated_images = noise_scheduler_.generate(
            model_pretrain,
            num_inference_steps=n_inference_timesteps,
            generator=generator,
            eta=1.0,
            batch_size=opt.batch_size,
            mode='f2d')
        # time.sleep(10)
        sample_dist = generated_images["sample_fmri_pt"]
        sample_resour = generated_images["sample_dti_pt"]
    # print(atlas.shape)
    '''
    network-aware start at epoch 46
    '''
    for i ,(inputs,labels) in enumerate(data_loader):
        # if 8825<i <8840 or 3620< i< 3640 or 160<i<170 or 320<i<340:
        #     continue
        if ((8825 <i <8840 ) or (7060<i<7070) or (5885<i<5890) or (3530<i<3535)) or\
                    (i<2000 and epoch ==73):
            continue

        # torch.cuda.empty_cache()
        data_time.update(time.time()-end_time)
        target_fMRI = inputs[0]
        mask_fMRI = inputs[1]
        affine_fMRI = inputs[2]
        target_dti = inputs[3]
        mask_dti = inputs[4]
        affine_dti = inputs[5]
        gen_images = model(sample_resour,sample_dist,
                         target_fMRI.type(torch.FloatTensor).unsqueeze(1).cuda(), \
                               target_dti.type(torch.FloatTensor).unsqueeze(1).cuda(),i, model_pretrain)
        loss = gen_images['loss']
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i in index_train and epoch % 1 == 0:

            # try:

                # index_d2f = np.random.randint(0, noise_pred["gen_fmri"].shape[0], size=1)
                index = np.random.randint(0,  target_fMRI.shape[0], size=1)
                # max_fmri = np.max(np.max(np.max(target_fMRI[index_d2f].squeeze().detach().cpu().numpy())))
                # max_dti = np.max(np.max(np.max(target_dti[index_f2d].squeeze().detach().cpu().numpy())))
                save_path = r'/data1/sxr/fmri_dti_synthesis/code/Refine_network_l1/Synthesis'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                refine_path = r'/data1/sxr/fmri_dti_synthesis/code/Refine_network_l1/Synthesis' + '/f_refine_epoch%d_iter%d.nii' % (
                    epoch, i)
                # nib.save(gen_path,MRI_gen_total )
                nib.Nifti1Image(((gen_images['gen_fmri'][index] + 1) * 0.5).squeeze().detach().cpu().numpy(),
                                affine_fMRI[index] .squeeze()).to_filename(refine_path)
                gen_path = r'/data1/sxr/fmri_dti_synthesis/code/Refine_network_l1/Synthesis' + '/f_diffusion_epoch%d_iter%d.nii' % (
                    epoch, i)
                # nib.save(gen_path,MRI_gen_total )
                nib.Nifti1Image(((gen_images['diffusion_fmri'][index]  + 1) * 0.5).squeeze().detach().cpu().numpy(),
                                affine_fMRI[index] .squeeze()).to_filename(gen_path)
                tar_path = r'/data1/sxr/fmri_dti_synthesis/code/Refine_network_l1/Synthesis' + '/f_tar_epoch%d_iter%d.nii' % (
                    epoch, i)
                nib.Nifti1Image((target_fMRI[index] .squeeze().detach().cpu().numpy() + 1) * 0.5,
                                affine_fMRI[index] .squeeze()).to_filename(tar_path)
                refine_path = r'/data1/sxr/fmri_dti_synthesis/code/Refine_network_l1/Synthesis' + '/d_refine_epoch%d_iter%d.nii' % (
                    epoch, i)
                # nib.save(gen_path,MRI_gen_total )
                nib.Nifti1Image(((gen_images['gen_dti'][index]  + 1) * 0.5).squeeze().detach().cpu().numpy(),
                                affine_dti[index] .squeeze()).to_filename(refine_path)
                gen_path = r'/data1/sxr/fmri_dti_synthesis/code/Refine_network_l1/Synthesis' + '/d_diffusion_epoch%d_iter%d.nii' % (
                    epoch, i)
                # nib.save(gen_path,MRI_gen_total )
                nib.Nifti1Image(((gen_images['diffusion_dti'][index] + 1) * 0.5).squeeze().detach().cpu().numpy(),
                                affine_dti[index] .squeeze()).to_filename(gen_path)
                tar_path = r'/data1/sxr/fmri_dti_synthesis/code/Refine_network_l1/Synthesis' + '/d_tar_epoch%d_iter%d_d.nii' % (
                epoch, i)
                nib.Nifti1Image((target_dti[index] .squeeze().detach().cpu().numpy() + 1) * 0.5,
                                affine_dti[index] .squeeze()).to_filename(tar_path)

        losses_log += loss.detach().item()
        if opt.mode_net == "pretrained classifier" or opt.mode_net == 'region-specific':
            checkpoint = 20
        elif opt.mode_net == 'image_generator':
            checkpoint = 1
            save_steps = 500
        if opt.save_weight:
            if epoch % checkpoint == 0 and i % save_steps ==0:
                if opt.mode_net == 'pretrained classifier' or opt.mode_net == 'region-specific':
                    save_dir = OsJoin(opt.result_path, opt.data_type, opt.mode_net,
                                      opt.model_name + str(opt.model_depth),
                                      'weights_%s_fold%s_%s_epoch%d' % (
                                      opt.category, str(fold_id), opt.features, opt.n_epochs))
                elif opt.mode_net == 'image_generator':
                    save_dir = OsJoin(opt.result_path, opt.data_type, opt.mode_net,
                                      opt.model_name + str(opt.model_depth),
                                      'weights_%s_fold%s_%s_epoch%d' % (
                                      opt.category, str(fold_id), opt.features, opt.n_epochs))
                    save_dir_old = OsJoin(opt.result_path, opt.data_type, opt.mode_net,
                                      opt.model_name + str(opt.model_depth),
                                      'weights_%s_fold%s_%s_epoch%d_step%d' % (
                                      opt.category, str(fold_id), opt.features, opt.n_epochs,i-10))


                elif opt.mode_net == 'text-image generator':
                    save_dir = OsJoin(opt.result_path, opt.data_type, 'total',
                                      opt.model_name + str(opt.model_depth),
                                      'weights_%s_fold%s_%s_epoch%d_step%d' % (
                                      opt.category, str(fold_id), opt.features, opt.n_epochs, i))
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_path = OsJoin(save_dir,
                                   '{}{}_weights_fold{}_epoch{}_step{}.pth'.format(opt.model_name, opt.model_depth,
                                                                            fold_id, epoch,i))

                save_path_old = OsJoin(save_dir,
                                       '{}{}_weights_fold{}_epoch{}_step{}.pth'.format(opt.model_name,
                                                                                       opt.model_depth,
                                                                                       fold_id, epoch, i-save_steps*2))
                if os.path.exists(save_path_old):
                    try:
                        os.remove(save_path_old)
                    except:
                        print('File has deleted')
                states = {
                    'fold': fold_id,
                    'epoch': epoch,
                    'arch': opt.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(states, save_path)
            # acc = 1
        losses.update(loss.data,inputs[0].size(0))

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i - 1),
            "loss_avg": losses_log / (i + 1),
            "loss": loss.detach().item(),
            "lr": optimizer.param_groups[0]['lr']
        })
        if opt.perceptual_loss == True:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss_gen {loss_gen.val:.4f} ({loss_gen.avg:.4f})\t'
                  'Loss_pe {loss_pe.val:.4f} ({loss_pe.avg:.4f})\t'
                                 .format(
                epoch, i + 1, len(data_loader), batch_time=batch_time,
                data_time=data_time, loss_gen=losses_gen, loss_pe=losses_pe))
        else:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            .format(
                epoch, i + 1, len(data_loader), batch_time=batch_time,\
                data_time=data_time, loss=losses))
        if i % writer_index == 0:
            writer.add_scalar('train/loss', losses_log / (i + 1), i + (epoch - 1) * len(data_loader))
            writer.add_scalar('train/lr', loss.detach().item(), i + (epoch - 1) * len(data_loader))
                
        batch_time.update(time.time()-end_time)
        end_time = time.time()

    try:
        epoch_logger.log({
            'epoch': epoch,
            'loss': round(losses.avg.item(), 4),
            'lr': optimizer.param_groups[0]['lr']
        })
    except:
            epoch_logger.log({
                'epoch': epoch,
                'loss': round(losses.avg, 4),
                'lr': optimizer.param_groups[0]['lr']
            })


    if opt.mode_net == "pretrained classifier" or opt.mode_net == 'region-specific':
        checkpoint = 20
    elif opt.mode_net == 'image_generator':
        checkpoint = 1
        save_steps = 10
    if opt.save_weight:
        if epoch % checkpoint == 0 :
            if opt.mode_net == 'pretrained classifier' or opt.mode_net == 'region-specific':
                save_dir = OsJoin(opt.result_path, opt.data_type, opt.mode_net,
                                  opt.model_name + str(opt.model_depth),
                                  'weights_%s_fold%s_%s_epoch%d' % (
                                      opt.category, str(fold_id), opt.features, opt.n_epochs))
            elif opt.mode_net == 'image_generator':
                save_dir = OsJoin(opt.result_path, opt.data_type, opt.mode_net,
                                  opt.model_name + str(opt.model_depth),
                                  'weights_%s_fold%s_%s_epoch%d' % (
                                      opt.category, str(fold_id), opt.features, opt.n_epochs))
            elif opt.mode_net == 'text-image generator':
                save_dir = OsJoin(opt.result_path, opt.data_type, 'total',
                                  opt.model_name + str(opt.model_depth),
                                  'weights_%s_fold%s_%s_epoch%d_step%d' % (
                                      opt.category, str(fold_id), opt.features, opt.n_epochs))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # if opt.pretrain ==True:
            #     epoch_save = epoch +3
            save_path = OsJoin(save_dir,
                               '{}{}_weights_fold{}_epoch{}.pth'.format(opt.model_name, opt.model_depth,
                                                                               fold_id, epoch))

            states = {
                'fold': fold_id,
                'epoch': epoch,
                'arch': opt.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(states, save_path)

