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
n_inference_timesteps = 250

'''
best performance at 83 epoch
'''


def perceptual_loss(gen, target):
    img_vgg_input, fmap_vgg_input = rearrange(gen, 'b c h w d -> (b c h) w d '), \
        rearrange(target, 'b c h w d -> (b c h) w d ')
    img_vgg_input, fmap_vgg_input = map(lambda t: repeat(t, 'b 1 ... -> b c ...', c=3),
                                                (img_vgg_input.unsqueeze(1),
                                                 fmap_vgg_input.unsqueeze(1)))
    vgg_ = vgg()
    img_vgg_feats = vgg_(img_vgg_input)
    recon_vgg_feats = vgg_(fmap_vgg_input)
    perceptual_loss = F.mse_loss(img_vgg_feats, recon_vgg_feats)
    return perceptual_loss

def plt_image(array):
    figure = plt.figure()
    plt.imshow(array)
    plt.axis('off')
    plt.close(figure)
    return figure
def normalize(array):
    max_ = np.max(np.max(array))
    min_ = np.min(np.min(array))
    return (array-max_)/(max_-min_)
def train_epoch(epoch, fold_id, data_loader, model, criterion,\
                opt, epoch_logger, batch_logger, writer,optimizer, global_step,
                noise_scheduler_, \
                scaler_, lr_scheduler_, Ema_, gamma_):
    print('train at epoch {}'.format(epoch))


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
    index_train = np.random.randint(1,len(data_loader),size=1)
    atlas_arr = nib.load(opt.mask_path).get_fdata()

    atlas_arr[atlas_arr>87]=0
    atlas_arr[atlas_arr>0]=1
    atlas = torch.from_numpy(np.array(atlas_arr)).type(torch.FloatTensor)
    # print(atlas.shape)
    '''
    network-aware start at epoch 46
    '''
    for i ,(inputs,labels) in enumerate(data_loader):
        # if 8825<i <8840 or 3620< i< 3640 or 160<i<170 or 320<i<340:
        #     continue
        if ((i< 6200 and epoch == 100 ) or (2520<i<2526) or (2352<i < 2358) \
                or (4410 < i < 4420) or (5870<i< 5900) or (2940<i<2947) or (11740<i<11800)\
                or (3920<i<3940)):
            continue
        torch.cuda.empty_cache()
        data_time.update(time.time()-end_time)
        # if opt.n_classes == 3:
        #     labels = labels.repeat(1,inputs[0].shape[1]).view(-1,3)
        # else:
        #     labels = labels.repeat(1, inputs[0].shape[1]).view(-1, 2)
        target_fMRI = inputs[0]
        mask_fMRI = inputs[1]
        affine_fMRI = inputs[2]
        target_dti = inputs[3]
        mask_dti = inputs[4]
        affine_dti = inputs[5]
        if opt.mode_net == 'image_generator' :

            noise_fMRI = torch.randn(target_fMRI.shape)
            noise_dti = torch.randn(target_dti.shape)
            #print(target_dti.shape)
            #print(target_fMRI.shape)
            timesteps = torch.randint(0,
                                      noise_scheduler_.num_train_timesteps,
                                      (noise_fMRI.shape[0],)).long()
            try:
                noisy_images_fMRI = noise_scheduler_.add_noise(target_fMRI, noise_fMRI,
                                                         timesteps)
                noisy_images_dti = noise_scheduler_.add_noise(target_dti, noise_dti,
                                                         timesteps)
            except:
                continue
            # print(atlas.shape)
            atlas_fmri = interpolate(atlas.unsqueeze(0).unsqueeze(1), [noisy_images_fMRI.shape[1],noisy_images_fMRI.shape[2],noisy_images_fMRI.shape[3]])
            atlas_dti = interpolate(atlas.unsqueeze(0).unsqueeze(1), [noisy_images_dti.shape[1], noisy_images_dti.shape[2],noisy_images_dti.shape[3]])
            optimizer.zero_grad()
            with (((autocast(enabled=opt.fp16_precision)))):#generate target modality images
                # try:
                    noise_pred = model(noisy_images_dti.type(torch.FloatTensor).unsqueeze(1).cuda(), \
                                            noisy_images_fMRI.type(torch.FloatTensor).unsqueeze(1).cuda(), \
                                            timesteps.cuda())
                    atlas_fmri, atlas_dti = map(lambda t: repeat(t, '1 ... -> b ...', b=noise_pred['sample_dti'].shape[0]),
                                                (atlas_fmri, atlas_dti)
                                                 )

                    loss_global =  F.l1_loss(noise_pred['sample_dti'], noise_dti.cuda().unsqueeze(1)) +\
                            F.l1_loss(noise_pred['sample_fmri'], noise_fMRI.cuda().unsqueeze(1))


                    loss_network = F.l1_loss(noise_pred['sample_dti']*atlas_dti.cuda(), noise_dti.cuda().unsqueeze(1)*atlas_dti.cuda()) +\
                            F.l1_loss(noise_pred['sample_fmri']*atlas_fmri.cuda() , noise_fMRI.cuda().unsqueeze(1)*atlas_fmri.cuda() )
                    loss = loss_global + loss_network

                    if opt.perceptual_loss == True:
                        perceptual_loss_dti = perceptual_loss(noise_pred['sample_dti'], noise_dti.cuda().unsqueeze(1))
                        perceptual_loss_fmri = perceptual_loss(noise_pred['sample_fmri'],
                                                                   noise_fMRI.cuda().unsqueeze(1))
                        perceptual_loss_global = perceptual_loss_dti + perceptual_loss_fmri
                        perceptual_loss_dti_net = perceptual_loss(noise_pred['sample_dti'] * atlas_dti.cuda(), \
                                                                  noise_dti.cuda().unsqueeze(1) * atlas_dti.cuda())
                        perceptual_loss_fmri_net = perceptual_loss(noise_pred['sample_fmri'] * atlas_fmri.cuda(), \
                                                                   noise_fMRI.cuda().unsqueeze(1) * atlas_fmri.cuda())
                        perceptual_loss_net = perceptual_loss_fmri_net + perceptual_loss_dti_net
                        loss = loss_global + loss_network + perceptual_loss_global + perceptual_loss_net
                        loss_pe = perceptual_loss_net + perceptual_loss_global
            if i in index_train and epoch % 100 == 0:
                model.eval()
                with torch.no_grad():
                    generator = torch.manual_seed(0)
                    generated_images = noise_scheduler_.generate(
                        model,
                        num_inference_steps=n_inference_timesteps,
                        generator=generator,
                        eta=1.0,
                        batch_size=opt.batch_size,
                        mode='f2d')
                    # generated_images_f2d = noise_scheduler_f2d.generate(
                    #     Ema_.ema_model,
                    #     num_inference_steps=n_inference_timesteps,
                    #     generator=generator,
                    #     eta=1.0,
                    #     batch_size=opt.batch_size,
                    #     mode= 'f2d')
                    try:

                        index_d2f = np.random.randint(0, generated_images["sample_fmri"].shape[0], size=1)
                        index_f2d = np.random.randint(0, generated_images["sample_dti"].shape[0], size=1)
                        max_fmri = np.max(np.max(np.max(target_fMRI[index_d2f].squeeze().detach().cpu().numpy())))
                        max_dti = np.max(np.max(np.max(target_dti[index_f2d].squeeze().detach().cpu().numpy())))
                        save_path = r'/home/b23sxr/fmri_dti_synthesis/code/network_generation_distribution2_scan30_supervise/Synthesis'
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        gen_path = r'/home/b23sxr/fmri_dti_synthesis/code/network_generation_distribution2_scan30_supervise/Synthesis' + '/train_epoch%d_iter%d_d2f.nii' % (
                        epoch, i)
                        # nib.save(gen_path,MRI_gen_total )
                        nib.Nifti1Image(generated_images['sample_fmri'][index_d2f].squeeze() * max_fmri,
                                        affine_dti[index_d2f].squeeze()).to_filename(gen_path)
                        tar_path = r'/home/b23sxr/fmri_dti_synthesis/code/network_generation_distribution2_scan30_supervise/Synthesis' + '/train_tar_epoch%d_iter%d_d2f.nii' % (
                        epoch, i)
                        nib.Nifti1Image((target_fMRI[index_d2f].squeeze().detach().cpu().numpy()+1)*0.5,
                                        affine_dti[index_d2f].squeeze()).to_filename(tar_path)
                        gen_path = r'/home/b23sxr/fmri_dti_synthesis/code/network_generation_distribution2_scan30_supervise/Synthesis' + '/train_epoch%d_iter%d_f2d.nii' % (
                        epoch, i)
                        # nib.save(gen_path,MRI_gen_total )
                        nib.Nifti1Image(generated_images['sample_dti'][index_f2d].squeeze()*max_dti, affine_dti[index_f2d].squeeze()).to_filename(gen_path)
                        tar_path = r'/home/b23sxr/fmri_dti_synthesis/code/network_generation_distribution2_scan30_supervise/Synthesis'+'/train_tar_epoch%d_iter%d_f2d.nii'%(epoch,i)
                        nib.Nifti1Image((target_dti[index_f2d].squeeze().detach().cpu().numpy()+1)*0.5, affine_dti[index_f2d].squeeze()).to_filename(tar_path)
                    except:
                        print(index_d2f)
                        print(generated_images['sample_dti'].shape)
                        continue
            try:
                scaler_.scale(loss).backward()
            except:
                continue
            scaler_.step(optimizer)
            scaler_.update()
            # scaler_.scale(loss_f2d).backward()
            # scaler_.step(optimizer)
            # scaler_.update()
            Ema_.update_params(gamma_)
            # Ema_d2f.update_params(gamma_d2f)
            gamma_ = Ema_.update_gamma(global_step))
            lr_scheduler_.step()
            losses_log += loss.detach().item()
            if opt.mode_net == "pretrained classifier" or opt.mode_net == 'region-specific':
                checkpoint = 20
            elif opt.mode_net == 'image_generator':
                checkpoint = 1
                save_steps = 200
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
        if opt.perceptual_loss == True:
            losses_pe.update(loss_pe.data,inputs[0].size(0))
        losses_gen.update(loss_l1.data, inputs[0].size(0))
        if opt.mode_net == 'image_generator':
            batch_logger.log({
                'epoch': epoch,
                'batch': i + 1,
                'iter': (epoch - 1) * len(data_loader) + (i - 1),
                "loss_avg": losses_log / (i + 1),
                "loss": loss.detach().item(),
                "lr": lr_scheduler_.get_last_lr()[0],
                "step": global_step,
                "gamma": gamma_
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
                      'Loss_gen {loss_gen.val:.4f} ({loss_gen.avg:.4f})\t'
                .format(
                    epoch, i + 1, len(data_loader), batch_time=batch_time,\
                    data_time=data_time, loss_gen=losses_gen))
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

