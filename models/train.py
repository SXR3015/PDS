import torch
# from torch.autograd import Variable
import os
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import nibabel as nib
from diffusers.optimization import get_scheduler
# import io
# from PIL import Image
import numpy as np
from utils import OsJoin
import time
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
# from models.my_model  import DM_MRI
from utils import AverageMeter,calculate_accuracy,generate_target_label,generate_neurodegeneration
# from models.scheduler import DDIMScheduler
# from models import ema
n_inference_timesteps = 250
def plt_image(array):
    figure = plt.figure()
    plt.imshow(array)
    plt.axis('off')
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # buf = io.BytesIO()
    # plt.savefig(buf, format='png')
    # buf.seek(0)
    # image = Image.open(buf)
    plt.close(figure)
    return figure
def normalize(array):
    max_ = np.max(np.max(array))
    min_ = np.min(np.min(array))
    return (array-max_)/(max_-min_)
def train_epoch(epoch, fold_id, data_loader, model, criterion,\
                opt, epoch_logger, batch_logger, writer,optimizer, global_step,
                noise_scheduler_f2d, noise_scheduler_d2f, \
                scaler_, lr_scheduler_, Ema_, gamma_):
    print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses= AverageMeter()
    losses_discr = AverageMeter()
    accuracies = AverageMeter()
    writer = writer
    losses_log = 0
    end_time = time.time()
    writer_index = np.random.randint(1,len(data_loader),size=1)
    index_train = np.random.randint(0, len(data_loader), size=1)
    for i ,(inputs,labels) in enumerate(data_loader):
        torch.cuda.empty_cache()
        data_time.update(time.time()-end_time)
        # if opt.n_classes == 3:
        #     labels = labels.repeat(1,inputs[0].shape[1]).view(-1,3)
        # else:
        #     labels = labels.repeat(1, inputs[0].shape[1]).view(-1, 2)
        target_fMRI = inputs[0]
        noise_fMRI = inputs[1]
        affine_fMRI = inputs[2]
        target_dti = inputs[3]
        noise_dti = inputs[4]
        affine_dti = inputs[5]
        # inputs[0] = inputs[0].view(-1,inputs[0].shape[2], inputs[0].shape[3])
        # inputs[1] = inputs[1].view(-1, inputs[1].shape[2], inputs[1].shape[3])
        labels = labels.type(torch.FloatTensor)
        target_MRI = [target_dti, target_fMRI ]
        inputs_noise = [noise_fMRI,noise_dti]
        affine = [affine_dti,affine_fMRI]
        # inputs_target = inputs[1].type(torch.FloatTensor).unsqueeze(1)
        # inputs = [inputs_noise, inputs_target]
        # if opt.mode_net == 'pretrained classifier' or opt.mode_net == 'region-specific':
        #    loss, outputs = model([inputs,labels])
        if opt.mode_net == 'image_generator' :

            noise_fMRI = torch.randn(target_fMRI.shape)
            noise_dti = torch.randn(target_dti.shape)
            #print(target_dti.shape)
            #print(target_fMRI.shape)
            timesteps = torch.randint(0,
                                      noise_scheduler_f2d.num_train_timesteps,
                                      (opt.batch_size,)).long()
            noisy_images_fMRI = noise_scheduler_f2d.add_noise(target_fMRI, noise_fMRI,
                                                     timesteps)
            noisy_images_dti = noise_scheduler_d2f.add_noise(target_dti, noise_dti,
                                                     timesteps)

            optimizer.zero_grad()
            with ((autocast(enabled=opt.fp16_precision))):#generate target modality images
                noise_pred = model(noisy_images_dti.type(torch.FloatTensor).unsqueeze(1).cuda(), \
                                        noisy_images_fMRI.type(torch.FloatTensor).unsqueeze(1).cuda(), \
                                        timesteps)
                 # = model(noisy_images_fMRI.type(torch.FloatTensor).unsqueeze(1).cuda(), \
                 #                        noisy_images_dti.type(torch.FloatTensor).unsqueeze(1).cuda(),\
                 #                        timesteps)["sample"]
                # loss_d2f = F.l1_loss(noise_pred_fMRI, noise_fMRI.cuda().unsqueeze(1))
                # loss_f2d = F.l1_loss(noise_pred_dti, noise_dti.cuda().unsqueeze(1))
                loss =  F.l1_loss(noise_pred['sample_dti'], noise_dti.cuda().unsqueeze(1)) +\
                        F.l1_loss(noise_pred['sample_fmri'], noise_fMRI.cuda().unsqueeze(1))
                #print(noise_pred_dti.shape)
                # loss = F.l1_loss(noise_pred_dti, noise_dti.cuda().unsqueeze(1))
            if i == index_train and epoch % 5 == 0:
                model.eval()
                with torch.no_grad():
                    generator = torch.manual_seed(0)
                    generated_images_d2f = noise_scheduler_d2f.generate(
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

                        index_d2f = np.random.randint(0, generated_images_d2f["sample_dti"].shape[0], size=1)
                        # index_f2d = np.random.randint(0, generated_images_f2d["sample"].shape[0], size=1)
                        max_fmri = np.max(np.max(np.max(target_fMRI[index_d2f].squeeze().detach().cpu().numpy())))
                        # max_dti = np.max(np.max(np.max(target_dti[index_f2d].squeeze().detach().cpu().numpy())))
                        save_path = r'/home/b23sxr/fmri_dti_synthesis/code/network_generation_distribution2/Synthesis'
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        gen_path = r'/home/b23sxr/fmri_dti_synthesis/code/network_generation_distribution2/Synthesis' + '/train_epoch%d_iter%d_d2f.nii' % (
                        epoch, i)
                        # nib.save(gen_path,MRI_gen_total )
                        nib.Nifti1Image(generated_images_d2f['sample_fmri'][index_d2f].squeeze() * max_fmri,
                                        affine_dti[index_d2f].squeeze()).to_filename(gen_path)
                        tar_path = r'/home/b23sxr/fmri_dti_synthesis/code/network_generation_distribution2/Synthesis' + '/train_tar_epoch%d_iter%d_d2f.nii' % (
                        epoch, i)
                        nib.Nifti1Image((target_fMRI[index_d2f].squeeze().detach().cpu().numpy()+1)*0.5,
                                        affine_dti[index_d2f].squeeze()).to_filename(tar_path)
                        gen_path = r'/home/b23sxr/fmri_dti_synthesis/code/network_generation_distribution2/Synthesis' + '/train_epoch%d_iter%d_f2d.nii' % (
                        epoch, i)
                        nib.save(gen_path,MRI_gen_total )
                        nib.Nifti1Image(generated_images_f2d['sample_dti'][index_f2d].squeeze()*max_dti, affine_dti[index_f2d].squeeze()).to_filename(gen_path)
                        tar_path = r'/home/b23sxr/fmri_dti_synthesis/code/network_generation_distribution2/Synthesis'+'/tar_epoch%d_iter%d_f2d.nii'%(epoch,i)
                        nib.Nifti1Image((target_dti[index_f2d].squeeze().detach().cpu().numpy()+1)*0.5, affine_dti[index_f2d].squeeze()).to_filename(tar_path)
                    except:
                        print(index_d2f)
                        print(generated_images_d2f['sample'].shape)
                        continue
            scaler_.scale(loss).backward()
            scaler_.step(optimizer)
            scaler_.update()
            # scaler_.scale(loss_f2d).backward()
            # scaler_.step(optimizer)
            # scaler_.update()
            Ema_.update_params(gamma_)
            # Ema_d2f.update_params(gamma_d2f)
            gamma_ = Ema_.update_gamma(global_step)
            # gamma_d2f = Ema_d2f.update_gamma(global_step)
            if opt.use_clip_grad:
                clip_grad_norm_(model.parameters(), 1.0)
            lr_scheduler_.step()
            losses_log += loss.detach().item()
            # acc = 1
        losses.update(loss.data,inputs[0].size(0))
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
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                                 .format(
                epoch, i + 1, len(data_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))
            if i % writer_index == 0:
                writer.add_scalar('train/loss', losses_log / (i + 1), i + (epoch - 1) * len(data_loader))
                writer.add_scalar('train/lr', loss.detach().item(), i + (epoch - 1) * len(data_loader))
                # fig, ax
                # figure = plt.figure()
                # plt.imshow(gen_smaple)
                # plt.axis('off')
                # plt.gca().xaxis.set_major_locator(plt.NullLocator())
                # plt.gca().yaxis.set_major_locator(plt.NullLocator())
                # writer.add_figure('train/gen_image', plt_image(gen_smaple), i + (epoch - 1) * len(data_loader))
                # plt.close()
                # writer.add_figure('train/gen_target_image', plt_image(gen_target_smaple), i + (epoch - 1) * len(data_loader))
                # plt.close()
                # writer.add_figure('train/sub_image', plt_image(sub_sample), i + (epoch - 1) * len(data_loader))
                # plt.close()
                # writer.add_figure('train/noise_image', plt_image(noise_smaple), i + epoch)
                # plt.close()
                # writer.add_figure('train/target_image', plt_image(target_smaple), i + (epoch - 1) * len(data_loader))
                # plt.close()
        batch_time.update(time.time()-end_time)
        end_time = time.time()



        # _, pred = outputs.topk(k=1, dim=1, largest=True)
        # pred_arr = torch.cat([pred_arr, pred], dim=0)
        # _, labels_ = labels.topk(k=1, dim=1, largest=True)
        # labels_arr = torch.cat([labels_arr, labels_.cuda().squeeze()], dim=0)
    # print('prediction :', end=' ')
    # for i in range(4, len(pred_arr)):
    #     print('%d\t'%(pred_arr[i]), end='')
    # print('\nlabel    :', end=' ')
    # for i in range(4, len(labels_arr)):
    #     print('%d\t'%(labels_arr[i]), end='')
    # print('\n')
    # labels_arr = torch.empty(4).cuda()
    # pred_arr = torch.empty(4, 1).cuda()
    if opt.mode_net == 'image_generator':
        epoch_logger.log({
            'epoch': epoch,
            'loss': round(losses.avg.item(), 4),
            'lr': optimizer.param_groups[0]['lr']
        })
    if opt.mode_net =="pretrained classifier" or opt.mode_net == 'region-specific':
        checkpoint =20
    elif  opt.mode_net == 'image_generator':
        checkpoint = 100
    if opt.save_weight:
        if epoch % checkpoint == 0:
            if opt.mode_net == 'pretrained classifier' or opt.mode_net == 'region-specific':
                save_dir = OsJoin(opt.result_path, opt.data_type, opt.mode_net, opt.model_name + str(opt.model_depth),
                                 'weights_%s_fold%s_%s_epoch%d' % (opt.category, str(fold_id), opt.features, opt.n_epochs))
            elif opt.mode_net == 'image_generator':
                save_dir =OsJoin(opt.result_path, opt.data_type, opt.mode_net, opt.model_name + str(opt.model_depth),
                                 'weights_%s_fold%s_%s_epoch%d' % (opt.category, str(fold_id), opt.features, opt.n_epochs))
            elif opt.mode_net == 'text-image generator':
                save_dir =OsJoin(opt.result_path, opt.data_type, 'total', opt.model_name + str(opt.model_depth),
                                 'weights_%s_fold%s_%s_epoch%d' % (opt.category, str(fold_id), opt.features, opt.n_epochs))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = OsJoin(save_dir,
                        '{}{}_weights_fold{}_epoch{}.pth'.format(opt.model_name, opt.model_depth, fold_id, epoch))
            states = {
                'fold': fold_id,
                'epoch': epoch,
                'arch': opt.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(states, save_path)



