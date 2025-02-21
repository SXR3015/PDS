import torch
from torch import nn
from models.model import unet
from models.model import unet_refine
# from models import resnet, pre_resnet, wide_resnet, densenet, simpleCNN
# from models import my_model
def generate_model(opt):
    assert opt.mode in ['score', 'feature']
    if opt.mode == 'score':
        last_fc = True
    elif opt.mode == 'feature':
        last_fc = False

    assert opt.model_name in ['resnet', 'preresnet', 'wideresnet', 'resnext', 'densenet', 'simpleCNN']

    if opt.model_name == 'resnet':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]

        if opt.model_depth == 10:
            # denoisor = unet.UNet(
            #     in_channel = opt.in_channel,
            #     out_channel = opt.out_channel,
            #     norm_groups = opt.norm_groups,
            #     inner_channel = opt.inner_channel,
            #     channel_mults = opt.channel_multiplier,
            #     attn_res = opt.attn_res,
            #     res_blocks = opt.res_blocks,
            #     dropout = opt.drop_rate,
            #     # image_size_fmri = opt.image_size_fMRI,
            #     # image_size_dti = opt.image_size_DTI,
            #     version=opt.version
            #      )
            # denoisor_fn = unet.UNet(
            #     in_channel=2,
            #     out_channel=1,
            #     norm_groups=opt.norm_groups,
            #     inner_channel=32,
            #     channel_mults=opt.channel_multiplier,
            #     attn_res=opt.attn_res,
            #     res_blocks=opt.res_blocks,
            #     dropout=opt.drop_rate,
            #     image_size_fmri=opt.image_size_fMRI,
            #     image_size_DTI=opt.image_size_fMRI,
            #     version=opt.version
            #      )
            # model = my_model.GaussianDiffusion(denoisor=denoisor
            #                                    # image_size_fMRI=opt.image_size_fmri,
            #                                    # image_size_DTI=opt.image_size_dti,
            #                                    # schedule_opt=[opt.schedule_start,opt.schedule_end],
            #                                    # denoisor_fn=None
            #                                      )
            if opt.refine == True:
                model = unet_refine.UNet(1, hidden_dims=[64, 128, 256, 512],
                         use_flash_attn=True)
            else:
                model= unet.UNet(1, hidden_dims=[64, 128, 256, 512],
                         use_flash_attn=True)
            # noise_scheduler = DDIMScheduler(num_train_timesteps=n_timesteps,
            #                                 beta_schedule="cosine")

    if not opt.no_cuda:
        model = model.cuda()
        #model = nn.DataParallel(model, device_ids=None)
        net_dict = model.state_dict()
    else:
        net_dict = model.state_dict()

    # load pretrain
    if opt.pretrain_diffusion_path != '' and opt.refine == False:
        print('loading pretrained model {}'.format(opt.resume_path))
        if opt.pretrain == True:
            # try:
                checkpoint = torch.load(opt.resume_path, weights_only=True, map_location='cuda:0')['state_dict']
                for key in list(checkpoint.keys()):
                    if 'module.' in key:
                        checkpoint[key.replace('module.', '')] = checkpoint[key]
                        del checkpoint[key]
                # opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
                # assert opt.arch == checkpoint['arch']
                # model.load_state_dict(checkpoint['state_dict'])
                model.load_state_dict(checkpoint)
                print('Load Model successfully')
    if opt.refine == True and opt.resume_path != '':
        checkpoint = torch.load(opt.resume_path, weights_only=True, map_location='cuda:0')['state_dict']
        for key in list(checkpoint.keys()):
            if 'module.' in key:
                checkpoint[key.replace('module.', '')] = checkpoint[key]
                del checkpoint[key]
        # opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
        # assert opt.arch == checkpoint['arch']
        # model.load_state_dict(checkpoint['state_dict'])
        model.load_state_dict(checkpoint)
        print('loading pretrained model {}'.format(opt.resume_path))
        print('Load Model successfully')
        # pretrain = torch.load(opt.pretrain_path)
        # pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
        #
        # net_dict.update(pretrain_dict)
        # model.load_state_dict(net_dict)
        #
        # new_parameters = []
        # for pname, p in model.named_parameters():
        #     for layer_name in opt.new_layer_names:
        #         if pname.find(layer_name) >= 0:
        #             new_parameters.append(p)
        #             break

        # except:
        #     print('Load Model unsuccessfully')

    return model, model.parameters()