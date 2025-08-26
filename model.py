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
        model.load_state_dict(checkpoint)
        print('loading pretrained model {}'.format(opt.resume_path))
        print('Load Model successfully')
    return model, model.parameters()
