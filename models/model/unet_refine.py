import math
import torch
from torch import nn
import torchvision
from models.model import unet
from torch.nn import functional as F
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
from models.scheduler import DDIMScheduler
# from scipy.ndimage import zoom
from torch.nn.functional import interpolate
from models.model._attention import Attention
from opts import parse_opts
n_inference_timesteps = 50
n_timesteps = 1000
opt = parse_opts()
def log(t, eps=1e-10):
    return torch.log(t + eps)
def hinge_gen_loss(fake):
    return -fake.mean()
def bce_discr_loss(fake, real):
        return (-log(1 - torch.sigmoid(fake)) - log(torch.sigmoid(real))).mean()

def get_downsample_layer(in_dim, hidden_dim, is_last):
    if not is_last:
        return nn.Sequential(
            Rearrange('b c (h p1) (w p2) (d p3)-> b (c p1 p2 p3) h w d', p1=2, p2=2, p3=2),
            nn.Conv3d(in_dim * 8, hidden_dim, 1))
    else:
        return nn.Conv3d(in_dim, hidden_dim, 3, padding=1)

def get_upsample_layer_fmri2dti(in_dim, hidden_dim, is_last):
        if not is_last:
            return nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                 nn.Conv3d(in_dim, hidden_dim, 3, padding=1))
        else:
            return nn.Sequential(nn.Upsample(scale_factor=1.5, mode='nearest'),
                                 nn.Conv3d(in_dim, hidden_dim, 3, padding=1))
        # if not is_last:
        #     return nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
        #                          nn.Conv3d(in_dim, hidden_dim, 3, padding=1))
        # else:
        #     return nn.Conv3d(in_dim, hidden_dim, 3, padding=1)


def get_upsample_layer_dti2fmri(in_dim, hidden_dim, is_last):
    if not is_last:
        return nn.Sequential(nn.Upsample(scale_factor=1.39, mode='nearest'),
                             nn.Conv3d(in_dim, hidden_dim, 3, padding=1))
    else:
        return nn.Sequential(nn.Upsample(scale_factor=1.39, mode='nearest'),
                             nn.Conv3d(in_dim, hidden_dim, 3, padding=1))
def get_attn_layer(in_dim, use_full_attn, use_flash_attn):
    if use_full_attn:
        return Attention(in_dim, use_flash_attn=use_flash_attn)
    else:
        return nn.Identity()

def skip_upsample_layer(in_dim, hidden_dim, is_last):
    if not is_last:
        return nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                             nn.Conv3d(in_dim, hidden_dim, 3, padding=1))
    else:
        return nn.Conv3d(in_dim, hidden_dim, 3, padding=1)
def get_upsample_layer(in_dim, hidden_dim, is_last):
    if not is_last:
        return nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                             nn.Conv3d(in_dim, hidden_dim, 3, padding=1))
    else:
        return nn.Conv3d(in_dim, hidden_dim, 3, padding=1)

    # if not is_last:
    #     return nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
    #                          nn.Conv3d(in_dim, hidden_dim, 3, padding=1))
    # else:
    #     return nn.Conv3d(in_dim, hidden_dim, 3, padding=1)

class ResidualBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 groups=8):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.residual_conv = nn.Conv3d(
            in_channels, out_channels=out_channels,
            kernel_size=1) if in_channels != out_channels else nn.Identity()

        self.conv1 = nn.Conv3d(in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding)
        self.conv2 = nn.Conv3d(out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding)

        self.norm1 = nn.GroupNorm(num_channels=out_channels, num_groups=groups)
        self.norm2 = nn.GroupNorm(num_channels=out_channels, num_groups=groups)
        self.nonlinearity = nn.SiLU()

    def forward(self, x):
        residual = self.residual_conv(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.nonlinearity(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.nonlinearity(x)

        return x + residual


class UNet(nn.Module):

    def __init__(self,
                 in_channels,
                 hidden_dims=[64, 128, 256, 512],
                 image_size_fmri=[64,64,36],
                 image_size_dti=[96, 96, 60],
                 use_flash_attn=False):
        super(UNet, self).__init__()

        self.sample_size = image_size_fmri
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.fmri_size = image_size_fmri
        self.dti_size = image_size_dti

        self.init_conv_d_resour = nn.Conv3d(in_channels,
                                   out_channels=hidden_dims[0],
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
        self.init_conv_d_dist = nn.Conv3d(in_channels,
                                   out_channels=hidden_dims[0],
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
        self.init_conv_f_dist  = nn.Conv3d(in_channels,
                                   out_channels=hidden_dims[0],
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
        self.init_conv_f_resour = nn.Conv3d(in_channels,
                                   out_channels=hidden_dims[0],
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)

        down_blocks_f = []

        in_dim = hidden_dims[0]
        for idx, hidden_dim in enumerate(hidden_dims[1:]):
            is_last = idx >= (len(hidden_dims) - 2)
            is_first = idx == 0
            use_attn = True if use_flash_attn else not is_first
            down_blocks_f.append(
                nn.ModuleList([
                    ResidualBlock(in_dim, in_dim),
                    ResidualBlock(in_dim, in_dim),
                    get_attn_layer(in_dim, use_attn, use_flash_attn),
                    get_downsample_layer(in_dim, hidden_dim, is_last)
                ]))
            in_dim = hidden_dim

        self.down_blocks_f = nn.ModuleList(down_blocks_f)
        down_blocks_d= []

        in_dim = hidden_dims[0]
        for idx, hidden_dim in enumerate(hidden_dims[1:]):
            is_last = idx >= (len(hidden_dims) - 2)
            is_first = idx == 0
            use_attn = True if use_flash_attn else not is_first
            down_blocks_d.append(
                nn.ModuleList([
                    ResidualBlock(in_dim, in_dim),
                    ResidualBlock(in_dim, in_dim),
                    get_attn_layer(in_dim, use_attn, use_flash_attn),
                    get_downsample_layer(in_dim, hidden_dim, is_last)
                ]))
            in_dim = hidden_dim

        self.down_blocks_d = nn.ModuleList(down_blocks_d)

        mid_dim = hidden_dims[-1]
        self.mid_block1_f = ResidualBlock(mid_dim, mid_dim)
        self.mid_attn_f  = Attention(mid_dim)
        self.mid_block2_f  = ResidualBlock(mid_dim, mid_dim)

        mid_dim = hidden_dims[-1]
        self.mid_block1_d = ResidualBlock(mid_dim, mid_dim)
        self.mid_attn_d = Attention(mid_dim)
        self.mid_block2_d = ResidualBlock(mid_dim, mid_dim)

        up_blocks_f = []
        in_dim = mid_dim
        for idx, hidden_dim in enumerate(list(reversed(hidden_dims[:-1]))):
            is_last = idx >= (len(hidden_dims) - 2)
            use_attn = True if use_flash_attn else not is_last
            up_blocks_f.append(
                nn.ModuleList([
                    ResidualBlock(in_dim + hidden_dim, in_dim),
                    ResidualBlock(in_dim + hidden_dim, in_dim),
                    get_attn_layer(in_dim, use_attn, use_flash_attn),
                    get_upsample_layer_dti2fmri(in_dim, hidden_dim, is_last)
                ]))
            in_dim = hidden_dim

        self.up_blocks_f = nn.ModuleList(up_blocks_f)

        up_blocks_d = []
        in_dim = mid_dim
        for idx, hidden_dim in enumerate(list(reversed(hidden_dims[:-1]))):
            is_last = idx >= (len(hidden_dims) - 2)
            use_attn = True if use_flash_attn else not is_last
            up_blocks_d.append(
                nn.ModuleList([
                    ResidualBlock(in_dim + hidden_dim, in_dim),
                    ResidualBlock(in_dim + hidden_dim, in_dim),
                    get_attn_layer(in_dim, use_attn, use_flash_attn),
                    get_upsample_layer_fmri2dti(in_dim, hidden_dim, is_last)
                ]))
            in_dim = hidden_dim

        self.up_blocks_d = nn.ModuleList(up_blocks_d)

        # self.out_block_f2d = ResidualBlock(hidden_dims[0] * 2, hidden_dims[0],
        #                                time_embed_dim)
        self.out_block_f = ResidualBlock(hidden_dims[0] * 2, hidden_dims[0])
        self.conv_out_f = nn.Conv3d(hidden_dims[0], out_channels=1, kernel_size=1)
        self.out_block_d = ResidualBlock(hidden_dims[0] * 2, hidden_dims[0])
        self.conv_out_d = nn.Conv3d(hidden_dims[0], out_channels=1, kernel_size=1)

    def vgg(self):
            vgg = torchvision.models.vgg16(pretrained=True)
            vgg.classifier = nn.Sequential(*vgg.classifier[:-2])
            _vgg = vgg.cuda()
            return _vgg
    def D_3View(self, data):
        if data.shape[0]>1:
            view1 = torch.mean(data, axis=2).squeeze().unsqueeze(1)
            view2 = torch.mean(data, axis=3).squeeze().unsqueeze(1)
            view3 = torch.mean(data, axis=4).squeeze().unsqueeze(1)
        else:
            view1 = torch.mean(data, axis=2).squeeze().unsqueeze(0).unsqueeze(1)
            view2 = torch.mean(data, axis=3).squeeze().unsqueeze(0).unsqueeze(1)
            view3 = torch.mean(data, axis=4).squeeze().unsqueeze(0).unsqueeze(1)
        view2 = interpolate(view2, [view3.shape[2], view3.shape[3]])
        view1 = interpolate(view1, [view3.shape[2], view3.shape[3]])
        view = torch.cat((view1, view2, view3), dim=1)
        return view


    def forward(self, sample_dti, sample_fmri, target_fMRI, target_dti,i, model_pretrain):
        if i% 500 ==0 :
            noise_scheduler_ = DDIMScheduler(num_train_timesteps=n_timesteps,
                                             beta_schedule="cosine")
            # model_pretrain = unet.UNet(1, hidden_dims=[64, 128, 256, 512],
            #                            use_flash_attn=True)
            # model_pretrain = model_pretrain.cuda()
            # model_pretrain.eval()
            with torch.no_grad():
                # checkpoint = torch.load(opt.pretrain_diffusion_path, map_location='cuda:0')['state_dict']
                # print('loading pretrained model {}'.format(opt.pretrain_diffusion_path))
                # for key in list(checkpoint.keys()):
                #     if 'module.' in key:
                #         checkpoint[key.replace('module.', '')] = checkpoint[key]
                #         del checkpoint[key]
                # model_pretrain.load_state_dict(checkpoint)

                generator = torch.manual_seed(0)
                generated_images = noise_scheduler_.generate(
                    model_pretrain,
                    num_inference_steps=n_inference_timesteps,
                    generator=generator,
                    eta=1.0,
                    batch_size=opt.batch_size,
                    mode='f2d')
                # time.sleep(10)
                sample_fmri = generated_images["sample_fmri_pt"]
                sample_dti  = generated_images["sample_dti_pt"]

        input_shape = [sample_dti.shape[2], sample_dti.shape[3], sample_dti.shape[4]]
        if input_shape == self.dti_size:
            sample_resour = target_fMRI
            sample_dist = target_dti
        else: # if first input is not fMRI, change position to calculate their output, the following code is fMRI first.
            tmp = sample_dist
            sample_dist  = target_fMRI
            sample_resour = target_dti

        # u-net for fmri2dti
        skips_d = []
        x_d = self.init_conv_d_resour(sample_resour)
        r_d = self.init_conv_d_dist(sample_dti)
        # x_dist_f2d = self.init_conv_f2d_dist(sample_dist)
        for block1, block2, attn, downsample in self.down_blocks_d:
            x_d = block1(x_d)
            skips_d.append(x_d)

            x_d = block2(x_d)
            x_d = attn(x_d)
            skips_d.append(x_d)

            x_d = downsample(x_d)

        x_d = self.mid_block1_d(x_d)
        x_d = self.mid_attn_d(x_d)
        x_d = self.mid_block2_d(x_d)

        for block1, block2, attn, upsample in self.up_blocks_d:
            skip_resize = interpolate(skips_d.pop(), [x_d.shape[2], x_d.shape[3], x_d.shape[4]])
            # x_d = torch.cat((x_d, skips_d.pop()), dim=1)
            x_d = torch.cat((x_d, skip_resize), dim=1)
            x_d = block1(x_d)
            skip_resize = interpolate(skips_d.pop(), [x_d.shape[2], x_d.shape[3], x_d.shape[4]])
            # x_d = torch.cat((x_d, skips_d.pop()), dim=1)
            x_d = torch.cat((x_d, skip_resize), dim=1)
            x_d = block2(x_d)
            x_d = attn(x_d)

            x_d = upsample(x_d)
        x_d = interpolate(x_d, [self.dti_size[0], self.dti_size[1], self.dti_size[2]])
        r_d = interpolate(r_d, [self.dti_size[0], self.dti_size[1], self.dti_size[2]])
        x_d = self.out_block_d(torch.cat((x_d, r_d), dim=1))
        gen_dti = self.conv_out_d(x_d)

        # u-net for dti2fmri

        skips_f = []
        x_f = self.init_conv_f_resour(sample_dist)
        r_f = self.init_conv_f_dist(sample_fmri)
        # x_dist_d2f = self.init_conv_d2f_dist(sample_resour)
        for block1, block2, attn, downsample in self.down_blocks_f:
            x_f = block1(x_f)
            skips_f.append(x_f)

            x_f = block2(x_f)
            x_f = attn(x_f)
            skips_f.append(x_f)

            x_f = downsample(x_f)

        x_f = self.mid_block1_f(x_f)
        x_f = self.mid_attn_f(x_f)
        x_f = self.mid_block2_f(x_f)

        for block1, block2, attn, upsample in self.up_blocks_f:
            # channel = skips_dti2fmri.pop().shape[1]
            skip_resize = interpolate(skips_f.pop(), [x_f.shape[2], x_f.shape[3], x_f.shape[4]])
            # x_f = torch.cat((x_f, skips_f.pop()), dim=1)
            x_f = torch.cat((x_f, skip_resize), dim=1)
            x_f = block1(x_f)
            skip_resize = interpolate(skips_f.pop(), [x_f.shape[2], x_f.shape[3], x_f.shape[4]])
            # x_f = torch.cat((x_f, skips_f.pop()), dim=1)
            x_f = torch.cat((x_f, skip_resize), dim=1)
            x_f = block2(x_f)
            x_f = attn(x_f)

            x_f = upsample(x_f)
        x_f = interpolate(x_f, [self.fmri_size[0], self.fmri_size[1], self.fmri_size[2]])
        r_f = interpolate(r_f, [self.fmri_size[0], self.fmri_size[1], self.fmri_size[2]])
        x_f = self.out_block_f(torch.cat((x_f, r_f), dim=1))
        gen_fmri = self.conv_out_f(x_f)
        '''
        loss
        '''
        # gen_loss_dti = hinge_gen_loss(self.discriminator(gen_dti))
        # gen_loss_fmri = hinge_gen_loss(self.discriminator(gen_fmri ))
        recon_loss_dti = (F.mse_loss(gen_dti, target_dti) + \
                          F.l1_loss(gen_dti, target_dti)) + bce_discr_loss(gen_dti, target_dti)
        recon_loss_fmri = F.mse_loss(gen_fmri, target_fMRI) + F.l1_loss(gen_fmri, target_fMRI) \
                                      + bce_discr_loss(gen_fmri, target_fMRI)

        vgg_ = self.vgg()
        gen_dti_view = self.D_3View(gen_dti)
        target_dti_view = self.D_3View(target_dti)
        gen_fmri_view = self.D_3View(gen_fmri)
        target_fmri_view = self.D_3View(target_fMRI)
        img_vgg_feats_dti = vgg_(gen_dti_view)
        recon_vgg_feats_dti = vgg_(target_dti_view)
        img_vgg_feats_fmri = vgg_(gen_fmri_view)
        recon_vgg_feats_fmri = vgg_(target_fmri_view)
        perceptual_loss_dti = F.mse_loss(img_vgg_feats_dti, recon_vgg_feats_dti)
        perceptual_loss_fmri = F.mse_loss(img_vgg_feats_fmri, recon_vgg_feats_fmri)
        gen_loss_dti_total =  recon_loss_dti + perceptual_loss_dti
        gen_loss_fmri_total = recon_loss_fmri + perceptual_loss_fmri
        loss = gen_loss_fmri_total + gen_loss_dti_total
        return {"gen_dti": gen_dti, "gen_fmri": gen_fmri, 'loss': loss,\
                'diffusion_fmri': sample_fmri, \
                'diffusion_dti': sample_dti \
                }
