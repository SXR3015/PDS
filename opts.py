import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root_path', default=r'/data1/sxr/fmri_dti_synthesis/code/Refine_network_l1', type=str, help='Root directory path of code')
    parser.add_argument(
        '--data_root_path', default=r'/data1/sxr/fmri_dti_synthesis/data_scan30', type=str, help='Root directory path of data')
    parser.add_argument(
        '--mode_net', default=r'image_generator', type=str, help='project mode: pretrained classifier, image_generator, or region-specific')
    parser.add_argument(
        '--pretrain', default=True, type=bool, help='path of pretrained classifier weight')
    parser.add_argument(
        '--pretrain_path', default='', type=str, help='path of pretrained classifier weight')
    parser.add_argument(
        '--resume_path', default='/data1/sxr/fmri_dti_synthesis/code'
                                 '/Refine_network_l1/results/DFC_CLINICAL/image_generator'
                                 '/resnet10'
                                 '/weights_HC_SCD_MCI_AD_fold1_ALFF_DFC_FA_FC_epoch300/'
                                 'resnet10_weights_fold1_epoch151.pth', type=str, help='path of pretrained classifier weight')
    parser.add_argument(
        '--result_path', default='results', type=str, help='Result directory path')
    parser.add_argument(
        '--event_path', default='events', type=str, help='Result directory path')
    parser.add_argument(
        '--pretrain_diffusion_path', default=r'//data1/sxr/fmri_dti_synthesis/code'
                                             r'/Refine_network_l1/results'
                                             r'/DFC_CLINICAL/image_generator/resnet10'
                                             r'/weights_HC_SCD_MCI_AD_fold1_ALFF_DFC_FA_FC_epoch105'
                                             r'/resnet10_weights_fold1_epoch112.pth'
                                ,
        type=str, help='Saved model (.pth) of previous training'
    )
    parser.add_argument(
        '--fold_id', default='2', type=str, help='Different data type directory')
    parser.add_argument(
        '--data_type', default='DFC_CLINICAL', type=str, help='FC or JPEG')
    parser.add_argument(
        '--category', default='HC_SCD_MCI_AD', type=str, help='Different data type directory')
    parser.add_argument(
        '--features', default='ALFF_DFC_FA_FC', type=str, help='Different data type directory')
    parser.add_argument(
        '--n_classes', default=4, type=int, help='Number of classes (an: 2, tri: 3)')
    parser.add_argument(
        '--n_fold', default=5, type=int, help='Number of cross validation fold')
    parser.add_argument(
        '--model_name', default='resnet', type=str, help='(resnet | preresnet | wideresnet | resnext | densenet | simpleCNN')
    parser.add_argument(
        '--model_depth', default=10, type=int,
        help='Depth of resnet (10 | 18 | 34 | 50 | 101); densenet (121, 169, 201, 264); simpleCNN(8)')
    parser.add_argument(
        '--new_layer_names',
        # default=['upsample1', 'cmp_layer3', 'upsample2', 'cmp_layer2', 'upsample3', 'cmp_layer1', 'upsample4', 'cmp_conv1', 'conv_seg'],
        default=['fc'], type=list, help='New layer except for backbone')
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=160, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument(
        '--resnet_shortcut', default='B', type=str, help='Shortcut type of resnet (A | B)')
    parser.add_argument(
        '--wide_resnet_k', default=2, type=int, help='Wide resnet k')
    parser.add_argument(
        '--refine', default=True, type=bool, help='Wide resnet k')
    parser.add_argument(
        '--resnext_cardinality', default=32, type=int, help='ResNeXt cardinality')
    parser.add_argument(
        '--manual_seed', default=1680, type=int, help='Manually set random seed')#1024
    parser.add_argument(
        '--learning_rate', default=1e-6, type=float, help= 'Initial learning rate')#学习率
    parser.add_argument(
        '--learning_rate_refine', default=1e-5, type=float, help='Initial learning rate')  # 学习率
    parser.add_argument(
        '--lr_decay_factor', default=0.02, type=float,
        help=' Factor by which the learning rate will be reduced. new_lr = lr * factor')
    parser.add_argument(
        '--weight_decay', default=5e-6, type=float, help='Weight Decay')
    parser.add_argument(
        '--lr_patience', default=10, type=int, help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.')
    parser.add_argument(
        '--batch_size', default=1, type=int, help='Batch Size')
    parser.add_argument('--temperature', default=0.07, type=float,
                        help='softmax temperature (default: 0.07)')
    parser.add_argument('--n_views', default=512, type=int, metavar='N',
                        help='Number of views for contrastive learning training.')
    parser.add_argument(
        '--n_epochs', default=300, type=int, help='Number of total epochs to run')
    parser.add_argument(
        '--n_epochs_pretrain', default=60, type=int, help='Number of total epochs to run')
    # parser.add_argument(
    #     '--n_epochs_pretrain', default=20, type=int, help='Number of total epochs to run')
    parser.add_argument(
        '--save_weight', default=True, type=int, help='wheather save the Trained model or not.')
    parser.add_argument(
        '--mode', default='score', type=str,
        help='Mode (score | feature). score outputs class scores. '
             'feature outputs features (after global average pooling).')
    parser.add_argument(
        '--no_train', action='store_true', help='If true, training is not performed.')
    parser.set_defaults(no_train=False)
    parser.add_argument(
        '--no_val', action='store_true', help='If true, validation is not performed.')
    parser.set_defaults(no_val=False)
    parser.add_argument(
        '--test', action='store_true', help='If true, test is performed.')
    parser.set_defaults(test=True)
    parser.add_argument(
        '--test_subset', default='test', type=str, help='Used subset in test (val | test)')
    parser.add_argument(
        '--drop_rate', default=1e-5, type=int, help='drop rate')
    parser.add_argument(
        '--perceptual_loss', default=False, type=bool, help='drop rate')
    parser.add_argument(
        '--in_channel', default=1, type=int, help='input channel')
    parser.add_argument(
        '--out_channel', default=1, type=int, help='output channel')
    parser.add_argument(
        '--inner_channel', default=1, type=int, help='1')
    parser.add_argument(
        '--norm_groups', default=48, type=int, help='1')
    parser.add_argument(
        '--res_blocks', default=2, type=int, help='1')
    parser.add_argument(
        '--channel_multiplier', default=[1,2,4,8,8], type=int, help='1')
    parser.add_argument(
        '--image_size_fmri', default=[64,64,36], type=int, help='image size of fMRI')
    parser.add_argument(
        '--image_size_dti', default=[96,96,60], type=int, help='image size of DTI')
    parser.add_argument(
        '--schedule_opt_start', default=5e-5, type=int, help='schedule start')
    parser.add_argument(
        '--schedule_opt_end', default=1e-2, type=int, help='schedule end')
    parser.add_argument(
        '--version', default='v1', type=str, help='U-net version ')
    parser.add_argument(
        '--attn_res', default=[16], type=int, help='attention res block')
    parser.add_argument(
        '--n_timesteps', default=1000, type=int, help='attention res block')
    parser.add_argument("--use_clip_grad", action='store_true')
    parser.add_argument(
        '--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.add_argument('--gamma',
                    default=0.996,
                    type=float,
                    help='Initial EMA coefficient')
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument('--fp16_precision',
                        action='store_true',
                        help='Whether to use 16-bit precision for GPU training')
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=100)
    parser.add_argument("--mask_path", type=str, default='/home/b23sxr/fmri_dti_synthesis/mask/Dosenbach160_3mm.nii')
    parser.set_defaults(no_cuda=False)
    args = parser.parse_args()

    return args
