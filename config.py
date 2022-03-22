from easydict import EasyDict as edict

default = edict()

# ----------------------------------------- 只修改这个就行 -----------------------------------------

default.snapshot_path = './snapshot/test18/'

# ---------------------------------------------------------------------------------------------------------------------------


# 以下内容不需要修改
default.vis_path = './' # 不改
default.data_path = './' # 不改

config = edict()
# setting for cycleGAN
# Hyper-parameters

config.multi_gpu = False
config.gpu_ids = [0,1,2]

# Setting path
config.snapshot_path = default.snapshot_path
# config.pretrained_path = default.snapshot_path
# config.data_path = default.data_path

# Setting training parameters
config.task_name = ""
config.G_LR = 2e-5
config.D_LR = 2e-5
config.beta1 = 0.5
config.beta2 = 0.999
config.c_dim = 2
config.num_epochs = 500
config.num_epochs_decay = 100
config.ndis = 1
config.lambda_A = 1.0
config.lambda_B =1.0
config.lambda_idt = 1
config.img_size = 256
config.g_conv_dim = 64
config.d_conv_dim = 64
config.g_repeat_num = 6
config.d_repeat_num = 3
config.checkpoint = ""
config.test_model = "51_2000"

# Setting datasets
dataset_config = edict()
dataset_config.name = 'MAKEUP'
dataset_config.img_size = 256



def merge_cfg_arg(config, args): # 改这里就行
    config.gpu_ids = [int(i) for i in args.gpus.split(',')]
    config.batch_size = args.batch_size
    config.vis_step = args.vis_step
    config.snapshot_step = args.snapshot_step
    config.ndis = args.ndis
    config.lambda_cls = args.lambda_cls
    config.lambda_A = args.lambda_rec
    # config.lambda_B = args.lambda_rec
    config.G_LR = args.LR
    config.D_LR = args.LR
    config.num_epochs_decay = args.decay
    config.num_epochs = args.epochs
    config.whichG = args.whichG
    config.task_name = args.task_name
    config.norm = args.norm
    # config.lambda_his = args.lambda_his
    config.lambda_vgg = args.lambda_vgg
    config.cls_list = [i for i in args.cls_list.split(',')]
    config.content_layer = [i for i in args.content_layer.split(',')]
    config.direct = args.direct
    config.g_repeat = args.g_repeat

    # Ture
    config.lips = args.lips
    config.brow = args.brow
    config.skin = args.skin
    config.eye = args.eye
    # lip, brow, eye 系数默认值
    config.lambda_his_lip = 1 # 1
    config.lambda_his_brow = 1 # 1
    config.lambda_his_eye = 1 # 1
    config.lambda_his_skin = 0.1 # 0.1



    config.data_path = args.data_path
    config.img_size = 256
    config.dataset = args.dataset
    # print(config)
    if "checkpoint" in config.items():
        config.checkpoint = args.checkpoint
    if "test_model" in config.items():
        config.test_model = args.test_model
    return config

