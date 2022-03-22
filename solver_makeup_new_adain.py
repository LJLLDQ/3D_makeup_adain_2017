
import torch.nn.functional as F
import torch.nn as nn

from model.Deep3DFaceRecon_pytorch.models.bfm import ParametricFaceModel
from model.Deep3DFaceRecon_pytorch.models.networks import ReconNetWrapper
from model.arcface_torch.backbones.iresnet import iresnet100

# from net_new_3 import Generator
from adain_net import Model
# from multiscalediscriminator import MultiscaleDiscriminator, MultiScaleGANLoss

import torch
import torch.nn.init as init
from torch.autograd import Variable
from torchvision.utils import save_image

import os
import time
import datetime

import net_new
from ops.histogram_matching import *
from ops.loss_added import GANLoss
from tqdm import tqdm as tqdm

import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter


class Solver_makeupGAN(object):
    def __init__(self, data_loaders, config,dataset_config):
        
        self.l1 = nn.L1Loss()
        self.log_writer = SummaryWriter()
        self.num_epochs = config.num_epochs
        self.data_loader_train = data_loaders[0]        
        self.criterionL1 = torch.nn.L1Loss()
        self.lambda_A = config.lambda_A
        self.norm = config.norm
        self.criterionGAN = GANLoss(use_lsgan=True, tensor =torch.cuda.FloatTensor)

        self.vgg=models.vgg16(pretrained=True)
        self.vgg.cuda()

        self.criterionL2 = torch.nn.MSELoss()
        self.batch_size = config.batch_size
        self.g_lr = config.G_LR
        self.d_lr = config.D_LR
        self.snapshot_step = config.snapshot_step
        self.vis_step = config.vis_step
        # self.MultiScaleGANLoss = MultiScaleGANLoss()

        self.task_name = config.task_name
        self.snapshot_path = config.snapshot_path + config.task_name
        self.lambda_idt = config.lambda_idt
        self.lambda_his_lip = config.lambda_his_lip
        self.lambda_his_eye = config.lambda_his_eye
        self.lambda_his_brow = config.lambda_his_brow
        self.lambda_his_skin = config.lambda_his_skin
        self.lambda_vgg = config.lambda_vgg
        # Model hyper-parameters
        self.img_size = config.img_size
        self.d_conv_dim = config.d_conv_dim
        self.d_repeat_num = config.d_repeat_num
        self.lips = config.lips
        self.skin = config.skin
        self.eye = config.eye
        self.brow = config.brow    

        if not os.path.exists(self.snapshot_path):
            os.makedirs(self.snapshot_path)        

        # Discriminator + optimizer
        setattr(self, "D", net_new.Discriminator(self.img_size, self.d_conv_dim, self.d_repeat_num, self.norm))
        getattr(self, "D").cuda()   
        setattr(self, "d" + "_optimizer", torch.optim.Adam(filter(lambda p: p.requires_grad, getattr(self, "D").parameters()), self.d_lr, [0.5, 0.999]))

        # Mutliscaler
        # self.D = MultiscaleDiscriminator()      
        # self.D = self.D.cuda()        
        # self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.g_lr, [0.5, 0.999])  

        # Generator + optimizer
        self.Generator = Model()
        self.Generator = self.Generator.cuda()        
        self.g_optimizer = torch.optim.Adam(self.Generator.parameters(), self.g_lr, [0.5, 0.999])

        # Weights initialization

        # self.Generator.apply(self.weights_init_xavier)
        # getattr(self, "D").apply(self.weights_init_xavier)
        # self.D.apply(self.weights_init_xavier)

        
        # sid loss
        self.f_3d_checkpoint_path = '/home/jl/ECCV_fighting/model/Deep3DFaceRecon_pytorch/checkpoints/epoch_20.pth'
        self.f_id_checkpoint_path = '/home/jl/ECCV_fighting/model/Deep3DFaceRecon_pytorch/ms1mv3_arcface_r100_fp16_backbone.pth'

        self.f_3d = ReconNetWrapper(net_recon='resnet50', use_last_fc=False)
        self.f_3d.load_state_dict(torch.load(self.f_3d_checkpoint_path, map_location='cpu')['net_recon'])
        self.f_3d.eval()
        self.f_3d.cuda()
        self.face_model = ParametricFaceModel()

        self.f_id = iresnet100(pretrained=False, fp16=False)
        self.f_id.load_state_dict(torch.load(self.f_id_checkpoint_path, map_location='cpu'))
        self.f_id.eval()
        self.f_id.cuda()



    def rebound_box(self, mask_A, mask_B, mask_A_center):
        # 制作空白图
        index_tmp = mask_A.nonzero() 

        x_A_index = index_tmp[:, 2]
        y_A_index = index_tmp[:, 3]
        index_tmp = mask_B.nonzero()
        x_B_index = index_tmp[:, 2]
        y_B_index = index_tmp[:, 3]
        mask_A_temp = mask_A.copy_(mask_A)
        mask_B_temp = mask_B.copy_(mask_B)

        mask_A_temp[: ,: ,min(x_A_index)-10:max(x_A_index)+11, min(y_A_index)-10:max(y_A_index)+11] =\
                            mask_A_center[: ,: ,min(x_A_index)-10:max(x_A_index)+11, min(y_A_index)-10:max(y_A_index)+11]
        mask_B_temp[: ,: ,min(x_B_index)-10:max(x_B_index)+11, min(y_B_index)-10:max(y_B_index)+11] =\
                            mask_A_center[: ,: ,min(x_B_index)-10:max(x_B_index)+11, min(y_B_index)-10:max(y_B_index)+11]
        mask_A_temp = self.to_var(mask_A_temp, requires_grad=False)
        mask_B_temp = self.to_var(mask_B_temp, requires_grad=False)
        return mask_A_temp, mask_B_temp



    def weights_init_xavier(self, m): # 初始化
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            init.xavier_normal(m.weight.data, gain=1.0)
        elif classname.find('Linear') != -1:
            init.xavier_normal(m.weight.data, gain=1.0)

    def save_models(self):
        torch.save(self.Generator.state_dict(), os.path.join(self.snapshot_path, '{}_{}_G.pth'.format(self.e + 1, self.i + 1)))
        torch.save(getattr(self, "D").state_dict(), os.path.join(self.snapshot_path, '{}_{}_D.pth'.format(self.e + 1, self.i + 1)))

    def vgg_forward(self, model, x):
        for i in range(18):
            x=model.features[i](x)
        return x

    def to_var(self, x, requires_grad=True):
        if torch.cuda.is_available():
            x = x.cuda()
            # print(1)
        if not requires_grad:
            return Variable(x, requires_grad=requires_grad)
        else:
            # print(1)
            return Variable(x)        

    def mask_preprocess(self, mask_A, mask_B):
        index_tmp = mask_A.nonzero()
        x_A_index = index_tmp[:, 2]
        y_A_index = index_tmp[:, 3]
        index_tmp = mask_B.nonzero()
        x_B_index = index_tmp[:, 2]
        y_B_index = index_tmp[:, 3]
        mask_A = self.to_var(mask_A, requires_grad=False)
        mask_B = self.to_var(mask_B, requires_grad=False)
        index = [x_A_index, y_A_index, x_B_index, y_B_index]
        index_2 = [x_B_index, y_B_index, x_A_index, y_A_index]
        return mask_A, mask_B, index, index_2       

    def criterionHis(self, input_data, target_data, mask_src, mask_tar, index):
        input_data = (self.de_norm(input_data) * 255).squeeze()
        target_data = (self.de_norm(target_data) * 255).squeeze()
        mask_src = mask_src.expand(1, 3, mask_src.size(2), mask_src.size(2)).squeeze()
        mask_tar = mask_tar.expand(1, 3, mask_tar.size(2), mask_tar.size(2)).squeeze()
        input_masked = input_data * mask_src
        target_masked = target_data * mask_tar
        input_match = histogram_matching(input_masked, target_masked, index)
        input_match = self.to_var(input_match, requires_grad=False)
        loss = self.criterionL1(input_masked, input_match)
        return loss             

    def de_norm(self, x):
        out = (x + 1) / 2
        return out.clamp(0, 1)

    def train(self):
        """Train StarGAN within a single dataset."""

        for self.e in tqdm(range(0, self.num_epochs)):
            for self.i, (img_A, img_B, mask_A, mask_B) in enumerate(tqdm(self.data_loader_train)): # img_A = non-makeup, img_B = makeup
            
                org_A = self.to_var(img_A, requires_grad=False)
                ref_B = self.to_var(img_B, requires_grad=False)
                # org_A_low = F.interpolate(org_A, scale_factor=0.25)
                # ref_B_low = F.interpolate(ref_B, scale_factor=0.25)      

                # fake_A, fake_A_low = self.Generator(org_A, ref_B) # fake_A是妆后图
                fake_A, adain_loss = self.Generator(org_A, ref_B) # fake_A是妆后图

            
                # ======================================================================== Train D ======================================================================== #
                ### fake_A ###
                ### org ###
                # Real
                out = getattr(self, "D")(ref_B)
                d_loss_real = self.criterionGAN(out, True)
                # Fake
                fake_A_D = fake_A.detach()
                out = getattr(self, "D")(fake_A_D)
                d_loss_fake =  self.criterionGAN(out, False)
                d_loss = (d_loss_real + d_loss_fake) * 0.5

                getattr(self, "d" + "_optimizer").zero_grad()
                d_loss.backward(retain_graph=True)
                getattr(self, "d" + "_optimizer").step()


                # ======================================================================== Train G ======================================================================== #




                # # ----------------------------------------------------------------------- identity loss -----------------------------------------------------------------------
                # # G should be identity if ref_B or org_A is fed
                idt_A1, _, = self.Generator(org_A, org_A)
                idt_B1, _, = self.Generator(ref_B, ref_B)
                          
                loss_idt_A1 = self.criterionL1(idt_A1, org_A) * self.lambda_A * self.lambda_idt
                loss_idt_B1 = self.criterionL1(idt_B1, ref_B) * self.lambda_A * self.lambda_idt

                # # loss_idt
                id_loss = loss_idt_A1 + loss_idt_B1



                # # # ----------------------------------------------------------------------- GAN loss -----------------------------------------------------------------------
                # # # fake_A, fake_A_low = self.Generator(org_A, ref_B) # fake_A是妆后图    

                pred_fake_org = getattr(self, "D")(fake_A) # 送进判别器
                g_A_loss_adv_org = self.criterionGAN(pred_fake_org, True)

                # pred_fake_low = getattr(self, "D")(fake_A_low) # 送进判别器
                # g_A_loss_adv_low = self.criterionGAN(pred_fake_low, True)
                # gan_loss = g_A_loss_adv_org + g_A_loss_adv_low * 0.5
                gan_loss = g_A_loss_adv_org

                # # # ----------------------------------------------------------------------- cycle loss -----------------------------------------------------------------------
                    
                rec_A, _, = self.Generator(fake_A, org_A)
                g_loss_rec_A_org = self.criterionL1(rec_A, org_A) * self.lambda_A
                cycle_loss = g_loss_rec_A_org


                # # # ----------------------------------------------------------------------- vgg loss -----------------------------------------------------------------------
                # # norm 一下
                org_A_vgg = self.de_norm(org_A)
                vgg_org = self.vgg_forward(self.vgg, self.de_norm(org_A)) # torch.Size([1, 512, 32, 32])
                vgg_org = Variable(vgg_org.data).detach()
                vgg_fake_A_org=self.vgg_forward(self.vgg, self.de_norm(fake_A))
                g_loss_A_vgg_org = self.criterionL2(vgg_fake_A_org, vgg_org) * self.lambda_A * self.lambda_vgg

                vgg_loss = g_loss_A_vgg_org

                # # # ----------------------------------------------------------------------------------------- color_histogram loss -----------------------------------------------------------------------------------------
                # # # fake_A = self.Generator(org_A, ref_B) 

                g_A_loss_his_lip = 0
                g_A_loss_his_brow = 0
                g_A_loss_his_skin = 0
                g_A_loss_his_eye = 0

                # # # Convert tensor to variable
                # # # 老位置
                # # # mask attribute: 0:background 1:face 2:left-eyebrown 3:right-eyebrown 4:left-eye 5: right-eye 6: nose 
                # # # 7: upper-lip 8: teeth 9: under-lip 10:hair 11: left-ear 12: right-ear 13: neck
                # # # 新位置
                # # # atts = [1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r',
                # # #         10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']

                for i in range(self.batch_size):

                    tmp_mask_A = mask_A[i].unsqueeze(dim=0) 
                    tmp_mask_B = mask_B[i].unsqueeze(dim=0)
                    tmp_fake_A = fake_A[i].unsqueeze(dim=0)
                    tmp_ref_B = ref_B[i].unsqueeze(dim=0)
                    tmp_org_A = org_A[i].unsqueeze(dim=0)
                                
                    if self.lips==True:
                        mask_A_lip = (tmp_mask_A==7).float() + (tmp_mask_A==9).float() # 上唇+下唇
                        mask_B_lip = (tmp_mask_B==7).float() + (tmp_mask_B==9).float() # 上唇+下唇                     
                        mask_A_lip, mask_B_lip, index_A_lip, index_B_lip = self.mask_preprocess(mask_A_lip, mask_B_lip)
                        g_A_lip_loss_his = self.criterionHis(tmp_fake_A, tmp_ref_B, mask_A_lip, mask_B_lip, index_A_lip) * self.lambda_his_lip     
                        g_A_loss_his_lip += g_A_lip_loss_his

                    if self.brow==True:
                        mask_A_brow = (tmp_mask_A==2).float() + (tmp_mask_A==3).float() # 左眉+右眉
                        mask_B_brow = (tmp_mask_B==2).float() + (tmp_mask_B==3).float() # 左眉+右眉
                        mask_A_brow, mask_B_brow, index_A_brow, index_B_brow = self.mask_preprocess(mask_A_brow, mask_B_brow)    
                        g_A_brow_loss_his = self.criterionHis(tmp_fake_A, tmp_ref_B, mask_A_brow, mask_B_brow, index_A_brow) * self.lambda_his_brow
                        g_A_loss_his_brow += g_A_brow_loss_his

                    # # skin loss 现在用这个，区别在于g_A_skin_loss_his的计算，fake_A的计算对象从ref_B变成了org_A
                    if self.skin==True:
                        mask_A_skin = (tmp_mask_A==1).float() + (tmp_mask_A==10).float() + (tmp_mask_A==14).float() # 脸+鼻子+脖子
                        mask_B_skin = (tmp_mask_B==1).float() + (tmp_mask_B==10).float() + (tmp_mask_B==14).float() # 脸+鼻子+脖子
                        mask_A_skin, mask_B_skin, index_A_skin, index_B_skin = self.mask_preprocess(mask_A_skin, mask_B_skin)
                        # 计算loss
                        g_A_skin_loss_his = self.criterionHis(tmp_fake_A, tmp_ref_B, mask_A_skin, mask_B_skin, index_A_skin) * self.lambda_his_skin
                        g_A_loss_his_skin += g_A_skin_loss_his

                    if self.eye==True:
                        mask_A_eye_left = (tmp_mask_A==4).float() # 左眼
                        mask_A_eye_right = (tmp_mask_A==5).float() # 右眼
                        mask_B_eye_left = (tmp_mask_B==4).float() # 左眼
                        mask_B_eye_right = (tmp_mask_B==5).float() # 右眼

                        # mask_A_center = (mask_A==1).float() + (mask_A==6).float()
                        # mask_B_center = (mask_B==1).float() + (mask_B==6).float()
                        # 对范围进行圈定
                        # mask_A_eye_left, mask_A_eye_right = self.rebound_box(mask_A_eye_left, mask_A_eye_right, mask_A_center)
                        # mask_B_eye_left, mask_B_eye_right = self.rebound_box(mask_B_eye_left, mask_B_eye_right, mask_B_center)

                        # 过mask_preprocess
                        mask_A_eye_left, mask_B_eye_left, index_A_eye_left, index_B_eye_left = self.mask_preprocess(mask_A_eye_left, mask_B_eye_left) # 左眼
                        mask_A_eye_right, mask_B_eye_right, index_A_eye_right, index_B_eye_right = self.mask_preprocess(mask_A_eye_right, mask_B_eye_right) # 右眼

                        # 计算loss
                        g_A_eye_left_loss_his = self.criterionHis(tmp_fake_A, tmp_org_A, mask_A_eye_left, mask_B_eye_left, index_A_eye_left) * self.lambda_his_eye
                        g_A_eye_right_loss_his = self.criterionHis(tmp_fake_A, tmp_org_A, mask_A_eye_right, mask_B_eye_right, index_A_eye_right) * self.lambda_his_eye
                        g_A_loss_his_eye += g_A_eye_left_loss_his + g_A_eye_right_loss_his

                # ----------------------------------------------------------------------- FINAl loss -----------------------------------------------------------------------

                # g_loss = id_loss + cycle_loss + vgg_loss + gan_loss + adain_loss + \
                        #  g_A_loss_his_lip + g_A_loss_his_brow + g_A_loss_his_eye + g_A_loss_his_skin
                g_loss = 10 * adain_loss + 10 * id_loss + cycle_loss + vgg_loss +  gan_loss +\
                         g_A_loss_his_lip * 2 + g_A_loss_his_eye + g_A_loss_his_skin
                # 优化
                self.g_optimizer.zero_grad()
                g_loss.backward()            
                # clip step
                # torch.nn.utils.clip_grad_norm_(self.Generator.parameters(), 5)                
                self.g_optimizer.step()

                # save iamge
                if (self.i + 1) % self.vis_step == 0:
                    image_list = []
                    image_list.append(org_A)
                    image_list.append(ref_B)                       
                    self.Generator.eval()
                    with torch.no_grad():   
                        # fake_A, _, = self.Generator(org_A, ref_B)
                        fake_A, _, = self.Generator(org_A, ref_B)

                    # print(fake_A.shape)
                    image_list.append(fake_A)
                    image_list = torch.cat(image_list, dim=3)
                    save_path = os.path.join('./test17/{}.jpg'.format(self.i + 1))
                    save_image(self.de_norm(image_list.data), save_path, nrow=1, padding=0, normalize=True)

            # Save model checkpoints
            if (self.e + 1) % self.snapshot_step == 0:
                self.save_models()

            # Decay learning rate
            # if (self.e + 1) > (self.num_epochs - self.num_epochs_decay):
            #     g_lr -= (self.g_lr / float(self.num_epochs_decay))
            #     d_lr -= (self.d_lr / float(self.num_epochs_decay))
            #     self.update_lr(g_lr, d_lr)
            #     print('Decay learning rate to g_lr: {}, d_lr:{}.'.format(g_lr, d_lr))
