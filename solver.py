import time
import datetime
import os
import torch
import torch.nn as nn
import os.path as osp
import torch.backends.cudnn as cudnn
import wandb
import random
import numpy as np
import cv2

torch.backends.cudnn.benchmark = True


from model import Unet
from model import ParsingInpaintNet
from model import Discriminator
from data_loader import get_loader
from torchvision import models
from torchvision.utils import save_image

vgg_activation = dict()


def get_activation(name):
    def hook(model, input, output):
        vgg_activation[name] = output.detach()

    return hook


class Solver(object):

    def __init__(self, config):
        """Initialize configurations."""

        assert torch.cuda.is_available()

        self.wandb = config['TRAINING_CONFIG']['WANDB'] == 'True'
        self.seed = config['TRAINING_CONFIG']['SEED']

        if self.seed != 0:
            print(f'set seed : {self.seed}')
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # https://hoya012.github.io/blog/reproducible_pytorch/
        else:
            print('do not set seed')

        self.train_loader = get_loader(config, 'seg', 'train')
        self.test_loader = get_loader(config, 'seg', 'test')

        #self.train_loader_stage2 = get_loader(config, 'inpaint', 'train')
        #self.test_loader_stag2 = get_loader(config, 'inpaint', 'test')

        self.H, self.W = config['MODEL_CONFIG']['IMG_SIZE'].split(",")
        self.H, self.W = int(self.H), int(self.W)

        self.epoch = config['TRAINING_CONFIG']['EPOCH']
        self.batch_size = config['TRAINING_CONFIG']['BATCH_SIZE']

        # vgg activation
        # self.target_layer = ['relu_3', 'relu_8', 'relu_13', 'relu_17']
        self.target_layer = ['relu_3', 'relu_8']

        for layer in self.target_layer:
            self.vgg.features[int(layer.split('_')[-1])].register_forward_hook(get_activation(layer))

        self.g_lr_stage1 = float(config['TRAINING_CONFIG']['G_LR_STAGE1'])
        self.g_lr_stage2 = float(config['TRAINING_CONFIG']['G_LR_STAGE2'])
        self.d_lr = float(config['TRAINING_CONFIG']['D_LR'])

        self.color_cate = ['bk', 'beige', 'black', 'blue', 'brown',
                           'gray', 'green', 'orange', 'pink', 'purple',
                           'red', 'white', 'yellow']

        # bgr
        self.color_rgb = np.array(
            [(255, 255, 255),  # 0=background
             (198, 226, 244),  # 1=beige
             (0, 0, 0),  # 2=black
             (255, 32, 0),  # 3=blue
             (77, 96, 128),  # 4=brown
             (128, 128, 128),  # gray
             (110, 198, 119),  # green
             (0, 131, 255),  # orange
             (193, 182, 255),  # pink
             (169, 81, 120),  # purple
             (36, 28, 237),  # red
             (239, 243, 243),  # white
             (0, 255, 255),  # yellow
             ])

        self.cate_class = 23
        self.color_class = 13

        self.cloth_cate = ['bk', 'T-shirt', 'bag', 'belt', 'blazer',
                           'blouse', 'coat', 'dress', 'face', 'hair',
                           'hat', 'jeans', 'legging', 'pants', 'scarf',
                           'shoe', 'shorts', 'skin', 'skirt', 'socks',
                           'stocking', 'sunglass', 'sweater']

        self.cloth_rgb = np.array(
            [(255, 255, 255),  # 0=background
             (198, 226, 244),  # 1=t-shirt
             (0, 0, 0),  # 2=bag
             (255, 32, 0),  # 3=belt
             (77, 96, 128),  # 4=blazer
             (128, 128, 128),  # blouse
             (110, 198, 119),  # coat
             (0, 131, 255),  # dress
             (193, 182, 255),  # face
             (169, 81, 120),  # hair
             (36, 28, 237),  # hat
             (239, 243, 243),  # jeans
             (0, 255, 255),  # legging
             (128, 128, 0),  # pants
             (255, 255, 0),  # scarf
             (102, 153, 51),  # shoe
             (0, 0, 128),  # shorts
             (153, 204, 255),  # skin
             (255, 255, 204),  # skirt
             (102, 0, 102),  # socks
             (255, 204, 204),  # stocking
             (51, 51, 51),  # sunglass
             (153, 102, 102),  # sunglass
             ])

        # stage 1
        self.lambda_g_fake = config['TRAINING_CONFIG']['LAMBDA_G_FAKE']
        self.lambda_g_gt = config['TRAINING_CONFIG']['LAMBDA_G_GT']

        self.lambda_d1_real = config['TRAINING_CONFIG']['LAMBDA_D1_REAL']
        self.lambda_d1_fake = config['TRAINING_CONFIG']['LAMBDA_D1_FAKE']
        self.lambda_d1_gp = config['TRAINING_CONFIG']['LAMBDA_D1_GP']

        # stage 2
        self.lambda_i_fake = config['TRAINING_CONFIG']['LAMBDA_I_FAKE']
        self.lambda_i_gt = config['TRAINING_CONFIG']['LAMBDA_I_GT']
        self.lambda_i_percep = config['TRAINING_CONFIG']['LAMBDA_I_PERCEP']

        self.lambda_d2_real = config['TRAINING_CONFIG']['LAMBDA_D2_REAL']
        self.lambda_d2_fake = config['TRAINING_CONFIG']['LAMBDA_D2_FAKE']
        self.lambda_d2_gp = config['TRAINING_CONFIG']['LAMBDA_D2_GP']

        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.ce_loss = nn.CrossEntropyLoss()

        self.d_critic = config['TRAINING_CONFIG']['D_CRITIC']
        self.g_critic = config['TRAINING_CONFIG']['G_CRITIC']

        self.optim = config['TRAINING_CONFIG']['OPTIM']
        self.beta1 = config['TRAINING_CONFIG']['BETA1']
        self.beta2 = config['TRAINING_CONFIG']['BETA2']

        self.gpu = config['TRAINING_CONFIG']['GPU']
        self.gpu = torch.device(f'cuda:{self.gpu}')

        # Directory
        self.train_dir  = config['TRAINING_CONFIG']['TRAIN_DIR']
        self.log_dir    = osp.join(self.train_dir, config['TRAINING_CONFIG']['LOG_DIR'])
        self.sample_dir = osp.join(self.train_dir, config['TRAINING_CONFIG']['SAMPLE_DIR'])
        self.result_dir = osp.join(self.train_dir, config['TRAINING_CONFIG']['RESULT_DIR'])
        self.model_dir  = osp.join(self.train_dir, config['TRAINING_CONFIG']['MODEL_DIR'])

        # Steps
        self.log_step       = config['TRAINING_CONFIG']['LOG_STEP']
        self.sample_step    = config['TRAINING_CONFIG']['SAMPLE_STEP']
        self.test_step      = config['TRAINING_CONFIG']['TEST_STEP']
        self.save_step      = config['TRAINING_CONFIG']['SAVE_STEP']
        self.save_start     = config['TRAINING_CONFIG']['SAVE_START']

        self.lr_decay_policy = config['TRAINING_CONFIG']['LR_DECAY_POLICY']
        print(f'lr_decay_policy : {self.lr_decay_policy}')

        if self.wandb:
            wandb.login(key='3b3fd7ec86b8f3f0f32f2d7a78456686d8755d99')
            wandb.init(project='dfd_gan_training', name=self.train_dir)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        #print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

        with open(osp.join(self.train_dir,'model_arch.txt'), 'a') as fp:
            print(model, file=fp)
            print(name, file=fp)
            print("The number of parameters: {}".format(num_params),file=fp)
            print('', file=fp)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.gpu)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

    def decode_segmap(self, image, domain):

        if domain == 'category':
            label_colors = self.cloth_rgb
            num_class = self.cate_class
        else:
            label_colors = self.color_rgb
            num_class = self.color_class

        r = np.zeros_like(image).astype(np.uint8)
        g = np.zeros_like(image).astype(np.uint8)
        b = np.zeros_like(image).astype(np.uint8)

        for l in range(0, num_class):
            idx = image == l
            r[idx] = label_colors[l, 0]
            g[idx] = label_colors[l, 1]
            b[idx] = label_colors[l, 2]
        rgb = np.stack([r, g, b], axis=0)
        rgb = np.squeeze(rgb)
        return np.transpose(rgb, (1, 2, 0))

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def train(self):
        self.train_stage1()
        self.train_stage2()

    def build_model_stage1(self):

        self.G = Unet(n_channels=21 + 1 + 3 + 1 + 1, n_classes=21).to(self.gpu)
        self.D1 = Discriminator().to(self.gpu)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr_stage1, (self.beta1, self.beta2))
        self.d_1_optimizer = torch.optim.Adam(self.D1.parameters(), self.d_lr, (self.beta1, self.beta2))

        if self.lr_decay_policy == 'LambdaLR':
            self.g_scheduler = torch.optim.lr_scheduler.LambdaLR(self.g_optimizer,
                                                                 lr_lambda=lambda epoch: 0.95 ** epoch)
            self.d_1_scheduler = torch.optim.lr_scheduler.LambdaLR(self.d_1_optimizer,
                                                                 lr_lambda=lambda epoch: 0.95 ** epoch)
        elif self.lr_decay_policy == 'ExponentialLR':
            self.g_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.g_optimizer, gamma=0.5)
            self.d_1_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.d_1_optimizer, gamma=0.5)
        elif self.lr_decay_policy == 'StepLR':
            self.g_scheduler = torch.optim.lr_scheduler.StepLR(self.g_optimizer, step_size=100, gamma=0.8)
            self.d_1_scheduler = torch.optim.lr_scheduler.StepLR(self.d_1_optimizer, step_size=100, gamma=0.8)
        else:
            self.g_scheduler = None
            self.d_1_scheduler = None

        self.print_network(self.G, 'Free-form parser')
        self.print_network(self.D1, 'Discriminator1')

    def build_model_stage2(self):

        del self.D1
        del self.g_scheduler, self.g_optimizer
        del self.d_1_scheduler, self.d_1_optimizer

        self.I = ParsingInpaintNet().to(self.gpu)
        self.D2 = Discriminator().to(self.gpu)
        self.G = self.G.eval()

        self.vgg = models.vgg19_bn(pretrained=True).to(self.gpu)

        self.i_optimizer = torch.optim.Adam(self.I.parameters(), self.g_lr_stage1, (self.beta1, self.beta2))
        self.d_2_optimizer = torch.optim.Adam(self.D2.parameters(), self.d_lr, (self.beta1, self.beta2))

        if self.lr_decay_policy == 'LambdaLR':
            self.i_scheduler = torch.optim.lr_scheduler.LambdaLR(self.i_optimizer,
                                                                 lr_lambda=lambda epoch: 0.95 ** epoch)
            self.d_2_scheduler = torch.optim.lr_scheduler.LambdaLR(self.d_2_optimizer,
                                                                   lr_lambda=lambda epoch: 0.95 ** epoch)
        elif self.lr_decay_policy == 'ExponentialLR':
            self.i_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.i_optimizer, gamma=0.5)
            self.d_2_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.d_2_optimizer, gamma=0.5)
        elif self.lr_decay_policy == 'StepLR':
            self.i_scheduler = torch.optim.lr_scheduler.StepLR(self.i_optimizer, step_size=100, gamma=0.8)
            self.d_2_scheduler = torch.optim.lr_scheduler.StepLR(self.d_2_optimizer, step_size=100, gamma=0.8)
        else:
            self.i_scheduler = None
            self.d_2_scheduler = None

        self.print_network(self.I, 'In-painter')
        self.print_network(self.D2, 'Discriminator2')

    def train_stage1(self):
        self.build_model_stage1()

        # Set data loader.
        data_loader = self.train_loader
        iterations = len(self.train_loader)
        print('iterations : ', iterations)
        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)

        # torch.LongTensor(int(id)), image, in_image, seg_map, inseg_map, bin, sketch, color, noise, mask
        _, _, _, f_seg_map, f_inseg_map, _, f_sketch, f_color, f_noise, f_mask = next(data_iter)
        f_seg_map, f_sketch, f_color = f_seg_map.to(self.gpu), f_sketch.to(self.gpu), f_color.to(self.gpu)
        f_inseg_map, f_noise, f_mask = f_inseg_map.to(self.gpu), f_noise.to(self.gpu), f_mask.to(self.gpu)

        start_time = time.time()
        print('Stage 1, Start training...')

        for e in range(self.epoch):
            for i in range(iterations):
                try:
                    _, _, _, seg_map, inseg_map, _, sketch, color, noise, mask = next(data_iter)
                except:
                    data_iter = iter(data_loader)
                    _, _, _, seg_map, inseg_map, _, sketch, color, noise, mask = next(data_iter)

                seg_map, inseg_map, sketch = seg_map.to(self.gpu), inseg_map.to(self.gpu), sketch.to(self.gpu)
                color, noise, mask = color.to(self.gpu), noise.to(self.gpu), mask.to(self.gpu)

                loss = dict()

                if (i + 1) % self.d_critic == 0:

                    data = torch.cat((inseg_map,sketch,color,noise,mask), 1)
                    fake_result = self.G(data)
                    real_score = self.D1(seg_map)
                    fake_score = self.D1(fake_result)

                    d_loss_real = -torch.mean(real_score)
                    d_loss_fake = torch.mean(fake_score)

                    alpha = torch.rand(seg_map.size(0), 1, 1, 1).to(self.gpu)
                    x_hat = (alpha * seg_map.data + (1 - alpha) * fake_result.data).requires_grad_(True)
                    out_src = self.D1(x_hat)
                    d_loss_gp = self.gradient_penalty(out_src, x_hat)
                    d_loss = self.lambda_d1_real * d_loss_real + self.lambda_d1_fake * d_loss_fake
                    d_loss += self.lambda_d1_gp * d_loss_gp
                    loss['D/loss_gp'] = self.lambda_d1_gp * d_loss_gp.item()
                    loss['D/d_loss'] = self.lambda_d1_gp * d_loss_gp.item()

                    if torch.isnan(d_loss):
                        raise Exception(f'd_loss_fake is nan at Epoch [{e + 1}/{self.epoch}] Iteration [{i + 1}/{iterations}]')

                    self.d_1_optimizer.zero_grad()
                    d_loss.backward()
                    self.d_1_optimizer.step()

                if (i + 1) % self.g_critic == 0:

                    data = torch.cat((inseg_map,sketch,color,noise,mask), 1)
                    fake_result = self.G(data)

                    fake_score = self.D1(fake_result)

                    g_loss = self.ce_loss(fake_result, seg_map)
                    g_loss_fake = self.lambda_g_fake * -torch.mean(fake_score)
                    g_loss += g_loss_fake

                    self.g_optimizer.zero_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    # Logging.
                    loss['G/loss_fake'] = self.lambda_g_fake * g_loss_fake.item()
                    loss['G/g_loss'] = g_loss.item()

                if self.wandb:
                    for tag, value in loss.items():
                        wandb.log({tag: value})

                if (i + 1) % self.log_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Epoch [{}/{}], Elapsed [{}], Iteration [{}/{}]".format(e + 1, self.epoch, et, i + 1,
                                                                                  iterations)
                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)

                if (e + 1) % self.sample_step == 0:
                    with torch.no_grad():

                        z_f = len(str(self.epoch))
                        epoch_str = str(e + 1).zfill(z_f)
                        sample_path = osp.join(self.sample_dir, epoch_str)
                        os.makedirs(sample_path, exist_ok=True)

                        f_inseg_map_l = f_inseg_map.chunk(self.batch_size, dim=0)
                        f_sketch_l = f_sketch.chunk(self.batch_size, dim=0)
                        f_color_l = f_color.chunk(self.batch_size, dim=0)
                        f_noise_l = f_noise.chunk(self.batch_size, dim=0)
                        f_mask_l = f_mask.chunk(self.batch_size, dim=0)
                        f_seg_map_l = f_seg_map.chunk(self.batch_size, dim=0)

                        composed = [f_seg_map_l, f_inseg_map_l, f_sketch_l, f_color_l, f_noise_l, f_mask_l]

                        for i, datas in enumerate(zip(*composed)):
                            gt = datas[0].to(self.gpu)
                            in_seg = datas[1]
                            in_data = torch.cat(datas[1:], 0).to(self.gpu)
                            fake_result = self.G(in_data)
                            om = torch.argmax(fake_result.squeeze(), dim=0).detach().cpu().numpy()
                            pred_rbg = self.decode_segmap(om, 'category')
                            in_seg_rbg = self.decode_segmap(in_seg, 'category')
                            gt_rbg = self.decode_segmap(gt.detach().cpu().numpy(), 'category')
                            # print(np.shape(img_np), np.shape(mask_rbg), np.shape(pred_rbg))
                            concat_img = cv2.hconcat([pred_rbg, in_seg_rbg, gt_rbg])
                            cv2.imwrite(osp.join(sample_path, f'{i}_fixed_result.jpg'), concat_img)
                        print('Saved real and fake images into {}...'.format(sample_path))

                # test step
                if (e + 1) % self.test_step == 0:
                    self.test_stage1(self.test_loader, e + 1)

                # Save model checkpoints.
                if (e + 1) % self.save_step == 0 and (e + 1) >= self.save_start:

                    e_len = len(str(self.epoch))
                    epoch_str = str(e + 1).zfill(e_len)
                    ckpt_path = osp.join(self.model_dir, '{}-model.ckpt'.format(epoch_str))
                    ckpt = dict()

                    ckpt['G'] = self.G.state_dict()
                    ckpt['D1'] = self.D1.state_dict()
                    ckpt['G_optim'] = self.g_optimizer.state_dict()
                    ckpt['D1_optim'] = self.d_1_optimizer.state_dict()

                    torch.save(ckpt, ckpt_path)
                    print('Saved model checkpoints into {}...'.format(self.model_dir))

                if self.wandb:
                    wandb.log({'G/lr': self.g_optimizer.param_groups[0]['lr']})
                    wandb.log({'D/lr': self.d_1_optimizer.param_groups[0]['lr']})

                if self.lr_decay_policy != 'None':
                    self.g_scheduler.step()
                    self.d_1_scheduler.step()
        print('Stage 1, Training done')

    def test_stage1(self, in_data_loader, epoch):

        z_f = len(str(self.epoch))
        epoch_str = str(epoch).zfill(z_f)

        result_path = osp.join(self.result_dir, epoch_str)
        os.makedirs(result_path, exist_ok=True)

        with torch.no_grad():
            for i, datas in enumerate(in_data_loader):
                # torch.LongTensor(int(id)), image, in_image, seg_map, inseg_map, bin, sketch, color, noise, mask
                # _, _, _, f_seg_map, f_inseg_map, f_sketch, f_color, f_noise, f_mask = next(data_iter)
                id = datas[0].item()
                gt = datas[3].to(self.gpu) # seg_map
                inseg_map = datas[4]
                temp_l = [datas[4], datas[6], datas[7], datas[8], datas[9]]
                # inseg_map, sketch, color, noise, mask
                in_data = torch.cat(temp_l, 0).to(self.gpu)
                fake_result = self.G(in_data)
                pred = torch.argmax(fake_result.squeeze(), dim=0).detach().cpu().numpy()
                pred_rbg = self.decode_segmap(pred, 'category')
                in_seg_rbg = self.decode_segmap(inseg_map, 'category')
                gt_rbg = self.decode_segmap(gt.detach().cpu().numpy(), 'category')
                concat_img = cv2.hconcat([pred_rbg, in_seg_rbg, gt_rbg])
                cv2.imwrite(osp.join(result_path, f'{id}_test_result.jpg'), concat_img)

    def train_stage2(self):
        self.build_model_stage2()

        # Set data loader.
        data_loader = self.train_loader
        iterations = len(self.train_loader)
        print('iterations : ', iterations)

        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)


        # torch.LongTensor(int(id)), image, in_image, seg_map, inseg_map, bin, sketch, color, noise, mask
        _, f_image, f_in_image, f_seg_map, f_inseg_map, f_bin, f_sketch, f_color, f_noise, f_mask = next(data_iter)
        f_image, f_bin, f_sketch = f_image.to(self.gpu), f_bin.to(self.gpu), f_sketch.to(self.gpu)
        f_color, f_noise, f_seg_map = f_color.to(self.gpu), f_noise.to(self.gpu), f_seg_map.to(self.gpu)
        f_inseg_map, f_in_image, f_mask = f_inseg_map.to(self.gpu), f_in_image.to(self.gpu), f_mask.to(self.gpu)

        start_time = time.time()
        print('Stage 2, Start training...')

        for e in range(self.epoch):
            for i in range(iterations):
                try:
                    _, image, in_image, seg_map, inseg_map, bin, sketch, color, noise, mask = next(data_iter)
                except:
                    data_iter = iter(data_loader)
                    _, image, in_image, seg_map, inseg_map, bin, sketch, color, noise, mask = next(data_iter)

                loss = dict()

                seg_map, inseg_map, sketch = seg_map.to(self.gpu), inseg_map.to(self.gpu), sketch.to(self.gpu)
                color, noise, mask = color.to(self.gpu), noise.to(self.gpu), mask.to(self.gpu)
                image, in_image, bin = image.to(self.gpu), in_image.to(self.gpu), bin.to(self.gpu)

                if (i + 1) % self.d_critic == 0:

                    data = torch.cat((inseg_map, sketch, color, noise, mask), 1)
                    fake_result = self.G(data).detach()

                    comp1 = torch.cat((in_image, bin), 1)
                    comp2 = torch.cat((sketch, color, noise), 1)
                    fake_image = self.I(fake_result, comp1, comp2)

                    real_score = self.D2(image)
                    fake_score = self.D2(fake_image)

                    d_loss_real = -torch.mean(real_score)
                    d_loss_fake = torch.mean(fake_score)

                    alpha = torch.rand(seg_map.size(0), 1, 1, 1).to(self.gpu)
                    x_hat = (alpha * seg_map.data + (1 - alpha) * fake_result.data).requires_grad_(True)
                    out_src = self.D1(x_hat)
                    d_loss_gp = self.gradient_penalty(out_src, x_hat)
                    d_loss = self.lambda_d1_real * d_loss_real + self.lambda_d1_fake * d_loss_fake
                    d_loss += self.lambda_d1_gp * d_loss_gp
                    loss['D/loss_gp'] = self.lambda_d1_gp * d_loss_gp.item()
                    loss['D/d_loss'] = self.lambda_d1_gp * d_loss_gp.item()

                    if torch.isnan(d_loss):
                        raise Exception(
                            f'd_loss_fake is nan at Epoch [{e + 1}/{self.epoch}] Iteration [{i + 1}/{iterations}]')

                    self.d_2_optimizer.zero_grad()
                    d_loss.backward()
                    self.d_2_optimizer.step()

                if (i + 1) % self.g_critic == 0:
                    data = torch.cat((inseg_map, sketch, color, noise, mask), 1)
                    fake_result = self.G(data).detach()

                    comp1 = torch.cat((in_image, bin), 1)
                    comp2 = torch.cat((sketch, color, noise), 1)
                    fake_image = self.I(fake_result, comp1, comp2)
                    fake_score = self.D2(fake_image)

                    fake_activation = dict()
                    real_activation = dict()

                    self.vgg(fake_image)
                    for layer in self.target_layer:
                        fake_activation[layer] = vgg_activation[layer]
                    vgg_activation.clear()

                    self.vgg(image)
                    for layer in self.target_layer:
                        real_activation[layer] = vgg_activation[layer]
                    vgg_activation.clear()

                    i_loss_percep = 0
                    for layer in self.target_layer:
                        i_loss_percep += self.l1_loss(fake_activation[layer], real_activation[layer])

                    i_loss = self.lambda_i_gt * self.l1_loss(fake_result, image)
                    i_loss += self.lambda_i_percep * i_loss_percep
                    i_loss_fake = self.lambda_g_fake * -torch.mean(fake_score)
                    i_loss += i_loss_fake

                    self.i_optimizer.zero_grad()
                    i_loss.backward()
                    self.g_optimizer.step()

                    # Logging.
                    loss['I/loss_fake'] = self.lambda_g_fake * i_loss_fake.item()
                    loss['I/loss_percep'] = self.lambda_i_percep * i_loss_percep.item()
                    loss['I/gt_loss'] = i_loss.item()

                if self.wandb:
                    for tag, value in loss.items():
                        wandb.log({tag: value})

                if (i + 1) % self.log_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Epoch [{}/{}], Elapsed [{}], Iteration [{}/{}]".format(e + 1, self.epoch, et, i + 1,
                                                                                  iterations)
                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)

                if (e + 1) % self.sample_step == 0:
                    with torch.no_grad():

                        z_f = len(str(self.epoch))
                        epoch_str = str(e + 1).zfill(z_f)
                        sample_path = osp.join(self.sample_dir, epoch_str)
                        os.makedirs(sample_path, exist_ok=True)

                        f_image = f_image.chunk(self.batch_size, dim=0)
                        f_in_image = f_in_image.chunk(self.batch_size, dim=0)
                        f_bin = f_bin.chunk(self.batch_size, dim=0)
                        f_inseg_map_l = f_inseg_map.chunk(self.batch_size, dim=0)
                        f_sketch_l = f_sketch.chunk(self.batch_size, dim=0)
                        f_color_l = f_color.chunk(self.batch_size, dim=0)
                        f_noise_l = f_noise.chunk(self.batch_size, dim=0)
                        f_mask_l = f_mask.chunk(self.batch_size, dim=0)
                        f_seg_map_l = f_seg_map.chunk(self.batch_size, dim=0)

                        composed = [f_image, f_in_image, f_bin,
                                    f_seg_map_l, f_inseg_map_l,
                                    f_sketch_l, f_color_l, f_noise_l, f_mask_l]

                        for i, datas in enumerate(zip(*composed)):
                            image_report = [datas[0], datas[1]] # gt, in_image
                            in_data = torch.cat(datas[4:9], 0).to(self.gpu)
                            fake_result = self.G(in_data).detach()
                            comp1 = torch.cat(datas[1:3], 1)
                            comp2 = torch.cat(datas[5:8], 1)
                            result_image = self.I(fake_result, comp1, comp2)
                            image_report.append(result_image)
                            x_concat = torch.cat(image_report, dim=3)
                            save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                        print('Saved real and fake images into {}...'.format(sample_path))

                # test step
                if (e + 1) % self.test_step == 0:
                    self.test_stage1(self.test_loader, e + 1)

                # Save model checkpoints.
                if (e + 1) % self.save_step == 0 and (e + 1) >= self.save_start:
                    e_len = len(str(self.epoch))
                    epoch_str = str(e + 1).zfill(e_len)
                    ckpt_path = osp.join(self.model_dir, '{}-model.ckpt'.format(epoch_str))
                    ckpt = dict()

                    ckpt['G'] = self.G.state_dict()
                    ckpt['I'] = self.I.state_dict()
                    ckpt['D2'] = self.D1.state_dict()
                    ckpt['G_optim'] = self.g_optimizer.state_dict()
                    ckpt['D2_optim'] = self.d_2_optimizer.state_dict()

                    torch.save(ckpt, ckpt_path)
                    print('Saved model checkpoints into {}...'.format(self.model_dir))

                if self.wandb:
                    wandb.log({'I/lr': self.i_optimizer.param_groups[0]['lr']})
                    wandb.log({'D/lr': self.d_2_optimizer.param_groups[0]['lr']})

                if self.lr_decay_policy != 'None':
                    self.i_scheduler.step()
                    self.d_2_scheduler.step()

        print('Stage 2, Training done')

    def test_stage2(self, in_data_loader, epoch):

        z_f = len(str(self.epoch))
        epoch_str = str(epoch).zfill(z_f)

        result_path = osp.join(self.result_dir, epoch_str)
        os.makedirs(result_path, exist_ok=True)

        with torch.no_grad():
            for i, datas in enumerate(in_data_loader):
                # torch.LongTensor(int(id)), image, in_image, seg_map, inseg_map, bin, sketch, color, noise, mask
                # _, _, _, f_seg_map, f_inseg_map, f_sketch, f_color, f_noise, f_mask = next(data_iter)
                id = datas[0].item()
                image_report = [datas[1], datas[2]]  # gt, in_image

                temp_l = [datas[4], datas[6], datas[7], datas[8], datas[9]]
                # inseg_map, sketch, color, noise, mask
                in_data = torch.cat(temp_l, 0).to(self.gpu)
                fake_result = self.G(in_data).detach()
                comp1 = torch.cat((datas[2], datas[5]), 1) # comp1 = torch.cat((in_image, bin), 1)
                comp2 = torch.cat((datas[6:9]), 1) # comp2 = torch.cat((sketch, color, noise), 1)
                result_image = self.I(fake_result, comp1, comp2)
                image_report.append(result_image)
                x_concat = torch.cat(image_report, dim=3)
                save_image(self.denorm(x_concat.data.cpu()), osp.join(result_path, f'{id}_test_result.jpg'), nrow=1, padding=0)
