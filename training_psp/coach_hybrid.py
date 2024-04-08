# Hybrid: Encoder-Decoder and Decoder-Encoder

import os
local_rank = int(os.environ["LOCAL_RANK"])
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math

matplotlib.use('Agg')

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from utils import common, train_utils
from criteria import id_loss, w_norm, moco_loss
from configs import data_configs
from datasets.ffhq_sketch_photo_dataset import TrainDataset as SketchTrain
from datasets.ffhq_sketch_photo_dataset import ValDataset as SketchVal
from datasets.ffhq_encode_dataset import TrainDataset as EncodeTrain
from datasets.ffhq_encode_dataset import ValDataset as EncodeVal
from datasets.latent_camera_dataset import WApproximateLatentDataset as WALatentSet
from datasets.latent_camera_dataset import WPlusLatentDataset as WPLatentSet
from datasets.latent_camera_dataset import WLatentDataset as WLatentSet
from datasets.latent_camera_dataset import get_cameras
from criteria.lpips.lpips import LPIPS
from models.psp_de_en_20 import pSp_eg3d
from training_psp.ranger import Ranger

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


class Avg_Loss:
    def __init__(self, keys):
        self.num = len(keys)
        self.loss_sum = dict.fromkeys(keys, 0.0)
        self.count = 0
    
    def add_loss(self, loss_dict):
        self.count += 1
        for key, value in loss_dict.items():
            if key in self.loss_sum.keys():
                self.loss_sum[key] += value
    
    def calc_loss_avg(self, count=None):
        self.loss_avg = dict()
        if count is None:
            for key, value in self.loss_sum.items():
                self.loss_avg[key] = value / self.count
        else:
            for key, value in self.loss_sum.items():
                self.loss_avg[key] = value / count
        return self.loss_avg
    
    def clear(self):
        for key in self.loss_sum.keys():
            self.loss_sum[key] = 0.0
        self.count = 0


class Coach:
    def __init__(self, opts):
        self.opts = opts

        self.global_step = 0
        self.epoch = 0
        if self.opts.checkpoint_path is not None:
            print("from checkpoint !")
            self.global_step = self.opts.start_iter
            self.epoch = self.opts.start_epo
        self.device = opts.device

        # Initialize network
        self.net = pSp_eg3d(self.opts).to(self.device)
        self.net = DDP(self.net, device_ids=[opts.local_rank], output_device=opts.local_rank, find_unused_parameters=True)

        # initial loss
        if self.opts.id_lambda > 0 and self.opts.moco_lambda > 0:
            raise ValueError('Both ID and MoCo loss have lambdas > 0! Please select only one to have non-zero lambda!')
        self.mse_loss = nn.MSELoss().to(self.device).eval()
        if self.opts.lpips_lambda > 0:
            print("LPIPS loss")
            self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
        if self.opts.id_lambda > 0:
            print("id loss")
            self.id_loss = id_loss.IDLoss().to(self.device).eval()
        if self.opts.w_norm_lambda > 0:
            print("w norm loss")
            self.w_norm_loss = w_norm.WNormLoss(start_from_latent_avg=self.opts.start_from_latent_avg).to(self.device).eval()
        if self.opts.moco_lambda > 0:  # not for human face images
            print("moco_loss")
            self.moco_loss = moco_loss.MocoLoss().to(self.device).eval()

        # Initialize optimizer
        self.optimizer = self.configure_optimizers()

        # Initialize dataset
        self.train_dataset, self.test_dataset, self.latent_dataset = self.configure_datasets()
        self.train_sampler = DistributedSampler(self.train_dataset,shuffle=True)
        self.latent_sampler = DistributedSampler(self.latent_dataset,shuffle=True)
        self.test_sampler = DistributedSampler(self.test_dataset)
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=self.opts.batch_size,
                                           num_workers=int(self.opts.workers),
                                           drop_last=True,
                                           sampler=self.train_sampler)
        self.latent_dataloader = DataLoader(self.latent_dataset,
                                           batch_size=1,
                                           num_workers=int(self.opts.sample_workers),
                                           drop_last=True,
                                           sampler=self.latent_sampler)
        self.test_dataloader = DataLoader(self.test_dataset,
                                          batch_size=self.opts.test_batch_size,
                                          shuffle=False,
                                          num_workers=int(self.opts.test_workers),
                                          drop_last=True,
                                          sampler=self.test_sampler)

        # Initialize logger
        log_dir = os.path.join(opts.exp_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.logger = SummaryWriter(log_dir=log_dir)

        # Initialize checkpoint dir
        self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_val_loss = None
        if self.opts.save_interval is None:
            self.opts.save_interval = self.opts.max_steps
        
        # avg, losses
        self.loss_keys = ['loss', "loss_id", "loss_l2", 'loss_lpips', 'loss_w_norm', 'latent_loss', 'latent_l2', "latent_std"]
        self.avg_loss = Avg_Loss(self.loss_keys)

    def train(self):  # En-De and Ge-En
        if self.opts.rank == 0:
            with open(os.path.join(self.opts.exp_dir, 'train_loss.txt'), 'w') as fp:
                print("epoch\titer\t", file=fp)
            with open(os.path.join(self.opts.exp_dir, 'val_loss.txt'), 'w') as fp:
                print("epoch\titer\tloss", file=fp)

        self.net.train()
        self.latent_iter = iter(self.latent_dataloader)
        while self.global_step < self.opts.max_steps:
            if self.opts.rank == 0:
                print("~" * 32)
                print("Start Epoch {}, train".format(self.epoch))
                print("~" * 32)
            self.net.train()
            self.train_sampler.set_epoch(self.epoch)  # shufful every epoch

            for batch_idx, batch in enumerate(self.train_dataloader):
                # """ En-De """
                self.optimizer.zero_grad()
                _, y, m, c = batch
                y, m, c = y.to(self.device).float(), m.to(self.device).float(), c.to(self.device).float()
                x = y
                y_hat, latent = self.net.forward(y, c, return_latents=True, resize=True)
                y_hat_masked = y_hat * m
                y_masked = y * m
                loss, loss_dict = self.calc_loss(x, y_masked, y_hat_masked, latent)
                self.avg_loss.add_loss(loss_dict)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.module.encoder.parameters(), max_norm=10.0, norm_type=2)
                self.optimizer.step()

                # """ Ge-En """
                self.optimizer.zero_grad()
                latent = next(self.latent_iter)
                latent = latent.to(self.device).float()
                cameras = get_cameras(self.opts.cam_batch_size).to(self.device).float()
                latents = self.net.module.forward_de_en(latent, cameras)
                ge_en_loss, ge_en_loss_dict = self.calc_loss_latent(latent, latents)
                self.avg_loss.add_loss(ge_en_loss_dict)
                ge_en_loss.backward()
                nn.utils.clip_grad_norm_(self.net.module.encoder.parameters(), max_norm=10.0, norm_type=2)
                self.optimizer.step()

                # Logging related
                if self.opts.rank == 0:
                    if self.global_step % self.opts.image_interval == 0:
                        self.parse_and_log_images(x, y, y_hat, title='images/train/faces')
                    if self.global_step % self.opts.board_interval == 0:
                        avg_loss = self.avg_loss.calc_loss_avg(self.opts.board_interval)
                        print("[Train LOSS]: epo{:0>2}-iter{:0>6}. En-De_loss {:.6f}. Ge-En_loss {:.6f}.".format(self.epoch, self.global_step, avg_loss["loss"], avg_loss["latent_loss"]))
                        with open(os.path.join(self.opts.exp_dir, 'train_loss.txt'), 'a') as fp:
                            store_str = "{}\t{}\t".format(self.global_step, self.global_step)
                            for key in self.loss_keys:
                                store_str += "{:.6f}\t".format(avg_loss[key])
                            print(store_str, file=fp)
                        self.avg_loss.clear()

                self.global_step += 1

            if self.opts.rank == 0:
                save_path = self.opts.exp_dir + "/DoneEpo{:0>2}-Iter{:0>6}.pth".format(self.epoch, self.global_step)
                torch.save(self.net.module.encoder.state_dict(), save_path)		

            self.validate()

            self.epoch += 1

        if self.opts.rank == 0:
            print('OMG, finished training!')
            print("End: epoch-{0:>2}-iter{0:>6}".format(self.epoch, self.global_step))

    def validate(self):
        torch.cuda.empty_cache()
        
        if self.opts.rank == 0:
            print("~" * 32)
            print("Epoch{} train done, start val".format(self.epoch))
            print("~" * 32)
        self.net.eval()
        agg_loss_dict = []
        for batch_idx, batch in enumerate(self.test_dataloader):

            with torch.no_grad():
                if self.opts.dataset_type != 'ffhq_encode':
                    x, y, m, c = batch
                    x, y, m, c = x.to(self.device).float(), y.to(self.device).float(), m.to(self.device).float(), c.to(self.device).float()
                    y_hat, latent = self.net.forward(x, c, return_latents=True, resize=True)
                else:
                    _, y, m, c = batch
                    y, m, c = y.to(self.device).float(), m.to(self.device).float(), c.to(self.device).float()
                    x = y
                    y_hat, latent = self.net.forward(y, c, return_latents=True, resize=True)

                y_hat_masked = y_hat * m
                y_masked = y * m
                loss, cur_loss_dict = self.calc_loss(x, y_masked, y_hat_masked, latent)
                if batch_idx % self.opts.board_interval == 0 and self.opts.rank == 0:
                    print()
                    print("[Val LOSS]: rank-{}-batch_idx-{}-loss:".format(self.opts.rank, batch_idx), loss)
            agg_loss_dict.append(cur_loss_dict)

            # Logging related
            if self.opts.rank == 0:
                self.parse_and_log_images(x, y, y_hat,
                                        title='images/test/faces',
                                        subscript='{:04d}'.format(batch_idx))
            
            if batch_idx > 1024:
                break

        loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
        if self.opts.rank == 0:
            self.log_metrics(loss_dict, prefix='test')
            # self.print_metrics(loss_dict, prefix='test')

            with open(os.path.join(self.opts.exp_dir, 'val_loss.txt'), 'a') as fp:
                print("{}\t{}\t{}".format(self.epoch, self.global_step, loss_dict["loss"]), file=fp)

        self.net.train()

        torch.cuda.empty_cache()

        return loss_dict
    
    def checkpoint_me(self, loss_dict, is_best):
        save_name = 'best_model.pt' if is_best else f'iteration_{self.global_step}.pt'
        save_dict = self.__get_save_dict()
        checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
        torch.save(save_dict, checkpoint_path)
        with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
            if is_best:
                f.write(f'**Best**: Step - {self.global_step}, Loss - {self.best_val_loss} \n{loss_dict}\n')
                if self.opts.use_wandb:
                    self.wb_logger.log_best_model()
            else:
                f.write(f'Step - {self.global_step}, \n{loss_dict}\n')

    def configure_optimizers(self):
        params = list(self.net.module.encoder.parameters())
        if self.opts.train_decoder:
            raise UserWarning("No decoder training !")
        if self.opts.optim_name == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
        else:
            optimizer = Ranger(params, lr=self.opts.learning_rate)
        return optimizer

    def configure_datasets(self):
        print("load dataset ...")
        train_dataset = EncodeTrain()
        test_dataset = EncodeVal()
        print("load latent ...")
        latent_dataset = WPLatentSet(20)  # 20, W+
        # latent_dataset = WALatentSet(20)  # 20, W~
        # latent_dataset = WLatentSet(20)  # 20, W
        print(f"Number of training samples: {len(train_dataset)}")
        print(f"Number of test samples: {len(test_dataset)}")
        print(f"Number of latent samples: {len(latent_dataset)}")
        return train_dataset, test_dataset, latent_dataset

    def calc_loss(self, x, y, y_hat, latent):
        loss_dict = {}
        loss = 0.0
        if self.opts.id_lambda > 0:
            loss_id, sim_improvement, id_logs = self.id_loss(y_hat, y)
            loss_dict['loss_id'] = float(loss_id)
            loss_dict['id_improve'] = float(sim_improvement)
            loss = loss_id * self.opts.id_lambda
        if self.opts.l2_lambda > 0:
            loss_l2 = F.mse_loss(y_hat, y)
            loss_dict['loss_l2'] = float(loss_l2)
            loss += loss_l2 * self.opts.l2_lambda
        if self.opts.lpips_lambda > 0:
            loss_lpips = self.lpips_loss(y_hat, y)
            loss_dict['loss_lpips'] = float(loss_lpips)
            loss += loss_lpips * self.opts.lpips_lambda
        if self.opts.lpips_lambda_crop > 0:
            loss_lpips_crop = self.lpips_loss(y_hat[:, :, 35:223, 32:220], y[:, :, 35:223, 32:220])
            loss_dict['loss_lpips_crop'] = float(loss_lpips_crop)
            loss += loss_lpips_crop * self.opts.lpips_lambda_crop
        if self.opts.l2_lambda_crop > 0:
            loss_l2_crop = F.mse_loss(y_hat[:, :, 35:223, 32:220], y[:, :, 35:223, 32:220])
            loss_dict['loss_l2_crop'] = float(loss_l2_crop)
            loss += loss_l2_crop * self.opts.l2_lambda_crop
        if self.opts.w_norm_lambda > 0:
            loss_w_norm = self.w_norm_loss(latent, self.net.module.latent_avg)
            loss_dict['loss_w_norm'] = float(loss_w_norm)
            loss += loss_w_norm * self.opts.w_norm_lambda
        if self.opts.moco_lambda > 0:
            loss_moco, sim_improvement, id_logs = self.moco_loss(y_hat, y, x)
            loss_dict['loss_moco'] = float(loss_moco)
            loss_dict['id_improve'] = float(sim_improvement)
            loss += loss_moco * self.opts.moco_lambda

        loss_dict['loss'] = float(loss)
        return loss, loss_dict

    def calc_loss_latent(self, latent, latents):
        loss_dict = {}
        loss = 0.0
        # latent, mse
        loss_latent_l2 = F.mse_loss(latent.repeat(latents.shape[0], 1, 1), latents)
        loss_dict['latent_l2'] = float(loss_latent_l2)
        loss += loss_latent_l2 * self.opts.l2_lambda
        # latent, std
        loss_latent_std = torch.std(latents, dim=0, )
        loss_latent_std = torch.mean(loss_latent_std)
        loss_dict["latent_std"] = float(loss_latent_std)
        loss += loss_latent_std * self.opts.latent_std_lambda

        loss_dict['latent_loss'] = float(loss)
        return loss, loss_dict

    def log_metrics(self, metrics_dict, prefix):
        for key, value in metrics_dict.items():
            self.logger.add_scalar(f'{prefix}/{key}', value, self.global_step)
        if self.opts.use_wandb:
            self.wb_logger.log(prefix, metrics_dict, self.global_step)

    def print_metrics(self, metrics_dict, prefix):
        print(f'Metrics for {prefix}, step {self.global_step}')
        for key, value in metrics_dict.items():
            print(f'\t{key} = ', value)

    def parse_and_log_images(self, x, y, y_hat, title, subscript=None, display_count=2):
        im_data = []
        for i in range(display_count):
            cur_im_data = {
                'input_face': common.tensor2sketch(x[i]),
                'target_face': common.tensor2im(y[i]),
                'output_face': common.tensor2im(y_hat[i]),
            }
            im_data.append(cur_im_data)
        self.log_images(title, im_data=im_data, subscript=subscript)

    def log_images(self, name, im_data, subscript=None, log_latest=False):
        fig = common.vis_faces(im_data)
        step = self.global_step
        if log_latest:
            step = 0
        if subscript:
            path = os.path.join(self.logger.log_dir, name, f'{subscript}_{step:04d}.jpg')
        else:
            path = os.path.join(self.logger.log_dir, name, f'{step:04d}.jpg')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)

    def __get_save_dict(self):
        save_dict = {
            'state_dict': self.net.state_dict(),
            'opts': vars(self.opts)
        }
        if self.opts.start_from_latent_avg:
            save_dict['latent_avg'] = self.net.latent_avg
        return save_dict
