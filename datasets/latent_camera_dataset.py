# UTF-8

import torch
from torch.utils.data import Dataset
import numpy as np
import random
from PIL import Image
from glob import glob
import torchvision.transforms as transforms
import math
from training.camera_utils import LookAtPoseSampler
import random


class WLatentDataset(Dataset):
	def __init__(self, num_laysers=1):
		print("W Latent ...")
		self.latents = np.load("/home/hzwu/Prj/EG3D/eg3d/networks/w_latents-2097152.npy")
		self.num_layers = num_laysers

	def __len__(self):
		return self.latents.shape[0]

	def __getitem__(self, index):
		latent = self.latents[index]  # shape(1, 512)
		latent = torch.FloatTensor(latent).repeat((self.num_layers, 1))  # shape(num_layers, 512)
		return latent


class WApproximateLatentDataset(Dataset):  # in W+, but close to W, refer to e4e
	def __init__(self, num_laysers=1, max_weight=0.1):
		print("W~ Latent ...")
		self.latents = np.load("/home/hzwu/Prj/EG3D/eg3d/networks/w_latents-2097152.npy")
		self.num_layers = num_laysers 
		self.len = self.latents.shape[0]
		self.max_weight = max_weight

	def __len__(self):
		return self.len

	def __getitem__(self, index):
		latent_base = self.latents[index]
		latent_base = torch.FloatTensor(latent_base)
		latent_shift_list = list()
		shift_weight_list = list()
		for _ in range(self.num_layers):
			latent_shift_list.append(self.latents[random.randrange(0, self.len)])
			shift_weight_list.append(self.max_weight * random.random())
		latent_shift = torch.FloatTensor(np.concatenate(latent_shift_list, axis=0))
		latent_shift_weight = torch.FloatTensor(np.array(shift_weight_list).reshape(-1, 1))
		latent = (1-latent_shift_weight) * latent_base + latent_shift_weight * latent_shift
		return latent


class WPlusLatentDataset(Dataset):  # W+
	def __init__(self, num_laysers=1):
		print("W+ Latent...")
		self.latents = np.load("/home/hzwu/Prj/EG3D/eg3d/networks/wp_latents-2097152.npy")
		self.num_layers = num_laysers
		self.len = self.latents.shape[0]

	def __len__(self):
		return self.len

	def __getitem__(self, index):
		latent = self.latents[index]
		latent = torch.FloatTensor(latent)
		return latent


def get_cameras(num_cameras=1):
	cams = np.load("/home/hzwu/data/cams_v20230408.npy")  # num_cameras, fixed
	camera_batch = cams[random.randint(0, 1000000)]
	camera_batch = torch.FloatTensor(camera_batch)
	return camera_batch


if __name__ == "__main__":
	print("print this only when test code: datasets/latent_camera_dataset.py")

	wa_latent_dataset = WApproximateLatentDataset(14, 0.1)
	print(len(wa_latent_dataset))
	latent = wa_latent_dataset[0]
	print(latent)
	latent = wa_latent_dataset[0]
	print(latent)
	latent = wa_latent_dataset[1]
	print(latent)

	# 
	init_args = ()
	init_kwargs = {'z_dim': 512, 'w_dim': 512, 'mapping_kwargs': {'num_layers': 2}, 'channel_base': 32768,
                       'channel_max': 512, 'fused_modconv_default': 'inference_only',
                       'rendering_kwargs': {'depth_resolution': 48, 'depth_resolution_importance': 48,
                                            'ray_start': 2.25, 'ray_end': 3.3, 'box_warp': 1, 'avg_camera_radius': 2.7,
                                            'avg_camera_pivot': [0, 0, 0.2], 'image_resolution': 512,
                                            'disparity_space_sampling': False, 'clamp_mode': 'softplus',
                                            'superresolution_module': 'training.superresolution.SuperresolutionHybrid8XDC',
                                            'c_gen_conditioning_zero': False, 'c_scale': 1.0,
                                            'superresolution_noise_mode': 'none', 'density_reg': 0.25,
                                            'density_reg_p_dist': 0.004, 'reg_type': 'l1', 'decoder_lr_mul': 1.0,
                                            'sr_antialias': True}, 'num_fp16_res': 0, 'sr_num_fp16_res': 4,
                       'sr_kwargs': {'channel_base': 32768, 'channel_max': 512,
                                     'fused_modconv_default': 'inference_only'}, 'conv_clamp': None, 'c_dim': 25,
                       'img_resolution': 512, 'img_channels': 3}
	model_path = "/home/hzwu/Prj/PSP-EG3D/networks/ffhqrebalanced512-128.pth"

	from training.triplane import TriPlaneGenerator

	eg3d = TriPlaneGenerator(*init_args, **init_kwargs).eval().requires_grad_(False).to("cuda:0")
	print('Loading decoder weights from {}'.format(model_path))
	rendering_kwargs = {'depth_resolution': 96, 'depth_resolution_importance': 96, 'ray_start': 2.25,
            'ray_end': 3.3, 'box_warp': 1, 'avg_camera_radius': 2.7, 'avg_camera_pivot': [0, 0, 0.2],
            'image_resolution': 512, 'disparity_space_sampling': False, 'clamp_mode': 'softplus',
            'superresolution_module': 'training.superresolution.SuperresolutionHybrid8XDC',
            'c_gen_conditioning_zero': False, 'c_scale': 1.0, 'superresolution_noise_mode': 'none',
            'density_reg': 0.25, 'density_reg_p_dist': 0.004, 'reg_type': 'l1', 'decoder_lr_mul': 1.0,
            'sr_antialias': True}
	ckpt = torch.load(model_path)
	eg3d.load_state_dict(ckpt['G_ema'], strict=False)
	eg3d.neural_rendering_resolution = 128
	eg3d.rendering_kwargs = rendering_kwargs

	num = 2
	latent = latent.unsqueeze(0).to("cuda:0").repeat((num, 1, 1))
	print(latent.shape)
	camera = get_cameras(num).to("cuda:0")
	print(camera.shape)
	for idx in range(8):
		img_batch = eg3d.synthesis(latent, get_cameras(num).to("cuda:0"))["image"]
		img_batch = (img_batch.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
		for i in range(num):
			Image.fromarray(img_batch[i].cpu().numpy(), 'RGB').save("/home/hzwu/results/test_EG3D_camera" +  "/test_sample_{}.jpg".format(num*idx+i))

