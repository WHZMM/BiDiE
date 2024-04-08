"""
test multi-view consistency
On Multi-PIE dataset
"""

import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import json
import sys
import pprint
import time
import torch
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm
import torchvision.transforms as transforms

sys.path.append(".")
sys.path.append("..")

from options.train_options import TrainOptions
from models.encoders.psp_eg3d_encoders import GradualStyleEncoder as encoder_20wp
from training.triplane_enhanced21 import TriPlaneGenerator


def triplane_arg():
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
	return init_args, init_kwargs


def cal_lec(encoder, eg3d, face_pool, latent, camera, boundary):
	latent_edited = latent + boundary
	image_edited = face_pool(eg3d.synthesis(latent_edited, camera)['image'])
	latent_edited_en_de = encoder(image_edited) + latent_avg 
	latent_loop = latent_edited_en_de - boundary  
	lec = (latent - latent_loop).norm(2, dim=(1, 2)).mean().to("cpu").numpy()
	return lec


if __name__ == '__main__':
	""" settings """
	batch_size = 1
	encoder_weight_paths = list()
	print_list = list() 
	temp_list = sorted(glob("/home/hzwu/results/model-28-70.pth"))
	encoder_weight_paths += temp_list[:]
	# temp_list = sorted(glob("/home/hzwu/results/PSP-hybrid-WA/*.pth"))
	# encoder_weight_paths += temp_list[:3]
	# print(encoder_weight_paths)
	eg3d_weight_path = "/home/hzwu/Prj/PSP-EG3D/networks/ffhqrebalanced512-128.pth"
	latent_avg_path = "/home/hzwu/Prj/EG3D/eg3d/networks/ffhqrebalanced512-128-w_avg-524288x8192.npy"
	max_std = 9999
	max_group = 100  # max, 7536!
	""" settings """

	# encoder
	opts = TrainOptions().parse()
	opts.encoder_type="GradualStyleEncoder"
	opts.input_nc = 3
	opts.label_nc = 3
	opts.n_styles = 20
	opts.start_from_latent_avg = True
	encoder = encoder_20wp(50, "ir_se", opts).eval().requires_grad_(False).to("cuda:0")
	init_args, init_kwargs = triplane_arg()
	# decoder(eg3d)
	eg3d = TriPlaneGenerator(*init_args, **init_kwargs).eval().requires_grad_(False).to("cuda:0")
	rendering_kwargs = {'depth_resolution': 96, 'depth_resolution_importance': 96, 'ray_start': 2.25,
			'ray_end': 3.3, 'box_warp': 1, 'avg_camera_radius': 2.7, 'avg_camera_pivot': [0, 0, 0.2],
			'image_resolution': 512, 'disparity_space_sampling': False, 'clamp_mode': 'softplus',
			'superresolution_module': 'training.superresolution21.SuperresolutionHybrid8XDC',
			'c_gen_conditioning_zero': False, 'c_scale': 1.0, 'superresolution_noise_mode': 'none',
			'density_reg': 0.25, 'density_reg_p_dist': 0.004, 'reg_type': 'l1', 'decoder_lr_mul': 1.0,
			'sr_antialias': True}
	ckpt = torch.load(eg3d_weight_path)
	eg3d.load_state_dict(ckpt['G_ema'], strict=False)
	eg3d.neural_rendering_resolution = 128
	eg3d.rendering_kwargs = rendering_kwargs
	# avg latent
	latent_avg = np.load(latent_avg_path)
	latent_avg = torch.tensor(latent_avg, dtype=torch.float32, device="cuda:0", requires_grad=False)
	latent_avg = latent_avg.unsqueeze(0)  # shape (1, 1, 512)

	# dataset
	data_root = "/home/hzwu/data_using/multiPIE_WHZseleted_EG3Dcrop"
	dict_camera = {0: "11_0", 1: "12_0", 2: "09_0", 3: "08_0", 4: "13_0", 5: "14_0", 6: "05_1", 7: "05_0", 8: "04_1", 9: "19_0", 10: "20_0", 11: "01_0", 12: "24_0", 13: "08_1", 14: "19_1"}
	groups = os.listdir(data_root)  # len=7536
	photo_transform = transforms.Compose([
		transforms.Resize((256, 256)),
		transforms.ToTensor(),
		transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

	for encoder_weight_path in encoder_weight_paths:
		# load every loop
		ckpt = torch.load(encoder_weight_path)
		print("load encoder:", encoder_weight_path)
		encoder.load_state_dict(ckpt)

		with torch.no_grad():
			std_sum = 0.0
			for idx_group in tqdm(range(len(groups))):
				if idx_group >= max_group:
					print("reach max {}, stop.".format(max_group))
					break
				group_name = groups[idx_group]
				triplane_list = list()
				for idx_cam in range(2, 11):
					img_name = "Cam{}.jpg".format(dict_camera[idx_cam])
					img_path = data_root + "/" + group_name + "/" + img_name
					img = Image.open(img_path).convert("RGB")
					img = photo_transform(img).to("cuda:0").unsqueeze(0)
					latent = encoder(img) + latent_avg 		
					image = eg3d.synthesis_part1(latent)
					triplane = eg3d._last_planes.clone()
					triplane_list.append(triplane)
				triplane_cat = torch.cat(triplane_list, dim=0, )
				triplane_std = torch.std(triplane_cat, dim=0, )
				triplane_std = torch.mean(triplane_std)
				std_sum += float(triplane_std)

		print_str = "triplane_std = {}\t{}".format(std_sum/idx_group, encoder_weight_path)
		print(print_str)
		print_list.append(print_str)
		
		print("every output")
		print("~" * 32)
		for print_str in print_list:
			print(print_str)
		print("~" * 32)

