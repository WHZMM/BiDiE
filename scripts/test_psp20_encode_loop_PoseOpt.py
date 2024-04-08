"""
for 20w++
encode: L2, masked LPIPS, id
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
from datasets.ffhq_encode_dataset import ValDataset
from criteria.lpips.lpips import LPIPS
from criteria import id_loss


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


def calc_loss_invert(models, photo_masked, photo_fake_masked):
    lpips, _, mse = models
    loss = 0.0
    loss_lpips = lpips(photo_masked, photo_fake_masked)
    loss += loss_lpips
    loss_l2 = mse(photo_masked, photo_fake_masked)
    loss += loss_l2
    loss_float = float(loss)
    return loss, loss_float, float(loss_l2), float(loss_lpips)


if __name__ == '__main__':
	""" settings """
	batch_size = 1
	encoder_weight_paths = list()
	print_list = list()
	# temp_list = sorted(glob("/home/hzwu/results/PSP-de-en-20-v3/200-800.pth"))
	# encoder_weight_paths += temp_list
	# temp_list = sorted(glob("/home/hzwu/results/PSP-hybrid-LossWOstd/*.pth"))
	# encoder_weight_paths += temp_list[2:]
	# temp_list = sorted(glob("/home/hzwu/results/PSP-hybrid-WA/*.pth"))
	# encoder_weight_paths += temp_list[:3]
	# temp_list = sorted(glob("/home/hzwu/results/PSP-hybrid-WP/*.pth"))
	# encoder_weight_paths += temp_list[:4]
	print(encoder_weight_paths)

	eg3d_weight_path = "/home/hzwu/Prj/PSP-EG3D/networks/ffhqrebalanced512-128.pth"
	latent_avg_path = "/home/hzwu/Prj/EG3D/eg3d/networks/ffhqrebalanced512-128-w_avg-524288x8192.npy"
	if_save_image = False
	if_save_latent = False
	if_reconstruction_loss = True
	max_l2 = 0.25
	max_l2 = 0.5625
	max_lpips = 0.8
	max_id = 0.9
	max_lec = 200
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
	eg3d = TriPlaneGenerator(*init_args, **init_kwargs).train().requires_grad_(True).to("cuda:0")
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
	latent_avg = np.load(latent_avg_path)  # shape (1, 512)
	latent_avg = torch.tensor(latent_avg, dtype=torch.float32, device="cuda:0", requires_grad=False)
	latent_avg = latent_avg.unsqueeze(0)  # shape (1, 1, 512)

	# dataset
	dataset = ValDataset()
	print("len(dataset):", len(dataset))
	face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))

	for encoder_weight_path in encoder_weight_paths:
		# load every loop
		ckpt = torch.load(encoder_weight_path)
		print("load encoder:", encoder_weight_path)
		encoder.load_state_dict(ckpt)

		if if_reconstruction_loss:
			total_mse = 0.0
			total_id = 0.0
			total_lpips = 0.0
			f_lpips_loss = LPIPS(net_type='alex').to("cuda:0").eval()
			f_id_loss = id_loss.IDLoss().to("cuda:0").eval()
			mse_loss = torch.nn.MSELoss().to("cuda:0").eval()

		total_num_1 = 0
		total_num_2 = 0
		for idx, data in enumerate(dataset):  # no mirror ver
			if idx >= len(dataset)/2:
				break

			flag_recon_error = False

			photo, _, mask, camera = data
			photo = photo.unsqueeze(0).to("cuda:0")
			camera = camera.unsqueeze(0).to("cuda:0")
			mask = mask.unsqueeze(0).to("cuda:0")
			
			photo_masked = photo * mask

			latent = encoder(photo) + latent_avg

			if if_save_latent:
				torch.save(latent, "/home/hzwu/results/val_result_latent/" + dataset.photo_paths[idx][-10:-4] + ".pt")
		
			image = eg3d.synthesis(latent, camera)['image']

			optimizer = torch.optim.Adam(list(camera), betas=(0.9, 0.999), lr=0.00005)
			tqdm_loop = tqdm(range(100))
			for i in tqdm_loop:
				photo_fake = face_pool(eg3d.synthesis(latent, camera)['image'])
				photo_fake_masked = photo_fake * mask
				loss, loss_float, l2_float, lpips_float = calc_loss_invert([f_lpips_loss, f_id_loss, mse_loss], photo_masked, photo_fake_masked)
				optimizer.zero_grad()
				eg3d.zero_grad()
				loss.backward()
				optimizer.step()
				tqdm_loop.set_postfix_str("Loss={:.4f}".format(loss_float))

			if if_reconstruction_loss:
				image_pooled = face_pool(image)
				image_pooled_masked = image_pooled * mask
				loss_id, _, _ = f_id_loss(image_pooled_masked, photo_masked)
				loss_l2 = torch.nn.functional.mse_loss(image_pooled_masked, photo_masked)
				loss_lpips = f_lpips_loss(image_pooled_masked, photo_masked)
				if (float(loss_l2) > max_l2 or float(loss_lpips) > max_lpips or float(loss_id) > max_id):
					print("[skip !!!!!], name={}".format(dataset.photo_paths[idx][-10:-4]))
					flag_recon_error = True
					save_path = "/home/hzwu/results/PSP-EG3D-encode20/val_result_img_error/" + dataset.photo_paths[idx][-10:-4] + ".jpg"
					img_out = Image.new("RGB", (1024, 512), )
					img_out.paste(Image.open(dataset.photo_paths[idx]), (0,0))
					image_PIL = (image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
					image_PIL = Image.fromarray(image_PIL[0].cpu().numpy(), 'RGB')
					img_out.paste(image_PIL, (512,0))
					img_out.save(save_path)
				else:
					total_mse += float(loss_l2)
					total_id += float(loss_id)
					total_lpips += float(loss_lpips)
					total_num_1 += 1
				
				print(encoder_weight_path, "avg loss:", total_mse/total_num_1, total_lpips/total_num_1, total_id/total_num_1)

		if if_reconstruction_loss:
			print_str = "l2 {} lpips {} id {}".format(total_mse/total_num_1, total_lpips/total_num_1, total_id/total_num_1)
			print(print_str)
			print_list.append(print_str)
		
		print("output all")
		print("~" * 32)
		for print_str in print_list:
			print(print_str)
		print("~" * 32)

