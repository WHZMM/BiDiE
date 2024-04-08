"""
for 20w++
L2, masked LPIPS, id
LEC (from E4E)
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


def cal_lec(encoder, eg3d, face_pool, latent, camera, boundary):
	latent_edited = latent + boundary 
	image_edited = face_pool(eg3d.synthesis(latent_edited, camera)['image'])
	latent_edited_en_de = encoder(image_edited) + latent_avg 
	latent_loop = latent_edited_en_de - boundary 
	lec = (latent - latent_loop).norm(2, dim=(1, 2)).mean().to("cpu").numpy()
	return lec


def cal_lec_every_latent(encoder, eg3d, face_pool, latent, camera, boundary):
	latent_edited = latent + boundary 
	image_edited = face_pool(eg3d.synthesis(latent_edited, camera)['image']) 
	latent_edited_en_de = encoder(image_edited) + latent_avg  
	latent_loop = latent_edited_en_de - boundary  
	lec = (latent - latent_loop).norm(2, dim=(2, )).mean().to("cpu").numpy() 
	return lec


if __name__ == '__main__':
	""" settings """
	batch_size = 1
	encoder_weight_paths = list()
	print_list = list()
	temp_list = sorted(glob("/home/hzwu/results/0408_2_B.pth"))
	encoder_weight_paths += temp_list
	# temp_list = sorted(glob("/home/hzwu/results//*.pth"))
	# encoder_weight_paths += temp_list[2:]
	print(encoder_weight_paths)

	eg3d_weight_path = "/home/hzwu/Prj/PSP-EG3D/networks/ffhqrebalanced512-128.pth"
	latent_avg_path = "/home/hzwu/Prj/EG3D/eg3d/networks/ffhqrebalanced512-128-w_avg-524288x8192.npy" 
	if_save_image = False 
	if_save_latent = False  
	if_reconstruction_loss = True  
	if_lec = True  
	max_l2 = 0.25 
	max_l2 = 0.5625  
	max_lpips = 0.8 
	max_id = 0.9  
	max_lec = 200 
	""" settings """

	opts = TrainOptions().parse()
	opts.encoder_type="GradualStyleEncoder"
	opts.input_nc = 3 
	opts.label_nc = 3
	opts.n_styles = 20  
	opts.start_from_latent_avg = True
	encoder = encoder_20wp(50, "ir_se", opts).eval().requires_grad_(False).to("cuda:0")
	init_args, init_kwargs = triplane_arg()
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
	latent_avg = np.load(latent_avg_path) 
	latent_avg = torch.tensor(latent_avg, dtype=torch.float32, device="cuda:0", requires_grad=False)
	latent_avg = latent_avg.unsqueeze(0)  

	dataset = ValDataset()
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
		if if_lec:
			total_lec_male = 0.0
			total_lec_female = 0.0
			total_lec_old = 0.0
			total_lec_young = 0.0
			boundary_old = np.load("/home/hzwu/results/interfaceGAN-EG3D/ffhqrebalanced512-128/age_boundary.npy")
			boundary_male = np.load("/home/hzwu/results/interfaceGAN-EG3D/ffhqrebalanced512-128/gender_boundary.npy")

		total_num_1 = 0
		total_num_2 = 0
		with torch.no_grad():
			for idx, data in tqdm(enumerate(dataset), total=len(dataset)/2):  # no mirror ver

				flag_recon_error = False 

				photo, _, mask, camera = data
				photo = photo.unsqueeze(0).to("cuda:0")
				camera = camera.unsqueeze(0).to("cuda:0")
				mask = mask.unsqueeze(0).to("cuda:0")

				latent = encoder(photo) + latent_avg 

				if if_save_latent: 
					torch.save(latent, "/home/hzwu/results/val_result_latent/" + dataset.photo_paths[idx][-10:-4] + ".pt")
			
				image = eg3d.synthesis(latent, camera)['image']

				if if_reconstruction_loss:
					image_pooled = face_pool(image)  
					image_pooled_masked = image_pooled * mask
					photo_masked = photo * mask
					loss_id, _, _ = f_id_loss(image_pooled_masked, photo_masked)
					loss_l2 = torch.nn.functional.mse_loss(image_pooled_masked, photo_masked)
					loss_lpips = f_lpips_loss(image_pooled_masked, photo_masked)
					if (float(loss_l2) > max_l2 or float(loss_lpips) > max_lpips or float(loss_id) > max_id):
						print("[skip !!!], name={}".format(dataset.photo_paths[idx][-10:-4]))
						flag_recon_error = True
						save_path = "/home/hzwu/results/val_result_img_error/" + dataset.photo_paths[idx][-10:-4] + ".jpg"
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

				if if_save_image and (not flag_recon_error): 
					save_path = "/home/hzwu/results/val_result_img/" + dataset.photo_paths[idx][-10:-4] + ".jpg"
					image_save = (image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
					image_save = Image.fromarray(image_save[0].cpu().numpy(), 'RGB')
					image_save.save(save_path)

				if if_lec and (not flag_recon_error): 
					# male -> female -> male
					tensor_male = torch.tensor(boundary_male).reshape((1, 1, -1)).to("cuda:0")  # (1, 1, 512)
					lec_male = cal_lec_every_latent(encoder, eg3d, face_pool, latent, camera, tensor_male)
					# female -> male -> female
					tensor_female = torch.tensor(0.0 - boundary_male).reshape((1, 1, -1)).to("cuda:0")  # (1, 1, 512)
					lec_female = cal_lec_every_latent(encoder, eg3d, face_pool, latent, camera, tensor_female)
					if (lec_female > max_lec):
						print("[MAYBE_ERROR], idx={}, loss_lec_female={}, name={}".format(idx, lec_female, dataset.photo_paths[idx][-10:-4]))
					# old -> young -> old
					tensor_old = torch.tensor(boundary_old).reshape((1, 1, -1)).to("cuda:0")  # (1, 1, 512)
					lec_old = cal_lec_every_latent(encoder, eg3d, face_pool, latent, camera, tensor_old)
					# young -> old -> young
					tensor_young = torch.tensor(0.0 - boundary_old).reshape((1, 1, -1)).to("cuda:0")  # (1, 1, 512)
					lec_young = cal_lec_every_latent(encoder, eg3d, face_pool, latent, camera, tensor_young)
					# lec
					if (lec_male > max_lec or lec_female > max_lec or lec_old > max_lec or lec_young > max_lec):  # recon skip!
						print("[error],skip,name={}".format(dataset.photo_paths[idx][-10:-4]))
						print("[error],value:{:.4f},{:.4f},{:.4f},{:.4f}".format(lec_male, lec_female, lec_old, lec_young))
					else:
						total_lec_male += lec_male
						total_lec_female += lec_female
						total_lec_old += lec_old
						total_lec_young += lec_young
						total_num_2 += 1

		if if_reconstruction_loss:
			print_str = "l2 {} lpips {} id {}".format(total_mse/total_num_1, total_lpips/total_num_1, total_id/total_num_1)
			print(print_str)
			print_list.append(print_str)
		if if_lec:
			print_str = "{} {} {} {}".format(total_lec_male/total_num_2, total_lec_female/total_num_2, total_lec_old/total_num_2, total_lec_young/total_num_2)
			print_str = encoder_weight_path + " " + print_str
			print(print_str)
			print_list.append(print_str)
		
		print("ouput all")
		print("~" * 32)
		for print_str in print_list:
			print(print_str)
		print("~" * 32)

