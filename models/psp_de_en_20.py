"""
This file defines the core research contribution
encoder, 20 layer
"""
import matplotlib
matplotlib.use('Agg')
import math

import numpy as np
import torch
from torch import nn
# from models.encoders import psp_encoders
from models.encoders import psp_eg3d_encoders
# from models.stylegan2.model import Generator
# from training.triplane import TriPlaneGenerator
from training.triplane_enhanced21 import TriPlaneGenerator
from torch_utils import misc
from configs.paths_config import model_paths


def get_keys(d, name):
	if 'state_dict' in d:
		d = d['state_dict']
	d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
	return d_filt


class pSp_eg3d(nn.Module):

	def __init__(self, opts):
		super(pSp_eg3d, self).__init__()
		self.set_opts(opts)
		# compute number of style inputs based on the output resolution
		# self.opts.n_styles = int(math.log(self.opts.output_size, 2)) * 2 - 2
		self.opts.n_styles = 20
		# Define architecture
		self.encoder = self.set_encoder()
		# self.decoder = Generator(self.opts.output_size, 512, 8)

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
		self.decoder = TriPlaneGenerator(*init_args, **init_kwargs).eval().requires_grad_(False).to("cuda:0")
		self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
		self.load_weights()

	def set_encoder(self):
		if self.opts.encoder_type == 'GradualStyleEncoder':  # use this
			encoder = psp_eg3d_encoders.GradualStyleEncoder(50, 'ir_se', self.opts)
		elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoW':
			encoder = psp_eg3d_encoders.BackboneEncoderUsingLastLayerIntoW(50, 'ir_se', self.opts)
		elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoWPlus':
			encoder = psp_eg3d_encoders.BackboneEncoderUsingLastLayerIntoWPlus(50, 'ir_se', self.opts)
		else:
			raise Exception('{} is not a valid encoders'.format(self.opts.encoder_type))
		return encoder

	def load_weights(self):
		if self.opts.checkpoint_path is not None:  # load from checkpoints
			print('load from checkpoints: {}'.format(self.opts.checkpoint_path))
			ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
			self.encoder.load_state_dict(ckpt)
		else:
			print('Loading encoders weights from irse50!')
			encoder_ckpt = torch.load(model_paths['ir_se50'])

			# if input to encoder is not an RGB image, do not load the input layer weights
			if self.opts.dataset_type != 'ffhq_encode':
				encoder_ckpt = {k: v for k, v in encoder_ckpt.items() if "input_layer" not in k}
			self.encoder.load_state_dict(encoder_ckpt, strict=False)

		# load EG3D
		print('Loading decoder weights from pretrained!')
		rendering_kwargs = {'depth_resolution': 96, 'depth_resolution_importance': 96, 'ray_start': 2.25,
				'ray_end': 3.3, 'box_warp': 1, 'avg_camera_radius': 2.7, 'avg_camera_pivot': [0, 0, 0.2],
				'image_resolution': 512, 'disparity_space_sampling': False, 'clamp_mode': 'softplus',
				'superresolution_module': 'training.superresolution.SuperresolutionHybrid8XDC',
				'c_gen_conditioning_zero': False, 'c_scale': 1.0, 'superresolution_noise_mode': 'none',
				'density_reg': 0.25, 'density_reg_p_dist': 0.004, 'reg_type': 'l1', 'decoder_lr_mul': 1.0,
				'sr_antialias': True}
		ckpt = torch.load("/home/hzwu/networks/ffhqrebalanced512-128.pth")
		self.decoder.load_state_dict(ckpt['G_ema'], strict=False)
		self.decoder.neural_rendering_resolution = 128
		self.decoder.rendering_kwargs = rendering_kwargs
		# done

		if self.opts.learn_in_w:
			self.__load_latent_avg(repeat=1)
			raise UserWarning("Wrong latant space (W)")
		else:
			print("load avg latent...")
			self.__load_latent_avg(repeat=self.opts.n_styles)

	# origin forward
	def forward(self, x, c, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
	            inject_latent=None, return_latents=False, alpha=None):
		if input_code:
			codes = x
		else:
			codes = self.encoder(x)
			codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

		input_is_latent = not input_code
		images = self.decoder.synthesis(codes, c)['image']

		if resize:
			images = self.face_pool(images)

		return images, codes

	# De-En
	def forward_de_en(self, latent, cameras, resize=True):
		self.decoder.synthesis_part1(latent)  # gen TriPlane, part1
		images = self.decoder.synthesis_part2(cameras)['image']  # forward part 2, gen images
		if resize:
			images = self.face_pool(images)
		
		latents = self.encoder(images) + self.latent_avg
		
		return latents

	def set_opts(self, opts):
		self.opts = opts

	def __load_latent_avg(self, repeat=None):  # load
		latent_np = np.load("/home/hzwu/networks/ffhqrebalanced512-128-w_avg-524288x8192.npy")
		print("/home/hzwu/networks/ffhqrebalanced512-128-w_avg-524288x8192.npy")
		self.latent_avg = torch.FloatTensor(latent_np).to(self.opts.device)  # shape (1, 512)
		if repeat is not None:
			self.latent_avg = self.latent_avg.repeat(repeat, 1)  # shape (14, 512)
			self.latent_avg = self.latent_avg.unsqueeze(0)  # shape (1, 14, 512)


if __name__ == "__main__":
	print("print this line when test the code: models/psp_de_en_20.py文件debug时候会运行")
	encoder_ckpt = torch.load('/home/hzwu/Download/About_PSP/model_ir_se50.pth')
	# if input to encoder is not an RGB image, do not load the input layer weights
	print(encoder_ckpt.keys())
	
	


