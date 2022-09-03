import torch
from torch import nn
from models.networks import latent_transformer
from models.stylegan2.model import Generator
import numpy as np

def get_keys(d, name):
	if 'state_dict' in d:
		d = d['state_dict']
	d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
	return d_filt


class StyleGANControler(nn.Module):

	def __init__(self, opts):
		super(StyleGANControler, self).__init__()
		self.set_opts(opts)
		# Define architecture
		
		if 'ffhq' in self.opts.stylegan_weights:
			self.style_num = 18
		elif 'car' in self.opts.stylegan_weights:
			self.style_num = 16
		elif 'cat' in self.opts.stylegan_weights:
			self.style_num = 14
		elif 'church' in self.opts.stylegan_weights:
			self.style_num = 14
		elif 'anime' in self.opts.stylegan_weights:
			self.style_num = 16
		else:
			self.style_num = 18 #Please modify to adjust network architecture to your pre-trained StyleGAN2
		
		self.encoder = self.set_encoder()
		if self.style_num==18:
			self.decoder = Generator(1024, 512, 8, channel_multiplier=2) 
		elif self.style_num==16:
			self.decoder = Generator(512, 512, 8, channel_multiplier=2)
		elif self.style_num==14:
			self.decoder = Generator(256, 512, 8, channel_multiplier=2)
			
		self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
		
		# Load weights if needed
		self.load_weights()

	def set_encoder(self):
		encoder = latent_transformer.Network(self.opts)
		return encoder

	def load_weights(self):
		if self.opts.checkpoint_path is not None:
			print('Loading from checkpoint: {}'.format(self.opts.checkpoint_path))
			ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
			self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
			self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
			self.__load_latent_avg(ckpt)
		else:
			print('Loading decoder weights from pretrained!')
			ckpt = torch.load(self.opts.stylegan_weights)
			self.decoder.load_state_dict(ckpt['g_ema'], strict=True)
			self.__load_latent_avg(ckpt, repeat=self.opts.style_num)
		
	def set_opts(self, opts):
		self.opts = opts

	def __load_latent_avg(self, ckpt, repeat=None):
		if 'latent_avg' in ckpt:
			self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
			if repeat is not None:
				self.latent_avg = self.latent_avg.repeat(repeat, 1)
		else:
			self.latent_avg = None
