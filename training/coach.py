import os
import math, random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from utils import common
from criteria.lpips.lpips import LPIPS
from models.StyleGANControler import StyleGANControler
from training.ranger import Ranger

from expansion.submission import Expansion
from expansion.utils.flowlib import point_vec

class Coach:
	def __init__(self, opts):
		self.opts = opts
		if self.opts.checkpoint_path is None:
			self.global_step = 0
		else:
			self.global_step = int(os.path.splitext(os.path.basename(self.opts.checkpoint_path))[0].split('_')[-1])

		self.device = 'cuda:0'  # TODO: Allow multiple GPU? currently using CUDA_VISIBLE_DEVICES
		self.opts.device = self.device

		# Initialize network
		self.net = StyleGANControler(self.opts).to(self.device)

		# Initialize loss
		if self.opts.lpips_lambda > 0:
			self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
		self.mse_loss = nn.MSELoss().to(self.device).eval()

		# Initialize optimizer
		self.optimizer = self.configure_optimizers()

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
			
		# Initialize optical flow estimator
		self.ex = Expansion()
		
		# Set flow normalization values
		if 'ffhq' in self.opts.stylegan_weights:
			self.sigma_f = 4
			self.sigma_e = 0.02
		elif 'car' in self.opts.stylegan_weights:
			self.sigma_f = 5
			self.sigma_e = 0.03
		elif 'cat' in self.opts.stylegan_weights:
			self.sigma_f = 12
			self.sigma_e = 0.04
		elif 'church' in self.opts.stylegan_weights:
			self.sigma_f = 8
			self.sigma_e = 0.02
		elif 'anime' in self.opts.stylegan_weights:
			self.sigma_f = 7
			self.sigma_e = 0.025
			
	def train(self, truncation = 0.3, sigma = 0.1, target_layers = [0,1,2,3,4,5]):
		
		x = np.array(range(0,256,16)).astype(np.float32)/127.5-1.
		y = np.array(range(0,256,16)).astype(np.float32)/127.5-1.
		xx, yy = np.meshgrid(x,y)
		grid = np.concatenate([xx[:,:,None],yy[:,:,None]], axis=2)
		grid = torch.from_numpy(grid[None,:]).cuda()
		grid = grid.repeat(self.opts.batch_size,1,1,1)
    					
		while self.global_step < self.opts.max_steps:
			with torch.no_grad():
				z1 = torch.randn(self.opts.batch_size,512).to("cuda")		
				z2 = torch.randn(self.opts.batch_size,self.net.style_num, 512).to("cuda")
    			
			x1, w1, f1 = self.net.decoder([z1],input_is_latent=False,randomize_noise=False,return_feature_map=True,return_latents=True,truncation=truncation, truncation_latent=self.net.latent_avg[0])
			x1 = self.net.face_pool(x1)
			x2, w2 = self.net.decoder([z2],input_is_latent=False,randomize_noise=False,return_latents=True, truncation_latent=self.net.latent_avg[0])
			x2 = self.net.face_pool(x2)
			w_mid = w1.clone()
			w_mid[:,target_layers] = w_mid[:,target_layers]+sigma*(w2[:,target_layers]-w_mid[:,target_layers])
			x_mid, _ = self.net.decoder([w_mid], input_is_latent=True, randomize_noise=False, return_latents=False)
			x_mid = self.net.face_pool(x_mid)
			
			flow, logexp = self.ex.run(x1.detach(),x_mid.detach())		
			flow_feature = torch.cat([flow/self.sigma_f, logexp/self.sigma_e], dim=1)
			f1 = F.interpolate(f1, (flow_feature.shape[2:]))
			f1 = F.grid_sample(f1, grid, mode='nearest', align_corners=True)
			flow_feature = F.grid_sample(flow_feature, grid, mode='nearest', align_corners=True)
			flow_feature = flow_feature.view(flow_feature.shape[0], flow_feature.shape[1], -1).permute(0,2,1)
			f1 = f1.view(f1.shape[0], f1.shape[1], -1).permute(0,2,1)

			self.net.train()
			self.optimizer.zero_grad()
			w_hat = self.net.encoder(w1[:,target_layers].detach(), flow_feature.detach(), f1.detach())
			loss, loss_dict, id_logs = self.calc_loss(w_hat, w_mid[:,target_layers].detach())
			loss.backward()
			self.optimizer.step()
    			
			w_mid[:,target_layers] = w_hat.detach()
			x_hat, _ = self.net.decoder([w_mid], input_is_latent=True, randomize_noise=False)
			x_hat = self.net.face_pool(x_hat)
			if self.global_step % self.opts.image_interval == 0 or (
					self.global_step < 1000 and self.global_step % 100 == 0):
				imgL_o = ((x1.detach()+1.)*127.5)[0].permute(1,2,0).cpu().numpy()
				flow = torch.cat((flow,torch.ones_like(flow)[:,:1]), dim=1)[0].permute(1,2,0).cpu().numpy()
				flowvis = point_vec(imgL_o, flow)
				flowvis = torch.from_numpy(flowvis[:,:,::-1].copy()).permute(2,0,1).unsqueeze(0)/127.5-1.
				self.parse_and_log_images(None, flowvis, x_mid, x_hat, title='trained_images')
				print(loss_dict)
    				
			if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
				self.checkpoint_me(loss_dict, is_best=False)
    						
			if self.global_step == self.opts.max_steps:
				print('OMG, finished training!')
				break
    
			self.global_step += 1

	def checkpoint_me(self, loss_dict, is_best):
		save_name = 'best_model.pt' if is_best else 'iteration_{}.pt'.format(self.global_step)
		save_dict = self.__get_save_dict()
		checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
		torch.save(save_dict, checkpoint_path)
		with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
			if is_best:
				f.write('**Best**: Step - {}, Loss - {:.3f} \n{}\n'.format(self.global_step, self.best_val_loss, loss_dict))
			else:
				f.write('Step - {}, \n{}\n'.format(self.global_step, loss_dict))

	def configure_optimizers(self):
		params = list(self.net.encoder.parameters())
		if self.opts.train_decoder:
			params += list(self.net.decoder.parameters())
		if self.opts.optim_name == 'adam':
			optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
		else:
			optimizer = Ranger(params, lr=self.opts.learning_rate)
		return optimizer

	def calc_loss(self, latent, w, y_hat=None, y=None):
		loss_dict = {}
		loss = 0.0
		id_logs = None
		
		if self.opts.l2_lambda > 0 and (y_hat is not None) and (y is not None):
			loss_l2 = F.mse_loss(y_hat, y)
			loss_dict['loss_l2'] = float(loss_l2)
			loss += loss_l2 * self.opts.l2_lambda
		if self.opts.lpips_lambda > 0 and (y_hat is not None) and (y is not None):
			loss_lpips = self.lpips_loss(y_hat, y)
			loss_dict['loss_lpips'] = float(loss_lpips)
			loss += loss_lpips * self.opts.lpips_lambda
		if self.opts.l2latent_lambda > 0:
			loss_l2 = F.mse_loss(latent, w)
			loss_dict['loss_l2latent'] = float(loss_l2)
			loss += loss_l2 * self.opts.l2latent_lambda

		loss_dict['loss'] = float(loss)
		return loss, loss_dict, id_logs
		
	def parse_and_log_images(self, id_logs, x, y, y_hat, title, subscript=None, display_count=1):
		im_data = []
		for i in range(display_count):
			cur_im_data = {
				'input_face': common.tensor2im(x[i]),
				'target_face': common.tensor2im(y[i]),
				'output_face': common.tensor2im(y_hat[i]),
			}
			if id_logs is not None:
				for key in id_logs[i]:
					cur_im_data[key] = id_logs[i][key]
			im_data.append(cur_im_data)
		self.log_images(title, im_data=im_data, subscript=subscript)


	def log_images(self, name, im_data, subscript=None, log_latest=False):
		fig = common.vis_faces(im_data)
		step = self.global_step
		if log_latest:
			step = 0
		if subscript:
			path = os.path.join(self.logger.log_dir, name, '{}_{:04d}.jpg'.format(subscript, step))
		else:
			path = os.path.join(self.logger.log_dir, name, '{:04d}.jpg'.format(step))
		os.makedirs(os.path.dirname(path), exist_ok=True)
		fig.savefig(path)
		plt.close(fig)

	def __get_save_dict(self):
		save_dict = {
			'state_dict': self.net.state_dict(),
			'opts': vars(self.opts)
		}

		save_dict['latent_avg'] = self.net.latent_avg
		return save_dict