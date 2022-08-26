import os
from argparse import Namespace
import numpy as np
import torch
import sys

sys.path.append(".")
sys.path.append("..")

from models.StyleGANControler import StyleGANControler

class demo():
	
	def __init__(self, checkpoint_path, truncation = 0.5, use_average_code_as_input = False):
		self.truncation = truncation
		self.use_average_code_as_input = use_average_code_as_input
		ckpt = torch.load(checkpoint_path, map_location='cpu')
		opts = ckpt['opts']
		opts['checkpoint_path'] = checkpoint_path
		self.opts = Namespace(**ckpt['opts'])
		
		self.net = StyleGANControler(self.opts)
		self.net.eval()
		self.net.cuda()
		self.target_layers = [0,1,2,3,4,5]
		
		self.w1 = None
		self.w1_after = None
		self.f1 = None

	def run(self):
		z1 = torch.randn(1,512).to("cuda")
		x1, self.w1, self.f1 = self.net.decoder([z1],input_is_latent=False,randomize_noise=False,return_feature_map=True,return_latents=True,truncation=self.truncation, truncation_latent=self.net.latent_avg[0])
		self.w1_after = self.w1.clone()
		x1 = self.net.face_pool(x1)
		result = ((x1.detach()[0].permute(1,2,0)+1.)*127.5).cpu().numpy()[:,:,::-1]
		return result
	
	def translate(self, dxy, sxsy=[0,0], stop_points=[], zoom_in=False, zoom_out=False):
		dz = -5. if zoom_in else 0.
		dz = 5. if zoom_out else dz
			
		dxyz = np.array([dxy[0],dxy[1],dz], dtype=np.float32)
		dxy_norm = np.linalg.norm(dxyz[:2], ord=2)
		dxyz[:2] = dxyz[:2]/dxy_norm
		vec_num = dxy_norm/10
		
		x = torch.from_numpy(np.array([[dxyz]],dtype=np.float32)).cuda()
		f1 = torch.nn.functional.interpolate(self.f1, (256,256))
		y = f1[:,:,sxsy[1],sxsy[0]].unsqueeze(0)

		if len(stop_points)>0:
			x = torch.cat([x, torch.zeros(x.shape[0],len(stop_points),x.shape[2]).cuda()], dim=1)
			tmp = []
			for sp in stop_points:
				tmp.append(f1[:,:,sp[1],sp[0]].unsqueeze(1))
			y = torch.cat([y,torch.cat(tmp, dim=1)],dim=1)

		if not self.use_average_code_as_input:
			w_hat = self.net.encoder(self.w1[:,self.target_layers].detach(), x.detach(), y.detach(), alpha=vec_num)
			w1 = self.w1.clone()
			w1[:,self.target_layers] = w_hat
		else:
			w_hat = self.net.encoder(self.net.latent_avg.unsqueeze(0)[:,self.target_layers].detach(), x.detach(), y.detach(), alpha=vec_num)
			w1 = self.w1.clone()
			w1[:,self.target_layers]  = self.w1.clone()[:,self.target_layers]  + w_hat - self.net.latent_avg.unsqueeze(0)[:,self.target_layers]

		x1, _ = self.net.decoder([w1], input_is_latent=True, randomize_noise=False)
		
		self.w1_after = w1.clone()
		x1 = self.net.face_pool(x1)
		result = ((x1.detach()[0].permute(1,2,0)+1.)*127.5).cpu().numpy()[:,:,::-1]
		return result
	
	def zoom(self, dz, sxsy=[0,0], stop_points=[]):
		vec_num = abs(dz)/5
		dz = 100*np.sign(dz)
		x = torch.from_numpy(np.array([[[1.,0,dz]]],dtype=np.float32)).cuda()
		f1 = torch.nn.functional.interpolate(self.f1, (256,256))
		y = f1[:,:,sxsy[1],sxsy[0]].unsqueeze(0)

		if len(stop_points)>0:
			x = torch.cat([x, torch.zeros(x.shape[0],len(stop_points),x.shape[2]).cuda()], dim=1)
			tmp = []
			for sp in stop_points:
				tmp.append(f1[:,:,sp[1],sp[0]].unsqueeze(1))
			y = torch.cat([y,torch.cat(tmp, dim=1)],dim=1)
			
		if not self.use_average_code_as_input:
			w_hat = self.net.encoder(self.w1[:,self.target_layers].detach(), x.detach(), y.detach(), alpha=vec_num)
			w1 = self.w1.clone()
			w1[:,self.target_layers] = w_hat
		else:
			w_hat = self.net.encoder(self.net.latent_avg.unsqueeze(0)[:,self.target_layers].detach(), x.detach(), y.detach(), alpha=vec_num)
			w1 = self.w1.clone()
			w1[:,self.target_layers]  = self.w1.clone()[:,self.target_layers]  + w_hat - self.net.latent_avg.unsqueeze(0)[:,self.target_layers]
		
		
		x1, _ = self.net.decoder([w1], input_is_latent=True, randomize_noise=False)
		
		x1 = self.net.face_pool(x1)
		result = ((x1.detach()[0].permute(1,2,0)+1.)*127.5).cpu().numpy()[:,:,::-1]
		return result
	
	def change_style(self):
		z1 = torch.randn(1,512).to("cuda")
		x1, w2 = self.net.decoder([z1],input_is_latent=False,randomize_noise=False,return_latents=True, truncation=self.truncation, truncation_latent=self.net.latent_avg[0])
		self.w1_after[:,6:] = w2.detach()[:,0]
		x1, _ = self.net.decoder([self.w1_after], input_is_latent=True, randomize_noise=False, return_latents=False)
		result = ((x1.detach()[0].permute(1,2,0)+1.)*127.5).cpu().numpy()[:,:,::-1]
		return result
	
	def reset(self):
		x1, _ = self.net.decoder([self.w1], input_is_latent=True, randomize_noise=False, return_latents=False)
		result = ((x1.detach()[0].permute(1,2,0)+1.)*127.5).cpu().numpy()[:,:,::-1]
		return result
		