import os
import matplotlib
import matplotlib.pyplot as plt
import sys
matplotlib.use('Agg')

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.autograd import Variable

from utils import common, train_utils
from criteria import id_loss, w_norm, moco_loss
from configs import data_configs
from datasets.images_dataset import ImagesDataset
from criteria.lpips.lpips import LPIPS
from models.psp import pSp
from training.ranger import Ranger
from torchvision.utils import save_image


class Coach:
	def __init__(self, opts):
		self.opts = opts

		self.global_step = 0

		self.device = 'cuda:0'  # TODO: Allow multiple GPU? currently using CUDA_VISIBLE_DEVICES
		self.opts.device = self.device

		# Initialize network
		self.net = pSp(self.opts).to(self.device)

		# Estimate latent_avg via dense sampling if latent_avg is not available
		if self.net.latent_avg is None:
			self.net.latent_avg = self.net.decoder.mean_latent(int(1e5))[0].detach()

		# Initialize loss
		if self.opts.id_lambda > 0 and self.opts.moco_lambda > 0:
			raise ValueError('Both ID and MoCo loss have lambdas > 0! Please select only one to have non-zero lambda!')

		self.mse_loss = nn.MSELoss().to(self.device).eval()
		if self.opts.lpips_lambda > 0:
			self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
		if self.opts.id_lambda > 0:
			self.id_loss = id_loss.IDLoss().to(self.device).eval()
		if self.opts.w_norm_lambda > 0:
			self.w_norm_loss = w_norm.WNormLoss(start_from_latent_avg=self.opts.start_from_latent_avg)
		if self.opts.moco_lambda > 0:
			self.moco_loss = moco_loss.MocoLoss().to(self.device).eval()

		# Initialize optimizer
		self.enc_optim, self.dec_optim = self.configure_optimizers()

		# Initialize dataset
		self.train_dataset, self.test_dataset = self.configure_datasets()
		self.train_dataloader = DataLoader(self.train_dataset,
										   batch_size=self.opts.batch_size,
										   shuffle=True,
										   num_workers=int(self.opts.workers),
										   drop_last=True)
		self.test_dataloader = DataLoader(self.test_dataset,
										  batch_size=self.opts.test_batch_size,
										  shuffle=False,
										  num_workers=int(self.opts.test_workers),
										  drop_last=True)

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

	def train(self):
		self.net.train()
		while self.global_step < self.opts.max_steps:
			for batch_idx, batch in enumerate(self.train_dataloader):



				x, y = batch
				x, y = x.to(self.device).float(), y.to(self.device).float()
				
				## VAE
				vae__=False
				if vae__:
					y_hat, latent = self.net.forward(x, return_latents=True)
					loss, loss_dict, id_logs = self.calc_loss(x, y, y_hat, latent)
					self.enc_optim.zero_grad(); self.dec_optim.zero_grad()
					loss.backward()

					nn.utils.clip_grad_norm_(self.net.encoder.parameters(), max_norm=1)
					nn.utils.clip_grad_norm_(self.net.decoder.parameters(), max_norm=1)
					self.enc_optim.step(); self.dec_optim.step()

				else:
					id_logs = None
					latent = torch.randn(4,10,512)

				## ENCODER UPDATE (AS DISCRIMINATOR)
				b, w, l = latent.size()
				with torch.no_grad():
					code = torch.randn(b,l).to(self.device)
					_, latent_input = self.net(code, input_code=True, return_latents=True)
					fake = self.net(latent_input, skip_encoder=True)
				fake_out = self.net(fake, skip_decoder=True)
				real_out = self.net(x, skip_decoder=True)

				b, w, l = fake_out.size()
				f_loss = fake_out[:,:,l//2:] + 0 * (1.0 / (2.0 * fake_out[:,:,l//2:].exp().pow(2.0) + 1e-5)) * (latent_input - fake_out[:,:,:l//2]).pow(2.0)
				r_loss = 0.5 * torch.mean(real_out[:, :, l//2:].exp() - real_out[:,:,l//2:] + real_out[:, :, :l//2].pow(2) - 1)
				dis_loss = torch.mean(f_loss + r_loss)
				self.enc_optim.zero_grad(); self.dec_optim.zero_grad()
				dis_loss.backward()
				nn.utils.clip_grad_norm_(self.net.encoder.parameters(), max_norm=1)
				self.enc_optim.step()

				# ## DECODER UPDATE (AS GENERATOR)
				b, w, l = latent.size()
				code = torch.randn(b,l).to(self.device)
				_, latent_input = self.net(code, input_code=True, return_latents=True)
				sample = self.net(latent_input, skip_encoder=True)
				sample_out = self.net(sample, skip_decoder=True)
				b,w,l = sample_out.size()
				adv_loss = 0.5 * torch.mean(sample_out[:, :, l//2:].exp() - sample_out[:,:,l//2:] + sample_out[:, :, :l//2].pow(2) - 1)
				adv_loss = torch.mean(adv_loss)
				self.enc_optim.zero_grad(); self.dec_optim.zero_grad()
				adv_loss.backward()
				nn.utils.clip_grad_norm_(self.net.decoder.parameters(), max_norm=1)
				self.dec_optim.step()

				# Logging related
				if self.global_step % self.opts.image_interval == 0 or (
						self.global_step < 1000 and self.global_step % 25 == 0):
					with torch.no_grad():
						codes = torch.randn(latent.size()).to(self.device)
						y_sample = self.net.forward(codes, input_code=True)
					if not vae__:
						y_hat = y_sample
					save_image(torch.cat([x, y, y_hat, y_sample]), f'images/train/faces/{batch_idx}', nrow=4, normalize=True, scale_each=True, pad_value=128, padding=1)
					# self.parse_and_log_images(id_logs, x, y, y_hat, y_sample, title='images/train/faces')
				if self.global_step % self.opts.board_interval == 0:
					if vae__:
						self.print_metrics(loss_dict, prefix='train')
						self.log_metrics(loss_dict, prefix='train')
					print(f"DIS LOSS: F: {torch.mean(f_loss).item()}, R: {torch.mean(r_loss).item()}")
					print("ADV LOSS: ", adv_loss.item())

				# Validation related
				val_loss_dict = None
				if self.opts.val_interval > 0:
					if self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:
						val_loss_dict = self.validate()
						if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss):
							self.best_val_loss = val_loss_dict['loss']
							self.checkpoint_me(val_loss_dict, is_best=True)

				# if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
				# 	if val_loss_dict is not None:
				# 		self.checkpoint_me(val_loss_dict, is_best=False)
				# 	else:
				# 		self.checkpoint_me(loss_dict, is_best=False)

				if self.global_step == self.opts.max_steps:
					print('OMG, finished training!')
					break

				self.global_step += 1

	def validate(self):
		self.net.eval()
		agg_loss_dict = []
		for batch_idx, batch in enumerate(self.test_dataloader):
			x, y = batch

			with torch.no_grad():
				x, y = x.to(self.device).float(), y.to(self.device).float()
				y_hat, latent = self.net.forward(x, return_latents=True)
				codes = torch.randn(latent.size()).to(self.device)
				y_sample = self.net.forward(codes, input_code=True)
				loss, cur_loss_dict, id_logs = self.calc_loss(x, y, y_hat, latent)
			agg_loss_dict.append(cur_loss_dict)

			# Logging related
			self.parse_and_log_images(id_logs, x, y, y_hat, y_sample,
									  title='images/test/faces',
									  subscript='{:04d}'.format(batch_idx))

			# For first step just do sanity test on small amount of data
			if self.global_step == 0 and batch_idx >= 4:
				self.net.train()
				return None  # Do not log, inaccurate in first batch

		loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
		self.log_metrics(loss_dict, prefix='test')
		self.print_metrics(loss_dict, prefix='test')

		self.net.train()
		return loss_dict

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
		params_enc = list(self.net.encoder.parameters())
		params_dec = list(self.net.decoder.parameters())

		enc_optim = torch.optim.Adam(params_enc, lr=self.opts.learning_rate)
		dec_optim = torch.optim.Adam(params_dec, lr=0.self.opts.learning_rate)
		return enc_optim, dec_optim

	def configure_datasets(self):
		if self.opts.dataset_type not in data_configs.DATASETS.keys():
			Exception('{} is not a valid dataset_type'.format(self.opts.dataset_type))
		print('Loading dataset for {}'.format(self.opts.dataset_type))
		dataset_args = data_configs.DATASETS[self.opts.dataset_type]
		transforms_dict = dataset_args['transforms'](self.opts).get_transforms()
		train_dataset = ImagesDataset(source_root=dataset_args['train_source_root'],
									  target_root=dataset_args['train_target_root'],
									  source_transform=transforms_dict['transform_source'],
									  target_transform=transforms_dict['transform_gt_train'],
									  opts=self.opts)
		test_dataset = ImagesDataset(source_root=dataset_args['test_source_root'],
									 target_root=dataset_args['test_target_root'],
									 source_transform=transforms_dict['transform_source'],
									 target_transform=transforms_dict['transform_test'],
									 opts=self.opts)
		print("Number of training samples: {}".format(len(train_dataset)))
		print("Number of test samples: {}".format(len(test_dataset)))
		return train_dataset, test_dataset

	def calc_loss(self, x, y, y_hat, latent):
		loss_dict = {}
		loss = 0.0
		id_logs = None

		if self.opts.kl_lambda > 0:
			b, w, l = latent.size()
			kl_loss = 0.5 * torch.mean(latent[:, :, l//2:].exp() - latent[:,:,l//2:] + latent[:, :, :l//2].pow(2) - 1)

			loss_dict['kl'] = float(kl_loss)
			loss += kl_loss * self.opts.kl_lambda
		if self.opts.id_lambda > 0:
			loss_id, sim_improvement, id_logs = self.id_loss(y_hat, y, x)
			loss_dict['loss_id'] = float(loss_id)
			loss_dict['id_improve'] = float(sim_improvement)
			loss += loss_id * self.opts.id_lambda
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
			loss_w_norm = self.w_norm_loss(latent, self.net.latent_avg)
			loss_dict['loss_w_norm'] = float(loss_w_norm)
			loss += loss_w_norm * self.opts.w_norm_lambda
		if self.opts.moco_lambda > 0:
			loss_moco, sim_improvement, id_logs = self.moco_loss(y_hat, y, x)
			loss_dict['loss_moco'] = float(loss_moco)
			loss_dict['id_improve'] = float(sim_improvement)
			loss += loss_moco * self.opts.moco_lambda

		loss_dict['loss'] = float(loss)
		return loss, loss_dict, id_logs

	def log_metrics(self, metrics_dict, prefix):
		for key, value in metrics_dict.items():
			self.logger.add_scalar('{}/{}'.format(prefix, key), value, self.global_step)

	def print_metrics(self, metrics_dict, prefix):
		print('Metrics for {}, step {}'.format(prefix, self.global_step))
		for key, value in metrics_dict.items():
			print('\t{} = '.format(key), value)

	def parse_and_log_images(self, id_logs, x, y, y_hat, y_sample, title, subscript=None, display_count=2):
		im_data = []
		for i in range(display_count):
			cur_im_data = {
				'input_face': common.log_input_image(x[i], self.opts),
				'target_face': common.tensor2im(y[i]),
				'output_face': common.tensor2im(y_hat[i]),
				'sample_face': common.tensor2im(y_sample[i])
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
		# save the latent avg in state_dict for inference if truncation of w was used during training
		if self.opts.start_from_latent_avg:
			save_dict['latent_avg'] = self.net.latent_avg
		return save_dict
