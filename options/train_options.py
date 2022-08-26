from argparse import ArgumentParser

class TrainOptions:

	def __init__(self):
		self.parser = ArgumentParser()
		self.initialize()

	def initialize(self):
		self.parser.add_argument('--exp_dir', type=str, help='Path to experiment output directory')

		self.parser.add_argument('--batch_size', default=1, type=int, help='Batch size for training')
		self.parser.add_argument('--learning_rate', default=0.001, type=float, help='Optimizer learning rate')
		self.parser.add_argument('--optim_name', default='ranger', type=str, help='Which optimizer to use')
		self.parser.add_argument('--train_decoder', default=False, type=bool, help='Whether to train the decoder model')

		self.parser.add_argument('--lpips_lambda', default=0., type=float, help='LPIPS loss multiplier factor')
		self.parser.add_argument('--l2_lambda', default=0, type=float, help='L2 loss multiplier factor')
		self.parser.add_argument('--l2latent_lambda', default=1.0, type=float, help='L2 loss multiplier factor')

		self.parser.add_argument('--stylegan_weights', default='pretrained_models/stylegan2-cat-config-f.pt', type=str, help='Path to StyleGAN model weights')
		self.parser.add_argument('--checkpoint_path', default=None, type=str, help='Path to pSp model checkpoint')

		self.parser.add_argument('--max_steps', default=60100, type=int, help='Maximum number of training steps')
		self.parser.add_argument('--image_interval', default=100, type=int, help='Interval for logging train images during training')
		self.parser.add_argument('--save_interval', default=10000, type=int, help='Model checkpoint interval')

		self.parser.add_argument('--style_num', default=14, type=int, help='The number of StyleGAN layers get latent codes ')
		self.parser.add_argument('--channel_multiplier', default=2, type=int, help='StyleGAN parameter')

	def parse(self):
		opts = self.parser.parse_args()
		return opts