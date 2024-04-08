"""
This file runs the main training/val loop
refer to : scripts/train_encode_ffhq20.py

commands：
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=22239 scripts/train_hybrid.py --exp_dir=/home/hzwu/results/hybrid
"""

import os
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import sys
import pprint

sys.path.append(".")
sys.path.append("..")

from options.train_options_hybrid import TrainOptions
from training_psp.coach_hybrid import Coach


def main():
	opts = TrainOptions().parse()
	
	# multi-GPU
	opts.rank = int(os.environ["RANK"])
	opts.local_rank = int(os.environ["LOCAL_RANK"])
	torch.cuda.set_device(opts.rank % torch.cuda.device_count())
	dist.init_process_group(backend="nccl")
	opts.device = torch.device("cuda", opts.local_rank)
	opts.world_size = dist.get_world_size()
	print(f"[init] == local rank: {opts.local_rank}, global rank: {opts.rank} ==")

	# opts.exp_dir = opts.exp_dir + time.strftime("-%m%d%H%M", time.localtime(time.time()))
	print("Save path:", opts.exp_dir)  #
	if os.path.exists(opts.exp_dir):
		# raise Exception('Oops... {} already exists'.format(opts.exp_dir))
		pass
	else:
		os.makedirs(opts.exp_dir)

	opts_dict = vars(opts)
	pprint.pprint(opts_dict)

	coach = Coach(opts)
	coach.train()


if __name__ == '__main__':
	main()
