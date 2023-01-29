import logging
import argparse
import json
import os
import pandas as pd
from typing import List
import torch
from model.model import GPT, GPTConfig, save_gpt_config
from prior.trainer import Trainer, TrainerConfig
from utils.utils import set_random_seed
from utils.dataset import load_smiles_from_list, get_tensor_dataset, SmilesCharDictionary
from model.sampler import sample
import moses

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger.addHandler(logging.NullHandler())


def train(training_set: List[str], validation_set: List[str], output_dir, n_epochs=10, lr=1e-3, batch_size=64,
		  n_layer=8, n_embd=512, n_head=8, max_len=140, device='cpu', num_workers=1, seed=42):

	logger.info(f'Running device:\t{device}')
	device = torch.device(device)
	set_random_seed(seed, device)

	# load data
	train_seqs, _ = load_smiles_from_list(training_set, max_len=max_len)
	valid_seqs, _ = load_smiles_from_list(validation_set, max_len=max_len)

	train_set = get_tensor_dataset(train_seqs)
	test_set = get_tensor_dataset(valid_seqs)

	sd = SmilesCharDictionary()
	n_characters = sd.get_char_num()
	block_size = max_len + 2  # add start & end

	# build network
	mconf = GPTConfig(n_characters, block_size=block_size, n_layer=n_layer, n_head=n_head, n_embd=n_embd)
	model = GPT(mconf)
	save_gpt_config(mconf, output_dir, 'gpt_model_config')  # save config for later use

	# initialize a trainer instance and kick off training
	tconf = TrainerConfig(learning_rate=lr, lr_decay=True, warmup_tokens=0.1*len(train_set)*max_len,
						  final_tokens=n_epochs*len(train_set)*max_len, output_dir=output_dir)
	trainer = Trainer(model, tconf)
	trainer.fit(train_set, test_set,
				n_epochs=n_epochs, batch_size=batch_size, num_workers=num_workers, save_model=True)
	return trainer.model


def run_eval(model: GPT, output_dir, max_len=140):
	logger.info(f'Generate samples...')
	smiles = sample(model, num_to_sample=15000, device='cuda', batch_size=64, max_seq_length=max_len)
	logger.info(f'Evaluate on moses...')
	metrics = moses.get_all_metrics(smiles)
	logger.info(metrics)
	# Save smiles
	df_smiles = pd.DataFrame(smiles, columns=['smiles'])
	df_smiles.to_csv(output_dir + "sampled.smiles")
	logger.info(f'Evaluation finished!')


def main(args):

	df_train = pd.read_csv(args.train_data)
	df_valid = pd.read_csv(args.valid_data)

	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	with open(os.path.join(args.output_dir, 'commandline_args.json'), 'w') as f:
		json.dump(args.__dict__, f, indent=2)

	logger.info(f"Training prior model started, the results are saved in {args.output_dir}")
	model = train(training_set=df_train['SMILES'].tolist(), validation_set=df_valid['SMILES'].tolist(),
				  output_dir=args.output_dir, n_epochs=args.n_epochs, lr=args.lr, batch_size=args.batch_size,
				  n_layer=args.n_layers, n_embd=args.n_embd, n_head=args.n_head,
				  device=args.device, max_len=args.max_len)

	logger.info(f'Training done, the trained model is in {args.output_dir}')
	if args.eval:
		run_eval(model, args.output_dir, max_len=args.max_len)


def parse_args():

	parser = argparse.ArgumentParser(description='Distribution learning benchmark for SMILES RNN',
									 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--train_data', '-t', type=str, help='Full path to SMILES file containing training data')
	parser.add_argument('--valid_data', '-v', type=str, help='Full path to SMILES file containing validation data')
	parser.add_argument('--output_dir', '-o', type=str, help='Output directory')

	optional = parser.add_argument_group('Optional')
	optional.add_argument('--n_epochs', default=10, type=int, help='Number of training epochs')
	optional.add_argument('--lr', default=1e-3, type=float, help='RNN learning rate')
	optional.add_argument('--n_layers', default=8, type=int, help='Number of layers for training')
	optional.add_argument('--batch_size', default=512, type=int, help='Size of a mini-batch for gradient descent')
	optional.add_argument('--n_embd', default=512, type=int, help='Number of embeddings for GPT model')
	optional.add_argument('--n_head', default=8, type=int, help='Number of attention heads for GPT model')
	optional.add_argument('--device', default='cuda', type=str, help='Use cuda or cpu, default=cuda')
	optional.add_argument('--max_len', default=140, type=int, help='Max length of a SMILES string')
	optional.add_argument('--eval', action="store_true", help='Evaluate with moses or not, default False')
	return parser.parse_args()


if __name__ == '__main__':
	args = parse_args()
	main(args)
