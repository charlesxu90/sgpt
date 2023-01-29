import os
import argparse
import logging
import json
from agent.agent_trainer import AgentTrainer
import time


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args):
    logger.info(f'device:\t{args.device}')
    logger.info('Training gpt agent started!')
    output_dir = args.output_dir + time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, 'commandline_args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if args.task == 'drd2':
        score_fns = ['Activity', 'SAScore', 'QED'] if args.mpo else ['Activity']
        weights = [3, 1, 1] if args.mpo else [1]
    elif args.task == 'ace2':
        score_fns = ['DockScore', 'SAScore', 'QED'] if args.mpo else ['DockScore']
        weights = [3, 1, 1] if args.mpo else [1]
    else:
        raise Exception("Task type not in ['drd2', 'ace2']")

    trainer = AgentTrainer(prior_path=args.prior, agent_path=args.agent, save_dir=output_dir, device=args.device,
                           learning_rate=args.lr, batch_size=args.batch_size, n_steps=args.n_steps, sigma=args.sigma,
                           experience_replay=args.er, max_seq_len=args.max_seq_len, score_fns=score_fns)
    trainer.train()
    logger.info(f"Training agent finished! Results saved to folder {output_dir}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prior', '-p', type=str, help='Path to prior checkpoint (.ckpt)')
    parser.add_argument('--agent', '-a', type=str, help='Path to agent checkpoint, likely prior (.ckpt)')
    parser.add_argument('--output_dir', '-o', type=str, help='Output directory')

    optional = parser.add_argument_group('Optional')
    optional.add_argument('--task', '-t', type=str, default='drd2', help='Task to run: drd2, ace2')
    optional.add_argument('--batch_size', type=int, default=64, help='Batch size (default is 64)')
    optional.add_argument('--n_steps', type=int, default=3000, help='Number of training steps (default is 3000)')
    optional.add_argument('--sigma', type=int, default=60, help='Sigma to update likelihood (default is 60)')
    optional.add_argument('--device', default='cuda', type=str, help='Use cuda or cpu, default=cuda')
    optional.add_argument('--lr', default=1e-4, type=float, help='Learning rate, default=1e-4')
    optional.add_argument('--er', action="store_true", help='Experience replay or not, default False')
    optional.add_argument('--mpo', action="store_true", help='Multiple properties or not, default False')
    optional.add_argument('--max_seq_len', default=100, type=int, help='Maximum sequence length, default=140')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
