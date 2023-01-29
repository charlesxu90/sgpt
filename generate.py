import argparse
import logging
from pathlib import Path
import pandas as pd
from model.model import load_gpt_model
from model.sampler import sample
import moses


def main(args):
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f'device:\t{args.device}')

    gpt_path = args.model_path
    out_path = args.out_file

    model_def = Path(gpt_path).with_suffix('.json')
    model = load_gpt_model(model_def, gpt_path, args.device, copy_to_cpu=True)

    logger.info(f'Generate samples...')
    smiles = sample(model, num_to_sample=args.num_to_sample, device=args.device, batch_size=args.batch_size,
                    max_seq_length=args.max_seq_length, temperature=args.temperature)
    if args.eval:
        logger.info(f'Evaluate on moses...')
        metrics = moses.get_all_metrics(smiles)
        logger.info(metrics)

    # Save smiles
    df_smiles = pd.DataFrame(smiles, columns=['smiles'])
    df_smiles.to_csv(out_path)

    logger.info(f'Generation finished!')


def get_args():
    parser = argparse.ArgumentParser(description='Generate SMILES from a GPT model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_path', type=str, help='Full path to GPT model')
    parser.add_argument('--out_file', type=str, help='Output file path')

    optional = parser.add_argument_group('Optional')
    optional.add_argument('--num_to_sample', default=30000, type=int, help='Molecules to sample, default=30000')
    optional.add_argument('--device', default='cuda', type=str, help='Use cuda or cpu, default=cuda')
    optional.add_argument('--batch_size', default=64, type=int, help='Batch_size during sampling, default=64')
    optional.add_argument('--max_seq_length', default=140, type=int, help='Maximum smiles length, default=100')
    optional.add_argument('--temperature', default=1.0, type=float, help='Temperature during sampling, default=1.0')
    optional.add_argument('--eval', action="store_true", help='Evaluate with moses or not, default False')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)
