import os
import argparse
import logging
import sys
import json
from agent.docker.ligand import Ligand
from agent.docker.ligprep_RDkit import RDkitLigandPreparator
from agent.docker.dock_AutoDockVina import AutodockVina

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger.addHandler(logging.NullHandler())


def ligands_from_console(smiles: str) -> list:
    ligand_smiles = smiles.split(';')
    return [Ligand(smile=smile, original_smile=smile, ligand_number=number_smile) for number_smile, smile
            in enumerate(ligand_smiles)]


def load_config(conf):
    with open(conf) as file:
        conf = file.read().replace("\r", "").replace("\n", "")
    return json.loads(conf)


def prep_ligands(smiles: str, prep_conf):
    ligands = ligands_from_console(smiles)
    logger.info(f"Totally {len(ligands)} ligands.")
    prep = RDkitLigandPreparator(ligands=ligands, input=prep_conf['input'], output=prep_conf['output'],
                                 parameters=prep_conf['RDkit'])
    prep.generate_3d_coordinates()
    prep.write_ligands()
    return prep


def run_docking(ligands, dock_conf):  # docking with AutoDock Vina

    docker = AutodockVina(output=dock_conf['output'], parameters=dock_conf['AutoDockVina'])
    clone_ligands = [lig.get_clone() for lig in ligands]
    docker.add_molecules(molecules=clone_ligands)
    docker.dock()

    # save results to files
    docker.write_docked_ligands()
    docker.write_result()
    return docker


def print_scores(docker, print_all=False):
    scores = docker.get_scores(best_only=not print_all)
    for score in scores:
        print(score, end="\n")
    logger.info(f"Printed {len(scores)} scores to console (print_all set to {args.print_all}).")


def main(args):
    if args.conf is None or not os.path.isfile(args.conf):
        raise Exception("Config file does not exist.")
    conf = load_config(args.conf)

    if args.print_scores is False and args.print_all:
        args.print_score = True

    if args.smiles is None:
        raise Exception("No input SMILES")

    # ligand preparation: SMILES -> List[LIGAND]
    prep_conf = conf['ligand_preparation']
    prep = prep_ligands(smiles=args.smiles, prep_conf=prep_conf)
    ligands = prep.get_ligands()

    # docking
    dock_conf = conf['docking']
    docker = run_docking(ligands, dock_conf)

    # printing
    if args.print_scores:
        print_scores(docker, print_all=args.print_all)

    sys.exit(0)


def parse_args():
    parser = argparse.ArgumentParser(description="Docker for the docking using AutoDock backends.")
    parser.add_argument("--conf", type=str, default=None, help="Path to docking configuration file (JSON).")
    parser.add_argument("--smiles", default=None, type=str, help="Input SMILES in command-line, separated by ';'.")

    optional = parser.add_argument_group('Optional')
    optional.add_argument("--print_scores", action="store_true", help="Print the scores Line-wise to the shell.")
    optional.add_argument("--print_all", action="store_true",
                          help="Print out scores for all conformers, not just the best. Use with -print_scores.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
