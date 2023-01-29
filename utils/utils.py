import os
import time
from datetime import timedelta
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def time_since(start_time):
    seconds = int(time.time() - start_time)
    return str(timedelta(seconds=seconds))


def get_path(base_dir, base_name, suffix):
    return os.path.join(base_dir, base_name + suffix)


def set_random_seed(seed, device):
    """
    Set the random seed for Numpy and PyTorch operations
    Args:
        seed: seed for the random number generators
        device: "cpu" or "cuda"
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)


def fraction_valid_smiles(smiles):
    n_valid = sum([1 if Chem.MolFromSmiles(smi) else 0 for smi in smiles])
    return n_valid / len(smiles)


def calculate_scaffold(can_smi):
    mol = Chem.MolFromSmiles(can_smi)
    if mol:
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaffold, isomericSmiles=False)
        except ValueError:
            scaffold_smiles = ''
    else:
        scaffold_smiles = ''
    return scaffold_smiles


def check_if_valid(smiles):
    return [idx for idx in range(len(smiles)) if Chem.MolFromSmiles(smiles[idx])]


def unique(arr):
    # Finds unique rows in arr and return their indices
    arr = arr.cpu().numpy()
    arr_ = np.ascontiguousarray(arr).view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))
    _, idxs = np.unique(arr_, return_index=True)
    if torch.cuda.is_available():
        return torch.LongTensor(np.sort(idxs)).cuda()
    return torch.LongTensor(np.sort(idxs))
