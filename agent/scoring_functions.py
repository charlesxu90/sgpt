import io
import subprocess
import warnings
import numpy as np
import pandas as pd
from rdkit import Chem, rdBase, DataStructs
from rdkit.Chem import AllChem, Descriptors, QED
import pickle
import gzip
from typing import List
import logging
from tqdm import tqdm

from agent.sascore.sascorer import calculateScore

rdBase.DisableLog('rdApp.error')
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)


def fingerprints_from_smiles(smiles: List, size=2048):
    """
        Create ECFP fingerprints of smiles, with validity check
    """
    fps = []
    valid_mask = []
    for i, smile in enumerate(smiles):
        mol = Chem.MolFromSmiles(smile)
        valid_mask.append(int(mol is not None))
        fp = fingerprints_from_mol(mol, size=size) if mol else np.zeros((1, size))
        fps.append(fp)

    fps = np.concatenate(fps, axis=0)
    return fps, valid_mask


def fingerprints_from_mol(molecule, radius=3, size=2048, hashed=False):
    """
        Create ECFP fingerprint of a molecule
    """
    if hashed:
        fp_bits = AllChem.GetHashedMorganFingerprint(molecule, radius, nBits=size)
    else:
        fp_bits = AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=size)
    fp_np = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp_bits, fp_np)
    return fp_np.reshape(1, -1)


class ActivityScorer:
    """Use pretrained activity classifier based on ECFP preprints for activity."""
    def __init__(self, model_path='./data/drd2/clf.pkl'):
        self.clf = pickle.load(open(model_path, "rb"))

    def __call__(self, smiles: List):
        fps, valid_mask = self.fingerprints_from_smiles(smiles)
        scores = self.clf.predict_proba(fps)[:, 1]
        scores = np.float32(scores * np.array(valid_mask))
        return scores, scores

    def fingerprints_from_smiles(self, smiles: List, size=2048):
        """
            Create ECFP fingerprints of smiles, with validity check
        """
        fps = []
        valid_mask = []
        for i, smile in enumerate(smiles):
            mol = Chem.MolFromSmiles(smile)
            valid_mask.append(int(mol is not None))
            fp = self.fingerprints_from_mol(mol, size=size) if mol else np.zeros((1, size), np.int32)
            fps.append(fp)

        fps = np.concatenate(fps, axis=0)
        return fps, valid_mask

    @staticmethod
    def fingerprints_from_mol(mol, radius=3, size=2048):
        fp = AllChem.GetMorganFingerprint(mol, radius, useCounts=True, useFeatures=True)
        nfp = np.zeros((1, size), np.int32)
        for idx, v in fp.GetNonzeroElements().items():
            nidx = idx % size
            nfp[0, nidx] += int(v)
        return nfp


class QEDScorer:
    def __call__(self, smiles):
        scores = []
        valid_mask = []
        for i, smile in enumerate(smiles):
            mol = Chem.MolFromSmiles(smile)
            valid_mask.append(int(mol is not None))
            try:
                qed_score = QED.qed(mol)
            except ValueError:
                qed_score = 0.0
            scores.append(qed_score)

        scores = np.float32(scores * np.array(valid_mask))
        return scores, scores


class SAScorer:
    def __init__(self, model_path='agent/sascore/SA_score_prediction.pkl.gz'):
        self.clf = pickle.load(gzip.open(model_path, "rb"))

    def __call__(self, smiles: List):
        descriptors, valid_mask = self._get_descriptors_from_smiles(smiles)
        scores = self.clf.predict_proba(descriptors)[:, 1]
        logger.debug("SA scores: " + str(scores * np.array(valid_mask)))
        scores = np.float32(scores * np.array(valid_mask))
        return scores, scores

    @staticmethod
    def _get_descriptors_from_smiles(smiles: List, radius=3, size=4096):  #
        """
            Add fingerprints together with SAscore and molecular weights
        """
        fps = []
        valid_mask = []
        for i, smile in enumerate(smiles):
            mol = Chem.MolFromSmiles(smile)
            valid_mask.append(int(mol is not None))
            fp = fingerprints_from_mol(mol, radius, size=size) if mol else np.zeros((1, size))
            others = np.array([calculateScore(mol), Descriptors.ExactMolWt(mol)]) if mol else np.zeros(2)
            prop_np = np.concatenate([others.T, fp.T[:, 0]])
            fps.append(prop_np)

        return fps, valid_mask


class DockScorer:
    def __init__(self, conf_path='./agent/docker/config.json', docker_path='agent.docker.docker'):
        self.conf_path = conf_path
        self.docker_path = docker_path
        self.params = {'low': -12, 'high': -8, 'k': 0.25}

    def __call__(self, smiles: List, batch_size=64):
        logger.info(f"Num of smiles: {len(smiles)}")
        num_smiles = len(smiles)
        if num_smiles <= batch_size:
            command = self.create_command(smiles)
            result = self.submit_command(command)
        else:  # Split into batches in case too many smiles
            number_batches = (num_smiles + batch_size - 1) // batch_size
            remaining_samples = num_smiles
            batch_start = 0
            result = []
            logger.info(f"List too long, Split into {number_batches} batches of size {batch_size}")
            for i in tqdm(range(number_batches), desc='Docking'):
                batch_size = min(batch_size, remaining_samples)
                batch_end = batch_start + batch_size

                command = self.create_command(smiles[batch_start:batch_end])
                result += self.submit_command(command)

                batch_start += batch_size
                remaining_samples -= batch_size

        scores = []
        for score in result:
            try:
                score = float(score)
            except ValueError:
                score = 0  # replace NA with 0
            scores.append(score)

        scores_tf = np.float32(self.transform_score(scores, self.params))
        raw_scores = np.float32(scores)
        return scores_tf, raw_scores

    def create_command(self, smiles: List):  # Initiate a docking process, which support multiprocessing
        concat_smiles = '"' + ';'.join(smiles) + '"'
        command = ' '.join(["python -m", self.docker_path, "--conf", self.conf_path, "--smiles", concat_smiles, "--print_scores"])
        return command

    @staticmethod
    def submit_command(command):
        with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              shell=True) as proc:
            wrapt_proc_in = io.TextIOWrapper(proc.stdin, 'utf-8')
            wrapt_proc_out = io.TextIOWrapper(proc.stdout, 'utf-8')
            result = [line.strip() for line in wrapt_proc_out.readlines()]
            wrapt_proc_in.close()
            wrapt_proc_out.close()
            proc.wait()
            proc.terminate()
        return result

    @staticmethod
    def transform_score(scores, params):
        def reverse_sigmoid(value, low, high, k) -> float:
            try:
                return 1 / (1 + 10 ** (k * (value - (high + low) / 2) * 10 / (high - low)))
            except:
                return 0

        scores_tf = [reverse_sigmoid(s_val, params['low'], params['high'], params['k']) for s_val in scores]
        return scores_tf


class ScoringFunctions:
    def __init__(self, scoring_func_names, weights=None):
        self.all_funcs = {'DockScore': DockScorer, 'Activity': ActivityScorer, 'QED': QEDScorer, 'SAScore': SAScorer}
        self.scoring_func_names = ['Activity'] if scoring_func_names is None else scoring_func_names
        self.weights = np.array([1] * len(self.scoring_func_names) if weights is None else weights)

    def scores(self, smiles: List, step: int, score_type='sum'):
        logger.debug(f"Num of SMILES: {len(smiles)}")
        scores, raw_scores = [], []
        for fn_name in self.scoring_func_names:
            score, raw_score = self.all_funcs[fn_name]()(smiles)
            scores.append(score)
            raw_scores.append(raw_score)
        scores = np.float32(scores).T
        raw_scores = np.float32(raw_scores).T

        if score_type == 'sum':
            final_scores = scores.sum(axis=1)
        elif score_type == 'product':
            final_scores = scores.prod(axis=1)
        elif score_type == 'weight':
            final_scores = (scores * self.weights / self.weights.sum()).sum(axis=1)
        else:
            raise Exception('Score type error!')

        np_step = np.ones(len(smiles)) * step
        scores_df = pd.DataFrame({'step': np_step, 'smiles': smiles, score_type: final_scores})
        scores_df[self.scoring_func_names] = pd.DataFrame(scores, index=scores_df.index)
        raw_names = [f'raw_{name}' for name in self.scoring_func_names]
        scores_df[raw_names] = pd.DataFrame(raw_scores, index=scores_df.index)
        return scores_df


def unit_tests():
    step = 1
    smiles = ['Cc', 'Cc1nc(N)nc(-c2cccnc2Nc2cnc(Cl)c(NS(C)(=O)=O)c2)n1', 'Nc1nonc1-c1nc2cc(O)ccc2n1C1CCC1',
              'C#CCCCn1c(Cc2cc(OC)c(OC)c(OC)c2Cl)nc2c(N)ncnc21']

    fps, valid_mask = fingerprints_from_smiles(smiles)
    assert valid_mask == [0, 1, 1, 1]

    drd2_act_model = ActivityScorer(model_path='./data/drd2/clf.pkl')
    print(drd2_act_model(smiles)[0])

    qed_scorer = QEDScorer()
    assert np.array_equal(qed_scorer(smiles)[0], np.float32([0., 0.540234, 0.73940235, 0.4455757]))
    print(f"QED test = {qed_scorer(['CC(C)CCN1CCC(CNc2cccc(Cl)c2)CC1'])}")

    sa_scorer = SAScorer(model_path='./agent/sascore/SA_score_prediction.pkl.gz')
    assert np.array_equal(sa_scorer(smiles)[0], np.float32([0.,  0.79, 0.82, 0.78]))

    scoring_function = ScoringFunctions(['Activity', 'QED', 'SAScore'], weights=[3, 1, 1])
    sum_scores_df = scoring_function.scores(smiles, step, score_type='sum')
    print("Sum of scores:")
    print(sum_scores_df)

    prod_scores_df = scoring_function.scores(smiles, step, score_type='product')
    print("Product of scores:")
    print(prod_scores_df)

    wt_scores_df = scoring_function.scores(smiles, step, score_type='weight')
    print("Weighted average of scores:")
    print(wt_scores_df)

    smiles = ['OC=C(C)Cc1ccccc1OC(=O)Nc1cccc(F)c1', 'C(=O)NCC(=O)NCC(C)Oc1cccc(-c2ccccc2)c1',
              'O=C(CO)NC(=O)NCCc1cccc2ccccc12', 'NCCC(=O)NCC1CCCc2cc(Br)ccc21', 'C(=O)Nc1cccc(C(=O)NCCNC(=O)c2ccccc2F)c1'
              'FCc1cncc(C(N)C(=O)c2ccc(Cl)cc2)c1', 'OC=CCC(=O)NCC(=O)Cc1ccccc1-c1ccccc1', 'OCc1cccc(OC)c1CNC(=O)Cc1ccccc1']

    ace2_dock_scorer = DockScorer()
    logger.info(ace2_dock_scorer(smiles))

    scoring_function = ScoringFunctions(['DockScore', 'QED', 'SAScore'], weights=[3, 1, 1])
    prod_scores_df = scoring_function.scores(smiles, step, score_type='product')
    print("Product of Docking, QED & SA scores:")
    print(prod_scores_df)

    wt_scores_df = scoring_function.scores(smiles, step, score_type='weight')
    print("Weighted average of Docking, QED & SA scores:")
    print(wt_scores_df)


if __name__ == "__main__":
    unit_tests()
