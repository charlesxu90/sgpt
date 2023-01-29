import os.path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.AtomPairs import Pairs
from rdkit import DataStructs

from utils.utils import calculate_scaffold, check_if_valid


class DiversityFilter:
    def __init__(self, min_similarity=0.4, min_score=0.4, max_bkt_size=10):
        self.scaffolds = {}  # scaffold -> smile -> score dictionary
        self.scaffolds_fingerprints = {}  # scaffold fingerprints used to calculate similarity
        self.min_similarity = min_similarity  # min similarity to existing scaffolds
        self.min_score = min_score  # min score to save to bucket
        self.max_bkt_size = max_bkt_size

    def score(self, smiles, scores):
        valid_idxs = check_if_valid(smiles)

        for idx in valid_idxs:
            cano_smile = Chem.MolToSmiles(Chem.MolFromSmiles(smiles[idx], sanitize=False), isomericSmiles=False)
            scaffold = calculate_scaffold(cano_smile)

            # check if a similar scaffold already exists, or create a new one
            scaffold = self.find_similar_scaffold(scaffold)

            scores[idx] = 0 if self.smile_exists(scaffold, cano_smile) else scores[idx]  # set 0 if smiles exists
            if scores[idx] >= self.min_score:
                self.add_to_memory(scaffold, cano_smile, scores[idx])  # Add even if bucket is full
                scores[idx] = self.penalize_score(scaffold, scores[idx])  # Penalize if a similar scaffold exists

        return scores

    def find_similar_scaffold(self, scaffold):
        """
            Return similar scaffold which already exists, else itself
        """
        if scaffold is not '':
            fp = Pairs.GetAtomPairFingerprint(Chem.MolFromSmiles(scaffold))

            fps = list(self.scaffolds_fingerprints.values())
            if len(fps) > 0:
                similarity = DataStructs.BulkDiceSimilarity(fp, fps)
                closest_idx = np.argmax(similarity)
                if similarity[closest_idx] >= self.min_similarity:
                    scaffold = list(self.scaffolds_fingerprints.keys())[closest_idx]
                    return scaffold

            self.scaffolds_fingerprints[scaffold] = fp
        return scaffold

    def smile_exists(self, scaffold, smile):
        """
            Return True if smile already in scaffold bucket, else False
        """
        if scaffold in self.scaffolds:
            if smile in self.scaffolds[scaffold]:
                return True
        return False

    def add_to_memory(self, scaffold, smile, score):
        """
            Add smile to scaffold bucket or create new bucket with the smile
        """
        if scaffold in self.scaffolds:
            self.scaffolds[scaffold][smile] = score
        else:
            self.scaffolds[scaffold] = {smile: score}

    def penalize_score(self, scaffold, score):
        """
            Set score to 0 if scaffold bucket is > max_bkt_size
        """
        bkt_size = len(self.scaffolds[scaffold]) if scaffold in self.scaffolds else 0
        if bkt_size > self.max_bkt_size:
            score = 0
        return score

    def save_to_csv(self, save_dir):
        """
            Save scaffold filter to a CSV file
        """
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        scaffold, smiles, scores = [], [], []
        for sca in self.scaffolds.keys():
            for smile in self.scaffolds[sca]:
                scaffold.append(sca)
                smiles.append(smile)
                scores.append(self.scaffolds[sca][smile])
        df = pd.DataFrame({'scaffold': scaffold, 'smiles': smiles, 'score': scores})
        df.to_csv(os.path.join(save_dir, "scaffold_memory.csv"), index=False)


def unit_test():
    df = DiversityFilter(max_bkt_size=1)
    smiles = ['CC', 'CCCC', 'Cc1ccccc1C', 'cc', 'Cc1ccccc1CC', 'Cc1cccc(c1)C']
    scores = [0.8, 0.7, 0.5, 0.1, 0.2, 0.5]
    assert df.score(smiles, scores) == [0.8, 0, 0.5, 0.1, 0.2, 0]  # 0,1,2,
    df.save_to_csv('./')


if __name__ == "__main__":
    unit_test()
