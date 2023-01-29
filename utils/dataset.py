import logging
import torch
from torch.utils.data import TensorDataset
import numpy as np
from rdkit import Chem

logger = logging.getLogger(__name__)


class SmilesCharDictionary(object):
    """
    A fixed dictionary for druglike SMILES.
    Enables smile<->token conversion.

    With a space:0 for padding, Q:1 as the start token and end_of_line \n:2 as the stop token.
    """

    PAD = ' '
    BEGIN = 'Q'
    END = '\n'

    def __init__(self) -> None:
        self.forbidden_symbols = {'Ag', 'Al', 'Am', 'Ar', 'At', 'Au', 'D', 'E', 'Fe', 'G', 'K', 'L', 'M', 'Ra', 'Re',
                                  'Rf', 'Rg', 'Rh', 'Ru', 'T', 'U', 'V', 'W', 'Xe',
                                  'Y', 'Zr', 'a', 'd', 'f', 'g', 'h', 'k', 'm', 'si', 't', 'te', 'u', 'v', 'y'}

        self.char_idx = {self.PAD: 0, self.BEGIN: 1, self.END: 2, '#': 20, '%': 22, '(': 25, ')': 24, '+': 26, '-': 27,
                         '.': 30,
                         '0': 32, '1': 31, '2': 34, '3': 33, '4': 36, '5': 35, '6': 38, '7': 37, '8': 40,
                         '9': 39, '=': 41, 'A': 7, 'B': 11, 'C': 19, 'F': 4, 'H': 6, 'I': 5, 'N': 10,
                         'O': 9, 'P': 12, 'S': 13, 'X': 15, 'Y': 14, 'Z': 3, '[': 16, ']': 18,
                         'b': 21, 'c': 8, 'n': 17, 'o': 29, 'p': 23, 's': 28,
                         "@": 42, "R": 43, '/': 44, "\\": 45, 'E': 46
                         }

        self.idx_char = {v: k for k, v in self.char_idx.items()}

        self.encode_dict = {"Br": 'Y', "Cl": 'X', "Si": 'A', 'Se': 'Z', '@@': 'R', 'se': 'E'}
        self.decode_dict = {v: k for k, v in self.encode_dict.items()}

    def allowed(self, smiles) -> bool:
        """
        Determine if smiles string has illegal symbols

        Args:
            smiles: SMILES string

        Returns:
            True if all legal
        """
        for symbol in self.forbidden_symbols:
            if symbol in smiles:
                logger.info('Forbidden symbol {:<2}  in  {}'.format(symbol, smiles))
                return False
        return True

    def encode(self, smiles: str) -> str:
        """
        Replace multi-char tokens with single tokens in SMILES string.

        Args:
            smiles: a SMILES string

        Returns:
            sanitized SMILE string with only single-char tokens
        """

        temp_smiles = smiles
        for symbol, token in self.encode_dict.items():
            temp_smiles = temp_smiles.replace(symbol, token)
        return temp_smiles

    def decode(self, smiles):
        """
        Replace special tokens with their multi-character equivalents.

        Args:
            smiles: SMILES string

        Returns:
            SMILES string with possibly multi-char tokens
        """
        temp_smiles = smiles
        for symbol, token in self.decode_dict.items():
            temp_smiles = temp_smiles.replace(symbol, token)
        return temp_smiles

    def get_char_num(self) -> int:
        """
        Returns:
            number of characters in the alphabet
        """
        return len(self.idx_char)

    @property
    def begin_idx(self) -> int:
        return self.char_idx[self.BEGIN]

    @property
    def end_idx(self) -> int:
        return self.char_idx[self.END]

    @property
    def pad_idx(self) -> int:
        return self.char_idx[self.PAD]

    def matrix_to_smiles(self, array):
        """
        Converts an matrix of indices into their SMILES representations
        Args:
            array: torch tensor of indices, one molecule per row

        Returns: a list of SMILES, without the termination symbol
        """
        smiles_strings = []

        for row in array:
            predicted_chars = []

            for j in row:
                next_char = self.idx_char[j.item()]
                if next_char == self.END:
                    break
                predicted_chars.append(next_char)

            smi = ''.join(predicted_chars)
            smi = self.decode(smi)
            smiles_strings.append(smi)

        return smiles_strings

    def smiles_to_matrix(self, smiles, max_len=100):
        """
        Converts a list of smiles into a matrix of indices

        Args:
            smiles: a list of SMILES, without the termination symbol
            max_len: the maximum length of smiles to encode, default=100

        Returns: a torch tensor of indices for all the smiles
        """
        batch_size = len(smiles)
        smiles = [self.BEGIN + self.encode(mol) + self.END for mol in smiles]
        idx_matrix = torch.zeros((batch_size, max_len))
        for i, mol in enumerate(smiles):
            enc_smi = self.BEGIN + self.encode(mol) + self.END
            for j in range(max_len):
                if j >= len(enc_smi):
                    break
                idx_matrix[i, j] = self.char_idx[enc_smi[j]]

        return idx_matrix.to(torch.int64)


def get_tensor_dataset(numpy_array):
    """
    Gets a numpy array of indices, convert it into a Torch tensor,
    divided it into inputs and targets and wrap it into a TensorDataset

    Args:
        numpy_array: to be converted

    Returns: a TensorDataset
    """

    tensor = torch.from_numpy(numpy_array).long()

    inp = tensor[:, :-1]
    target = tensor[:, 1:]

    return TensorDataset(inp, target)


def remove_duplicates(list_with_duplicates):
    """
    Removes the duplicates and keeps the ordering of the original list.
    For duplicates, the first occurrence is kept and the later occurrences are ignored.

    Args:
        list_with_duplicates: list that possibly contains duplicates

    Returns:
        A list with no duplicates.
    """

    unique_set = set()
    unique_list = []
    for element in list_with_duplicates:
        if element not in unique_set:
            unique_set.add(element)
            unique_list.append(element)

    return unique_list


def load_smiles_from_list(smiles_list, rm_invalid=True, rm_duplicates=True, max_len=100):
    """
    Given a list of SMILES strings, provides a zero padded NumPy array
    with their index representation. Sequences longer than `max_len` are
    discarded. The final array will have dimension (all_valid_smiles, max_len+2)
    as a beginning and end of sequence tokens are added to each string.

    Args:
        smiles_list: a list of SMILES strings
        rm_invalid: bool if True remove invalid smiles from final output. Note that if True the length of the output
          does not
          equal the size of the input  `smiles_list`. Default True
        rm_duplicates: bool if True return remove duplicates from final output. Note that if True the length of the
          output does not equal the size of the input  `smiles_list`. Default True
        max_len: dimension 1 of returned array, sequences will be padded

    Returns:
        sequences:list a numpy array of SMILES character indices
        valid_mask: list of len(smiles_list) - a boolean mask vector indicating if each index maps to a valid smiles
    """
    sd = SmilesCharDictionary()

    # filter valid smiles strings
    valid_smiles = []
    valid_mask = [False] * len(smiles_list)
    for i, s in enumerate(smiles_list):
        s = s.strip()
        if sd.allowed(s) and len(s) <= max_len:
            valid_smiles.append(s)
            valid_mask[i] = True
        else:
            if not rm_invalid:
                valid_smiles.append('C')  # default placeholder

    if rm_duplicates:
        unique_smiles = remove_duplicates(valid_smiles)
    else:
        unique_smiles = valid_smiles

    # max len + two chars for start token 'Q' and stop token '\n'
    max_seq_len = max_len + 2
    num_seqs = len(unique_smiles)
    logger.info(f'Number of sequences: {num_seqs}, max length: {max_len}')

    # allocate the zero matrix to be filled
    sequences = np.zeros((num_seqs, max_seq_len), dtype=np.int32)

    for i, mol in enumerate(unique_smiles):
        enc_smi = sd.BEGIN + sd.encode(mol) + sd.END
        for c in range(len(enc_smi)):
            try:
                sequences[i, c] = sd.char_idx[enc_smi[c]]
            except KeyError as e:
                logger.info(f'KeyError: {mol}, key: {i}, {enc_smi[c]}')

    return sequences, valid_mask


def rnn_start_token_vector(batch_size, device='cpu'):
    """
    Returns a vector of start tokens.
    This vector can be used to start sampling a batch of SMILES strings.

    Args:
        batch_size: how many SMILES will be generated at the same time
        device: cpu | cuda

    Returns:
        a tensor (batch_size x 1) containing the start token
    """
    sd = SmilesCharDictionary()
    return torch.LongTensor(batch_size, 1).fill_(sd.begin_idx).to(device)


class Experience(object):
    """Class for prioritized experience replay that remembers the highest scored sequences
       seen and samples from them with probabilities relative to their scores.
       Used to train agent.
       """
    def __init__(self, max_size=100):
        self.memory = []
        self.max_size = max_size
        self.sd = SmilesCharDictionary()  # using to replace voc

    def add_experience(self, experience):
        """Experience should be a list of (smiles, score, prior likelihood) tuples"""
        self.memory.extend(experience)
        if len(self.memory)>self.max_size:
            # Remove duplicates
            idxs, smiles = [], []
            for i, exp in enumerate(self.memory):
                if exp[0] not in smiles:
                    idxs.append(i)
                    smiles.append(exp[0])
            self.memory = [self.memory[idx] for idx in idxs]
            # Retain highest scores
            self.memory.sort(key = lambda x: x[1], reverse=True)
            self.memory = self.memory[:self.max_size]
            logger.info("\nBest score in memory: {:.2f}".format(self.memory[0][1]))

    def sample(self, n):
        """Sample a batch size n of experience"""
        if len(self.memory) < n:
            raise IndexError('Size of memory ({}) is less than requested sample ({})'.format(len(self), n))
        else:
            scores = [x[1] for x in self.memory]
            sample = np.random.choice(len(self), size=n, replace=False, p=scores/np.sum(scores))  # random sampling
            sample = [self.memory[i] for i in sample]
            smiles = [x[0] for x in sample]
            scores = [x[1] for x in sample]
            prior_likelihood = [x[2] for x in sample]
        idx_matrix = self.sd.smiles_to_matrix(smiles)
        return idx_matrix, np.array(scores), np.array(prior_likelihood)

    def initiate_from_file(self, fname, scoring_function, Prior):
        """Adds experience from a file with SMILES
           Needs a scoring function and an RNN to score the sequences.
           Using this feature means that the learning can be very biased
           and is typically advised against."""
        with open(fname, 'r') as f:
            smiles = []
            for line in f:
                smile = line.split()[0]
                if Chem.MolFromSmiles(smile):
                    smiles.append(smile)
        scores = scoring_function(smiles)
        idx_matrix = self.sd.smiles_to_matrix(smiles)
        prior_likelihood, _ = Prior.likelihood(idx_matrix.long())  # Need to update
        prior_likelihood = prior_likelihood.data.cpu().numpy()
        new_experience = zip(smiles, scores, prior_likelihood)
        self.add_experience(new_experience)

    def print_memory(self, path):
        """Prints the memory."""
        print("\n" + "*" * 80 + "\n")
        print("         Best recorded SMILES: \n")
        print("Score     Prior log P     SMILES\n")
        with open(path, 'w') as f:
            f.write("SMILES Score PriorLogP\n")
            for i, exp in enumerate(self.memory[:100]):
                if i < 50:
                    print("{:4.2f}   {:6.2f}        {}".format(exp[1], exp[2], exp[0]))
                    f.write("{} {:4.2f} {:6.2f}\n".format(*exp))
        print("\n" + "*" * 80 + "\n")

    def __len__(self):
        return len(self.memory)
