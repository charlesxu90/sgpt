import torch
import torch.nn.functional as F
from utils.utils import set_random_seed
from model.model import GPT
from utils.dataset import SmilesCharDictionary, rnn_start_token_vector
from tqdm import tqdm

sd = SmilesCharDictionary()


def _sample_batch(model: GPT, batch_size: int, device, max_seq_length, temperature) -> torch.Tensor:
    x = rnn_start_token_vector(batch_size, device)
    indices = torch.zeros((batch_size, max_seq_length), dtype=torch.long).to(device)

    for char in range(max_seq_length):
        logits, _ = model(x)
        logits = logits[:, -1, :] / temperature

        probs = F.softmax(logits, dim=-1)
        distribution = torch.distributions.Categorical(probs=probs)
        action = distribution.sample()

        indices[:, char] = action.squeeze()
        x = indices[:, :char + 1]  # assign x with all sequence generated

    return indices


def sample(model: GPT, num_to_sample=10000, device='cpu', batch_size=64, max_seq_length=140, temperature=1.0, seed=42):
    set_random_seed(seed, device)

    # Round up division to get the number of batches that are necessary:
    number_batches = (num_to_sample + batch_size - 1) // batch_size
    remaining_samples = num_to_sample

    indices = torch.LongTensor(num_to_sample, max_seq_length).to(device)

    model.eval()
    with torch.no_grad():
        batch_start = 0
        for i in tqdm(range(number_batches), desc='Sampling'):
            batch_size = min(batch_size, remaining_samples)
            batch_end = batch_start + batch_size

            indices[batch_start:batch_end, :] = _sample_batch(model, batch_size, device, max_seq_length, temperature)

            batch_start += batch_size
            remaining_samples -= batch_size

        return sd.matrix_to_smiles(indices)
