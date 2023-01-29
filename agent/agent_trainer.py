import os
import argparse
import logging
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from agent.diversity_filter import DiversityFilter
from utils.utils import fraction_valid_smiles, unique
from utils.dataset import SmilesCharDictionary, Experience, rnn_start_token_vector
from model.model import load_gpt_model, save_gpt_model
from prior.trainer import TrainerConfig
from agent.scoring_functions import ScoringFunctions


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentTrainer:
    def __init__(self, prior_path, agent_path, save_dir, device='cuda', learning_rate=0.0001, batch_size=64, n_steps=3000,
                 sigma=60, max_seq_len=100, score_fns=None, weights=None, score_type='weight', df_min_score=0.4,
                 experience_replay=False, exp_max_size=30):
        logger.info("Initializing agent trainer...")
        self.prior_path = prior_path
        self.agent_path = agent_path
        self.save_dir = save_dir
        self.device = device
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.sigma = sigma
        self.experience_replay = experience_replay
        self.max_seq_len = max_seq_len
        self.sd = SmilesCharDictionary()

        self.prior_model, self.agent_model = self.load_pretrain_models()
        self.prior_model = self.prior_model.module if hasattr(self.prior_model, "module") else self.prior_model
        self.agent_model = self.agent_model.module if hasattr(self.agent_model, "module") else self.agent_model
        self.tconf = TrainerConfig(learning_rate=self.learning_rate, lr_decay=True)
        self.optimizer = self.agent_model.configure_optimizers(self.tconf)  # Use adamW with lr_decay
        self.agent_model = torch.nn.DataParallel(self.agent_model).to(self.device)  # Enable using multiple GPUs

        if score_fns is None:
            score_fns, weights = ['Activity'], [1]
        self.scoring_function = ScoringFunctions(score_fns, weights=weights)
        self.score_type = score_type
        self.scaffold_filter = DiversityFilter(min_score=df_min_score)  # Init diversity filter
        if experience_replay:
            self.experience = Experience(max_size=exp_max_size)  # Init experience
        self.writer = SummaryWriter(self.save_dir)

    def load_pretrain_models(self):
        logger.info("Loading pretrained models")
        model_def = Path(self.prior_path).with_suffix('.json')
        logger.info(f"Loading prior & agent to device {self.device}")
        try:
            prior = load_gpt_model(model_def, self.prior_path, self.device, copy_to_cpu=False)
            agent = load_gpt_model(model_def, self.agent_path, self.device, copy_to_cpu=False)
            return prior, agent
        except:
            raise Exception(f"Device '{self.device}' or model not available")

    def save_step(self, step, scores_df, agent_likelihoods, prior_likelihoods, augmented_likelihoods):
        """
            Save step to a CSV file
        """
        scores_df['step'] = step * np.ones(len(scores_df))
        scores_df['agent_likelihood'] = agent_likelihoods.data.cpu().numpy()
        scores_df['prior_likelihood'] = prior_likelihoods.data.cpu().numpy()
        scores_df['augmented_likelihood'] = augmented_likelihoods.data.cpu().numpy()
        scores_df.to_csv(os.path.join(self.save_dir, f"step_{step}_smiles.csv"), index=False)

    def nll_loss(self, inputs, targets):
        """
            Custom Negative Log Likelihood loss that returns loss per example, rather than for the entire batch.

            Args:
                inputs : (batch_size, num_classes) *Log probabilities of each class*
                targets: (batch_size) *Target class index*

            Outputs:
                loss : (batch_size) *Loss for each example*
        """
        target_expanded = torch.zeros(inputs.size()).to(self.device)
        target_expanded.scatter_(1, targets.contiguous().view(-1, 1).detach(), 1.0)  # One_hot encoding
        loss = torch.sum(target_expanded * inputs, 1)
        return loss

    def sample(self, model, num_samples: int):
        """
            Sample molecules from agent and calculate likelihood
            Args:
                model: model to sample from
                num_samples: number of samples to produce for each step, i.e. batch_size

            Returns:
                sample_idxes: a list of SMILES indexes, with no beginning nor end symbols
                log_probs: log likelihood for SMILES generated
            """
        finished = torch.zeros(num_samples).byte().to(self.device)
        sequences = []

        x = rnn_start_token_vector(num_samples, self.device)
        log_probs = torch.zeros(num_samples).to(self.device)
        for step in range(self.max_seq_len):
            logits, _ = model(x)
            prob = F.softmax(logits[:, -1, :], dim=-1)  # only for last time-step
            sampled_idx = Categorical(probs=prob).sample().squeeze()
            sequences.append(sampled_idx.view(-1, 1))
            x = torch.cat(sequences, 1)  # assign x with all sequence generated

            log_probs += self.nll_loss(prob.log(), sampled_idx)  # update log_probs

            # Stop if EOS sampled
            end_sampled = (sampled_idx == self.sd.end_idx).detach()  # Check if end in current step
            finished = torch.ge(finished + end_sampled, 1)  # Check if all ends sampled
            if torch.prod(finished) == 1:
                break

        sequences = torch.cat(sequences, 1)
        return sequences.detach(), log_probs

    def likelihood(self, model, sample_idxes):
        """
        Retrieves the likelihood of a given sequence
            Args: x
                model: GPT model to calculate likelihood
                sample_idxes: A list of smiles of batch_size length
                device: Device used
            Outputs:
                log_probs : (batch_size) Log likelihood for each example
        """

        x = sample_idxes.to(self.device)
        num_samples, seq_length = x.size()
        log_probs = torch.zeros(num_samples).to(self.device)

        for step in range(1, seq_length):
            logits, _ = model(x[:, :step])
            log_prob = F.log_softmax(logits[:, -1, :], dim=-1).squeeze()
            log_probs += self.nll_loss(log_prob, x[:, step])

        return log_probs

    def replay_experience(self, loss, agent_likelihoods, prior_likelihoods, smiles, scores):
        if len(self.experience) > 4:  # Sample experiences and update loss
            exp_smiles, exp_scores, exp_prior_likelihoods = self.experience.sample(4)
            exp_agent_likelihoods = self.likelihood(self.agent_model, exp_smiles)
            exp_augmented_likelihood = exp_prior_likelihoods + self.sigma * exp_scores
            exp_loss = torch.pow((exp_augmented_likelihood - exp_agent_likelihoods), 2)
            loss = torch.cat((loss, exp_loss), 0)
            agent_likelihoods = torch.cat((agent_likelihoods, exp_agent_likelihoods), 0)

        prior_likelihoods = prior_likelihoods.data.cpu().numpy()
        new_experience = zip(smiles, scores, prior_likelihoods)
        self.experience.add_experience(new_experience)  # Add new experience
        return loss, agent_likelihoods

    def train(self):
        for param in self.prior_model.parameters():  # Don't update Prior
            param.requires_grad = False

        logger.info("Starting training agent...")
        for step in range(self.n_steps):
            sample_idxes, agent_likelihoods = self.sample(self.agent_model, self.batch_size)  # Sample from agent
            uniq_ids = unique(sample_idxes)  # Remove duplicates
            uniq_token_seqs = sample_idxes[uniq_ids]
            smiles = self.sd.matrix_to_smiles(uniq_token_seqs)
            agent_likelihoods = agent_likelihoods[uniq_ids]
            prior_likelihoods = self.likelihood(self.prior_model, uniq_token_seqs)

            scores_df = self.scoring_function.scores(smiles, step, score_type=self.score_type)
            scores = scores_df[self.score_type].to_numpy()
            # scores = self.scaffold_filter.score(smiles, scores_df[self.score_type].to_numpy())  # save in diversity filter

            augmented_likelihoods = prior_likelihoods + self.sigma * torch.from_numpy(scores).to(self.device)
            loss = torch.pow((augmented_likelihoods - agent_likelihoods), 2)

            if self.experience_replay:
                loss, agent_likelihood = self.replay_experience(loss, agent_likelihoods, prior_likelihoods, smiles, scores)

            loss = loss.mean()
            loss -= 5 * 1e3 * (1/agent_likelihoods).mean()  # Penalize small likelihoods, stimulate learning

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent_model.parameters(), self.tconf.grad_norm_clip)
            self.optimizer.step()

            logger.info(f"Step {step}, Valid %: {fraction_valid_smiles(smiles) * 100:4.1f}, "
                        + f"Max score: {max(scores):6.2f}, Mean score: {scores.mean():6.2f}, ")
            self.writer.add_scalar('Valid (%)', fraction_valid_smiles(smiles) * 100, step + 1)
            self.writer.add_scalar('Max score', max(scores), step + 1)
            self.writer.add_scalar('Mean score', scores.mean(), step + 1)

            self.save_step(step, scores_df, agent_likelihoods, prior_likelihoods, augmented_likelihoods)

            if step % 250 == 0 and step != 0:  # save model every 250 steps
                save_gpt_model(self.agent_model, self.save_dir, f'Agent_{step}')

        save_gpt_model(self.agent_model, self.save_dir, f'Agent_final')
        self.scaffold_filter.save_to_csv(self.save_dir)
