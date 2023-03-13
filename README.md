# Optimization of binding affinities in chemical space with transformer and deep reinforcement learning

This is the code of **SGPT-RL**, a tool for chemical design using transformer and deep reinforcement learning. Through employing GPT model as the policy network, SGPT-RL can learn scaffolds patterns in exploring the chemical space.

![Workflow of SGPT-RL](./pipeline.png)

## Installation
### Clone & create environment
```shell
git clone https://github.com/charlesxu90/sgpt.git
cd sgpt

conda env create -f environment.yml
conda activate sgpt-env
```
#### Download data and models using git-lfs
In the conda environment, [git-lfs](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage) is installed. It can be used to download the raw data and pretrained models associated with this repository.

```shell
git lfs pull
```
## Install Openbabel
```shell
sudo apt-get install -y openbabel
```
Need to remove the default `openbabel` in the conda environment if there's one.

## Running the code

Commands to retrain the models & generate molecules:
```shell

# Train prior on Moses dataset
python train_prior.py --train_data data/moses/train.csv --valid_data data/moses/test.csv --n_epochs 10 --output_dir result/prior --eval --n_embd 256 --n_head 8 --n_layers 8 --batch_size 1024

# Train an agent to optimize DRD2 activity
python train_agent.py -p data/prior/gpt_model_10_0.126.pt -a data/prior/gpt_model_10_0.126.pt  -o result/drd2- -t drd2 --sigma 60

# Train an agent to optimize ACE2 docking score
python train_agent.py -p data/prior/gpt_model_10_0.126.pt -a data/prior/gpt_model_10_0.126.pt  -o result/ace2- -t ace2 --sigma 60  --n_steps 1000

# Generate molecules from pretrained models
python generate.py --model_path data/prior/gpt_model_10_0.126.pt --out_file result/prior/sgpt-10000.csv --num_to_sample 10000

```

## License

This code is licensed under [MIT License](./LICENSE).
