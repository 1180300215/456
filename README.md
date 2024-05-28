# Transformer Against Opponent (TAO)

This is an anonymous code repository for the paper **Towards Offline Opponent Modeling with In-context Learning**.

## Environment dependencies installation
Please follow the steps below to install environment dependencies. Installation on Ubuntu 18.04 LTS is recommended.
```bash
# using Anaconda to create a virtual environment
conda create --name tao python=3.8
conda activate tao

# install dependencies (might be tricky for some packages)
pip install -r requirements.txt

# install environments
./install_envs.sh
```
We record our experiments using [wandb](https://wandb.ai/site?utm_source=google&utm_medium=cpc&utm_campaign=Performance-Max&utm_content=site&gclid=CjwKCAjwlqOXBhBqEiwA-hhitGcG5-wtdqoNgKyWdNpsRedsbEYyK9NeKcu8RFym6h8IatTjLFYliBoCbikQAvD_BwE) and recommend referring to the [wandb quickstart documentation](https://docs.wandb.ai/quickstart) for the setup process.
&nbsp;

## Offline dataset download
### Markov Soccer (MS)
Please download the *offline dataset* from the [anonymous link](https://osf.io/35cwy/?view_only=75573b84362442449036a11ec023d26d). Then, place the downloaded offline dataset in the following path:
```bash
.
├── envs
│   ├── markov_soccer
│   │   ├── data
│   │   │   └── offline_dataset_MS_5oppo_10k.pkl
│   │   └── ...
│   └── ...
└── ...
```
### Particleworld Adversary (PA)
Please download the *offline dataset* from the [anonymous link](https://osf.io/35cwy/?view_only=75573b84362442449036a11ec023d26d). Then, place the downloaded offline dataset in the following path:
```bash
.
├── envs
│   ├── multiagent_particle_envs
│   │   ├── data
│   │   │   └── offline_dataset_PA_5oppo_10k.pkl
│   │   └── ...
│   └── ...
└── ...
```

## Offline Stage 1: Policy Embedding Learning
Train **Opponent Policy Encoder (OPE)** based on the offline dataset. See [config.py](offline_stage_1/config.py) for specific experimental configurations.
```bash
cd offline_stage_1
python train.py
```

## Offline Stage 2: Opponent-aware Response Policy Training
Train **In-context Control Decoder (ICD)** based on the offline dataset and the trained **OPE**. See [config.py](offline_stage_2/config.py) for specific experimental configurations.
```bash
cd ../offline_stage_2
python train.py
```

## Deployment Stage: In-context Opponent Adapting
Deploy the pre-trained **OPE** and **ICD** into a new environment and test against a non-stationary unknown opponent with the help of an **Opponent Context Window (OCW)**.
```bash
cd ../deployment_stage
python test.py
```
Additionally, pre-trained model weight files are available at `model/MS-pretrained_models` and `model/PA-pretrained_models`.
## Remark
The code for Offline Stage 2 is based on [Decision-Transformer](https://github.com/kzl/decision-transformer). The code for the MS environment and some opponent policies of MS are based on [Competitive Policy Gradient (CoPG) Algorithm](https://github.com/manish-pra/copg/tree/master). The code for the PA environment is based on [Multi-Agent Particle Environments (MPE)](https://github.com/openai/multiagent-particle-envs).