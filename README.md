# Transformer MPC
Real-world robot navigation in human-centric environments remains an unsolved problem. Model Predictive Control (MPC) has emerged as a powerful control strategy in the field of robotics, offering the capability to optimize performance while considering constraints and predicting future behavior. The combination of deep learning techniques, particularly Transformer architectures, with MPC, has the potential to significantly improve the efficiency and applicability of control policies in real-world systems. This project presents Transformer-MPC, an approach that integrates Transformer-based attention mechanisms into the MPC framework for a context-aware, learnable control policy. The architecture is trained end-to-end via imitation learning to mimic expert demonstrations.

![](https://github.com/J4nn1K/transformer-mpc/blob/main/docs/figures/architecture.png)

## Training & Inference
Clone this repo, install dependencies, and instal the python module `transformer-mpc`.
```
pip install -r requirements.txt
pip install -e .
```
It is recommended to train on TPUs since this implementation uses JAX and the dataset is large. Instructions on how to set up a TPU can be found in this repo's [docs](https://github.com/J4nn1K/transformer-mpc/tree/main/docs). With Google's [TPU Research Cloud](https://sites.research.google/trc/about/) you can apply for on-demand Cloud TPU access free of charge for 30 days.

To train the model make changes to `/src/config` and run:
```
python3 main.py
```
Currently [Weights & Biases](https://wandb.ai/site) is used for logging. You will be prompted to log in with a W&B account.

Up until this point the training performance is insufficient for the model to be deployed for inference.

## Dataset
![](https://github.com/J4nn1K/transformer-mpc/blob/main/docs/figures/data.png)
The dataset that the models were trained on consists of three different sensor measurements: Occupancy Grids (100x100x1), RGB Images (640x480x3), and Depth Images (640x480x1).

Download the dataset:
```
export FILE_ID=1oeb7QHveAzVp08Jwiv7pepLB9geRFUHO
export FILENAME=obstacles.npz

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=${FILE_ID}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${FILE_ID}" -O ${FILENAME} && rm -rf /tmp/cookies.txt
```