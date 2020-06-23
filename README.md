# NIM
Open source implementation for NIM: Neural Input Modeling

## Installation

1. Install [Anaconda](https://www.anaconda.com/)
2. Create a conda environment: `conda create --name nim python=3.8`
3. Install `numpy`: `conda install numpy`
4. Install `click`: `conda install -c conda-forge click`
5. Install PyTorch: `conda install pytorch -c pytorch`. See [PyTorch website](https://pytorch.org/get-started/locally/) for GPU support, etc.

## Usage

1. Create a synthetic dataset with 1,000 length 100 sequences and values are sampled from i.i.d. Gamma: `python create_data.py -n 1000 -t 10 data`
2. Train the neural network: `python train.py --network nvm --hidden-size 32 --data data.npy --transform transform:log_nvm --save nvm`. For more options, run `python train.py --help`
3. Generate new samples and save in `samples.npy`: `python generate.py --network=nvm.json -n 1000 -t 10 --transform=transform:exp samples.npy`
