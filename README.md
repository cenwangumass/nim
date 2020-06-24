# NIM
Open source implementation for NIM: Neural Input Modeling

## Usage

1. Install NIM: `pip install git+https://github.com/cenwangumass/nim.git`
2. In the `examples` directory, create a synthetic dataset with 1,000 length 100 sequences and values are sampled from i.i.d. Gamma: `python create_data.py -n 1000 -t 10 data.npy`
3. Train the neural network: `python train.py nvm data.npy --transform transform:log_nvm`. For more options, run `python train.py --help`
4. Generate new samples and save in `samples.npy`: `python generate.py nvm.json samples.npy -n 1000 -t 10 --transform=transform:exp`
