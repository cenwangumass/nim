# NIM
Open source implementation for NIM: Neural Input Modeling

## Usage

1. Install NIM: `pip install git+https://github.com/cenwangumass/nim.git`
2. Enter `examples` directory: `cd examples/`
3. Create a synthetic dataset with 1,000 length 100 sequences and values are sampled from i.i.d. Gamma: `python create_data.py -n 1000 -t 10 data`
4. Train the neural network: `python train.py --network nvm --hidden-size 32 --data data.npy --transform transform:log_nvm --save nvm`. For more options, run `python train.py --help`
5. Generate new samples and save in `samples.npy`: `python generate.py --network=nvm.json -n 1000 -t 10 --transform=transform:exp samples.npy`
