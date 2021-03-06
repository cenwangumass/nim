import importlib

import click
import numpy as np
import torch
from torch.optim import Adam

from nim import NVM, NVL


def train_epoch(net, x, optimizer, batch_size):
    n = x.shape[0]

    total_loss = 0
    n_batches = n // batch_size
    for batch in range(n_batches):
        start = batch * batch_size
        end = (batch + 1) * batch_size
        x_minibatch = x[start:end]

        optimizer.zero_grad()
        output = net(x_minibatch)
        loss = net.loss_function(x_minibatch, *output)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()

    return total_loss / n


def train(net, data, optimizer, epochs, batch_size, device):
    device = torch.device(device)
    net = net.to(device)
    net.train()
    x = torch.from_numpy(data).float().to(device)

    for epoch in range(1, epochs + 1):
        loss = train_epoch(net, x, optimizer, batch_size)

        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, Loss: {loss}")


@click.command()
@click.option("--hidden-size", default=32, type=int)
@click.option("--transform")
@click.option("--lr", default=1e-3, type=float)
@click.option("--epochs", default=1000, type=int)
@click.option("--gpu", default=False, is_flag=True)
@click.option("--save")
@click.argument("network")
@click.argument("data")
def main(network, data, hidden_size, transform, lr, epochs, gpu, save):
    x = np.load(data)

    if transform is not None:
        module_name, function_name = transform.split(":")
        module = importlib.import_module(module_name)
        function = getattr(module, function_name)

        x = function(x)

    n = x.shape[0]

    if network == "nvm":
        net = NVM(1, hidden_size)
    elif network == "nvl":
        net = NVL(1, hidden_size)
    else:
        raise ValueError("unknown network architecture")
    optimizer = Adam(net.parameters(), lr=lr)

    train(
        net=net,
        data=x,
        optimizer=optimizer,
        epochs=epochs,
        batch_size=n,
        device="cuda" if gpu else "cpu",
    )

    if save is None:
        save = network
    net.save(save)


if __name__ == "__main__":
    main()
