import importlib
import json

import click
import numpy as np
import torch

from nim import NVM, NVL


def load(filename):
    with open(filename) as f:
        description = json.load(f)

    type = description.pop("type")
    if type == "nvm":
        network_class = NVM
    elif type == "nvl":
        network_class = NVL
    else:
        raise ValueError("unknown network architecture")

    net = network_class(**description)

    weights_filename = filename.replace(".json", ".pth")
    net.load_state_dict(torch.load(weights_filename, map_location="cpu"))

    return net


@click.command()
@click.option("--transform")
@click.option("-n", required=True, type=int)
@click.option("-t", required=True, type=int)
@click.argument("network", type=click.Path())
@click.argument("save")
def main(network, transform, n, t, save):
    net = load(network)

    y = net.generate((n, t)).numpy()

    if transform is not None:
        module_name, function_name = transform.split(":")
        module = importlib.import_module(module_name)
        function = getattr(module, function_name)

        y = function(y)

    np.save(save, y)


if __name__ == "__main__":
    main()
