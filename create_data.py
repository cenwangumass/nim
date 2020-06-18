import click
import numpy as np


def create_data(n, t):
    return np.random.gamma(2, 2, size=(n, t))


@click.command()
@click.option("-n", required=True, type=int)
@click.option("-t", required=True, type=int)
@click.argument("filename")
def main(n, t, filename):
    data = create_data(n, t)
    np.save(filename, data)


if __name__ == "__main__":
    main()
