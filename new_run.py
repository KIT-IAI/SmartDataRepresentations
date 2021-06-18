import numpy as np
import matplotlib.pyplot as plt

import wandb


def main():

    x = np.arange(10)
    y = np.square(np.arange(10)) + np.random.rand(10) * 10

    wandb.init(project='ci_test')

    for x_i, y_i in zip(x, y):
        wandb.log({'x': x_i, 'y': y_i})

    table = wandb.Table(data=list(zip(x, y)), columns=['x', 'y'])
    wandb.log(
        {
            'my_table_plot': wandb.plot.line(table, 'x', 'y', title='test1')
        }
    )

    wandb.log({'param': 4})

if __name__ == '__main__':
    main()