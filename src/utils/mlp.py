import torch.nn as nn

def construct_mlp(sizes, activation, output_activation=nn.Identity):

    layers = []

    for i in range(len(sizes-1)):
        act = activation if i != len(sizes) - 2 else output_activation
        layers += [nn.Linear(size[i], size[i+1]), act()]

    return nn.Sequential(*layers)
