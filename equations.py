import numpy as np


def flops(image_size, batch):
    return batch * (17712 * image_size**2 + 313344)


def memory(image_size, batch):
    return 4181312 + 76 * batch * image_size**2


def bytes_moved(image_size, batch):
    return 532 * batch * image_size**2 + 8592 * batch + 4181264


def latency(image_size, batch, theta):
    bias, theta_f, theta_d = theta

    f = flops(image_size, batch)
    d = bytes_moved(image_size, batch)

    return bias + np.maximum(f / theta_f, d / theta_d)


def energy(image_size, batch, theta_energy):
    theta_f, theta_d, theta_base, theta_latency = theta_energy

    f = flops(image_size, batch)
    d = bytes_moved(image_size, batch)
    t = latency(image_size, batch, theta_latency)

    return theta_f * f + theta_d * d + theta_base * t
