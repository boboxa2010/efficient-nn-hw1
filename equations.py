def flops(image_size, batch): 
    return batch * (17712 * image_size**2 + 313344)


def memory(image_size, batch):
    return 4181312 + 76 * batch * image_size**2


def latency(image_size, batch, theta):
    pass


def energy(image_size, batch, theta_energy): 
    pass
