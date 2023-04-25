import torch

# Turns a dictionary into a class
class Dict2Class(object):
    def __init__(self, d):
        for key in d:
            setattr(self, key, d[key])


def num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



hash_key_1 = torch.tensor(0x58a849af6cbf585a, dtype=torch.long)
hash_key_2 = torch.tensor(0x5ca118cb4d828a5e, dtype=torch.long)
def hash64(x):
    # Convert the input to a 64-bit unsigned integer tensor
    x = torch.tensor(x, dtype=torch.long)
    # Use the PyTorch's bitwise XOR operator to generate a hash value
    # We use two random 64-bit integers as the hash key
    # Note: the key can be any fixed 64-bit integer or a set of integers
    h = torch.bitwise_xor(x ^ hash_key_1.to(x.device), hash_key_2.to(x.device))
    return h