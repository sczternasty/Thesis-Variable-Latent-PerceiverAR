import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import gzip
import numpy as np

def enwik8(path=None, n_train=int(90e6), n_valid=int(5e6), n_test=int(5e6)):
    cwd = os.getcwd()
    if path is None:
        path = cwd + '\\data\\enwik8.gz'

    with gzip.open(path) if path.endswith('.gz') else open(path) as file:
        X = np.frombuffer(file.read(n_train + n_valid + n_test), dtype=np.uint8)
        trX, vaX, teX = np.split(X, [n_train, n_train + n_valid])
        return torch.from_numpy(trX.copy()), torch.from_numpy(vaX.copy()), torch.from_numpy(teX.copy())

def sample_batch(data, length=100, batch_size=32):
    starts = torch.randint(size=(batch_size,), low=0, high=data.size(0) - length - 1)

    X = [data[start:start + length] for start in starts]
    y = [data[start + 1:start + length + 1] for start in starts]

    X = torch.cat([s.unsqueeze(dim=0) for s in X], dim=0).to(torch.long)
    y = torch.cat([s.unsqueeze(dim=0) for s in y], dim=0).to(torch.long)

    return X, y

def sample(probs, temp=1.0):
    if temp == 0.0:
        return probs.argmax()

    p = F.softmax(probs / temp, dim=0)
    out = torch.multinomial(p, 1)
    return out


def sample_sequence(model, seed, cross_seq_len, max_context, length=600, temp=0.8, device='cpu'):
    seq = seed.detach().clone()

    print('Context ->>', end='', flush=True)
    for c in seed:
        print(str(chr(max(32, c))), end='', flush=True)
    print('<<-- Context ', end='', flush=True)

    model = model.to(device)
    seq = seq.to(device)
    model.eval()

    with torch.no_grad():
        for _ in range(length):
            X = seq[-max_context:]
            if cross_seq_len:
                output = model(X.unsqueeze(0), cross_seq_len)
            else:
                output = model(X.unsqueeze(0))


            c = sample(output[0, -1, :], temp)


            print(str(chr(max(32, c))), end='', flush=True)

            seq = torch.cat([seq, c], dim=0)


    print()
    return seq[-length:].detach().cpu()
