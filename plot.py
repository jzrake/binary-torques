#!/usr/bin/env python3

import struct
import numpy as np
import argparse



def load(filename):
    with open(filename, 'rb') as f:
        dtype = struct.unpack('8s', f.read(8))[0].decode('utf-8').strip('\x00')
        rank = struct.unpack('i', f.read(4))[0]
        dims = struct.unpack('i' * rank, f.read(4 * rank))
        data = f.read()
        return np.frombuffer(data, dtype=dtype).reshape(dims)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", nargs='+')

    args = parser.parse_args()
    x = load(args.filename[0])

    import matplotlib.pyplot as plt

    for fname in args.filename[1:]:
        y = load(fname)
        plt.plot(x, y, '-o', mfc='none')

    plt.show()
