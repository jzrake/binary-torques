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
    parser.add_argument("filenames", nargs='+')
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    if len(args.filenames) == 2:
        x = load(args.filenames[0])
        y = load(args.filenames[1])
        plt.plot(x, y, '-o', mfc='none')
        plt.show()

    else:
        y = load(args.filenames[0])
        print(y.shape) 
 
