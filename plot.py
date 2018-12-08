#!/usr/bin/env python3

import os
import struct
import numpy as np
import argparse



def load_ndfile(filename):
    with open(filename, 'rb') as f:
        dtype = struct.unpack('8s', f.read(8))[0].decode('utf-8').strip('\x00')
        rank = struct.unpack('i', f.read(4))[0]
        dims = struct.unpack('i' * rank, f.read(4 * rank))
        data = f.read()
        return np.frombuffer(data, dtype=dtype).reshape(dims)



def load_checkpoint(btdir):
    database = dict()

    for patch in os.listdir(btdir):

        fd = os.path.join(btdir, patch)
        pd = dict()

        for field in os.listdir(fd):
            fe = os.path.join(fd, field)
            pd[field] = load_ndfile(fe)

        database[patch] = pd

    return database



def imshow_database(database):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    for key, patch in database.items():
        X = patch['cell_coords'][:,:,0]
        Y = patch['cell_coords'][:,:,1]
        D = patch['conserved'][:,:,0]

        extent = [X[0,0], X[-1,0], Y[0,0], Y[0,-1]]

        ax1.imshow(D.T, origin='bottom', extent=extent, vmin=0.0, vmax=1.1)

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs='+')
    args = parser.parse_args()

    db = load_checkpoint(args.filenames[0])
    imshow_database(db)
