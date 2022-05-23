#!/home/alal/anaconda3/envs/tf/bin/python3

# shebang configured for my machine - replace with yours or install dependencies
# into local python and call as
# python3 colourTransfer.py <source> <target>

# %%
import os, sys
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import ot
rng = np.random.RandomState(42)

# %%
def im2mat(img):
    """Converts an image to matrix (one pixel per line)"""
    return img.reshape((img.shape[0] * img.shape[1], img.shape[2]))

def mat2im(X, shape):
    """Converts back a matrix to an image"""
    return X.reshape(shape)

def minmax(img):
    return np.clip(img, 0, 1)

# %%
def colour_transport(source, target, nb = 500,
        mixed = "output.jpg"):
    # can make this a command line application
    I1 = plt.imread(source).astype(np.float64)
    I2 = plt.imread(target).astype(np.float64)
    if source.split(".")[-1] == "jpg":
        I1 = I1/256
    if target.split(".")[-1] == "jpg":
        I2 = I2/256

    X1 = im2mat(I1); X2 = im2mat(I2)

    # take sample
    idx1 = rng.randint(X1.shape[0], size=(nb,))
    idx2 = rng.randint(X2.shape[0], size=(nb,))
    Xs = X1[idx1, :]; Xt = X2[idx2, :]

    # transport computation
    ot_emd = ot.da.EMDTransport()
    ot_emd.fit(Xs=Xs, Xt=Xt)
    transp_Xs_emd = ot_emd.transform(Xs=X1)
    I1t = minmax(mat2im(transp_Xs_emd, I1.shape))

    if mixed: # path can be null to show output
        # write
        plt.figure(dpi = 200)
        plt.imshow(I1t)
        plt.axis('off')
        plt.savefig(mixed, bbox_inches='tight')
    else:
        return I1t

# %%
if __name__ == '__main__':
    a, b = sys.argv[1], sys.argv[2]
    print(f"making {a} look like {b}")
    colour_transport(a, b)
