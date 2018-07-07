# Lec: 2-10
# https://udemy.com/deep-learning-convolutional-neural-networks-theano-tensorflow
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
# from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from datetime import datetime

def convolve2d_long(X, W):
    t0 = datetime.now()
    n1, n2 = X.shape
    m1, m2 = W.shape
    Y = np.zeros((n1 + m1 - 1, n2 + m2 - 1))
    for i in range(n1 + m1 - 1):
        for ii in range(m1):
            for j in range(n2 + m2 - 1):
                for jj in range(m2):
                    if i >= ii and j >= jj and i - ii < n1 and j - jj < n2:
                        Y[i,j] += W[ii,jj]*X[i - ii,j - jj]
    print("elapsed time:", (datetime.now() - t0))
    return Y


def convolve2d_normal(X, W):
    t0 = datetime.now()
    n1, n2 = X.shape
    m1, m2 = W.shape
    Y = np.zeros((n1 + m1 - 1, n2 + m2 - 1))
    for i in range(n1):
        for j in range(n2):
            Y[i:i+m1,j:j+m2] += X[i,j]*W
    print("elapsed time:", (datetime.now() - t0))
    return Y

# same size as input
def convolve2d_same(X, W):
    n1, n2 = X.shape
    m1, m2 = W.shape
    Y = np.zeros((n1 + m1 - 1, n2 + m2 - 1))
    for i in range(n1):
        for j in range(n2):
            Y[i:i+m1,j:j+m2] += X[i,j]*W
    ret = Y[m1//2:-m1//2+1,m2//2:-m2//2+1]
    assert(ret.shape == X.shape)
    return ret

# smaller than input
def convolve2d_smaller(X, W):
    n1, n2 = X.shape
    m1, m2 = W.shape
    Y = np.zeros((n1 + m1 - 1, n2 + m2 - 1))
    for i in range(n1):
        for j in range(n2):
            Y[i:i+m1,j:j+m2] += X[i,j]*W
    ret = Y[m1-1:-m1+1,m2-1:-m2+1]
    return ret

# ----------------------------------------------------------------------------
# main

# load the famous Lena image
img = mpimg.imread('lena.png')

# what does it look like?
# plt.imshow(img)
# plt.show()

# make it B&W
bw = img.mean(axis=2)
plt.imshow(bw, cmap='gray')
plt.title("original B&W")
plt.show()


# create a Gaussian filter
N1 = N2 = 20
center = N1/2
W = np.zeros((N1, N2))
for i in range(N1):
    for j in range(N2):
        dist = (i - center)**2 + (j - center)**2  # square distance from the center
        W[i, j] = np.exp(-dist / 50.)
W /= W.sum() # normalize the kernel


# let's see what the filter looks like
plt.imshow(W, cmap='gray')
plt.title("Gaussian filter")
plt.show()

# convolution : normal
out = convolve2d_normal(bw, W)
plt.imshow(out, cmap='gray')
plt.title("convolve2d_normal")
plt.show()

# convolution : same size
out = convolve2d_same(bw, W)
plt.imshow(out, cmap='gray')
plt.title("convolve2d_same")
plt.show()

# convolution : smaller size
out = convolve2d_smaller(bw, W)
plt.imshow(out, cmap='gray')
plt.title("convolve2d_smaller")
plt.show()

# convolution : long time
# out = convolve2d_long(bw, W)
# plt.imshow(out, cmap='gray')
# plt.title("convolve2d_long")
# plt.show()



# what's that weird black stuff on the edges? let's check the size of output
# print(out.shape)
# after convolution, the output signal is N1 + N2 - 1

# try it in color
out = np.zeros(img.shape)
W /= W.sum()
for i in range(3):
    out[:,:,i] = convolve2d_same(img[:,:,i], W)
plt.imshow(out)
plt.title("convolve2d in colr")
plt.show()



