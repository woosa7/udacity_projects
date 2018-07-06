# Lec: 2-8
# https://udemy.com/deep-learning-convolutional-neural-networks-theano-tensorflow
from __future__ import print_function, division
from builtins import range
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import  sys

# load the famous Lena image
img = mpimg.imread('lena.png')
print(img.shape)

# original image
plt.imshow(img)
plt.show()


# make it B&W
bw = img.mean(axis=2)
plt.imshow(bw)
plt.show()

plt.imshow(bw, cmap='gray')
plt.show()


# create a 2-dimentional Gaussian filter
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
plt.show()

# now the convolution
out = convolve2d(bw, W)
plt.imshow(out, cmap='gray')
plt.show()

# what's that weird black stuff on the edges? let's check the size of output
# 512 --> 531
print(out.shape)
# after convolution, the output signal is N1 + N2 - 1


# we can also just make the output the same size as the input
out = convolve2d(bw, W, mode='same')
plt.imshow(out, cmap='gray')
plt.show()
print(out.shape)


# in color
out3 = np.zeros(img.shape)
print(out3.shape)
for i in range(3):
    out3[:,:,i] = convolve2d(img[:,:,i], W, mode='same')
# out3 /= out3.max() # can also do this if you didn't normalize the kernel
plt.imshow(out3)
plt.show() # does not look like anything
