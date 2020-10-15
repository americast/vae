import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
plt.switch_backend('agg')


def merge_images(images, size):
    # merge a mini-batch of images into a single grid of images
    H, W, C = images.shape[1], images.shape[2], images.shape[3]
    merged_img = np.zeros((H * size[0], W * size[1], C))

    for idx, img in enumerate(images):
        i = idx // size[1]  # row number
        j = idx % size[1]   # column number

        merged_img[H * i: H * (i+1), W * j: W * (j+1), :] = img

    return merged_img


def imsave(X, path):
    # save the batch of images in X as a single image in path
    grid_sizes = {4: (2, 2),
                  8: (2, 4),
                  16: (4, 4),
                  32: (4, 8),
                  64: (8, 8),
                  128: (8, 16),
                  256: (16, 16),
                  512: (16, 32),
                  1024: (32, 32)}
    X = X[:256]
    N = X.shape[0]
    size = grid_sizes.get(N, (1, N))

    imgs = (X.to('cpu').numpy().transpose(0, 2, 3, 1) + 1.)/2.
    img = merge_images(imgs, size)
    plt.figure()
    plt.imshow(img)
    plt.savefig(path)
    plt.close()

def imsave_custom(X, path):
    X = X.squeeze(0) 
    img = (X.to('cpu').numpy().transpose(1, 2, 0) + 1.)/2.
    plt.imsave(path, img)
