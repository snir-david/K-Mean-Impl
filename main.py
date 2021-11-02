# Snir David Nahari 205686538
import sys
import numpy as np
import matplotlib.pyplot as plt

def reshape(image_fname, centroids_fname):
    centroids = np.loadtxt(centroids_fname)  # load centroids
    orig_pixels = plt.imread(image_fname)
    pixels = orig_pixels.astype(float) / 255
    # Reshape the image(128x128x3) into an Nx3 matrix where N = number of pixels.
    return pixels.reshape(-1, 3), centroids

# def find_closest_centroids(pixel, centroids):



def k_means(pixels_list, centroids_list, output_file):
    print(pixels_list)
    print(centroids_list)
    # for i in range(19):
    #     for pixel in pixels_list:


if __name__ == '__main__':
    pixels, centroids = reshape(sys.argv[1], sys.argv[2])
    k_means(pixels, centroids, sys.argv[3])
