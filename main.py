# Snir David Nahari 205686538
import sys
import numpy as np
from numpy import inf
import matplotlib.pyplot as plt


def reshape(image_fname, centroids_fname):
    centroids = np.loadtxt(centroids_fname)  # load centroids
    orig_pixels = plt.imread(image_fname)
    pixels = orig_pixels.astype(float) / 255
    # Reshape the image(128x128x3) into an Nx3 matrix where N = number of pixels.
    return pixels.reshape(-1, 3), centroids


def find_closest_centroids(pixel, centroids):
    min_dist = inf
    centroid_id = -1
    for centroid in centroids:
        distance = abs(pixel[0] - centroid[0]) + abs(pixel[1] - centroid[1]) + abs(pixel[2] - centroid[2])
        if distance < min_dist:
            min_dist = distance
            centroid_id = np.where(centroids == centroid)
    return centroid_id[0][0]


def k_means(pixels_list, centroids_list, output_file):
    pixels_to_centroids = []
    for i in range(len(centroids_list)):
        pixels_to_centroids.append([])
    for i in range(19):
        for pixel in pixels_list:
            id = find_closest_centroids(pixel, centroids_list)
            pixels_to_centroids[id].append(pixel)


if __name__ == '__main__':
    pixels, centroids = reshape(sys.argv[1], sys.argv[2])
    k_means(pixels, centroids, sys.argv[3])
