# Snir David Nahari 205686538
import sys
import numpy as np
from numpy import inf, sqrt
import matplotlib.pyplot as plt


def reshape(image_fname, centroids_fname):
    centroids = np.loadtxt(centroids_fname)  # load centroids
    orig_pixels = plt.imread(image_fname)
    pixels = orig_pixels.astype(float) / 255
    # Reshape the image(128x128x3) into an Nx3 matrix where N = number of pixels.
    return pixels.reshape(-1, 3), centroids


def find_new_centroids(pixel_to_centroids_list, old_cent):
    new_centroids = []
    for i in range(len(pixel_to_centroids_list)):
        sum_pixels = sum(pixel_to_centroids_list[i])
        num_of_pixels = len(pixel_to_centroids_list[i])
        if num_of_pixels > 0:
            new_cent = sum_pixels / num_of_pixels
            new_centroids.append(new_cent.round(4))
        else:
            new_centroids.append(old_cent[i])

    return new_centroids


def find_closest_centroids(pixel, centroids):
    min_dist = inf
    centroid_id = -1
    for centroid in centroids:
        distance = sqrt(
            pow(pixel[0] - centroid[0], 2) + pow(pixel[1] - centroid[1], 2) + pow(pixel[2] - centroid[2], 2))
        if distance < min_dist:
            min_dist = distance
            centroid_id = np.where(centroids == centroid)
    return centroid_id[0][0]


def k_means(pixels_list, centroids_list, output_file):
    new_cent = []
    coverage = 0
    old_cent = centroids_list
    for i in range(20):
        if coverage < 2:
            pixels_to_centroids = []
            for j in range(len(centroids_list)):
                pixels_to_centroids.append([])
            for pixel in pixels_list:
                id = find_closest_centroids(pixel, centroids_list)
                pixels_to_centroids[id].append(pixel)
            old_cent = centroids_list
            new_cent = find_new_centroids(pixels_to_centroids, centroids_list)
            print("iter: ")
            print(i)
            print(new_cent)
            compare = old_cent == new_cent
            if compare.all():
                coverage += 1
                print(coverage)
                centroids_list = new_cent
            else:
                centroids_list = new_cent


if __name__ == '__main__':
    pixels, centroids = reshape(sys.argv[1], sys.argv[2])
    k_means(pixels, centroids, sys.argv[3])
