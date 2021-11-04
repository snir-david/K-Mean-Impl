# Snir David Nahari 205686538
import sys
import numpy as np
from numpy import inf, sqrt
import matplotlib.pyplot as plt


def initialize_centroids(num_of_cent):
    return np.random.rand(num_of_cent, 3)


# drawing plot according to the num of iteration and loss cost
def draw_plot(x, y):
    plt.plot(x, y)
    plt.xlabel("Iteration")
    plt.ylabel("Cost Function")
    plt.show()


# cost function - checking the distance of each pixel from it's centroid and summing the distance
def kmean_loss_function(pixels_to_cents, cent_list):
    cost = 0
    for i in range(len(cent_list)):
        for pixel in pixels_to_cents[i]:
            cost += np.linalg.norm(pixel - cent_list[i])
    return cost


def reshape(image_fname, centroids_fname):
    centroids = np.loadtxt(centroids_fname)  # load centroids
    orig_pixels = plt.imread(image_fname)
    pixels = orig_pixels.astype(float) / 255
    # Reshape the image(128x128x3) into an Nx3 matrix where N = number of pixels.
    return pixels.reshape(-1, 3), centroids


def save_image(pixels_list, centroids, pix_cents, shape):
    pixels = np.zeros(len(pixels_list))
    for i in range(len(pixels_list)):
        pixels[i] = centroids[pix_cents.index(pixels_list[i])]
    pic = pixels.reshape(shape)
    plt.imsave("newImage.jpeg", pic)


# giving pixel to centroids list, calculating mean of each centroid and rounding to 4 decimal points
def find_new_centroids(pixel_to_centroids_list, old_cent):
    new_centroids = []
    for i in range(len(pixel_to_centroids_list)):
        if len(pixel_to_centroids_list[i]):
            mean = np.mean(pixel_to_centroids_list[i], axis=0)
            new_centroids.append(mean.round(4))
        else:
            new_centroids.append(old_cent[i])
    return new_centroids


# getting pixel and centroids list - return the id of the closest centroid
def find_closest_centroids(pixel, centroids_list):
    min_dist = inf
    idx = -1
    for i in range(len(centroids_list)):
        distance = np.linalg.norm(pixel - centroids_list[i]) ** 2
        if distance < min_dist:
            min_dist = distance
            idx = i
    return idx


# K-Mean Algorithm - getting pixels list, centroids list and output file name.
# return void, result will be in output file.
def k_means(pixels_list, centroids_list, output_file):
    loss = []
    iteration = []
    # opening new file for output file for writing the results
    out_file = open(output_file, 'w+')
    # variable that will check convergence before 20 iterations
    convergence = False
    # for 20 iterations or convergence - cluster pixels to centroids and update centroids.
    for i in range(20):
        if not convergence:
            iteration.append(i)
            # initialize list of list (each centroids has list in the primary list)
            pixels_to_centroids = []
            for j in range(len(centroids_list)):
                pixels_to_centroids.append([])
            # iterating pixel in pixels list and finding closest centroid in the current iteration
            for pixel in pixels_list:
                # getting closest centroid id
                idx = find_closest_centroids(pixel, centroids_list)
                # adding pixel to the right centroid list
                pixels_to_centroids[idx].append(pixel)
            # saving old centroids list for comparing and checking convergence
            prev_cent = centroids_list
            # getting new centroids by mean function
            new_cent = find_new_centroids(pixels_to_centroids, centroids_list)
            loss.append(kmean_loss_function(pixels_to_centroids, new_cent))
            out_file.write(f"[iter {i}]:{','.join([str(i) for i in new_cent])}\n")
            # checking if some value changed from last iteration
            if np.array_equal(prev_cent, new_cent):
                convergence = True
                centroids_list = new_cent
            else:
                centroids_list = new_cent
    # draw_plot(iteration, loss)
    out_file.close()


if __name__ == '__main__':
    pixels, centroids = reshape(sys.argv[1], sys.argv[2])
    k_means(pixels, centroids, sys.argv[3])
