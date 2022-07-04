import numpy as np


def get_pixel_vector(center_x, center_y, radius, image_width):
    kernel_size = int(radius * 2 + 1)
    vector = np.zeros(kernel_size ** 2, dtype=int)
    for y in range(0, kernel_size):
        for x in range(0, kernel_size):
            index = (center_x + x - radius) + (center_y + y - radius) * image_width
            vector[x + y * kernel_size] = index
    return vector


def zca_whitening(inputs):
    inputs -= np.mean(inputs, axis=0)
    sigma = np.dot(inputs.T, inputs) / inputs.shape[0]
    u, s, v = np.linalg.svd(sigma)
    epsilon = 0.1
    zca_matrix = np.dot(np.dot(u, np.diag(1.0/np.sqrt(s + epsilon))), u.T).astype(np.float32)

    i = 0
    while i < inputs.shape[0]:
        next_i = min(inputs.shape[0], i+100000)
        inputs[i:next_i] = np.dot(inputs[i:next_i], zca_matrix.T)
        i = next_i

    return inputs
