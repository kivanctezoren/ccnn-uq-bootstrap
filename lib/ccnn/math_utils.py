import numpy as np


def get_pixel_vector(center_x, center_y, radius, image_width):
    kernel_size = int(radius * 2 + 1)
    vector = np.zeros(kernel_size ** 2, dtype=int)
    for y in range(0, kernel_size):
        for x in range(0, kernel_size):
            index = (center_x + x - radius) + (center_y + y - radius) * image_width
            vector[x + y * kernel_size] = index
    return vector
