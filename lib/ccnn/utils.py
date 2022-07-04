import math
import numexpr as ne
import numpy as np

from numpy import linalg as la


# TODO: Check for numpy to torch conversion in tensors to enable GPU usage

class RandomFourierTransformer:
    transform_matrix = 0
    transform_bias = 0
    n_components = 0
    gamma = 0

    def __init__(self, gamma, n_components):
        self.n_components = n_components
        self.gamma = gamma

    def fit(self, X):
        d = X.shape[1]
        self.transform_matrix = np.random.normal(loc=0, scale=math.sqrt(2*self.gamma), size=(d, self.n_components)).astype(np.float32)
        self.transform_bias = (np.random.rand(1, self.n_components) * 2 * math.pi).astype(np.float32)

    def transform(self, Y):
        ny = Y.shape[0]
        angle = np.dot(Y, self.transform_matrix)
        bias = self.transform_bias
        factor = np.float32(math.sqrt(2.0 / self.n_components))
        return ne.evaluate("factor*cos(angle+bias)")


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


def transform_and_pooling(patch, transformer, selected_group_size, gamma, nystrom_dim,
                          patch_per_side, pooling_size, pooling_stride):
    n = patch.shape[0]
    patch_per_image = patch.shape[1]
    selected_channel_num = patch.shape[2]
    pixel_per_patch = patch.shape[3]
    group_num = len(selected_group_size)  # TODO: always 1?
    feature_dim = group_num * nystrom_dim

    # construct Nystroem transformer
    patch = patch.reshape((n*patch_per_image, selected_channel_num, pixel_per_patch))
    psi = np.zeros((n*patch_per_image, group_num, nystrom_dim), dtype=np.float32)
    if transformer[0] == 0:
        transformer = np.empty(group_num, dtype=object)
        sum_value = 0
        for i in range(group_num):
            transformer[i] = RandomFourierTransformer(gamma=gamma, n_components=nystrom_dim)
            sub_patch = patch[:, sum_value:sum_value + selected_group_size[i]].reshape((n * patch_per_image, selected_group_size[i] * pixel_per_patch)) / math.sqrt(selected_group_size[i])

            transformer[i].fit(X=sub_patch)
            sum_value += selected_group_size[i]

    # Nystrom transformation
    sum_value = 0
    for i in range(group_num):
        sub_patch = patch[:, sum_value:sum_value + selected_group_size[i]].reshape((n * patch_per_image, selected_group_size[i] * pixel_per_patch)) / math.sqrt(selected_group_size[i])
        psi[:, i] = transformer[i].transform(Y=sub_patch)
        sum_value += selected_group_size[i]
    psi = psi.reshape((n, patch_per_image, feature_dim))
    # tprint("    transformation completes")

    # pooling
    pooling_per_side = int(patch_per_side/pooling_stride)
    pooling_per_image = pooling_per_side * pooling_per_side
    psi_pooling = np.zeros((n, pooling_per_image, feature_dim), dtype=np.float32)

    for pool_y in range(0, pooling_per_side):
        range_y = np.array(range(pool_y*pooling_stride, min(pool_y*pooling_stride+pooling_size, patch_per_side)))
        for pool_x in range(0, pooling_per_side):
            range_x = np.array(range(pool_x*pooling_stride, min(pool_x*pooling_stride+pooling_size, patch_per_side)))
            pooling_id = pool_x + pool_y * pooling_per_side
            index = []
            for y in range_y:
                for x in range_x:
                    index.append(x + y*patch_per_side)
            psi_pooling[:, pooling_id] = np.average(psi[:, np.array(index)], axis=1)

    # normalization
    psi_pooling = psi_pooling.reshape((n*pooling_per_image, feature_dim))
    psi_pooling -= np.mean(psi_pooling, axis=0)
    psi_pooling /= la.norm(psi_pooling) / math.sqrt(n*pooling_per_image)
    psi_pooling = psi_pooling.reshape((n, pooling_per_image*feature_dim))

    return psi_pooling.astype(np.float16), transformer


def low_rank_matrix_regression(x_train, y_train, x_test, y_test, d1, d2, reg, n_iter, learning_rate, ratio):
    n_train = x_train.shape[0]
    cropped_d1 = int(d1*ratio*ratio)
    A = np.zeros((9, cropped_d1*d2), dtype=np.float32) # 9-(d1*d2)
    A_sum = np.zeros((9, cropped_d1*d2), dtype=np.float32) # 9-(d1*d2)
    computation_time = 0

    for t in range(n_iter):
        mini_batch_size = 50
        batch_size = 10

        start = time.time()
        for i in range(0, batch_size):
            index = np.random.randint(0, n_train, mini_batch_size)
            x_sample = random_crop(x_train[index], d1, d2, ratio) # batch-(d1*d2)
            y_sample = y_train[index, 0:9] # batch-9

            # stochastic gradient descent
            XA = np.dot(x_sample, A.T)
            eXA = ne.evaluate("exp(XA)")
            # eXA = np.exp(XA)
            eXA_sum = np.sum(eXA, axis=1).reshape((mini_batch_size, 1)) + 1
            diff = ne.evaluate("eXA/eXA_sum - y_sample")
            grad_A = np.dot(diff.T, x_sample) / mini_batch_size
            # grad_A = np.dot((eXA/eXA_sum - y_sample).T, x_sample) / mini_batch_size
            A -= learning_rate * grad_A

        # projection to trace norm ball
        A, U, s, V = project_to_trace_norm(A, reg, cropped_d1, d2)
        end = time.time()
        computation_time += end - start

        A_sum += A
        if (t+1) % 250 == 0:
            dim = np.sum(s[0:25]) / np.sum(s)
            A_avg = A_sum / 250
            loss, error_train, error_test = evaluate_classifier(central_crop(x_train, d1, d2, ratio),
                                                                central_crop(x_test, d1, d2, ratio), y_train, y_test, A_avg)
            A_sum = np.zeros((9, cropped_d1*d2), dtype=np.float32)

            # debug
            tprint("iter " + str(t+1) + ": loss=" + str(loss) + ", train=" + str(error_train) + ", test=" + str(error_test) + ", dim=" + str(dim))
            # print(str(computation_time) + "\t" + str(error_test))

    A_avg, U, s, V = project_to_trace_norm(np.reshape(A_avg, (9*cropped_d1, d2)), reg, cropped_d1, d2)
    dim = min(np.sum((s > 0).astype(int)), 25)
    return V[0:dim]
