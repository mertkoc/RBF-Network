"""
Mert Koc
koc.15@osu.edu

"""
import numpy as np
import matplotlib.pyplot as plt
import os


# A simple RBF network since this is the final layer is a linear layer, it is not possible to use backpropagation
# algorithm; instead, we need to use linear regression.

class RBFNetwork:
    def __init__(self, input_size, output_size, learning_rate):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.W = np.random.uniform(-1, 1, (self.output_size, self.input_size + 1))  # Consider the bias term too
        self._output = np.zeros((self.output_size, 1))
        self.total_loss = 0

    def update_weights(self, input, desired):
        error = desired - self._output
        self.total_loss += 0.5 * error ** 2
        self.W += self.learning_rate * np.dot(error, input.T)

    def output(self, input):
        self.input = input.copy()
        self._output = np.dot(self.W, input)

    @staticmethod
    def Gaussian_activation(value, mean, vars):
        return np.exp((-1 / (2 * vars)) * ((value - mean) ** 2))

    def train(self, data_pts, data_list, max_epoch, means, variances):
        # Sample data
        means = means[:, 0].reshape((-1, 1))
        for epoch in range(max_epoch):
            index = np.random.choice(len(data_list), len(data_list), replace=False)
            for i in index:
                activation = self.Gaussian_activation(data_pts[i], means, variances).flatten()
                network_input = np.append(activation, [1], axis=0).reshape((-1, 1))
                self.output(network_input)
                self.update_weights(network_input, data_list[i])

    def test(self, data_pts, means, variances):
        means = means[:, 0].reshape((-1, 1))
        outputs = np.zeros(len(data_pts))
        for i in range(len(data_pts)):
            activation = self.Gaussian_activation(data_pts[i], means, variances).flatten()
            network_input = np.append(activation, [1], axis=0).reshape((-1, 1))
            self.output(network_input)
            outputs[i] = self._output.item()

        return outputs


def give_min_distances_cluster(data, means):
    for i in range(means.shape[0]):
        if i == 0:
            # Directly assign the distance to min_dist
            min_dist = ((data - means[i, 0]) ** 2, i)
        else:
            dist = (data - means[i, 0]) ** 2
            if dist <= min_dist[0]:
                min_dist = (dist, i)

    return min_dist[1]


def update_cluster_centers(means, data_list, cluster_ids):
    means = means.copy()
    for i in range(len(data_list)):
        means[cluster_ids[i], 0] = \
            (means[cluster_ids[i], 0] * means[cluster_ids[i], 1] + data_list[i]) / (means[cluster_ids[i, 1]] + 1)
        means[cluster_ids[i], 1] += 1  # Add one element to it

    return means


def K_means(data_list, K):
    # 1. Choose a set of K cluster centers randomly
    # Two methods from wikipedia: Forgy and Random Partition. The page says that Random Partition tends to
    # cluster around the center of the data (which is really what happened; therefore, I'll try the other one
    # k_arange = np.arange(K)
    # cluster_ids = np.random.choice(k_arange, len(data_list)).tolist()  # Choose which cluster to place the data
    cluster_ids = [-1 for x in range(len(data_list))]
    centers = np.random.choice(data_list, (K, 1), replace=False)
    clustering_done = True
    counts = np.zeros((K, 1))
    means = np.append(centers, counts, axis=1)  # One for counter the other for means
    # Now place them and calculate the means
    # update_cluster_centers(means, data_list, cluster_ids)
    # for i in range(len(data_list)):
    #     means[cluster_ids[i], 0] = \
    #         (means[cluster_ids[i], 0] * means[cluster_ids[i], 1] + data_list[i]) / (means[cluster_ids[i], 1] + 1)
    #     means[cluster_ids[i], 1] += 1  # Add one element to it
    while True:
        # 2. Assign the N to the K clusters using the Eucledean distance
        for i in range(len(data_list)):
            new_id = give_min_distances_cluster(data_list[i], means)
            if new_id != cluster_ids[i]:
                clustering_done = False  # We need to go on
                # 2. Update the cluster centers
                if cluster_ids[i] != -1:
                    means[cluster_ids[i], 0] = \
                        (means[cluster_ids[i], 0] * means[cluster_ids[i], 1] - data_list[i]) / (
                                means[cluster_ids[i], 1] - 1)
                    means[cluster_ids[i], 1] -= 1  # We have removed one item from current cluster
                cluster_ids[i] = new_id
                means[cluster_ids[i], 0] = \
                    (means[cluster_ids[i], 0] * means[cluster_ids[i], 1] + data_list[i]) / (
                            means[cluster_ids[i], 1] + 1)
                # means[cluster_ids[i], 0] += data_list[i]
                means[cluster_ids[i], 1] += 1  # Add one element to it

        # # Normalize centers
        # means[:,0] /= means[:,1]

        # 3. If any of the previous cluster centers changes, go back to step 2, otherwise stop
        if clustering_done:
            break
        else:
            clustering_done = True

    return cluster_ids, means


if __name__ == '__main__':
    folder_name = f"figures"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    # K = 4
    # data_list = np.random.normal(100, 0.33, (100,)).tolist() + np.random.normal(0, 0.33, (100,)).tolist() + \
    #             np.random.normal(20, 0.33, (100,)).tolist() + np.random.normal(50, 0.33, (100,)).tolist()
    # cluster_ids, means = K_means(data_list, K)
    # variances = np.zeros(K)
    # for i in range(len(data_list)):
    #     variances[cluster_ids[i]] += (means[cluster_ids[i], 0] - data_list[i]) ** 2
    #
    # variances = variances / means[:, 1]
    #
    # print(means)
    # print(variances)

    # Create datapoints
    # K = 2
    # K = 4
    # K = 7
    # K = 11
    # K = 16
    K_list = [2, 4, 7, 11, 16]

    # learning_rate = 0.01
    # learning_rate = 0.02
    learning_rate_list = [0.01, 0.02]

    for K in K_list:
        for learning_rate in learning_rate_list:
            data_pts = np.random.uniform(0, 1, 75)
            data_list = 0.5 + 0.4 * np.sin(2 * np.pi * data_pts)
            # Corrupt the data with noise
            noise = np.random.uniform(-0.1, 0.1, 75)
            data_list_corrupted = noise + data_list

            cluster_ids, means = K_means(data_list, K)
            variances = np.zeros(K)
            for i in range(len(data_list)):
                variances[cluster_ids[i]] += (means[cluster_ids[i], 0] - data_list[i]) ** 2

            variances = variances / means[:, 1]
            mask = np.ones(variances.shape, dtype=np.bool)
            mask_idx = []
            for i in range(K):
                if means[i, 1] == 1:
                    mask[i] = False
                    mask_idx.append(i)
            for i in mask_idx:
                variances[i] = np.mean(variances[mask])
            variances = variances.reshape((-1, 1))

            network = RBFNetwork(K, 1, learning_rate)
            network.train(data_pts, data_list_corrupted, max_epoch=100, means=means, variances=variances)
            print(f"Total loss: {network.total_loss[0, 0]:.2f}")
            test_subject = np.arange(0, 1.001, step=0.01)
            # test_subject = np.arange(0, 2.001, step=0.01)
            output_pts = network.test(test_subject, means, variances)

            plt.plot(data_pts, data_list, 'go', label='sampled points')
            plt.plot(test_subject, 0.5 + 0.4 * np.sin(2 * np.pi * test_subject),
                     label='function', color='b')
            for count, var_x in enumerate(variances):
                plt.plot(test_subject,
                         network.W[:, count] * network.Gaussian_activation(test_subject, means[count, 0], var_x),
                         label=f"gaussian{count}")
            plt.plot(test_subject, output_pts, color='r', label='generated by RBF')
            plt.legend(loc='best')
            plt.title(f"RBF Network for K={K}, learning rate={learning_rate}, loss={network.total_loss[0, 0]:.2f}")
            plt.ylabel('output')
            plt.xlabel('input')
            plt.savefig(
                f"{folder_name}/k={K}_learning_rate{learning_rate:.2f}.png",
                bbox_inches='tight')
            plt.close()

    # Second case

            dmax = max(means[:, 0]) - min(means[:, 0])
            variances = np.ones(K)
            variances = variances * (dmax**2 / (2 * K))
            # variances = np.zeros(K)
            # for i in range(len(data_list)):
            #     variances[cluster_ids[i]] += (means[cluster_ids[i], 0] - data_list[i]) ** 2
            #
            # variances = variances / means[:, 1]
            variances = variances.reshape((-1, 1))

            network = RBFNetwork(K, 1, learning_rate)
            network.train(data_pts, data_list_corrupted, max_epoch=100, means=means, variances=variances)
            print(f"Total loss2: {network.total_loss[0, 0]:.2f}")
            # test_subject = np.arange(0, 1.001, step=0.01)
            output_pts = network.test(test_subject, means, variances)

            plt.plot(data_pts, data_list, 'go', label='sampled points')
            plt.plot(test_subject, 0.5 + 0.4 * np.sin(2 * np.pi * test_subject),
                     label='function', color='b')
            plt.plot(test_subject, output_pts, color='r', label='generated by RBF')
            for count, var_x in enumerate(variances):
                plt.plot(test_subject,
                         network.W[:, count] * network.Gaussian_activation(test_subject, means[count, 0], var_x),
                         label=f"gaussian{count}")
            plt.legend(loc='best')
            plt.title(f"RBF Network for K={K}, learning rate={learning_rate}, loss={network.total_loss[0, 0]:.2f}")
            plt.ylabel('output')
            plt.xlabel('input')
            plt.savefig(
                f"{folder_name}/second-case_k={K}_learning_rate{learning_rate:.2f}.png",
                bbox_inches='tight')
            plt.close()
