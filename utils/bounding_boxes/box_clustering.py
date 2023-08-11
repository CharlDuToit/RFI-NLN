
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def k_means_clustering(data, max_iterations, num_clusters, convergence_error):
    # Initialize means
    x_min, y_min = np.min(data, axis=0)
    x_max, y_max = np.max(data, axis=0)
    x_means = np.linspace(x_min, x_max, num_clusters)
    y_means = np.linspace(y_min, y_max, num_clusters)
    means = np.column_stack((x_means, y_means))

    # Initialize variables
    num_points = data.shape[0]
    distances = np.zeros((num_points, num_clusters))
    loss = np.inf
    prev_loss = np.inf
    iteration = 0

    # Run k-means clustering
    while iteration < max_iterations and (abs(1 - loss / prev_loss) > convergence_error or iteration == 0):
        iteration += 1
        prev_loss = loss

        # Calculate distances
        for i in range(num_clusters):
            distances[:, i] = np.linalg.norm(data - means[i], axis=1)

        # Assign points to clusters
        assignments = np.argmin(distances, axis=1)

        # Check if any means have no points assigned to them
        empty_means = []
        for i in range(num_clusters):
            if np.sum(assignments == i) == 0:
                empty_means.append(i)

        # Remove empty means and reduce number of clusters
        if len(empty_means) > 0:
            means = np.delete(means, empty_means, axis=0)
            num_clusters -= len(empty_means)
            distances = np.delete(distances, empty_means, axis=1)

        # Update means
        for i in range(num_clusters):
            means[i] = np.mean(data[assignments == i], axis=0)

        # Calculate loss
        loss = np.sum(distances[np.arange(num_points), assignments])

    return means, assignments,  loss, iteration


def plot_clusters(data, means, assignments, filename='./clusters.png'):
    # Get number of clusters
    num_clusters = means.shape[0]

    # Create color map for clusters
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, num_clusters))

    # Create scatter plot
    for i in range(num_clusters):
        cluster_data = data[assignments == i]
        #plt.scatter(cluster_data[:, 0], cluster_data[:, 1], c=[colors[i]], s=50, alpha=0.8, label='Cluster {}'.format(i+1))
        #plt.scatter(cluster_data[:, 0], cluster_data[:, 1], c=colors[i], s=50, alpha=0.8, label='Cluster {}'.format(i+1))
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], s=50, alpha=0.8, label='Cluster {}'.format(i+1))

        #plt.scatter(means[i, 0], means[i, 1], marker='x', s=200, c=[colors[i]])
        #plt.scatter(means[i, 0], means[i, 1], marker='x', s=200, c=colors[i])
        #plt.scatter(means[i, 0], means[i, 1], marker='x', s=200)

    # Add legend and labels
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.savefig(filename)
    plt.close('all')
    # plt.show()


def read_widths_heights(filename):
    df = pd.read_csv(filename)
    widths = df['width']
    heights = df['height']
    points = np.column_stack((widths, heights))
    return points


def main():
    f = './lofar_width_heights_test_64.csv'
    f = './lofar_boxes/lofar_width_heights_test_64.csv'
    points = read_widths_heights(f)
    means, assignments, loss, iteration = k_means_clustering(points, 10, 4, 0.01)
    plot_clusters(points, means, assignments)


if __name__ == '__main__':
    main()

