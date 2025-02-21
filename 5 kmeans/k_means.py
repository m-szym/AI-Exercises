import numpy as np


def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def initialize_centroids_forgy(data, k):
    # TODO implement random initialization
    idx = np.random.choice(range(data.shape[0]), k, replace=False)
    return data[idx]


def initialize_centroids_kmeans_pp(data, k):
    # TODO implement kmeans++ initizalization
    centroids = np.zeros((k, data.shape[1]))
    centroids[0] = data[np.random.choice(range(data.shape[0]))]

    i = 1
    while i < k:
        max_distance = 0
        best_candidate = None

        for candidate in data:
            if np.any(centroids == candidate):
                continue

            candidate_distance = np.sum(np.sqrt(np.sum((candidate - centroids) ** 2)))

            if candidate_distance > max_distance:
                max_distance = candidate_distance
                best_candidate = candidate

        centroids[i] = best_candidate
        i += 1

    return centroids


def assign_to_cluster(data, centroids):
    # TODO find the closest cluster for each data point
    assignments = np.zeros((data.shape[0], ), dtype=np.int32)

    for i, observation in enumerate(data):
        min_centroid_distance = np.inf
        centroid_idx = None

        for j, centroid in enumerate(centroids):
            distance = euclidean_distance(observation, centroid)

            if distance < min_centroid_distance:
                min_centroid_distance = distance
                centroid_idx = j

        assignments[i] = centroid_idx

    return assignments


def update_centroids(data, assignments):
    # TODO find new centroids based on the assignments
    new = []
    for i in np.unique(assignments):
        mask = [True if j == i else False for j in assignments]
        new.append(np.mean(data[mask], axis=0))

    return np.array(new)


def mean_intra_distance(data, assignments, centroids):
    return np.sqrt(np.sum((data - centroids[assignments, :]) ** 2))


def k_means(data, num_centroids, kmeansplusplus=False, max_iterations=100):
    # centroids initizalization
    if kmeansplusplus:
        centroids = initialize_centroids_kmeans_pp(data, num_centroids)
    else:
        centroids = initialize_centroids_forgy(data, num_centroids)

    assignments = assign_to_cluster(data, centroids)
    for i in range(max_iterations):  # max number of iteration = 100
        print(f"Intra distance after {i} iterations: {mean_intra_distance(data, assignments, centroids)}")
        centroids = update_centroids(data, assignments)
        new_assignments = assign_to_cluster(data, centroids)
        if np.all(new_assignments == assignments):  # stop if nothing changed
            break
        else:
            assignments = new_assignments

    return new_assignments, centroids, mean_intra_distance(data, new_assignments, centroids)
