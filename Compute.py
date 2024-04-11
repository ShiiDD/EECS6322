from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn


def separate(Z, y, n_clusters):
    """
    Separates the dataset Z into clusters based on the predictions y.

    Args:
        Z: The dataset, a tensor of shape (num_samples, features).
        y: The cluster assignments for each sample in Z.
        n_clusters: The number of clusters.

    Returns:
        A dictionary where each key corresponds to a cluster index and
        its value is a list of samples belonging to that cluster.
    """
    separated_clusters = defaultdict(list)
    for i in range(n_clusters):
        for j, cluster_id in enumerate(y):
            if cluster_id == i:
                # Detach and convert tensor to numpy for processing
                separated_clusters[i].append(Z[j].cpu().detach().numpy())
    return separated_clusters


def initialize_D(Z, y, n_clusters, d):
    """
    Initializes the subspace matrix D using SVD on separated clusters.

    Args:
        Z: The dataset, a tensor of shape (num_samples, features).
        y: The cluster assignments for each sample in Z.
        n_clusters: The number of clusters.
        d: The dimensionality of the subspace.

    Returns:
        The initialized subspace matrix D.
    """
    separated_clusters = separate(Z, y, n_clusters)
    D_matrix = np.zeros([n_clusters * d, n_clusters * d])
    print("Initializing D")
    for i in range(n_clusters):
        cluster_data = np.array(separated_clusters[i])
        # Perform SVD on the data of the current cluster
        u, _, _ = np.linalg.svd(cluster_data.T)
        # Fill the corresponding block in D_matrix
        D_matrix[:, i * d:(i + 1) * d] = u[:, 0:d]

    print("Shape of D: ", D_matrix.T.shape)
    print("Initialization of D Finished")
    return D_matrix


class Constraint1(nn.Module):
    """
    A module for enforcing first constraint on D.
    """

    def __init__(self):
        super(Constraint1, self).__init__()

    def forward(self, D):
        """
        Calculates the 1st-D constraint loss.

        Args:
            D: The subspace matrix.

        Returns:
            The 1st-D constraint loss.
        """
        I = torch.eye(D.shape[1], device=D.device)
        loss_d1_constraint = torch.norm(torch.mm(D.T, D) * I - I)
        return 1e-3 * loss_d1_constraint


class Constraint2(nn.Module):
    """
    A module for enforcing second constraint between subspaces of D.
    """

    def __init__(self):
        super(Constraint2, self).__init__()

    def forward(self, D, d, n_clusters):
        """
        Calculates the 2nd-D constraint loss.

        Args:
            D: The subspace matrix.
            d: The dimensionality of the subspace.
            n_clusters: The number of clusters.

        Returns:
            The 2nd-D constraint loss.
        """
        S = torch.ones(D.shape[1], D.shape[1], device=D.device)
        zero_block = torch.zeros(d, d, device=D.device)
        for i in range(n_clusters):
            S[i * d:(i + 1) * d, i * d:(i + 1) * d] = zero_block
        loss_d2_constraint = torch.norm(torch.mm(D.T, D) * S)
        return 1e-3 * loss_d2_constraint
