from __future__ import print_function, division
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from torch.optim import Adam
import warnings
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils import LoadDataset, dump_dataset, cluster_acc
from AutoEncoderFramework import AutoEncoder
from Compute import initialize_D, Constraint1, Constraint2

warnings.filterwarnings("ignore")


class EDESC(nn.Module):

    def __init__(self,
                 n_enc_1,
                 n_enc_2,
                 n_enc_3,
                 n_dec_1,
                 n_dec_2,
                 n_dec_3,
                 n_input,
                 n_z,
                 n_clusters,
                 num_sample,
                 pretrain_path=''):
        super(EDESC, self).__init__()
        self.pretrain_path = pretrain_path
        self.n_clusters = n_clusters

        self.ae = AutoEncoder(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)

        # Subspace bases proxy
        self.D = Parameter(torch.Tensor(n_z, n_clusters))

    def pretrain(self, path=''):
        if path == '':
            pretrain_ae(self.ae)
            self.pretrain_path = f'{args.dataset_base_path}/{args.dataset.lower()}.pkl'
            self.ae.load_state_dict(torch.load(self.pretrain_path, map_location='cpu'))
        else:
            # Load pre-trained weights
            self.ae.load_state_dict(torch.load(self.pretrain_path, map_location='cpu'))
        print('Load pre-trained model from', path)

    def forward(self, x):

        x_bar, z = self.ae(x)
        d = args.d
        s = None
        eta = args.eta

        # Calculate subspace affinity
        for i in range(self.n_clusters):

            si = torch.sum(torch.pow(torch.mm(z, self.D[:, i * d:(i + 1) * d]), 2), 1, keepdim=True)
            if s is None:
                s = si
            else:
                s = torch.cat((s, si), 1)
        s = (s + eta * d) / ((eta + 1) * d)
        s = (s.t() / torch.sum(s, 1)).t()
        return x_bar, s, z

    def total_loss(self, x, x_bar, z, pred, target, dim, n_clusters, beta):

        # Reconstruction loss
        reconstr_loss = F.mse_loss(x_bar, x)

        # Subspace clustering loss
        kl_loss = F.kl_div(pred.log(), target)

        # Constraints
        d_cons1 = Constraint1()
        d_cons2 = Constraint2()
        loss_d1 = d_cons1(self.D)
        loss_d2 = d_cons2(self.D, dim, n_clusters)

        # Total_loss
        total_loss = reconstr_loss + beta * kl_loss + loss_d1 + loss_d2
        # w/o L_Recon
        # total_loss = beta * kl_loss + loss_d1 + loss_d2
        # w/o L_Sub
        # total_loss = reconstr_loss + loss_d1 + loss_d2
        # w/o Constraints
        # total_loss = reconstr_loss + beta * kl_loss

        return total_loss


def refined_subspace_affinity(s):
    weight = s ** 2 / s.sum(0)
    return (weight.t() / weight.sum(1)).t()


def pretrain_ae(model):
    train_loader = DataLoader(dataset, batch_size=512, shuffle=True)
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr)
    for epoch in range(50):
        total_loss = 0.
        for batch_idx, (x, _, _) in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()
            x_bar, z = model(x)
            loss = F.mse_loss(x_bar, x)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        print("Epoch {} loss={:.4f}".format(epoch, total_loss / (batch_idx + 1)))
        save_path = f'{args.dataset_base_path}/{args.dataset.lower()}.pkl'
        torch.save(model.state_dict(), save_path)
    args.pretrain_path = save_path
    print("Model saved to {}.".format(save_path))
    print("args.pretrain_path: ", args.pretrain_path)


def train_EDESC():
    model = EDESC(
        n_enc_1=500,
        n_enc_2=500,
        n_enc_3=1000,
        n_dec_1=1000,
        n_dec_2=500,
        n_dec_3=500,
        n_input=args.n_input,
        n_z=args.n_z,
        n_clusters=args.n_clusters,
        num_sample=args.num_sample,
        pretrain_path=args.pretrain_path).to(device)
    start = time.time()

    # Load pre-trained model
    model.pretrain(args.pretrain_path)
    optimizer = Adam(model.parameters(), lr=args.lr)

    # Cluster parameter initiate
    data = dataset.x
    y = dataset.y
    data = torch.Tensor(data).to(device)
    x_bar, hidden = model.ae(data)
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=10)

    # Get clusters from K-means
    y_pred = kmeans.fit_predict(hidden.data.cpu().numpy())
    print("Initial Cluster Centers: ", y_pred)

    # Initialize D
    D = initialize_D(hidden, y_pred, args.n_clusters, args.d)
    D = torch.tensor(D).to(torch.float32)
    accmax = 0
    nmimax = 0
    y_pred_last = y_pred
    model.D.data = D.to(device)
    loss_values = []  # For storing loss values

    model.train()

    for epoch in range(200):
        x_bar, s, z = model(data)

        # Update refined subspace affinity
        tmp_s = s.data
        s_tilde = refined_subspace_affinity(tmp_s)

        # Evaluate clustering performance
        y_pred = tmp_s.cpu().detach().numpy().argmax(1)
        delta_label = np.sum(y_pred != y_pred_last).astype(
            np.float32) / y_pred.shape[0]
        y_pred_last = y_pred
        acc = cluster_acc(y, y_pred)
        nmi = nmi_score(y, y_pred)
        if acc > accmax:
            accmax = acc
        if nmi > nmimax:
            nmimax = nmi
        print('Iter {}'.format(epoch), ':Current Acc {:.4f}'.format(acc),
              ':Max Acc {:.4f}'.format(accmax), ', Current nmi {:.4f}'.format(nmi), ':Max nmi {:.4f}'.format(nmimax))

        ############## Total loss function ######################
        loss = model.total_loss(data, x_bar, z, pred=s, target=s_tilde, dim=args.d, n_clusters=args.n_clusters,
                                beta=args.beta)
        loss_values.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    end = time.time()
    print('Running time: ', end - start)

    # Plotting with dark red line as requested
    plt.plot(loss_values, color='darkred', linewidth=1)
    plt.xlabel('Epoch')  # Making x-axis label bold
    plt.ylabel('Objective Function Value')  # Making y-axis label bold
    plt.grid(True)

    # Setting x-axis limits
    plt.xlim(0, 200)

    # Making the border of the plot bold
    ax = plt.gca()  # Get current axes
    for spine in ax.spines.values():
        spine.set_linewidth(2)  # Adjusting the linewidth to make it look bold

    # Save the figure
    plt.savefig(f'./{args.dataset}_convergence.png', dpi=600)

    return accmax, nmimax


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EDESC PyTorch implementation',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', type=str, default='REUTERS',
                        choices=['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100', 'STL10', 'REUTERS'],
                        help='Dataset name')

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--d', default=5, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--pretrain_path', type=str, default='', help='Path for saving pre-trained models')
    parser.add_argument('--beta', default=0.1, type=float, help='coefficient of subspace affinity loss')
    args = parser.parse_args()

    # Dataset-specific configurations
    dataset_configurations = {
        'MNIST': {'n_clusters': 10, 'n_input': 784, 'num_sample': 70000},
        'FashionMNIST': {'n_clusters': 10, 'n_input': 784, 'num_sample': 70000},
        'CIFAR10': {'n_clusters': 10, 'n_input': 2048, 'num_sample': 60000},  # ResNet50 get 2048-dimensional features
        'CIFAR100': {'n_clusters': 20, 'n_input': 2048, 'num_sample': 60000},
        'STL10': {'n_clusters': 10, 'n_input': 2048, 'num_sample': 13000},
        'REUTERS': {'n_clusters': 4, 'n_input': 2000, 'num_sample': 10000}
    }

    # Dynamically set the paths based on the dataset
    if args.pretrain_path != '':
        args.pretrain_path = f'{args.pretrain_path}/{args.dataset.lower()}.pkl'
    args.dataset_base_path = f'data/{args.dataset.lower()}'
    args.dataset_path = f'{args.dataset_base_path}/{args.dataset.lower()}.npy'

    if not os.path.exists(args.dataset_path):
        dump_dataset(args.dataset)  # Assuming this function saves the dataset to dataset_path
    dataset = LoadDataset(args.dataset.lower(), args.dataset_path)  # Load the dataset

    # Ensure the directories exist (for pre-trained models)
    os.makedirs(args.dataset_base_path, exist_ok=True)

    # Adjust configurations based on the chosen dataset
    args.n_clusters = dataset_configurations[args.dataset]['n_clusters']
    args.n_input = dataset_configurations[args.dataset]['n_input']
    args.num_sample = dataset_configurations[args.dataset]['num_sample']
    args.eta = args.d
    args.n_z = args.d * args.n_clusters

    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    print(args)

    bestacc = 0
    bestnmi = 0
    for i in range(10):
        acc, nmi = train_EDESC()
        if acc > bestacc:
            bestacc = acc
        if nmi > bestnmi:
            bestnmi = nmi
    print('Best ACC {:.4f}'.format(bestacc), ' Best NMI {:4f}'.format(bestnmi))
