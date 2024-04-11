from torchvision import datasets, transforms, models
import torch
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment as linear_assignment
from torch.utils.data import Dataset
import numpy as np
import os
from tqdm import tqdm


def extract_features_resnet50(dataloader):
    """
    Extracts features using the ResNet50 model for datasets.
    """
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the ResNet50 model and move it to the chosen device
    resnet50 = models.resnet50(pretrained=True)
    # Remove the final fully connected layer to use the model as a feature extractor
    resnet50 = torch.nn.Sequential(*(list(resnet50.children())[:-1]))

    resnet50 = resnet50.to(device)
    resnet50.eval()  # Set the model to evaluation mode
    features = []

    for inputs, _ in tqdm(dataloader, desc="Extracting features"):
        # Transfer inputs to the device
        inputs = inputs.to(device)

        # Extract features
        with torch.no_grad():
            # Forward pass through the modified ResNet50
            outputs = resnet50(inputs)
            # Reshape the outputs to flatten the feature tensors
            outputs = outputs.view(outputs.size(0), -1)
        features.append(outputs.cpu().numpy())

    features = np.concatenate(features, axis=0)
    return features


def load_dataset_with_resnet50(dataset_name, batch_size=512):
    """
    Loads datasets and extracts features using ResNet50.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Handling datasets differently based on their specific parameters
    if dataset_name == 'STL10':
        dataset = datasets.STL10(root='./data', split='train', download=True, transform=transform)
    else:
        dataset = datasets.__dict__[dataset_name](root='./data', train=True, download=True, transform=transform)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    features = extract_features_resnet50(dataloader)

    # Check if the dataset has 'targets' or 'labels' attribute
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        labels = np.array(dataset.labels)
    else:
        raise AttributeError(f"The {dataset_name} dataset does not have 'targets' or 'labels' attribute.")

    # Save features and labels to .npy files
    data_labels = {'data': features, 'label': labels}
    save_path = f"./data/{dataset_name.lower()}/{dataset_name.lower()}.npy"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        np.save(f, data_labels)
    f.close()

    return features, labels


def dump_dataset(dataset_name, **kwargs):
    """
    Generalized function for dumping datasets. Currently supports CIFAR10, CIFAR100, STL10, MNIST, and FashionMNIST.
    Reuters dataset is already provided in the repository.
    """
    if dataset_name in ['CIFAR10', 'CIFAR100', 'STL10']:
        return load_dataset_with_resnet50(dataset_name, **kwargs)
    elif dataset_name in ['MNIST', 'FashionMNIST']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = datasets.__dict__[dataset_name](root='./data', train=True, download=True, transform=transform)
        dataloader = DataLoader(dataset, batch_size=kwargs.get('batch_size', 512), shuffle=False)
        features, labels = [], []
        for data, target in dataloader:
            features.append(data.numpy())
            labels.extend(target.numpy())
        features = np.concatenate(features, axis=0)
        labels = np.array(labels)

        # Save features and labels to .pkl file
        data_labels = {'data': features, 'label': labels}
        save_path = f"./data/{dataset_name.lower()}/{dataset_name.lower()}.npy"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            np.save(f, data_labels)
        f.close()

        return features, labels
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def load_dataset_xy(dataset_name, data_path):
    """
    Loads a dataset from a .npy file.

    Parameters:
    - dataset_name: The name of the dataset to load.
    - data_path: The base path where the dataset is stored.

    Returns:
    - x: The features of the dataset.
    - y: The labels of the dataset.
    """
    if not os.path.exists(data_path):
        print(f"Dataset file {data_path} not found.")
        return None, None

    data = np.load(data_path, allow_pickle=True).item()
    x = data['data']
    y = data['label']

    # Reshape if necessary - this part can be adjusted based on how you've structured your data
    if x.ndim > 2:
        x = x.reshape((x.shape[0], -1)).astype('float32')
    if y.ndim > 1:
        y = y.reshape((y.size,))

    print((f"{dataset_name} samples", x.shape))
    return x, y


class LoadDataset(Dataset):

    def __init__(self, dataset_name, dataset_path):
        self.x, self.y = load_dataset_xy(dataset_name, dataset_path)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])), torch.from_numpy(
            np.array(self.y[idx])), torch.from_numpy(np.array(idx))


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(row_ind, col_ind)]) * 1.0 / y_pred.size
