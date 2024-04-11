# EECS6322 - EDESC
This final project is a reproduction of EECS6322 for [Efficient Deep Embedded Subspace Clustering](https://openaccess.thecvf.com/content/CVPR2022/html/Cai_Efficient_Deep_Embedded_Subspace_Clustering_CVPR_2022_paper.html).

# Parameter Settings
The parameters below can be modified for this experiment:
```python
parser.add_argument('--dataset', type=str, default='REUTERS',
                        choices=['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100', 'STL10', 'REUTERS'],
                        help='Dataset name')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--d', default=5, type=int)
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--pretrain_path', type=str, default='', help='Path for saving pre-trained weights')
parser.add_argument('--beta', default=0.1, type=float, help='coefficient of subspace affinity loss')
```
# Implementation
The `requirements.txt` is provided. Run the code using the following command:
```python
# --dataset: default='REUTERS', choices=['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100', 'STL10', 'REUTERS']
python EDESC_main.py
```
# Dataset
The REUTERS dataset is available in the data folder. Code for downloading other datasets is also provided in `utils.py`.
