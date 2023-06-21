import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

train_data_root = ''
val_data_root = ''
label_root = ''


def pairwise_euclidean_distance(x: torch.Tensor):
    """
    generate N x N node distance matrix
    :param x: a tensor of size N x C (N nodes with C feature dimension)
    :return: a tensor of N x N (distance matrix for each node pair)
    """
    assert isinstance(x, torch.Tensor)
    assert len(x.shape) == 2
    x = x.float()

    x_transpose = torch.transpose(x, dim0=0, dim1=1)
    x_inner = torch.matmul(x, x_transpose)
    x_inner = -2 * x_inner
    x_square = torch.sum(x ** 2, dim=1, keepdim=True)
    x_square_transpose = torch.transpose(x_square, dim0=0, dim1=1)
    dis = x_square + x_inner + x_square_transpose
    return dis


def neighbor_distance(x: torch.Tensor, k_nearest, dis_metric=pairwise_euclidean_distance):
    """
    construct hyperedge for each node in x matrix. Each hyperedge contains a node and its k-1 nearest neighbors.
    :param x: N x C matrix. N denotes node number, and C is the feature dimension.
    :param k_nearest:
    :return:
    """

    assert len(x.shape) == 2, 'should be a tensor with dimension (N x C)'

    # N x C
    node_num = x.size(0)
    dis_matrix = dis_metric(x)
    _, nn_idx = torch.topk(dis_matrix, k_nearest-1, dim=1, largest=False)
    self_node = torch.arange(node_num).unsqueeze(dim=1)
    nn_idx = torch.cat((nn_idx, self_node), 1)
    nn_idx = nn_idx.reshape(-1)
    hyedge_idx = torch.arange(node_num).to(x.device).unsqueeze(0).repeat(k_nearest, 1).transpose(1, 0).reshape(-1)
    # H = torch.stack([nn_idx.reshape(-1), hyedge_idx])
    h = torch.zeros(node_num, node_num)
    for i in range(nn_idx.size(0)):
        h[nn_idx[i]][hyedge_idx[i]] = 1.0
    return h


class UnitData(Dataset):
    def __init__(self, train_flag: bool, data_root=None) -> None:
        self.train = train_flag
        if data_root is None:
            if self.train is True:
                self.data_root = train_data_root
            else:
                self.data_root = val_data_root
        self.label_root = label_root

    def __getitem__(self, index: int):
        # print(sorted(os.listdir(self.data_root))[index])
        feat = np.load(os.path.join(self.data_root, sorted(os.listdir(self.data_root))[index]))
        label2 = pd.read_csv(os.path.join(self.label_root, sorted(os.listdir(self.label_root))[index]), sep='\t')['label2'].tolist()
        label1 = pd.read_csv(os.path.join(self.label_root, sorted(os.listdir(self.label_root))[index]), sep='\t')['label1'].tolist()
        feat_tensor = torch.Tensor(feat)
        gamma_tensor_2 = torch.zeros(159, feat_tensor.size(0))
        gamma_tensor_1 = torch.zeros(29, feat_tensor.size(0))

        for idx, l2 in enumerate(label2):
            gamma_tensor_2[l2][idx] = 1.0
        for id, l1 in enumerate(label1):
            gamma_tensor_1[l1-1][id] = 1.0
        label1 = [l-1 for l in label1]
        label2 = torch.Tensor(label2)
        label1 = torch.Tensor(label1)
        H_tensor = neighbor_distance(feat_tensor, 10)

        return feat_tensor, H_tensor, gamma_tensor_2, gamma_tensor_1, label2, label1

    def __len__(self) -> int:
        if len(os.listdir(self.data_root)) == len(os.listdir(self.label_root)):
            return len(os.listdir(self.data_root))
        else:
            print('the number of the data file or label file is incorrect')


if __name__ == '__main__':
    unit_data = UnitData(True)
    # print(type(clkcf_data.query_token_list), clkcf_data.query_token_list)
    # print(type(clkcf_data.item_token_list), clkcf_data.item_token_list)
    dataloader = DataLoader(unit_data, batch_size=1, shuffle=False)

    print(len(dataloader))
    for feat, H, gamma2, gamma1, _, _ in dataloader:
        exit(0)

