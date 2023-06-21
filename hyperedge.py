import torch


def count_node(H, node_num=None):
    return H[0].max().item() + 1 if node_num is None else node_num


def count_hyedge(H, hyedge_num=None):
    return H[1].max().item() + 1 if hyedge_num is None else hyedge_num


def degree_node(H, node_num=None):
    '''
    node_idx, edge_idx = H
    node_num = count_node(H, node_num)
    src = torch.ones_like(node_idx).float().to(H.device)
    out = torch.zeros(node_num).to(H.device)
    return out.scatter_add(0, node_idx, src).long()
    # return torch.zeros(node_num).scatter_add(0, node_idx, torch.ones_like(node_idx).float()).long()
    '''
    # H = H.squeeze(0)
    tmp = torch.sum(H, dim=1)
    degree_node_matrix = torch.diag(tmp)
    return degree_node_matrix


def degree_hyedge(H: torch.Tensor, hyedge_num=None):
    '''
    node_idx, hyedge_idx = H
    hyedge_num = int(count_hyedge(H, hyedge_num=hyedge_num))
    src = torch.ones_like(hyedge_idx).float().to(H.device)
    out = torch.zeros(hyedge_num).to(H.device)
    return out.scatter_add(0, hyedge_idx.long(), src).long()
    '''
    # H = H.squeeze(0)
    tmp = torch.sum(H, dim=0)
    degree_hyedge_matrix = torch.diag(tmp)
    return degree_hyedge_matrix
