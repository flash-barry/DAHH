import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from models.hyedge import degree_hyedge, degree_node


class DAHHConv(nn.Module):
    def __init__(self, in_ch, out_ch, bias=True) -> None:
        super().__init__()
        self.theta = Parameter(torch.Tensor(in_ch, out_ch))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ch))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.theta)
        nn.init.zeros_(self.bias)

    def gen_hyedge_ft(self, x: torch.Tensor, H: torch.Tensor, hyedge_weight=None):
        '''
        ft_dim = x.size(1)
        node_idx, hyedge_idx = H
        hyedge_num = count_hyedge(H)

        # a vector to normalize hyperedge feature
        hyedge_norm = 1.0 / degree_hyedge(H).float()
        if hyedge_weight is not None:
            hyedge_norm *= hyedge_weight
        hyedge_norm = hyedge_norm[hyedge_idx.long()]

        x = x[node_idx.long()] * hyedge_norm.unsqueeze(1)
        x = torch.zeros(hyedge_num, ft_dim).to(x.device).scatter_add(0, hyedge_idx.unsqueeze(1).repeat(1, ft_dim), x)
        return x
        '''
        degree_hyedge_matrix = degree_hyedge(H)
        hyedge_norm = degree_hyedge_matrix.inverse()
        tmp = torch.matmul(H, hyedge_norm)
        return torch.matmul(tmp.T, x)

    def gen_node_ft(self, x: torch.Tensor, H: torch.Tensor):
        '''
        ft_dim = x.size(1)
        node_idx, hyedge_idx = H
        node_num = count_node(H)

        # a vector to normalize node feature
        node_norm = 1.0 / degree_node(H).float()
        node_norm = node_norm[node_idx]

        x = x[hyedge_idx] * node_norm.unsqueeze(1)
        x = torch.zeros(node_num, ft_dim).to(x.device).scatter_add(0, node_idx.unsqueeze(1).repeat(1, ft_dim), x)
        return x
        '''
        degree_node_matrix = degree_node(H)
        node_norm = degree_node_matrix.inverse()
        tmp = torch.matmul(node_norm, H)
        return torch.matmul(tmp, x)

    def forward(self, x, H: torch.Tensor, hyedge_weight=None):
        assert len(x.shape) == 2, 'the input of HyperConv should be N x C'
        # feature transform
        x = x.matmul(self.theta)

        # generate hyperedge feature from node feature
        x = self.gen_hyedge_ft(x, H, hyedge_weight)
        # generate node feature from hyperedge feature
        x = self.gen_node_ft(x, H)

        if self.bias is not None:
            return x + self.bias
        else:
            return x

class MLPClassifier(nn.Module):
    
    def __init__(self, dim_fea, n_hiddens, n_category, dropout_rate):
        super().__init__()
        self.fcs = nn.Sequential(
            nn.Linear(dim_fea, n_hiddens),
            nn.Tanh(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(n_hiddens, n_category),
        )
    
    def forward(self, x):
        return self.fcs(x)

class DAHH(nn.Module):
    def __init__(self, emb_dim=768, N_2=159, N_1=29, hiddens=None, dropout=None):
        super(DAHH, self).__init__()
        if dropout is None:
            dropout = [0.5, 0.5]
        if hiddens is None:
            hiddens = [768, 768]
        self.dropout = dropout

        self.DAHH_convs = list()
        for h in hiddens:
            self.DAHH_convs.append(DAHHConv(emb_dim, h))
        self.DAHH_convs = nn.ModuleList(self.DAHH_convs)

        self.output_2 = MLPClassifier(emb_dim * 2, 512, N_2, 0.5)
        self.output_1 = MLPClassifier(emb_dim * 2, 512, N_1, 0.5)

    def aggregate(self, X_i_new, Gamma_i):
        return torch.matmul(Gamma_i, X_i_new)

    def forward(self, H_i, X_i, Gamma_2, Gamma_1):
        """
        H_i: n_V1 x n_E1
        X_i: n_V1 x emb_dim
        """
        X_i_new = X_i
        for dahh_conv, dropout_rate in zip(self.DAHH_convs, self.dropout):
            X_i_new = dahh_conv(X_i_new, H_i)
            X_i_new = F.leaky_relu(X_i_new, inplace=True)
            X_i_new = F.dropout(X_i_new, dropout_rate)
            X_i_new += X_i

        S_i = self.aggregate(X_i_new, Gamma_2)
        P_i = self.aggregate(X_i_new, Gamma_1)
        S_i = torch.mean(S_i, 0).repeat(X_i_new.size()[0], 1)
        P_i = torch.mean(P_i, 0).repeat(X_i_new.size()[0], 1)

        x_hat_primary = torch.cat((X_i_new, P_i), 1)
        x_hat_second  = torch.cat((X_i_new, S_i), 1)

        output1 = self.output_1(x_hat_primary)
        output2 = self.output_2(x_hat_second)

        return output1, output2

class LblPred(nn.Module):
    def __init__(self, S: torch.Tensor, Phi: torch.Tensor, emb_dim=768, N_1=29, N_2=159):
        super().__init__()
        self.S = S / 1000  # N_2 x d  Secondary
        self.P = torch.matmul(Phi, self.S)  # N_1 x d  Primary

        self.S = torch.reshape(self.S, (1, -1))
        self.P = torch.reshape(self.P, (1, -1))

        self.pri_pred_layer = nn.Linear(emb_dim * (N_1 + 1), N_1)
        self.sec_pred_layer = nn.Linear(emb_dim * (N_2 + 1), N_2)

    def get_nearest_cls(self, x: torch.Tensor) -> torch.Tensor:
        eu_dis_list = list()
        for s in self.S:
            eu_dis_list.append(F.pairwise_distance(s, x, p=2))
        nearest_s = self.S[eu_dis_list.index(min(eu_dis_list))]
        x_hat = torch.mean(torch.stack([nearest_s, x], 0), 0)
        return x_hat

    def forward(self, in_feat):

        self.S = self.S.repeat(in_feat.size()[0], 1)
        self.P = self.P.repeat(in_feat.size()[0], 1)
        x_hat_primary = torch.cat((in_feat, self.P), 1)
        x_hat_secondary = torch.cat((in_feat, self.S), 1)

        x_hat_primary_acti = F.leaky_relu(x_hat_primary)
        x_hat_secondary_acti = F.leaky_relu(x_hat_secondary)
        output_p = self.pri_pred_layer(x_hat_primary_acti)
        output_s = self.sec_pred_layer(x_hat_secondary_acti)
        output_1 = F.softmax(output_p, dim=1)
        output_2 = F.softmax(output_s, dim=1)
        output_1 = output_1.max(1)[1]
        output_2 = output_2.max(1)[1]
        return F.softmax(output_p, dim=1),\
               F.softmax(output_s, dim=1)


class JointLoss(nn.Module):
    def __init__(self):
        super(JointLoss, self).__init__()
        self.celoss = nn.CrossEntropyLoss()

    def forward(self, p1, y1, p2, y2):
        return self.celoss(p1, y1) + self.celoss(p2, y2)
