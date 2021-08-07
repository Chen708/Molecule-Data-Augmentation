from models.gcn import GCN
import os.path as osp
import math
import torch
from torch import nn
from torch._C import AggregationType
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn.conv.message_passing import MessagePassing
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv
from torch_geometric.nn import MessagePassing

from data_aug.dataset import ATOM_LIST, CHIRALITY_LIST, BOND_LIST, BONDDIR_LIST

num_atom_type = len(ATOM_LIST) + 1 # including the extra mask tokens
num_chirality_tag = len(CHIRALITY_LIST)
num_bond_type = len(BOND_LIST) + 1 # including aromatic and self-loop edge
num_bond_direction = len(BONDDIR_LIST)

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]


class GATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, emb_dim, aggr="add"):
        super(GATConv, self).__init__()
        self.aggr = aggr 
        self.conv1 = GATConv(in_channels, 8, heads=8, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(8 * 8, out_channels, heads=1, concat=False,
                             dropout=0.6)
        self.emb_dim = emb_dim

        self.weight = Parameter(torch.Tensor(emb_dim,emb_dim))
        self.bias = Parameter(torch.Tensor(emb_dim))
        self.reset_parameters()

        self.edge_embedding1 = nn.Embedding(num_bond_type,1)
        self.edge_embedding2 = nn.Embedding(num_bond_direction,1)
        
        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
        return F.log_softmax(x, dim=-1)

    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.weight.size(-2) + self.weight.size(-1)))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.fill_(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GATConv(dataset.num_features, dataset.num_classes).to(device)


class GAT(nn.Module):
    def __init__(self,task='classification', num_layer=5, emb_dim=300, 
        feat_dim=256, pool='mean', drop_ratio=0, **kwargs):
        super(GAT, self).__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.drop_ratio = drop_ratio
        self.task = task

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)

        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        
        if self.task == 'classification':
            self.pred_lin = nn.Sequential(
                nn.Linear(self.feat_dim, self.feat_dim//2), 
                nn.ReLU(inplace=True),
                nn.Linear(self.feat_dim//2, 2)
            )
        elif self.task == 'regression':
            self.pred_lin = nn.Sequential(
                nn.Linear(self.feat_dim, self.feat_dim//2), 
                nn.ReLU(inplace=True),
                nn.Linear(self.feat_dim//2, self.feat_dim//2),
                nn.ReLU(inplace=True),
                nn.Linear(self.feat_dim//2, 1)

    def test(data):
        model.eval()
        out, accs = model(data.x, data.edge_index), []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
           acc = float((out[mask].argmax(-1) == data.y[mask]).sum() / mask.sum())
           accs.append(acc)
        return accs


