import os
import csv
import math
import time
import random
import networkx as nx
import numpy as np
from copy import deepcopy

import torch
import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms

from torch_scatter import scatter
from torch_geometric.data import Data, Dataset, DataLoader

import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles


ATOM_LIST = list(range(1,100))

CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]

BOND_LIST = [
    BT.SINGLE, 
    BT.DOUBLE, 
    BT.TRIPLE, 
    BT.AROMATIC,
    BT.UNSPECIFIED,
]

BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT,
    Chem.rdchem.BondDir.EITHERDOUBLE
]


def _generate_scaffold(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold


def generate_scaffolds(smiles_data, log_every_n=1000):
    scaffolds = {}
    data_len = len(smiles_data)
    print(data_len)

    print("About to generate scaffolds")
    for ind, smiles in enumerate(smiles_data):
        if ind % log_every_n == 0:
            print("Generating scaffold %d/%d" % (ind, data_len))
        scaffold = _generate_scaffold(smiles)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [ind]
        else:
            scaffolds[scaffold].append(ind)

    # Sort from largest to smallest scaffold sets
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]
    # print(scaffold_sets)
    return scaffold_sets


def scaffold_split(smiles_data, valid_size, test_size, seed=None, log_every_n=1000):
    train_size = 1.0 - valid_size - test_size
    scaffold_sets = generate_scaffolds(smiles_data)

    train_cutoff = train_size * len(smiles_data)
    valid_cutoff = (train_size + valid_size) * len(smiles_data)
    train_inds: List[int] = []
    valid_inds: List[int] = []
    test_inds: List[int] = []

    print("About to sort in scaffold sets")
    for scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) > train_cutoff:
            if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
                test_inds += scaffold_set
            else:
                valid_inds += scaffold_set
        else:
            train_inds += scaffold_set
    return train_inds, valid_inds, test_inds


def read_smiles(data_path, target, task):
    smiles_data, labels = [], []
    with open(data_path) as csv_file:
        # csv_reader = csv.reader(csv_file, delimiter=',')
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            if i != 0:
                # smiles = row[3]
                smiles = row['smiles']
                label = row[target]
                mol = Chem.MolFromSmiles(smiles)
                if mol != None and label != '':
                    smiles_data.append(smiles)
                    if task == 'classification':
                        labels.append(int(label))
                    elif task == 'regression':
                        labels.append(float(label))
                    else:
                        ValueError('task must be either regression or classification')
    print(len(smiles_data))
    return smiles_data, labels


class MolDataset(Dataset):
    def __init__(self, smiles_data, labels, task, test_mode=True, aug_time=1):
        super(Dataset, self).__init__()
        # self.smiles_data, self.labels = read_smiles(data_path, target, task)
        # self.task = task
        self.smiles_data = smiles_data
        self.labels = labels
        self.task = task
        self.test_mode = test_mode
        self.aug_time = aug_time
        if self.test_mode:
            self.aug_time = 1
        assert type(aug_time) == int
        assert aug_time >= 1

        self.reproduce_seeds = list(range(self.__len__()))
        np.random.shuffle(self.reproduce_seeds)

    def __getitem__(self, index):
        if self.test_mode:
            true_index = index
        else:
            true_index = index // self.aug_time
        
        # set random seed
        seed = self.reproduce_seeds[index]
        random.seed(seed)
        np.random.seed(seed)

        mol = Chem.MolFromSmiles(self.smiles_data[true_index])
        mol = Chem.AddHs(mol)

        N = mol.GetNumAtoms()
        M = mol.GetNumBonds()

        type_idx = []
        chirality_idx = []
        atomic_number = []
        for atom in mol.GetAtoms():
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            atomic_number.append(atom.GetAtomicNum())

        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
        x = torch.cat([x1, x2], dim=-1)

        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            # edge_type += 2 * [MOL_BONDS[bond.GetBondType()]]
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)
        if self.task == 'classification':
            y = torch.tensor(self.labels[true_index], dtype=torch.long).view(1,-1)
        elif self.task == 'regression':
            y = torch.tensor(self.labels[true_index], dtype=torch.float).view(1,-1)
        data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)

        if self.test_mode:
            return data
        
        if index % self.aug_time == 0:
            return data
        else:
            # random mask a subgraph of the molecule
            node_mask_ratio = random.uniform(0, 0.25)
            edge_mask_ratio = random.uniform(0, 0.25)
            num_mask_nodes = max([1, math.floor(node_mask_ratio*N)])
            num_mask_edges = max([0, math.floor(edge_mask_ratio*M)])
            mask_nodes = random.sample(list(range(N)), num_mask_nodes)
            mask_edges_single = random.sample(list(range(M)), num_mask_edges)
            mask_edges = [2*i for i in mask_edges_single] + [2*i+1 for i in mask_edges_single]

            x_mask = deepcopy(x)
            for atom_idx in mask_nodes:
                x_mask[atom_idx,:] = torch.tensor([len(ATOM_LIST), 0])
            edge_index_mask = torch.zeros((2, 2*(M-num_mask_edges)), dtype=torch.long)
            edge_attr_mask = torch.zeros((2*(M-num_mask_edges), 2), dtype=torch.long)
            count = 0
            for bond_idx in range(2*M):
                if bond_idx not in mask_edges:
                    edge_index_mask[:,count] = edge_index[:,bond_idx]
                    edge_attr_mask[count,:] = edge_attr[bond_idx,:]
                    count += 1
            data_aug = Data(x=x_mask, y=y, edge_index=edge_index_mask, edge_attr=edge_attr_mask)
            return data_aug

    def __len__(self):
        return len(self.smiles_data) * self.aug_time


class MolDatasetWrapper(object):
    def __init__(self, batch_size, num_workers, valid_size, test_size, aug_time, data_path, target, task):
        super(object, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.test_size = test_size
        self.aug_time = aug_time
        self.target = target
        self.task = task
        self.smiles_data, self.labels = read_smiles(data_path, target, task)
        self.smiles_data = np.asarray(self.smiles_data)
        self.labels = np.asarray(self.labels)

    def get_data_loaders(self):
        train_idx, valid_idx, test_idx = scaffold_split(self.smiles_data, self.valid_size, self.test_size)

        # define dataset
        train_set = MolDataset(self.smiles_data[train_idx], self.labels[train_idx], test_mode=False, aug_time=self.aug_time, task=self.task)
        valid_set = MolDataset(self.smiles_data[valid_idx], self.labels[valid_idx], test_mode=True, task=self.task)
        test_set = MolDataset(self.smiles_data[test_idx], self.labels[test_idx], test_mode=True, task=self.task)

        train_loader = DataLoader(
            train_set, batch_size=self.batch_size, 
            num_workers=self.num_workers, drop_last=True, shuffle=True
        )
        valid_loader = DataLoader(
            valid_set, batch_size=self.batch_size, 
            num_workers=self.num_workers, drop_last=False
        )
        test_loader = DataLoader(
            test_set, batch_size=self.batch_size, 
            num_workers=self.num_workers, drop_last=False
        )

        return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    data_path = 'data/chem_dataset/bbbp/raw/BBBP.csv'
    # dataset = MolDataset(data_path=data_path)
    # print(dataset)
    # print(dataset.__getitem__(0))
    dataset = MolDatasetWrapper(
        batch_size=4, num_workers=4, valid_size=0.1, test_size=0.1,
        data_path=data_path, target='p_np', task='classification')
    train_loader, valid_loader, test_loader = dataset.get_data_loaders()
    for bn, data in enumerate(train_loader):
        print(data)
        break