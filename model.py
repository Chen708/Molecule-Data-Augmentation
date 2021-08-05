def get_model(task_name, config):
    if config['type'] == 'gin':
        from models.gine import GINE
        return GINE(task_name, **config)
    elif config['type'] == 'gcn':
        from models.gcn import GCN
        return GCN(task_name, **config)
    elif config['type'] == 'gat':
        from models.gat import GAT
        return GAT(task_name, **config)