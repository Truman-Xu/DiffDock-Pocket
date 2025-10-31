import os
import socket
import torch
from torch import nn
import pickle
import torch.multiprocessing as mp
from models.binding_affinty_pred import TensorProductScoreModel as AffinityPredModel
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from utils.diffusion_utils import set_time
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef

class SubsampledDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        self.processed = os.path.join(root, 'processed')
        if not os.path.exists(self.processed):
            os.makedirs(self.processed)
        self.raw = os.path.join(root, 'raw')
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return [p for p in os.listdir(self.raw) if p.endswith('.pt')]
        
    @property
    def processed_file_names(self):
        return [p for p in os.listdir(self.processed) if p.endswith('.pt')]

    def process(self):
        label_dir = os.path.join(root, 'lit-pcba-labels.pkl')
        with open(label_dir, 'rb') as p:
            labels = pickle.load(p)

        cur_idx = 0
        for raw_path in tqdm(self.raw_paths):
            if not raw_path.endswith('.pt'):
                continue
            graphs = torch.load(raw_path)
            total = len(graphs)
            iter = tqdm(enumerate(graphs.items(), start=cur_idx), total=total)
            for i, (k, g) in iter:
                iter.set_description(f'processing: {raw_path} {i}')
                g.y = torch.tensor(labels[k], dtype=torch.float16)
                torch.save(g, os.path.join(self.processed, f'data_{i}.pt'))
            cur_idx = i
            del graphs

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed, f'data_{idx}.pt'))
        data.y = torch.tensor(data.y, dtype=torch.float16)
        return data

def ddp_setup(rank: int, world_size: int):
    """
    Args:
       rank: Unique identifier of each process
      world_size: Total number of processes
    """
    # os.environ["MASTER_ADDR"] = socket.gethostname()
    # os.environ["MASTER_ADDR"] = "192.168.0.102"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["NCCL_SOCKET_IFNAME"]= "eno1,lo"
    os.environ['NCCL_SOCKET_FAMILY'] = 'AF_INET' #Force using IP v4 addr for init socket
    os.environ["MASTER_PORT"] = "65400"
    torch.cuda.set_device(rank)
    # initialize the process group
    init_process_group(backend='nccl', rank=rank, world_size=world_size)

def ddp_train(rank, world_size, model, n_epochs, batch_size, data_dir):
    print(f"Running DDP on rank {rank}.")
    # torch.cuda.set_device(rank)
    model.to(rank)
    ddp_setup(rank, world_size)
    # process_group = torch.distributed.new_group(ranks)
    # sync_bn_module = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group)
    model = DDP(model, device_ids=[rank])
    
    train_graphs = SubsampledDataset(data_dir)
    train_sampler = DistributedSampler(train_graphs, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(
        train_graphs, batch_size=batch_size, shuffle=False,
        sampler = train_sampler
    )
    num_positives = sum(data.y.item() for data in train_graphs)
    num_negatives = len(train_graphs) - num_positives
    pos_weight = torch.tensor([num_negatives / num_positives])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion.to(rank)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)
    # --- Training Loop ---
    for epoch in range(n_epochs):
        model.train()
        for batch in train_loader:
            batch.to(rank)
            set_time(
                batch,
                0, 0, 0, 0, 0,
                batchsize=batch_size, all_atoms=True,
                asyncronous_noise_schedule=False, device='cuda',
                include_miscellaneous_atoms=False
            )
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits.squeeze(), batch.y)
            loss.backward()
            optimizer.step()
        print(f"Rank {rank}, Epoch {epoch}, Loss: {loss / len(train_loader)}")

    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    model = AffinityPredModel(
        device='cuda', 
        cross_max_distance=80,
        ns=24,
        nv=6,
        norm_by_sigma=False,
        num_conv_layers=5,
        atom_max_neighbors=8,
        flexible_sidechains=True,
        sh_lmax=1,
        lm_embedding_type='esm',
    )
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    n_epochs = 2
    batch_size = 4
    world_size = 4
    data_dir = '/home/ziqiaoxu/DiffDock-Pocket/subsample_graphs/'
    mp.spawn(ddp_train, args=(world_size, model, n_epochs, batch_size, data_dir), nprocs=world_size, join=True)
