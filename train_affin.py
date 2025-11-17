import os
import json
import torch
from torch import nn
import pickle
from tqdm import tqdm
import torch.multiprocessing as mp
from models.binding_affinty_pred import TensorProductScoreModel as AffinityPredModel
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from utils.diffusion_utils import set_time
import torch.optim as optim
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, matthews_corrcoef
)
from make_graphs import LigReceptorDataset

def init_weights(m):
    if isinstance(m, nn.Linear):
        # nn.init.kaiming_normal_ is usually preferred for ReLU
        # 'fan_in' keeps the variance of the activations stable
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0) # Initialize bias to zero

class SubsampledDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        self.root = root
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
        return [p for p in os.listdir(self.processed) if (p.endswith('.pt') and p.startswith('data_'))]

    def process(self):
        no_contact = []
        label_dir = os.path.join(self.root, 'lit-pcba-labels.pkl')
        with open(label_dir, 'rb') as p:
            labels = pickle.load(p)

        idx = 0
        for raw_path in tqdm(self.raw_paths):
            if not raw_path.endswith('.pt'):
                continue
            graphs = torch.load(raw_path)
            iter = tqdm(graphs.items())
            for k, g in iter:
                iter.set_description(f'processing: {raw_path} {idx}')
                if len(g['flexResidues']['pdbIds']) < 2:
                    no_contact.append(g.name)
                    continue
                g.y = torch.tensor(labels[k], dtype=torch.float16)
                torch.save(g, os.path.join(self.processed, f'data_{idx}.pt'))
                idx+=1
                
            del graphs

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(
            os.path.join(self.processed, f'data_{idx}.pt'), weights_only=False
        )
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
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["NCCL_SOCKET_IFNAME"]= "lo"
    # os.environ['NCCL_SOCKET_FAMILY'] = 'AF_INET' #Force using IP v4 addr for init socket
    os.environ["MASTER_PORT"] = "29500"
    # os.environ["NCCL_DEBUG"] = "INFO"
    torch.cuda.set_device(rank)
    # initialize the process group
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def load_dataset(data_dir):
    with open('sampled_target_lig_ids_131k.json', 'r') as f:
        lig_id_dict = json.load(f)

    pop_targets = ['ADRB2']
    print("Removing targets:", pop_targets)
    for target in pop_targets:
        lig_id_dict.pop(target, None)

    structure_dir = 'furyal_docked_poses'
    lm_embeddings = torch.load(os.path.join(structure_dir, 'esm_lm_embeddings.pt'))
    dataset = LigReceptorDataset(
        root=data_dir, lig_id_dict=lig_id_dict, 
        lm_embeddings=lm_embeddings, structure_dir=structure_dir
    )
    return dataset

def load_ddp_state_dict(model_path):
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    updated_state_dict = {}
    for key in state_dict.keys():
        new_key = key[len('module.'): ] if key.startswith('module.') else key
        updated_state_dict[new_key] = state_dict[key]
    return updated_state_dict

def ddp_train(rank, world_size, model, n_epochs, batch_size, data_dir, model_dir):
    print(f"Running DDP on rank {rank}.")
    # torch.cuda.set_device(rank)
    model.to(rank)
    ddp_setup(rank, world_size)
    
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    graphs = load_dataset(data_dir)
    # graphs = SubsampledDataset(data_dir)

    n_val = int(len(graphs)*0.2)
    n_train = len(graphs)-n_val
    train_graphs, val_graphs = torch.utils.data.random_split(
        graphs, [n_train, n_val], 
        # generator=torch.Generator().manual_seed(42)
    )
    train_sampler = DistributedSampler(train_graphs, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(
        train_graphs, batch_size=batch_size, shuffle=False, drop_last=True, sampler=train_sampler
    )
    val_loader = DataLoader(
        val_graphs, batch_size=batch_size, drop_last=True
    )
    # --- Loss and Optimizer ---
    pos_weight = torch.tensor([31.0337])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion.to(rank)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)
    # --- Training Loop ---
    for epoch in tqdm(range(n_epochs), disable=(rank != 0)):
        model.train()
        for batch in tqdm(train_loader, disable=(rank != 0)):
            batch.to(rank)
            set_time(
                batch,
                0, 0, 0, 0, 0,
                batchsize=batch_size, all_atoms=True,
                asyncronous_noise_schedule=False, device=rank,
                include_miscellaneous_atoms=False
            )
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits.squeeze(), batch.y)
            if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)):
                print(batch.name , batch.y, logits)
                continue
            loss.backward()
            optimizer.step()
        print(f"Rank {rank}, Epoch {epoch}, Loss: {loss / len(train_loader)}")

        model.eval()
        all_preds, all_labels = [], []
        total_val_loss = 0
        problem_batches = []
        with torch.no_grad():
            for batch in val_loader:
                batch.to(rank)
                set_time(
                    batch,
                    0, 0, 0, 0, 0,
                    batchsize=batch_size, all_atoms=True,
                    asyncronous_noise_schedule=False, device=rank,
                    include_miscellaneous_atoms=False
                )
                logits = model(batch)
                loss = criterion(logits.squeeze(), batch.y)
                if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)):
                    # print("NaN or Inf detected in validation loss, skipping this batch.")
                    # print(batch.name , batch.y, logits)
                    problem_batches.append(batch.name)
                    continue
                total_val_loss += loss.item()
                
                probs = torch.sigmoid(logits).cpu().numpy()
                labels = batch.y.cpu().numpy()
                all_preds.extend(probs)
                all_labels.extend(labels)

        avg_val_loss = total_val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        print(f'Epoch: {epoch}, Rank {rank}, Problem Batches: {len(problem_batches)}')
        # --- Evaluation Metrics ---
        # Use a 0.5 threshold on probabilities for binary metrics
        binary_preds = [1 if p > 0.5 else 0 for p in all_preds]
        
        auc_roc = roc_auc_score(all_labels, all_preds)
        auc_pr = average_precision_score(all_labels, all_preds)
        f1 = f1_score(all_labels, binary_preds)
        mcc = matthews_corrcoef(all_labels, binary_preds)
        score_dict = {
            'AUC-ROC': auc_roc,
            'AUC-PR': auc_pr,
            'F1': f1,
            'MCC': mcc
        }
        save_path = os.path.join(model_dir, f'affinity_model_epoch_{epoch}_metrics.pt')
        if rank == 0:
            torch.save(score_dict, save_path)
            print(
                f"Epoch {epoch}: Val Loss: {avg_val_loss:.4f}, AUC-ROC: {auc_roc:.4f}, "
                f"AUC-PR: {auc_pr:.4f}, F1: {f1:.4f}, MCC: {mcc:.4f}"
            )
            if epoch % 10 == 0:
                torch.save(
                    model.state_dict(), 
                    os.path.join(model_dir, f'affinity_model_epoch_{epoch}.pt')
                )
            if epoch == n_epochs - 1:
                torch.save(
                    model.state_dict(), 
                    os.path.join(model_dir, f'affinity_model_final.pt')
                )
        torch.cuda.empty_cache()

    torch.distributed.destroy_process_group()

def ddp_main(model_path, data_dir, model_save_dir):
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
    weights = load_ddp_state_dict(model_path)
    model.load_state_dict(weights)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    n_epochs = 100
    batch_size = 8
    world_size = 8
    mp.spawn(
        ddp_train, 
        args=(world_size, model, n_epochs, batch_size, data_dir, model_save_dir), 
        nprocs=world_size, 
        join=True
    )

def train_affinity_model(
    model, data_dir, model_dir, num_epochs=50, batch_size=4, 
    device='cuda', lr=1e-5, init_weight=False
):
    graphs = SubsampledDataset(data_dir)
    n_val = int(len(graphs)*0.2)
    n_train = len(graphs)-n_val
    train_graphs, val_graphs = torch.utils.data.random_split(
        graphs, [n_train, n_val], 
        # generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_graphs, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_graphs, batch_size=batch_size, drop_last=True
    )
    # --- Loss and Optimizer ---
    pos_weight = torch.tensor([31.0337])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion.to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=lr, weight_decay=1e-5
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=10
    )
    # --- Training Loop ---
    if init_weight:
        model.apply(init_weights)
    model.to(device)
    for epoch in tqdm(range(num_epochs)):
        model.train()
        for batch in tqdm(train_loader):
            batch.to(device)
            set_time(
                batch,
                0, 0, 0, 0, 0,
                batchsize=batch_size, all_atoms=True,
                asyncronous_noise_schedule=False, device=device,
                include_miscellaneous_atoms=False
            )
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits.squeeze(), batch.y)
            # if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)):
            #     print(batch.name , batch.y, logits)
            #     continue
            loss.backward()
            optimizer.step()

        model.eval()
        all_preds, all_labels = [], []
        total_val_loss = 0
        problem_batches = []
        with torch.no_grad():
            for batch in val_loader:
                batch.to(device)
                set_time(
                    batch,
                    0, 0, 0, 0, 0,
                    batchsize=batch_size, all_atoms=True,
                    asyncronous_noise_schedule=False, device=device,
                    include_miscellaneous_atoms=False
                )
                logits = model(batch)
                loss = criterion(logits.squeeze(), batch.y)
                if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)):
                    # print("NaN or Inf detected in validation loss, skipping this batch.")
                    print(batch.name , batch.y, logits)
                    # problem_batches.append(batch.name)
                    continue
                total_val_loss += loss.item()
                
                probs = torch.sigmoid(logits).cpu().numpy()
                labels = batch.y.cpu().numpy()
                all_preds.extend(probs)
                all_labels.extend(labels)

        avg_val_loss = total_val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        # --- Evaluation Metrics ---
        # Use a 0.5 threshold on probabilities for binary metrics
        binary_preds = [1 if p > 0.5 else 0 for p in all_preds]
        
        auc_roc = roc_auc_score(all_labels, all_preds)
        auc_pr = average_precision_score(all_labels, all_preds)
        f1 = f1_score(all_labels, binary_preds)
        mcc = matthews_corrcoef(all_labels, binary_preds)
        score_dict = {
            'AUC-ROC': auc_roc,
            'AUC-PR': auc_pr,
            'F1': f1,
            'MCC': mcc
        }
        save_path = os.path.join(model_dir, f'affinity_model_epoch_{epoch}_metrics.pt')
        torch.save(score_dict, save_path)
        print(
            f"Epoch {epoch}: Val Loss: {avg_val_loss:.4f}, AUC-ROC: {auc_roc:.4f}, "
            f"AUC-PR: {auc_pr:.4f}, F1: {f1:.4f}, MCC: {mcc:.4f}"
        )
        if epoch % 10 == 0:
            torch.save(
                model.state_dict(), 
                os.path.join(model_dir, f'affinity_model_epoch_{epoch}.pt')
            )
        if epoch == num_epochs - 1:
            torch.save(
                model.state_dict(), 
                os.path.join(model_dir, f'affinity_model_final.pt')
            )
        torch.cuda.empty_cache()

def single_train():
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
    n_epochs = 100
    batch_size = 64
    lr=1e-4
    data_dir = '/home/ziqiaoxu/DiffDock-Pocket/subsample_graphs/'
    model_dir = '/home/ziqiaoxu/DiffDock-Pocket/affin_models_proper_init/'
    train_affinity_model(
        model, data_dir, model_dir, 
        num_epochs=n_epochs, batch_size=batch_size, device='cuda', lr=lr
    )

if __name__ == '__main__':
    model_path = '/home/ziqiaoxu/DiffDock-Pocket/affin_models_ddp/affinity_model_epoch_10.pt'
    data_dir = '/home/ziqiaoxu/DiffDock-Pocket/all_train_samples/'
    save_dir = '/home/ziqiaoxu/DiffDock-Pocket/affin_models_all/'
    ddp_main(model_path, data_dir, save_dir)
