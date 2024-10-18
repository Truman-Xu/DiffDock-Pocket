import os
import torch
import yaml
import numpy as np
import Bio
from functools import partial
from argparse import Namespace
from copy import deepcopy
from datasets.pdbbind import NoAtomCloseToLigandException
from datasets.process_mols import (
    get_fullrec_graph, get_lig_graph_with_matching, 
    extract_receptor_structure, set_sidechain_rotation_masks
)
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl, get_t_schedule
from utils.utils import get_model, get_default_device, ensure_device
from utils.sampling import randomize_position, sampling

from datasets.process_mols import extract_receptor_structure
from esm import FastaBatchedDataset, pretrained
from torch_geometric.data import HeteroData


SORTING_DICT = {
    "ALA": ["N", "CA", "C", "O", "CB"],
    "ARG": ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
    "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"],
    "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"],
    "CYS": ["N", "CA", "C", "O", "CB", "SG"],
    "GLN": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"],
    "GLU": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"],
    "GLY": ["N", "CA", "C", "O"],
    "HIS": ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"],  
    "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"],  
    "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"],
    "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"],
    "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE"],
    "MSE": ["N", "CA", "C", "O", "CB", "CG", "SE", "CE"],
    "PHE": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],  
    "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD"],
    "SER": ["N", "CA", "C", "O", "CB", "OG"],
    "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2"],  
    "TRP": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"], 
    "TYR": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
    "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2"],
}

class PocketSelector(Bio.PDB.Select):
    def __int__(self, pocket, radius, all_atoms):
        self.pocket = pocket
        self.radius = radius
        self.all_atoms = all_atoms

    def accept_residue(self, residue):
        if self.all_atoms:
            return (np.linalg.norm(np.array([a.coord for a in residue.child_list]) - self.pocket, axis=1) < self.radius).any()
        return np.linalg.norm(residue.child_dict["CA"].coord - self.pocket) < self.radius

def order_atoms_in_residue(res, atom):
    """
    An order function that sorts atoms of a residue.
    Atoms N, CA, C, O always come first, thereafter the rest of the atoms are sorted according
    to how they appear in the chemical components. Hydrogens come last and are not sorted.
    """

    if atom.name == "OXT":
        return 999
    elif atom.element == "H":
        return 1000

    if res.resname in SORTING_DICT:
        if atom.name in SORTING_DICT[res.resname]:
            return SORTING_DICT[res.resname].index(atom.name)
    else:
        raise Exception("Unknown residue", res.resname)
    raise Exception(f"Could not find atom {atom.name} in {res.resname}")

def _sort_atoms_by_element(rec_model):
    for chain in rec_model:
        for res in chain:
            res.child_list.sort(key=lambda atom: order_atoms_in_residue(res, atom))

def _remove_hs(rec_model):
    for chain in rec_model:
        for res in chain:
            atoms_to_remove = []
            for atom in res:
                if atom.element == 'H':
                    atoms_to_remove.append(atom)
            for atom in atoms_to_remove:
                res.detach_child(atom.id)

def _calculate_binding_pocket(
    receptor, ligand, buffer, pocket_cutoff, skip_no_pocket_atoms=False
):
    d = torch.cdist(receptor, ligand)
    label = torch.any(d < pocket_cutoff, axis=1)
    if label.any():
        center_pocket = receptor[label].mean(axis=0)
    else:
        if skip_no_pocket_atoms:
            raise NoAtomCloseToLigandException(pocket_cutoff)

        print("No pocket residue below minimum distance ", pocket_cutoff, "taking closest at", d.min())
        center_pocket = receptor[d.min(axis=1)[0].argmin()]

    # TODO: train a model that uses a sphere around the ligand, and not the distance to the pocket? maybe better testset performance
    radius_pocket = torch.linalg.norm(ligand - center_pocket[None, :], axis=1)

    return center_pocket, radius_pocket.max() + buffer  # add a buffer around the binding pocket

def _get_flexdist_cutoff_func(rec, ligand, flexdist, mode, pocket_cutoff):  # mode can be either "L2 or prism"
        if mode == "L2":
            # compute distance cutoff with l2 distance 
            pocket_center, pocket_radius_buffered = _calculate_binding_pocket(rec, ligand, flexdist, pocket_cutoff)
            def L2_distance_metric(atom:Bio.PDB.Atom.Atom):
                return torch.linalg.vector_norm(torch.tensor(atom.coord)-pocket_center.squeeze()) <= pocket_radius_buffered

            return L2_distance_metric
        
        elif mode == "prism":
            xMin, yMin, zMin = torch.min(ligand, dim=0).values - flexdist
            xMax, yMax, zMax = torch.max(ligand, dim=0).values + flexdist
            def prism_distance_metric(atom:Bio.PDB.Atom.Atom):
                atom_coord = torch.tensor(atom.coord)
                if (xMin <= atom_coord[0] <= xMax) * (yMin <= atom_coord[1] <= yMax) * (zMin <= atom_coord[2] <= zMax):
                    # check distance to ligand atoms akin to gnina, valid as hydrogens are removed during graph construction
                    return torch.any(torch.linalg.vector_norm(ligand - atom_coord, ord=2, dim=1) < flexdist)
                else: 
                    return False 
            return prism_distance_metric
        else:
            raise NotImplementedError(f"The distancec metric {mode} is not implemented.")

def get_complex_graph(
    receptor_model, 
    lm_embedding_chains, # list
    ligand, # rdkit Mol
    ligand_id, # str
    pocket_center = None, # tensor
    predefined_flexible_sidechains = None
):

    # IMPORTANT: The indices between experimental_receptor and computational_receptor are not a 1:1 mapping
    # So we sort the atoms by element name, such that they are equal
    _remove_hs(receptor_model)
    _sort_atoms_by_element(receptor_model)

    complex_graph = HeteroData()
    complex_graph['name'] = f'{receptor_model.parent.id}_{receptor_model.id}_{ligand_id}'
    get_lig_graph_with_matching(
        ligand, complex_graph, popsize=20, maxiter=20, matching=False,
        keep_original=False, num_conformers=1, remove_hs=True
    )

    # use the c-alpha atoms to define the pocket
    # use the holo structure to define the pocket
    rec_atoms_for_pocket = torch.tensor(
        np.array([a.coord for a in receptor_model.get_atoms() if a.name == 'CA']),
        dtype=complex_graph['ligand'].pos.dtype)

    selector = None
    if pocket_center is not None:  # change to predefined pocket if any
        pocket_center = torch.tensor(pocket_center)
        molecule_center = torch.mean(complex_graph['ligand'].pos, dim=0)
        pocket_radius = torch.max(
            torch.linalg.vector_norm(complex_graph['ligand'].pos - molecule_center.unsqueeze(0), dim=1))
    else:
        pocket_center, pocket_radius = _calculate_binding_pocket(
            rec_atoms_for_pocket,
            complex_graph['ligand'].pos, 0,
            pocket_cutoff=5.0,
            skip_no_pocket_atoms=False
        )

        pocket_radius_buffered = pocket_radius + 10.0 # Angstrom

        # center-dist mode
        selector = PocketSelector()
        selector.pocket = pocket_center.cpu().detach().numpy()
        selector.radius = pocket_radius_buffered.item()
        selector.all_atoms = True

    receptor_structure = receptor_model.parent.copy()
    receptor = receptor_structure.models[0]
    receptor, rec_coords, c_alpha_coords, n_coords, c_coords, misc_coords, misc_features, lm_embeddings = extract_receptor_structure(
        receptor, ligand, cutoff=np.inf,
        lm_embedding_chains=lm_embedding_chains,
        include_miscellaneous_atoms=False,
        all_atom=True,
        selector=selector)

    if lm_embeddings is not None and len(c_alpha_coords) != len(lm_embeddings):
        raise ValueError(
            f"LM embeddings for complex {complex_graph['name']} did not have "
            "the right length for the protein."
        )

    get_fullrec_graph(
        receptor, rec_coords, c_alpha_coords, n_coords, c_coords, misc_coords, misc_features, 
        complex_graph,
        c_alpha_cutoff=15.0, 
        c_alpha_max_neighbors=24,
        remove_hs=True,
        lm_embeddings=lm_embeddings
    )

    # select flexible sidechains in receptor
    if predefined_flexible_sidechains is not None:
        predefined_flexible_sidechains = predefined_flexible_sidechains.split('-')
        accept_atom_function = lambda atom: f"{atom.parent.get_full_id()[2]}:{atom.parent.get_full_id()[3][1]}" in predefined_flexible_sidechains
    else:
        accept_atom_function = _get_flexdist_cutoff_func(
            rec_atoms_for_pocket, complex_graph['ligand'].pos,
            flexdist=3.5, 
            mode='prism',
            pocket_cutoff=5.0
        )

    complex_graph = set_sidechain_rotation_masks(
        complex_graph, receptor, accept_atom_function, remove_hs=True
    )

    protein_center = pocket_center[None, :]
    # Center the protein around the specified pos
    complex_graph['receptor'].pos -= protein_center
    complex_graph['atom'].pos -= protein_center
    complex_graph['ligand'].pos -= protein_center
    complex_graph.original_center = protein_center

    return complex_graph, ligand

@ensure_device
def compute_ESM_embeddings(model, alphabet, labels, sequences, device=None):
    # settings used
    toks_per_batch = 4096
    repr_layers = [33]
    truncation_seq_length = 1022

    dataset = FastaBatchedDataset(labels, sequences)
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=alphabet.get_batch_converter(truncation_seq_length),
        batch_sampler=batches
    )

    assert all(
        -(model.num_layers + 1) <= i <= model.num_layers for i in repr_layers
    )
    repr_layers = [
        (i + model.num_layers + 1) % (model.num_layers + 1) for i in repr_layers
    ]
    embeddings = {}

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(
                f"Processing {batch_idx + 1} of {len(batches)} "
                "batches ({toks.size(0)} sequences)"
            )
            if device is not None:
                toks = toks.to(device)

            out = model(toks, repr_layers=repr_layers, return_contacts=False)
            representations = {layer: t for layer, t in out["representations"].items()}

            for i, label in enumerate(labels):
                truncate_len = min(truncation_seq_length, len(strs[i]))
                embeddings[label] = representations[33][i, 1: truncate_len + 1].clone()

            del representations

    del dataset, data_loader

    return embeddings

class AutoDiffDocker:
    def __init__(self, model_cache_dir = '.cache/model', tag = 'v1.0.0'):
        base_model_dir = os.path.join(model_cache_dir, tag)
        self.device = get_default_device()

        sigma_schedule = 'expbeta'
        inference_steps = 30
        inf_sched_alpha = 1
        inf_sched_beta = 1
        t_max = 1
        tr_schedule = get_t_schedule(
            sigma_schedule=sigma_schedule, inference_steps=inference_steps,
            inf_sched_alpha=inf_sched_alpha, inf_sched_beta=inf_sched_beta,
            t_max=t_max
        )

        temp_sampling_tr = 0.9766350103728372
        temp_psi_tr = 1.5102572175711826
        temp_sampling_rot = 6.077432837220868
        temp_psi_rot = 0.8141168207563049
        temp_sampling_tor = 6.761568162335063
        temp_psi_tor = 0.7661845361370018
        temp_sampling_sc_tor = 1.4487910576602347
        temp_psi_sc_tor = 1.339614553802453
        temp_sigma_data = 0.48884149503636976

        with open(f'{base_model_dir}/score_model/model_parameters.yml') as f:
            score_model_args = Namespace(**yaml.full_load(f))
        with open(f'{base_model_dir}/confidence_model/model_parameters.yml') as f:
            filtering_args = Namespace(**yaml.full_load(f))

        t_to_sigma = partial(t_to_sigma_compl, args=score_model_args)

        infer_model = get_model(
            score_model_args, 
            self.device, 
            t_to_sigma=t_to_sigma, 
            no_parallel=True
        )
        state_dict = torch.load(
            f'{base_model_dir}/score_model/best_ema_inference_epoch_model.pt', 
            map_location=self.device
        )
        infer_model.load_state_dict(state_dict, strict=True)
        infer_model = infer_model.to(self.device)
        infer_model.eval()

        filtering_model = get_model(
            filtering_args, 
            self.device, 
            t_to_sigma=t_to_sigma, 
            no_parallel=True, 
            confidence_mode=True
        )
        state_dict = torch.load(
            f'{base_model_dir}/confidence_model/best_model.pt', 
            map_location=torch.device('cpu')
        )
        filtering_model.load_state_dict(state_dict, strict=True)
        filtering_model = filtering_model.to(self.device)
        filtering_model.eval()

        self._randomize_position = partial(
            randomize_position, 
            no_torsion=score_model_args.no_torsion, 
            no_random=False,
            tr_sigma_max=score_model_args.tr_sigma_max,
            flexible_sidechains=score_model_args.flexible_sidechains
        )

        self._sampling = partial(
            sampling,
            model=infer_model,
            inference_steps=30,
            tr_schedule=tr_schedule, 
            rot_schedule=tr_schedule,
            tor_schedule=tr_schedule, 
            sidechain_tor_schedule=tr_schedule,
            t_schedule=None,
            t_to_sigma=t_to_sigma, 
            model_args=score_model_args,
            confidence_model=filtering_model,
            device=self.device,
            visualization_list=None,
            sidechain_visualization_list=None,
            no_random=False,
            ode=False, 
            filtering_data_list=None,
            filtering_model_args=filtering_args,
            asyncronous_noise_schedule=score_model_args.asyncronous_noise_schedule,
            no_final_step_noise=False,
            temp_sampling=[
                temp_sampling_tr, temp_sampling_rot,
                temp_sampling_tor, temp_sampling_sc_tor
            ],
            temp_psi=[
                temp_psi_tr,
                temp_psi_rot,
                temp_psi_tor,
                temp_psi_sc_tor
            ],
            flexible_sidechains=score_model_args.flexible_sidechains
        )

    def sample_single_graph(self, complex_graph, batch_size=5):
        complex_graph_copy = deepcopy(complex_graph)
        self._randomize_position([complex_graph_copy])
        result_data_list, confidence = self._sampling(
            data_list=[complex_graph_copy],
            batch_size = batch_size
        )
        return result_data_list, confidence

    def esm_embeddings_from_chains(self, chains):
        model_location = "esm2_t33_650M_UR50D"
        model, alphabet = pretrained.load_model_and_alphabet(model_location)

        model.eval()
        if self.device is not None:
            model = model.to(self.device)

        all_labels, all_sequences = [], []

        # More efficient to calculate embeddings in batches
        # So we split the chains up for each protein complex,
        # create a numbered label for each chain, and then
        # make a list of lists at the end.
        for chain in chains:
            all_labels.append('-'.join([str(x) for x in chain.full_id]))
            all_sequences.append(str(chain.can_seq))

        lm_embeddings = compute_ESM_embeddings(
            model, alphabet, all_labels, all_sequences,
            device=self.device
        )

        del model
        return list(lm_embeddings.values())