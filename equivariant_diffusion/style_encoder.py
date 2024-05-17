import torch
import torch.nn as nn
import torch.nn.functional as F
from equivariant_diffusion.egnn_new import EGNN, GNN
from equivariant_diffusion.egnn_new import unsorted_segment_sum
import numpy as np
from torch_scatter import scatter_mean

class StyleEncoder(nn.Module):
    def __init__(self, atom_nf, residue_nf,
                 n_dims, joint_nf=16, hidden_nf=64, device='cpu',
                 act_fn=torch.nn.SiLU(), n_layers=4, attention=False,
                 condition_time=True, tanh=False, mode='egnn_dynamics',
                 norm_constant=0, inv_sublayers=2, sin_embedding=False,
                 normalization_factor=100, aggregation_method='sum',
                 update_pocket_coords=True, edge_cutoff_ligand=None,
                 edge_cutoff_pocket=None, edge_cutoff_interaction=None,
                 reflection_equivariant=True, edge_embedding_dim=None):
        super().__init__()
        self.mode = mode
        self.edge_cutoff_l = edge_cutoff_ligand
        self.edge_cutoff_p = edge_cutoff_pocket
        self.edge_cutoff_i = edge_cutoff_interaction
        self.edge_nf = edge_embedding_dim

        self.atom_encoder = nn.Sequential(
            nn.Linear(atom_nf, 2 * atom_nf),
            act_fn,
            nn.Linear(2 * atom_nf, joint_nf)
        )


        self.residue_encoder = nn.Sequential(
            nn.Linear(residue_nf, 2 * residue_nf),
            act_fn,
            nn.Linear(2 * residue_nf, joint_nf)
        )


        self.edge_nf = 0 if self.edge_nf is None else self.edge_nf

        # if condition_time:
        #     dynamics_node_nf = joint_nf + 1
        # else:
        #     print('Warning: dynamics model is _not_ conditioned on time.')
        dynamics_node_nf = joint_nf

        
        self.egnn = EGNN(
            in_node_nf=dynamics_node_nf, in_edge_nf=self.edge_nf,
            hidden_nf=hidden_nf, device=device, act_fn=act_fn,
            n_layers=n_layers, attention=attention, tanh=tanh,
            norm_constant=norm_constant,
            inv_sublayers=inv_sublayers, sin_embedding=sin_embedding,
            normalization_factor=normalization_factor,
            aggregation_method=aggregation_method,
            reflection_equiv=reflection_equivariant,
            style_encoder=True
        )
        self.node_nf = dynamics_node_nf
        self.update_pocket_coords = update_pocket_coords


        self.device = device
        self.n_dims = n_dims
        self.condition_time = condition_time

        self.final_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf),
        )

    @staticmethod
    def remove_mean_batch(x, indices):
        mean = scatter_mean(x, indices, dim=0)
        x = x - mean[indices]
        return x

    def forward(self, xh_atoms, mask_atoms):

        x_atoms = xh_atoms[:, :self.n_dims].clone()
        x_atoms = self.remove_mean_batch(x_atoms, mask_atoms)

        h_atoms = xh_atoms[:, self.n_dims:].clone()
        h_atoms = self.atom_encoder(h_atoms)

        # x_residues = xh_residues[:, :self.n_dims].clone()
        # h_residues = xh_residues[:, self.n_dims:].clone()

        # embed atom features and residue features in a shared space
        # h_residues = self.residue_encoder(h_residues)

        # combine the two node types
        # x = torch.cat((x_atoms, x_residues), dim=0)
        # h = torch.cat((h_atoms, h_residues), dim=0)
        # mask = torch.cat([mask_atoms, mask_residues])
        x = x_atoms
        h = h_atoms
        mask = mask_atoms


        # get edges of a complete graph
        edges = self.get_edges(mask_atoms, x_atoms)
        assert torch.all(mask[edges[0]] == mask[edges[1]])


        edge_types = None

        
        update_coords_mask = None if self.update_pocket_coords \
            else torch.cat((torch.ones_like(mask_atoms),
                            torch.zeros_like(mask_residues))).unsqueeze(1)
        h_final, x_final = self.egnn(h, x, edges,
                                        update_coords_mask=update_coords_mask,
                                        batch_mask=mask, edge_attr=edge_types)


        h_ligand = h_final[:len(mask_atoms)]
        h_pocket = h_final[len(mask_atoms):]

       
        h_ligand_avg_pool = unsorted_segment_sum(h_ligand, mask_atoms, mask_atoms.max() + 1, normalization_factor=1, aggregation_method='mean')
        style_ligand = self.final_mlp(h_ligand_avg_pool)


        return style_ligand

    def get_edges(self, batch_mask_ligand, x_ligand):
        adj_ligand = batch_mask_ligand[:, None] == batch_mask_ligand[None, :]
        # adj_pocket = batch_mask_pocket[:, None] == batch_mask_pocket[None, :]
        # adj_cross = batch_mask_ligand[:, None] == batch_mask_pocket[None, :]

        if self.edge_cutoff_l is not None:
            adj_ligand = adj_ligand & (torch.cdist(x_ligand, x_ligand) <= self.edge_cutoff_l)

        # if self.edge_cutoff_p is not None:
        #     adj_pocket = adj_pocket & (torch.cdist(x_pocket, x_pocket) <= self.edge_cutoff_p)

        # if self.edge_cutoff_i is not None:
        #     adj_cross = adj_cross & (torch.cdist(x_ligand, x_pocket) <= self.edge_cutoff_i)

        # adj = torch.cat((torch.cat((adj_ligand, adj_cross), dim=1),
        #                  torch.cat((adj_cross.T, adj_pocket), dim=1)), dim=0)
        edges = torch.stack(torch.where(adj_ligand), dim=0)

        return edges