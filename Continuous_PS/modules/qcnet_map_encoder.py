# Copyright (c) 2023, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Dict

import torch
import torch.nn as nn
from torch_cluster import radius_graph
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData

import sys
sys.path.append('/home/niutian/.vscode-server/QCNet-main')
from layers.attention_layer import AttentionLayer
from layers.fourier_embedding import FourierEmbedding
from utils import angle_between_2d_vectors
from utils import merge_edges
from utils import weight_init
from utils import wrap_angle


class QCNetMapEncoder(nn.Module):

    def __init__(self,
                 dataset: str,
                 input_dim: int,
                 hidden_dim: int,
                 num_historical_steps: int,
                 pl2pl_radius: float,
                 num_freq_bands: int,
                 num_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float) -> None:
        super(QCNetMapEncoder, self).__init__()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.pl2pl_radius = pl2pl_radius
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout

        if dataset == 'argoverse_v2':
            if input_dim == 2:
                input_dim_x_pt = 1
                input_dim_x_pl = 0
                input_dim_r_pt2pl = 3
                input_dim_r_pl2pl = 3
            elif input_dim == 3:
                input_dim_x_pt = 2
                input_dim_x_pl = 1
                input_dim_r_pt2pl = 4
                input_dim_r_pl2pl = 4
            else:
                raise ValueError('{} is not a valid dimension'.format(input_dim))
        else:
            raise ValueError('{} is not a valid dataset'.format(dataset))

        if dataset == 'argoverse_v2':
            self.type_pt_emb = nn.Embedding(17, hidden_dim)
            self.side_pt_emb = nn.Embedding(3, hidden_dim)
            self.type_pl_emb = nn.Embedding(4, hidden_dim)
            self.int_pl_emb = nn.Embedding(3, hidden_dim)
        else:
            raise ValueError('{} is not a valid dataset'.format(dataset))
        self.type_pl2pl_emb = nn.Embedding(5, hidden_dim)
        #FourierEmbedding are concatenated with the semantic attributes and passed through a (MLP)
        self.x_pt_emb = FourierEmbedding(input_dim=input_dim_x_pt, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.x_pl_emb = FourierEmbedding(input_dim=input_dim_x_pl, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_pt2pl_emb = FourierEmbedding(input_dim=input_dim_r_pt2pl, hidden_dim=hidden_dim,
                                            num_freq_bands=num_freq_bands)
        self.r_pl2pl_emb = FourierEmbedding(input_dim=input_dim_r_pl2pl, hidden_dim=hidden_dim,
                                            num_freq_bands=num_freq_bands)
        self.pt2pl_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.pl2pl_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=False, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.apply(weight_init)

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        pos_pt = data['map_point']['position'][:, :self.input_dim].contiguous() #[742, 2] contiguous()防止后续被改变 sampled points for polygon
        orient_pt = data['map_point']['orientation'].contiguous() #[742]
        pos_pl = data['map_polygon']['position'][:, :self.input_dim].contiguous() #[35,2]
        orient_pl = data['map_polygon']['orientation'].contiguous() #[35]
        orient_vector_pl = torch.stack([orient_pl.cos(), orient_pl.sin()], dim=-1) #[35,2]

        if self.dataset == 'argoverse_v2':
            if self.input_dim == 2:
                x_pt = data['map_point']['magnitude'].unsqueeze(-1) #[742,1] unsqueeze(-1)表示最后一个维度增加维数
                x_pl = None
            elif self.input_dim == 3:
                x_pt = torch.stack([data['map_point']['magnitude'], data['map_point']['height']], dim=-1)
                x_pl = data['map_polygon']['height'].unsqueeze(-1)
            else:
                raise ValueError('{} is not a valid dimension'.format(self.input_dim))
            x_pt_categorical_embs = [self.type_pt_emb(data['map_point']['type'].long()), #long表示转化为长整数，仍是tensor，其中数字由unit8变为int
                                     self.side_pt_emb(data['map_point']['side'].long())] #list[torch.Size([742, 64]),torch.Size([742, 64])]
            x_pl_categorical_embs = [self.type_pl_emb(data['map_polygon']['type'].long()),
                                     self.int_pl_emb(data['map_polygon']['is_intersection'].long())] #list[torch.Size([35, 64]),torch.Size([35, 64])]
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))

        #transform each polar coordinate into Fourier features,to facilitate learning high-frequency signals.
        #x_pt的计算
        x_pt = self.x_pt_emb(continuous_inputs=x_pt, categorical_embs=x_pt_categorical_embs) #FourierEmbedding [742, 64]
        #x_pl的计算
        x_pl = self.x_pl_emb(continuous_inputs=x_pl, categorical_embs=x_pl_categorical_embs) #[35, 64]
        #计算relative pt2pl
        edge_index_pt2pl = data['map_point', 'to', 'map_polygon']['edge_index'] #[2, 742] 是index
        rel_pos_pt2pl = pos_pt[edge_index_pt2pl[0]] - pos_pl[edge_index_pt2pl[1]] #[742, 2]
        rel_orient_pt2pl = wrap_angle(orient_pt[edge_index_pt2pl[0]] - orient_pl[edge_index_pt2pl[1]]) #[742]
        if self.input_dim == 2:
            r_pt2pl = torch.stack( #连接
                [torch.norm(rel_pos_pt2pl[:, :2], p=2, dim=-1), #按照最后一维求2范数 [742,2]变为[742]
                 angle_between_2d_vectors(ctr_vector=orient_vector_pl[edge_index_pt2pl[1]],
                                          nbr_vector=rel_pos_pt2pl[:, :2]), #[742]
                 rel_orient_pt2pl], dim=-1) #[742, 3]
        elif self.input_dim == 3:
            r_pt2pl = torch.stack(
                [torch.norm(rel_pos_pt2pl[:, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=orient_vector_pl[edge_index_pt2pl[1]],
                                          nbr_vector=rel_pos_pt2pl[:, :2]),
                 rel_pos_pt2pl[:, -1],
                 rel_orient_pt2pl], dim=-1)
        else:
            raise ValueError('{} is not a valid dimension'.format(self.input_dim))
        r_pt2pl = self.r_pt2pl_emb(continuous_inputs=r_pt2pl, categorical_embs=None) #FourierEmbedding[742, 64]
        # 计算relative pl2pl
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        edge_index_pl2pl = data['map_polygon', 'to', 'map_polygon']['edge_index'].to(device) #[2, 64]
        edge_index_pl2pl_radius = radius_graph(x=pos_pl[:, :2], r=self.pl2pl_radius,
                                               batch=data['map_polygon']['batch'] if isinstance(data, Batch) else None,
                                               loop=False, max_num_neighbors=300).to(device) #[2, 1098]
        type_pl2pl = data['map_polygon', 'to', 'map_polygon']['type'].to(device) #[64]
        type_pl2pl_radius = type_pl2pl.new_zeros(edge_index_pl2pl_radius.size(1), dtype=torch.uint8).to(device) #[1098]
        edge_index_pl2pl, type_pl2pl = merge_edges(edge_indices=[edge_index_pl2pl_radius, edge_index_pl2pl],edge_attrs=[type_pl2pl_radius, type_pl2pl], reduce='max') #[2,1098][1098]
        rel_pos_pl2pl = pos_pl[edge_index_pl2pl[0]] - pos_pl[edge_index_pl2pl[1]] #[1098, 2]
        rel_orient_pl2pl = wrap_angle(orient_pl[edge_index_pl2pl[0]] - orient_pl[edge_index_pl2pl[1]]) #[1098]
        if self.input_dim == 2:
            r_pl2pl = torch.stack(
                [torch.norm(rel_pos_pl2pl[:, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=orient_vector_pl[edge_index_pl2pl[1]],
                                          nbr_vector=rel_pos_pl2pl[:, :2]),
                 rel_orient_pl2pl], dim=-1) #[1098, 3]
        elif self.input_dim == 3:
            r_pl2pl = torch.stack(
                [torch.norm(rel_pos_pl2pl[:, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=orient_vector_pl[edge_index_pl2pl[1]],
                                          nbr_vector=rel_pos_pl2pl[:, :2]),
                 rel_pos_pl2pl[:, -1],
                 rel_orient_pl2pl], dim=-1)
        else:
            raise ValueError('{} is not a valid dimension'.format(self.input_dim))
        r_pl2pl = self.r_pl2pl_emb(continuous_inputs=r_pl2pl, categorical_embs=[self.type_pl2pl_emb(type_pl2pl.long())]) #FourierEmbedding [1098, 64]

        for i in range(self.num_layers):
            #attention-based pooling on the embeddings of sampled points within each map polygon,To further produce polygon-level representations for lanes and crosswalks
            x_pl = self.pt2pl_layers[i]((x_pt, x_pl), r_pt2pl, edge_index_pt2pl) #[35, 64] [M,D]
            # Self-Attention for Map Encoding,model the relationships among map elements
            x_pl = self.pl2pl_layers[i](x_pl, r_pl2pl, edge_index_pl2pl) #[35, 64]
        x_pl = x_pl.repeat_interleave(repeats=self.num_historical_steps,
                                      dim=0).reshape(-1, self.num_historical_steps, self.hidden_dim) #[35, 50, 64]

        return {'x_pt': x_pt, 'x_pl': x_pl} #[742, 64],[35, 50, 64]
