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
from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn
from torch_cluster import radius
from torch_cluster import radius_graph
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData
from torch_geometric.utils import dense_to_sparse
from torch_geometric.utils import subgraph

from layers.attention_layer import AttentionLayer
from layers.fourier_embedding import FourierEmbedding
from utils import angle_between_2d_vectors
from utils import weight_init
from utils import wrap_angle


class QCNetAgentEncoder(nn.Module):

    def __init__(self,
                 dataset: str,
                 input_dim: int,
                 hidden_dim: int,
                 num_historical_steps: int,
                 time_span: Optional[int],
                 pl2a_radius: float,
                 a2a_radius: float,
                 num_freq_bands: int,
                 num_layers: int, #default=2
                 num_heads: int,
                 head_dim: int,
                 dropout: float) -> None:
        super(QCNetAgentEncoder, self).__init__()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps #50
        self.time_span = time_span if time_span is not None else num_historical_steps
        self.pl2a_radius = pl2a_radius
        self.a2a_radius = a2a_radius
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout

        if dataset == 'argoverse_v2':
            input_dim_x_a = 4
            input_dim_r_t = 4
            input_dim_r_pl2a = 3
            input_dim_r_a2a = 3
        else:
            raise ValueError('{} is not a valid dataset'.format(dataset))

        if dataset == 'argoverse_v2':
            self.type_a_emb = nn.Embedding(10, hidden_dim)
        else:
            raise ValueError('{} is not a valid dataset'.format(dataset))
        self.x_a_emb = FourierEmbedding(input_dim=input_dim_x_a, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_t_emb = FourierEmbedding(input_dim=input_dim_r_t, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_pl2a_emb = FourierEmbedding(input_dim=input_dim_r_pl2a, hidden_dim=hidden_dim,
                                           num_freq_bands=num_freq_bands)
        self.r_a2a_emb = FourierEmbedding(input_dim=input_dim_r_a2a, hidden_dim=hidden_dim,
                                          num_freq_bands=num_freq_bands)
        self.t_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=False, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.pl2a_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.a2a_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=False, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.apply(weight_init)

    def forward(self,
                data: HeteroData,
                map_enc: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        mask = data['agent']['valid_mask'][:, :self.num_historical_steps].contiguous() #[30,50]
        pos_a = data['agent']['position'][:, :self.num_historical_steps, :self.input_dim].contiguous() #[30, 50, 2]
        motion_vector_a = torch.cat([pos_a.new_zeros(data['agent']['num_nodes'], 1, self.input_dim), #[30,1,2],[30,49,2]按2维进行拼接
                                     pos_a[:, 1:] - pos_a[:, :-1]], dim=1) #[30, 50, 2] agent (历史) motion相对向量
        head_a = data['agent']['heading'][:, :self.num_historical_steps].contiguous() #[30, 50] agent角度
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1) #[30, 50, 2] 角度的坐标系
        pos_pl = data['map_polygon']['position'][:, :self.input_dim].contiguous() #[35, 2]
        orient_pl = data['map_polygon']['orientation'].contiguous() #[35]
        if self.dataset == 'argoverse_v2':
            vel = data['agent']['velocity'][:, :self.num_historical_steps, :self.input_dim].contiguous() #[30, 50, 2] 速度
            length = width = height = None
            categorical_embs = [
                self.type_a_emb(data['agent']['type'].long()).repeat_interleave(repeats=self.num_historical_steps,
                                                                                dim=0), #[30,64]重复repeats次，在0维上
            ] #list[tensor[1500,64]] agent类型的embadding
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))

        if self.dataset == 'argoverse_v2':
            #the polar coordinates of all geometric attributes relative to the spatial point and direction referenced by the element’s local frame
            x_a = torch.stack(
                [torch.norm(motion_vector_a[:, :, :2], p=2, dim=-1), #motion vector 对最后一维求2范数，相当于计算出了斜边
                 angle_between_2d_vectors(ctr_vector=head_vector_a, nbr_vector=motion_vector_a[:, :, :2]), #向量相对位置
                 torch.norm(vel[:, :, :2], p=2, dim=-1), #velocity vector
                 angle_between_2d_vectors(ctr_vector=head_vector_a, nbr_vector=vel[:, :, :2])], dim=-1) #[30, 50, 4]the velocity and motion vector of each agent state
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))
        x_a = self.x_a_emb(continuous_inputs=x_a.view(-1, x_a.size(-1)), categorical_embs=categorical_embs) #[1500, 64] Fourier features are concatenated with an agent’s category and passed through a (MLP) to obtain an embedding.
        x_a = x_a.view(-1, self.num_historical_steps, self.hidden_dim) #[30,50,64]
        #Relative Spatial-Temporal Positional Embedding for r(time s to t, agent i to i )
        pos_t = pos_a.reshape(-1, self.input_dim) #[1500, 2]
        head_t = head_a.reshape(-1) #[1500]
        head_vector_t = head_vector_a.reshape(-1, 2) #[1500, 2]
        mask_t = mask.unsqueeze(2) & mask.unsqueeze(1) #[30,50,1]&[30,1,50][30, 50, 50] unsqueeze扩展
        edge_index_t = dense_to_sparse(mask_t)[0] #[2, 26186] tuple[0]
        edge_index_t = edge_index_t[:, edge_index_t[1] > edge_index_t[0]] #[2, 12737]
        edge_index_t = edge_index_t[:, edge_index_t[1] - edge_index_t[0] <= self.time_span] #[2, 5627] edge 选出观测范围内出现的agent
        rel_pos_t = pos_t[edge_index_t[0]] - pos_t[edge_index_t[1]] #[5627, 2] relative p
        rel_head_t = wrap_angle(head_t[edge_index_t[0]] - head_t[edge_index_t[1]]) #[5627] relative orientation
        r_t = torch.stack( #[5627, 4]
            [torch.norm(rel_pos_t[:, :2], p=2, dim=-1), #relative p
             angle_between_2d_vectors(ctr_vector=head_vector_t[edge_index_t[1]], nbr_vector=rel_pos_t[:, :2]), #relative direction
             rel_head_t, #relative orientation
             edge_index_t[0] - edge_index_t[1]], dim=-1) #time gap
        r_t = self.r_t_emb(continuous_inputs=r_t, categorical_embs=None) #[5627, 64]
        #Relative Spatial-Temporal Positional Embedding for r( j to i)
        pos_s = pos_a.transpose(0, 1).reshape(-1, self.input_dim) #[1500, 2]
        head_s = head_a.transpose(0, 1).reshape(-1) #[1500]
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2) #[1500, 2]
        mask_s = mask.transpose(0, 1).reshape(-1) #[1500]
        pos_pl = pos_pl.repeat(self.num_historical_steps, 1) #[1750, 2]
        orient_pl = orient_pl.repeat(self.num_historical_steps) #[1750]
        if isinstance(data, Batch):
            batch_s = torch.cat([data['agent']['batch'] + data.num_graphs * t
                                 for t in range(self.num_historical_steps)], dim=0)
            batch_pl = torch.cat([data['map_polygon']['batch'] + data.num_graphs * t
                                  for t in range(self.num_historical_steps)], dim=0)
        else:
            batch_s = torch.arange(self.num_historical_steps, #返回长度50，类型为pos_a.device（cpu）类型的张量
                                   device=pos_a.device).repeat_interleave(data['agent']['num_nodes']) #[1500]
            batch_pl = torch.arange(self.num_historical_steps,
                                    device=pos_pl.device).repeat_interleave(data['map_polygon']['num_nodes']) #[1750]
        edge_index_pl2a = radius(x=pos_s[:, :2], y=pos_pl[:, :2], r=self.pl2a_radius, batch_x=batch_s, batch_y=batch_pl,
                                 max_num_neighbors=300) #[2, 10512]
        edge_index_pl2a = edge_index_pl2a[:, mask_s[edge_index_pl2a[1]]] #[2, 10103]
        rel_pos_pl2a = pos_pl[edge_index_pl2a[0]] - pos_s[edge_index_pl2a[1]] #[10103, 2]
        rel_orient_pl2a = wrap_angle(orient_pl[edge_index_pl2a[0]] - head_s[edge_index_pl2a[1]]) #[10103]
        r_pl2a = torch.stack( #[10103, 3]
            [torch.norm(rel_pos_pl2a[:, :2], p=2, dim=-1), #relative p
             angle_between_2d_vectors(ctr_vector=head_vector_s[edge_index_pl2a[1]], nbr_vector=rel_pos_pl2a[:, :2]),#relative direction
             rel_orient_pl2a], dim=-1) #relative orientation
        r_pl2a = self.r_pl2a_emb(continuous_inputs=r_pl2a, categorical_embs=None) #[10103, 64]
        #Relative Spatial-Temporal Positional Embedding r(time t to t, agent j to i )
        edge_index_a2a = radius_graph(x=pos_s[:, :2], r=self.a2a_radius, batch=batch_s, loop=False,
                                      max_num_neighbors=300) #[2, 17102]
        edge_index_a2a = subgraph(subset=mask_s, edge_index=edge_index_a2a)[0] #[2, 5624]
        rel_pos_a2a = pos_s[edge_index_a2a[0]] - pos_s[edge_index_a2a[1]] #[5624, 2]
        rel_head_a2a = wrap_angle(head_s[edge_index_a2a[0]] - head_s[edge_index_a2a[1]]) #[5624]
        r_a2a = torch.stack( #[5624, 3]
            [torch.norm(rel_pos_a2a[:, :2], p=2, dim=-1), #relative p
             angle_between_2d_vectors(ctr_vector=head_vector_s[edge_index_a2a[1]], nbr_vector=rel_pos_a2a[:, :2]), #relative direction
             rel_head_a2a], dim=-1) #relative orientation
        r_a2a = self.r_a2a_emb(continuous_inputs=r_a2a, categorical_embs=None) #[5624, 64]
        #
        for i in range(self.num_layers): # i=0,1
            x_a = x_a.reshape(-1, self.hidden_dim) #-1表示自动补足该维度 [1500, 64]
            #Temporal Attn
            x_a = self.t_attn_layers[i](x_a, r_t, edge_index_t) #[1500, 64] 和num_layers无关
            x_a = x_a.reshape(-1, self.num_historical_steps,
                              self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim) #[1500, 64]
            #Agent-Map Attn
            x_a = self.pl2a_attn_layers[i]((map_enc['x_pl'].transpose(0, 1).reshape(-1, self.hidden_dim), x_a), r_pl2a,
                                           edge_index_pl2a) #[1500, 64]
            #Social Attn
            x_a = self.a2a_attn_layers[i](x_a, r_a2a, edge_index_a2a)  #[1500, 64]
            x_a = x_a.reshape(self.num_historical_steps, -1, self.hidden_dim).transpose(0, 1) #transpose交换矩阵的两个维度 [30, 50, 64]

        return {'x_a': x_a} #torch.Size([30, 50, 64])

