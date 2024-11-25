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
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform

from utils import wrap_angle


class TargetBuilder(BaseTransform):

    def __init__(self,
                 num_historical_steps: int,
                 num_future_steps: int) -> None:
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps

    def __call__(self, data: HeteroData) -> HeteroData:
        origin = data['agent']['position'][:, self.num_historical_steps - 1] #[30,3]
        theta = data['agent']['heading'][:, self.num_historical_steps - 1] #[30] 是弧度，不是角度 360°=2pi
        cos, sin = theta.cos(), theta.sin()
        rot_mat = theta.new_zeros(data['agent']['num_nodes'], 2, 2) #[30,2,2]Returns a Tensor of size size filled with 0. By default, the returned Tensor has the same torch.dtype and torch.device as this tensor
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = -sin
        rot_mat[:, 1, 0] = sin
        rot_mat[:, 1, 1] = cos
        data['agent']['target'] = origin.new_zeros(data['agent']['num_nodes'], self.num_future_steps, 4)

        data['agent']['target'][..., :2] = torch.bmm(data['agent']['position'][:, self.num_historical_steps:, :2] - origin[:, :2].unsqueeze(1), rot_mat) #计算两个tensor的矩阵乘法，torch.bmm(a,b),tensor a 的size为(b,h,w),tensor b的size为(b,w,m) 也就是说两个tensor的第一维是相等的，然后第一个数组的第三维和第二个数组的第二维度要求一样，对于剩下的则不做要求，输出维度 （b,h,m）,unsqueeze表示返回一个新的张量，对输入的既定位置插入维度 1
        if data['agent']['position'].size(2) == 3:
            data['agent']['target'][..., 2] = (data['agent']['position'][:, self.num_historical_steps:, 2] - origin[:, 2].unsqueeze(-1))
        data['agent']['target'][..., 3] = wrap_angle(data['agent']['heading'][:, self.num_historical_steps:] - theta.unsqueeze(-1))
        return data
