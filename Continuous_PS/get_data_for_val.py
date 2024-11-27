import os
import pickle
import torch
import math
import numpy as np
import pandas as pd

from av2.geometry.interpolate import compute_midpoint_line
from av2.map.map_api import ArgoverseStaticMap
from av2.map.map_primitives import Polyline
from av2.utils.io import read_json_file
from tqdm import tqdm

from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union
from urllib import request
import numpy as np
import pandas as pd
from torch_geometric.data import Dataset
from torch_geometric.data import HeteroData
from torch_geometric.data import extract_tar
from typing import Any, List, Optional


def wrap_angle(
        angle: torch.Tensor,
        min_val: float = -math.pi,
        max_val: float = math.pi) -> torch.Tensor:
    return min_val + (angle + max_val) % (max_val - min_val)

def get_agent_features(df):
    predict_unseen_agents=False
    num_historical_steps=50
    num_future_steps=60
    num_steps = num_historical_steps + num_future_steps
    dim=3
    split= 'val'
    vector_repr=True
    _agent_types = ['vehicle', 'pedestrian', 'motorcyclist', 'cyclist', 'bus', 'static', 'background',
                                'construction', 'riderless_bicycle', 'unknown']
    if not predict_unseen_agents:  # filter out agents that are unseen during the historical time steps
        historical_df = df[df['timestep'] < num_historical_steps]
        agent_ids = list(historical_df['track_id'].unique())
        df = df[df['track_id'].isin(agent_ids)]
    else:
        agent_ids = list(df['track_id'].unique())

    num_agents = len(agent_ids)
    av_idx = agent_ids.index('AV')

    # initialization
    valid_mask = torch.zeros(num_agents, num_steps, dtype=torch.bool)
    current_valid_mask = torch.zeros(num_agents, dtype=torch.bool)
    predict_mask = torch.zeros(num_agents, num_steps, dtype=torch.bool)
    agent_id: List[Optional[str]] = [None] * num_agents
    agent_type = torch.zeros(num_agents, dtype=torch.uint8)
    agent_category = torch.zeros(num_agents, dtype=torch.uint8)
    position = torch.zeros(num_agents, num_steps, dim, dtype=torch.float)
    heading = torch.zeros(num_agents, num_steps, dtype=torch.float)
    velocity = torch.zeros(num_agents, num_steps, dim, dtype=torch.float)

    for track_id, track_df in df.groupby('track_id'):  #track_df表示该id下的df
        agent_idx = agent_ids.index(track_id)
        agent_steps = track_df['timestep'].values

        valid_mask[agent_idx, agent_steps] = True
        current_valid_mask[agent_idx] = valid_mask[agent_idx, num_historical_steps - 1]
        predict_mask[agent_idx, agent_steps] = True
        if vector_repr:  # a time step t is valid only when both t and t-1 are valid
            valid_mask[agent_idx, 1: num_historical_steps] = (
                    valid_mask[agent_idx, :num_historical_steps - 1] &
                    valid_mask[agent_idx, 1: num_historical_steps])
            valid_mask[agent_idx, 0] = False
        predict_mask[agent_idx, :num_historical_steps] = False  #要预测的是true，过去的是false
        if not current_valid_mask[agent_idx]:
            predict_mask[agent_idx, num_historical_steps:] = False  #如果还没预测物体就消失了，则是要预测的未来是false

        agent_id[agent_idx] = track_id
        agent_type[agent_idx] = _agent_types.index(track_df['object_type'].values[0])
        agent_category[agent_idx] = track_df['object_category'].values[0]
        position[agent_idx, agent_steps, :2] = torch.from_numpy(np.stack([track_df['position_x'].values,
                                                                          track_df['position_y'].values],
                                                                         axis=-1)).float()
        heading[agent_idx, agent_steps] = torch.from_numpy(track_df['heading'].values).float()
        velocity[agent_idx, agent_steps, :2] = torch.from_numpy(np.stack([track_df['velocity_x'].values,
                                                                          track_df['velocity_y'].values],
                                                                         axis=-1)).float()

    if split == 'test':
        predict_mask[current_valid_mask
                     | (agent_category == 2)
                     | (agent_category == 3), num_historical_steps:] = True

    return {
            'num_nodes': num_agents,
            'av_index': av_idx,
            'valid_mask': valid_mask,
            'predict_mask': predict_mask,
            'id': agent_id,
            'type': agent_type,
            'category': agent_category,
            'position': position,
            'heading': heading,
            'velocity': velocity,
        }

def side_to_directed_lineseg(
        query_point: torch.Tensor,
        start_point: torch.Tensor,
        end_point: torch.Tensor) -> str:
    cond = ((end_point[0] - start_point[0]) * (query_point[1] - start_point[1]) -
            (end_point[1] - start_point[1]) * (query_point[0] - start_point[0]))
    if cond > 0:
        return 'LEFT'
    elif cond < 0:
        return 'RIGHT'
    else:
        return 'CENTER'
    
def safe_list_index(ls: List[Any], elem: Any) -> Optional[int]:
    try:
        return ls.index(elem)
    except ValueError:
        return None

def get_map_features(map_api,centerlines) :
    dim=3
    _polygon_types = ['VEHICLE', 'BIKE', 'BUS', 'PEDESTRIAN']
    _polygon_is_intersections = [True, False, None]
    _point_types = ['DASH_SOLID_YELLOW', 'DASH_SOLID_WHITE', 'DASHED_WHITE', 'DASHED_YELLOW',
                                'DOUBLE_SOLID_YELLOW', 'DOUBLE_SOLID_WHITE', 'DOUBLE_DASH_YELLOW', 'DOUBLE_DASH_WHITE',
                                'SOLID_YELLOW', 'SOLID_WHITE', 'SOLID_DASH_WHITE', 'SOLID_DASH_YELLOW', 'SOLID_BLUE',
                                'NONE', 'UNKNOWN', 'CROSSWALK', 'CENTERLINE']
    _point_sides = ['LEFT', 'RIGHT', 'CENTER']
    _polygon_to_polygon_types = ['NONE', 'PRED', 'SUCC', 'LEFT', 'RIGHT']
    lane_segment_ids = map_api.get_scenario_lane_segment_ids()
    cross_walk_ids = list(map_api.vector_pedestrian_crossings.keys())
    polygon_ids = lane_segment_ids + cross_walk_ids
    num_polygons = len(lane_segment_ids) + len(cross_walk_ids) * 2
    # initialization
    polygon_position = torch.zeros(num_polygons, dim, dtype=torch.float)
    polygon_orientation = torch.zeros(num_polygons, dtype=torch.float)
    polygon_height = torch.zeros(num_polygons, dtype=torch.float)
    polygon_type = torch.zeros(num_polygons, dtype=torch.uint8)
    polygon_is_intersection = torch.zeros(num_polygons, dtype=torch.uint8)
    point_position: List[Optional[torch.Tensor]] = [None] * num_polygons
    point_orientation: List[Optional[torch.Tensor]] = [None] * num_polygons
    point_magnitude: List[Optional[torch.Tensor]] = [None] * num_polygons
    point_height: List[Optional[torch.Tensor]] = [None] * num_polygons
    point_type: List[Optional[torch.Tensor]] = [None] * num_polygons
    point_side: List[Optional[torch.Tensor]] = [None] * num_polygons
    for lane_segment in map_api.get_scenario_lane_segments():
        lane_segment_idx = polygon_ids.index(lane_segment.id)
        centerline = torch.from_numpy(centerlines[lane_segment.id].xyz).float()
        polygon_position[lane_segment_idx] = centerline[0, :dim]
        polygon_orientation[lane_segment_idx] = torch.atan2(centerline[1, 1] - centerline[0, 1],
                                                            centerline[1, 0] - centerline[0, 0])
        polygon_height[lane_segment_idx] = centerline[1, 2] - centerline[0, 2]
        polygon_type[lane_segment_idx] = _polygon_types.index(lane_segment.lane_type.value)
        polygon_is_intersection[lane_segment_idx] = _polygon_is_intersections.index(
            lane_segment.is_intersection)
        left_boundary = torch.from_numpy(lane_segment.left_lane_boundary.xyz).float()
        right_boundary = torch.from_numpy(lane_segment.right_lane_boundary.xyz).float()
        point_position[lane_segment_idx] = torch.cat([left_boundary[:-1, :dim],
                                                      right_boundary[:-1, :dim],
                                                      centerline[:-1, :dim]], dim=0)
        left_vectors = left_boundary[1:] - left_boundary[:-1]
        right_vectors = right_boundary[1:] - right_boundary[:-1]
        center_vectors = centerline[1:] - centerline[:-1]
        point_orientation[lane_segment_idx] = torch.cat([torch.atan2(left_vectors[:, 1], left_vectors[:, 0]),
                                                         torch.atan2(right_vectors[:, 1], right_vectors[:, 0]),
                                                         torch.atan2(center_vectors[:, 1], center_vectors[:, 0])],
                                                        dim=0)
        point_magnitude[lane_segment_idx] = torch.norm(torch.cat([left_vectors[:, :2],
                                                                  right_vectors[:, :2],
                                                                  center_vectors[:, :2]], dim=0), p=2, dim=-1)
        point_height[lane_segment_idx] = torch.cat([left_vectors[:, 2], right_vectors[:, 2], center_vectors[:, 2]],
                                                   dim=0)
        left_type = _point_types.index(lane_segment.left_mark_type.value)
        right_type = _point_types.index(lane_segment.right_mark_type.value)
        center_type = _point_types.index('CENTERLINE')
        point_type[lane_segment_idx] = torch.cat(
            [torch.full((len(left_vectors),), left_type, dtype=torch.uint8),
             torch.full((len(right_vectors),), right_type, dtype=torch.uint8),
             torch.full((len(center_vectors),), center_type, dtype=torch.uint8)], dim=0)
        point_side[lane_segment_idx] = torch.cat(
            [torch.full((len(left_vectors),), _point_sides.index('LEFT'), dtype=torch.uint8),
             torch.full((len(right_vectors),), _point_sides.index('RIGHT'), dtype=torch.uint8),
             torch.full((len(center_vectors),), _point_sides.index('CENTER'), dtype=torch.uint8)], dim=0)
    for crosswalk in map_api.get_scenario_ped_crossings():
        crosswalk_idx = polygon_ids.index(crosswalk.id)
        edge1 = torch.from_numpy(crosswalk.edge1.xyz).float()
        edge2 = torch.from_numpy(crosswalk.edge2.xyz).float()
        start_position = (edge1[0] + edge2[0]) / 2
        end_position = (edge1[-1] + edge2[-1]) / 2
        polygon_position[crosswalk_idx] = start_position[:dim]
        polygon_position[crosswalk_idx + len(cross_walk_ids)] = end_position[:dim]
        polygon_orientation[crosswalk_idx] = torch.atan2((end_position - start_position)[1],
                                                         (end_position - start_position)[0])
        polygon_orientation[crosswalk_idx + len(cross_walk_ids)] = torch.atan2((start_position - end_position)[1],
                                                                               (start_position - end_position)[0])
        polygon_height[crosswalk_idx] = end_position[2] - start_position[2]
        polygon_height[crosswalk_idx + len(cross_walk_ids)] = start_position[2] - end_position[2]
        polygon_type[crosswalk_idx] = _polygon_types.index('PEDESTRIAN')
        polygon_type[crosswalk_idx + len(cross_walk_ids)] = _polygon_types.index('PEDESTRIAN')
        polygon_is_intersection[crosswalk_idx] = _polygon_is_intersections.index(None)
        polygon_is_intersection[crosswalk_idx + len(cross_walk_ids)] = _polygon_is_intersections.index(None)
        if side_to_directed_lineseg((edge1[0] + edge1[-1]) / 2, start_position, end_position) == 'LEFT':
            left_boundary = edge1
            right_boundary = edge2
        else:
            left_boundary = edge2
            right_boundary = edge1
        num_centerline_points = math.ceil(torch.norm(end_position - start_position, p=2, dim=-1).item() / 2.0) + 1
        centerline = torch.from_numpy(
            compute_midpoint_line(left_ln_boundary=left_boundary.numpy(),
                                  right_ln_boundary=right_boundary.numpy(),
                                  num_interp_pts=int(num_centerline_points))[0]).float()
        point_position[crosswalk_idx] = torch.cat([left_boundary[:-1, :dim],
                                                   right_boundary[:-1, :dim],
                                                   centerline[:-1, :dim]], dim=0)
        point_position[crosswalk_idx + len(cross_walk_ids)] = torch.cat(
            [right_boundary.flip(dims=[0])[:-1, :dim],
             left_boundary.flip(dims=[0])[:-1, :dim],
             centerline.flip(dims=[0])[:-1, :dim]], dim=0)
        left_vectors = left_boundary[1:] - left_boundary[:-1]
        right_vectors = right_boundary[1:] - right_boundary[:-1]
        center_vectors = centerline[1:] - centerline[:-1]
        point_orientation[crosswalk_idx] = torch.cat(
            [torch.atan2(left_vectors[:, 1], left_vectors[:, 0]),
             torch.atan2(right_vectors[:, 1], right_vectors[:, 0]),
             torch.atan2(center_vectors[:, 1], center_vectors[:, 0])], dim=0)
        point_orientation[crosswalk_idx + len(cross_walk_ids)] = torch.cat(
            [torch.atan2(-right_vectors.flip(dims=[0])[:, 1], -right_vectors.flip(dims=[0])[:, 0]),
             torch.atan2(-left_vectors.flip(dims=[0])[:, 1], -left_vectors.flip(dims=[0])[:, 0]),
             torch.atan2(-center_vectors.flip(dims=[0])[:, 1], -center_vectors.flip(dims=[0])[:, 0])], dim=0)
        point_magnitude[crosswalk_idx] = torch.norm(torch.cat([left_vectors[:, :2],
                                                               right_vectors[:, :2],
                                                               center_vectors[:, :2]], dim=0), p=2, dim=-1)
        point_magnitude[crosswalk_idx + len(cross_walk_ids)] = torch.norm(
            torch.cat([-right_vectors.flip(dims=[0])[:, :2],
                       -left_vectors.flip(dims=[0])[:, :2],
                       -center_vectors.flip(dims=[0])[:, :2]], dim=0), p=2, dim=-1)
        point_height[crosswalk_idx] = torch.cat([left_vectors[:, 2], right_vectors[:, 2], center_vectors[:, 2]],
                                                dim=0)
        point_height[crosswalk_idx + len(cross_walk_ids)] = torch.cat(
            [-right_vectors.flip(dims=[0])[:, 2],
             -left_vectors.flip(dims=[0])[:, 2],
             -center_vectors.flip(dims=[0])[:, 2]], dim=0)
        crosswalk_type = _point_types.index('CROSSWALK')
        center_type = _point_types.index('CENTERLINE')
        point_type[crosswalk_idx] = torch.cat([
            torch.full((len(left_vectors),), crosswalk_type, dtype=torch.uint8),
            torch.full((len(right_vectors),), crosswalk_type, dtype=torch.uint8),
            torch.full((len(center_vectors),), center_type, dtype=torch.uint8)], dim=0)
        point_type[crosswalk_idx + len(cross_walk_ids)] = torch.cat(
            [torch.full((len(right_vectors),), crosswalk_type, dtype=torch.uint8),
             torch.full((len(left_vectors),), crosswalk_type, dtype=torch.uint8),
             torch.full((len(center_vectors),), center_type, dtype=torch.uint8)], dim=0)
        point_side[crosswalk_idx] = torch.cat(
            [torch.full((len(left_vectors),), _point_sides.index('LEFT'), dtype=torch.uint8),
             torch.full((len(right_vectors),), _point_sides.index('RIGHT'), dtype=torch.uint8),
             torch.full((len(center_vectors),), _point_sides.index('CENTER'), dtype=torch.uint8)], dim=0)
        point_side[crosswalk_idx + len(cross_walk_ids)] = torch.cat(
            [torch.full((len(right_vectors),), _point_sides.index('LEFT'), dtype=torch.uint8),
             torch.full((len(left_vectors),), _point_sides.index('RIGHT'), dtype=torch.uint8),
             torch.full((len(center_vectors),), _point_sides.index('CENTER'), dtype=torch.uint8)], dim=0)
    num_points = torch.tensor([point.size(0) for point in point_position], dtype=torch.long)
    point_to_polygon_edge_index = torch.stack(
        [torch.arange(num_points.sum(), dtype=torch.long),
         torch.arange(num_polygons, dtype=torch.long).repeat_interleave(num_points)], dim=0)
    polygon_to_polygon_edge_index = []
    polygon_to_polygon_type = []
    for lane_segment in map_api.get_scenario_lane_segments():
        lane_segment_idx = polygon_ids.index(lane_segment.id)
        pred_inds = []
        for pred in lane_segment.predecessors:
            pred_idx = safe_list_index(polygon_ids, pred)
            if pred_idx is not None:
                pred_inds.append(pred_idx)
        if len(pred_inds) != 0:
            polygon_to_polygon_edge_index.append(
                torch.stack([torch.tensor(pred_inds, dtype=torch.long),
                             torch.full((len(pred_inds),), lane_segment_idx, dtype=torch.long)], dim=0))
            polygon_to_polygon_type.append(
                torch.full((len(pred_inds),), _polygon_to_polygon_types.index('PRED'), dtype=torch.uint8))
        succ_inds = []
        for succ in lane_segment.successors:
            succ_idx = safe_list_index(polygon_ids, succ)
            if succ_idx is not None:
                succ_inds.append(succ_idx)
        if len(succ_inds) != 0:
            polygon_to_polygon_edge_index.append(
                torch.stack([torch.tensor(succ_inds, dtype=torch.long),
                             torch.full((len(succ_inds),), lane_segment_idx, dtype=torch.long)], dim=0))
            polygon_to_polygon_type.append(
                torch.full((len(succ_inds),), _polygon_to_polygon_types.index('SUCC'), dtype=torch.uint8))
        if lane_segment.left_neighbor_id is not None:
            left_idx = safe_list_index(polygon_ids, lane_segment.left_neighbor_id)
            if left_idx is not None:
                polygon_to_polygon_edge_index.append(
                    torch.tensor([[left_idx], [lane_segment_idx]], dtype=torch.long))
                polygon_to_polygon_type.append(
                    torch.tensor([_polygon_to_polygon_types.index('LEFT')], dtype=torch.uint8))
        if lane_segment.right_neighbor_id is not None:
            right_idx = safe_list_index(polygon_ids, lane_segment.right_neighbor_id)
            if right_idx is not None:
                polygon_to_polygon_edge_index.append(
                    torch.tensor([[right_idx], [lane_segment_idx]], dtype=torch.long))
                polygon_to_polygon_type.append(
                    torch.tensor([_polygon_to_polygon_types.index('RIGHT')], dtype=torch.uint8))
    if len(polygon_to_polygon_edge_index) != 0:
        polygon_to_polygon_edge_index = torch.cat(polygon_to_polygon_edge_index, dim=1)
        polygon_to_polygon_type = torch.cat(polygon_to_polygon_type, dim=0)
    else:
        polygon_to_polygon_edge_index = torch.tensor([[], []], dtype=torch.long)
        polygon_to_polygon_type = torch.tensor([], dtype=torch.uint8)
    map_data = {
        'map_polygon': {},
        'map_point': {},
        ('map_point', 'to', 'map_polygon'): {},
        ('map_polygon', 'to', 'map_polygon'): {},
    }
    map_data['map_polygon']['num_nodes'] = num_polygons
    map_data['map_polygon']['position'] = polygon_position
    map_data['map_polygon']['orientation'] = polygon_orientation
    if dim == 3:
        map_data['map_polygon']['height'] = polygon_height
    map_data['map_polygon']['type'] = polygon_type
    map_data['map_polygon']['is_intersection'] = polygon_is_intersection
    if len(num_points) == 0:
        map_data['map_point']['num_nodes'] = 0
        map_data['map_point']['position'] = torch.tensor([], dtype=torch.float)
        map_data['map_point']['orientation'] = torch.tensor([], dtype=torch.float)
        map_data['map_point']['magnitude'] = torch.tensor([], dtype=torch.float)
        if dim == 3:
            map_data['map_point']['height'] = torch.tensor([], dtype=torch.float)
        map_data['map_point']['type'] = torch.tensor([], dtype=torch.uint8)
        map_data['map_point']['side'] = torch.tensor([], dtype=torch.uint8)
    else:
        map_data['map_point']['num_nodes'] = num_points.sum().item()
        map_data['map_point']['position'] = torch.cat(point_position, dim=0)
        map_data['map_point']['orientation'] = torch.cat(point_orientation, dim=0)
        map_data['map_point']['magnitude'] = torch.cat(point_magnitude, dim=0)
        if dim == 3:
            map_data['map_point']['height'] = torch.cat(point_height, dim=0)
        map_data['map_point']['type'] = torch.cat(point_type, dim=0)
        map_data['map_point']['side'] = torch.cat(point_side, dim=0)
    map_data['map_point', 'to', 'map_polygon']['edge_index'] = point_to_polygon_edge_index
    map_data['map_polygon', 'to', 'map_polygon']['edge_index'] = polygon_to_polygon_edge_index
    map_data['map_polygon', 'to', 'map_polygon']['type'] = polygon_to_polygon_type
    return map_data




def get_data(raw_file_name):
    root="/home/niutian/原data/Argoverse 2 Motion Forecasting Dataset"
    split= 'val'
    train_raw_dir = os.path.join(root, split, 'raw')
    _raw_dir = train_raw_dir
    _raw_file_names = [name for name in os.listdir(_raw_dir) if os.path.isdir(os.path.join(_raw_dir, name))]
    train_processed_dir = os.path.join(root, split, 'processed')
    _processed_dir = train_processed_dir
    _processed_file_names = [name for name in os.listdir(_processed_dir) if os.path.isfile(os.path.join(_processed_dir, name)) and name.endswith(('pkl', 'pickle'))]
    # raw_file_name='6a115008-a464-4c1c-aa55-72ffa71b4177'
    df = pd.read_parquet(os.path.join(_raw_dir, raw_file_name, f'scenario_{raw_file_name}.parquet'))
    map_dir = Path(_raw_dir) / raw_file_name
    map_path = map_dir / sorted(map_dir.glob('log_map_archive_*.json'))[0]
    map_data = read_json_file(map_path)
    centerlines = {lane_segment['id']: Polyline.from_json_data(lane_segment['centerline'])
                for lane_segment in map_data['lane_segments'].values()}
    map_api = ArgoverseStaticMap.from_json(map_path)
    data = dict()
    data['scenario_id'] = df['scenario_id'].values[0]
    data['city'] = df['city'].values[0]
    data['agent'] = get_agent_features(df)
    data.update(get_map_features(map_api, centerlines))

    origin = data['agent']['position'][:, 50 - 1] #[30,3]
    theta = data['agent']['heading'][:, 50 - 1] #[30]
    cos, sin = theta.cos(), theta.sin()
    rot_mat = theta.new_zeros(data['agent']['num_nodes'], 2, 2) #[30,2,2]Returns a Tensor of size size filled with 0. By default, the returned Tensor has the same torch.dtype and torch.device as this tensor
    rot_mat[:, 0, 0] = cos
    rot_mat[:, 0, 1] = -sin
    rot_mat[:, 1, 0] = sin
    rot_mat[:, 1, 1] = cos
    data['agent']['target'] = origin.new_zeros(data['agent']['num_nodes'], 60, 4)
    data['agent']['target'][..., :2] = torch.bmm(data['agent']['position'][:, 50:, :2] -
                                                origin[:, :2].unsqueeze(1), rot_mat)
    if data['agent']['position'].size(2) == 3:
        data['agent']['target'][..., 2] = (data['agent']['position'][:, 50:, 2] -
                                        origin[:, 2].unsqueeze(-1))
    data['agent']['target'][..., 3] = wrap_angle(data['agent']['heading'][:, 50:] -
                                                theta.unsqueeze(-1))
    
    return data

# if __name__ == '__main__':
#     f_save = open("/data/niutian/QCNet_MCTS_data/groundtruth_val_02.pkl", 'wb')  #new
#     pickle.dump(data, f_save) #new
#     f_save.close()