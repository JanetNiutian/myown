from pathlib import Path
import pickle
import numpy as np
import torch.nn.functional as F
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple
import io
import cv2
from PIL import Image as img
from PIL.Image import Image
import copy
import torch
import torch.nn.functional as F

from val_for_MCTS import qc_for_state
import sys
sys.path.append('../')
import av2.geometry.interpolate as interp_utils

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np

from av2.datasets.motion_forecasting.data_schema import (
    ArgoverseScenario,
    ObjectType,
    TrackCategory,
)
from av2.map.map_api import ArgoverseStaticMap
from av2.utils.typing import NDArrayFloat, NDArrayInt
from matplotlib.patches import Rectangle
from numpy.typing import NDArray
from shapely.geometry import LineString

from av2.datasets.motion_forecasting import scenario_serialization
# from av2.datasets.motion_forecasting.viz.scenario_visualization import (visualize_scenario,)
from av2.map.map_api import ArgoverseStaticMap

from ego_optimal_frenet_scene7 import get_ego_traj
from get_data_for_val import get_data
from cost_func import cost_all



raw_file_name = '3c375e5e-c981-4e7d-9da0-c4d8e3a8fc24'
data = get_data(raw_file_name)
nobjs = data['agent']['num_nodes']
max_t = 60
dt = 0.1


# 对于这个场景，obs是静止的，所以不用放在loop中，所以可以用loop独立算出自车一系列轨迹
# 以1s为一个window，创建x_ego,y_ego,heading_ego,vx_ego,vy_ego，不断计算往里填充，一直得到未来6s的
# obs的传入需要和他车的颗粒度对应，目前为1s，而自车traj的更新可以小于1s,当前先设置成1s
# 对于其他场景，每次循环，需要检测他车的位置，是否在obs范围内，如果在的话，会影响自车的frenet生成

x_ego,y_ego,heading_ego,vx_ego,vy_ego = get_ego_traj()
x_ego = [item for sublist in x_ego for item in sublist]
y_ego = [item for sublist in y_ego for item in sublist]
heading_ego = [item for sublist in heading_ego for item in sublist]


def get_forecasted_traj():
    forecasted_all = np.empty((max_t,nobjs,5))
    heading_all = np.empty((110,nobjs-1))
    for n in range(nobjs-1):
        heading_all[:50,n] = data['agent']['heading'][n][:50]
    h_s = copy.deepcopy(data)
    h_sp = copy.deepcopy(h_s)
    for j in range(0,6): # max_t
        qc_output = qc_for_state(h_s)
        eval_mask=3
        pi=qc_output['pi']
        pi_eval = F.softmax(pi[eval_mask], dim=-1)
        pi_eval = pi_eval.cpu().numpy()
        # max_value = np.max(pi_eval)
        # m_p_i = np.argwhere(pi_eval == max_value).flatten()
        random_index = np.random.choice([0, 1, 2, 3, 4, 5], p=pi_eval.ravel())
        for n in range(nobjs-1):
            theta=h_s['agent']['heading'][n][49]
            cos=math.cos(theta)
            sin=math.sin(theta)
            origin=h_s['agent']['position'][n][49]
            loc_refine_pos=qc_output['loc_refine_pos'][n].cpu()

            x=loc_refine_pos[random_index,:10,0]*cos-loc_refine_pos[random_index,:10,1]*sin+origin[0].unsqueeze(0) # loc_refine_pos[6,60,2]
            y=loc_refine_pos[random_index,:10,0]*sin+loc_refine_pos[random_index,:10,1]*cos+origin[1].unsqueeze(0) # [random_index,0,0]
            
            # heading = qc_output['loc_refine_head'][n,random_index,:10,0].cpu()+theta.unsqueeze(0) # [23,6,60,1]

            h_sp['agent']['position'][n,:40,:]=h_s['agent']['position'][n,10:50,:]
            h_sp['agent']['position'][n,40:50,0]=x
            h_sp['agent']['position'][n,40:50,1]=y
            h_sp['agent']['position'][n,50:100,:]=h_s['agent']['position'][n,60:,:]
            h_sp['agent']['position'][n,100:,:]=h_s['agent']['position'][n,109,:].unsqueeze(0)

            v_x = (h_sp['agent']['position'][n,40:50,0]-h_sp['agent']['position'][n,39:49,0])/0.1
            v_y = (h_sp['agent']['position'][n,40:50,1]-h_sp['agent']['position'][n,39:49,1])/0.1

            h_sp['agent']['velocity'][n,:40,:]=h_s['agent']['velocity'][n,10:50,:]
            h_sp['agent']['velocity'][n,40:50,0]=v_x
            h_sp['agent']['velocity'][n,40:50,1]=v_y
            h_sp['agent']['velocity'][n,50:100,:]=h_s['agent']['velocity'][n,60:,:]
            h_sp['agent']['velocity'][n,100:,:]=h_s['agent']['velocity'][n,109,:].unsqueeze(0)

            for hn in range(10):
                speed = math.sqrt(v_x[hn]**2 + v_y[hn]**2)
                min_speed_threshold=1.0
                if speed > min_speed_threshold:
                    heading_all[50+j*10+hn,n]= compute_heading(v_x[hn], v_y[hn])
                else:
                    heading_all[50+j*10+hn,n] = heading_all[50+j*10+hn-1,n]


            h_sp['agent']['heading'][n,:40]= h_s['agent']['heading'][n,10:50]
            h_sp['agent']['heading'][n,40:50]= torch.FloatTensor(heading_all[j*10:j*10+10,n])
            # qc_output['loc_refine_head'][n,random_index,:10,0].cpu()+theta.unsqueeze(0)   
            # torch.FloatTensor(heading)
            h_sp['agent']['heading'][n,50:100]=h_s['agent']['heading'][n,60:]
            h_sp['agent']['heading'][n,100:]=h_s['agent']['heading'][n,109].unsqueeze(0)
        
            forecasted_all[j*10:j*10+10,n,0]=x # j,n,0
            forecasted_all[j*10:j*10+10,n,1]=y # j,n,0
            # forecasted_all[j*10:j*10+10,n,2]=torch.FloatTensor(heading)
            forecasted_all[j*10:j*10+10,n,3]=v_x
            forecasted_all[j*10:j*10+10,n,4]=v_y
        
        for i in range(10):

            h_sp['agent']['position'][-1,:49,:]=h_s['agent']['position'][-1,1:50,:]
            h_sp['agent']['position'][-1,49,0]=torch.tensor(x_ego[j*10+i])
            h_sp['agent']['position'][-1,49,1]=torch.tensor(y_ego[j*10+i])
            h_sp['agent']['position'][-1,50:109,:]=h_s['agent']['position'][-1,51:,:]
            h_sp['agent']['position'][-1,109,:]=h_s['agent']['position'][-1,109,:]

            h_sp['agent']['heading'][-1,:49]=h_s['agent']['heading'][-1,1:50]
            h_sp['agent']['heading'][-1,49]=torch.tensor(heading_ego[j*10+i])
            h_sp['agent']['heading'][-1,50:109]=h_s['agent']['heading'][-1,51:]
            h_sp['agent']['heading'][-1,109]=h_s['agent']['heading'][-1,109]
        
            h_sp['agent']['velocity'][-1,:49,:]=h_s['agent']['velocity'][-1,1:50,:]
            h_sp['agent']['velocity'][-1,49,0]=torch.tensor(vx_ego[j*10+i])
            h_sp['agent']['velocity'][-1,49,1]=torch.tensor(vy_ego[j*10+i])
            h_sp['agent']['velocity'][-1,50:109,:]=h_s['agent']['velocity'][-1,51:,:]
            h_sp['agent']['velocity'][-1,109,:]=h_s['agent']['velocity'][-1,109,:]

            forecasted_all[j*10+i,-1,0] = x_ego[j*10+i]
            forecasted_all[j*10+i,-1,1] = y_ego[j*10+i]
            forecasted_all[j*10+i,-1,2] = heading_ego[j*10+i]
            forecasted_all[j*10+i,-1,3] = vx_ego[j*10+i]
            forecasted_all[j*10+i,-1,4] = vy_ego[j*10+i]
            h_s = copy.deepcopy(h_sp)

    forecasted_all[:,:(nobjs-1),2] = heading_all[50:,:]
    comfort_cost,collision_cost = cost_all(forecasted_all,nobjs) 
    print(comfort_cost,collision_cost) 
    return forecasted_all

class LowPassFilter:
    def __init__(self, alpha):
        self.alpha = alpha  # 滤波系数，介于0和1之间，值越小滤波效果越强
        self.last_val = None  # 上一次的滤波结果

    def filter(self, val):
        if self.last_val is None:
            result = val
        else:
            result = self.alpha * val + (1 - self.alpha) * self.last_val
        self.last_val = result
        return result
    
def compute_heading(vx, vy):
    # 实例化滤波器，alpha根据需要调整
    vx_filter = LowPassFilter(alpha=0.5)
    vy_filter = LowPassFilter(alpha=0.5)

    # 更新速度值并获取滤波后的速度
    vx_filtered = vx_filter.filter(vx)  # vx是当前的速度x分量
    vy_filtered = vy_filter.filter(vy)  # vy是当前的速度y分量
    
    # 使用滤波后的速度计算航向角
    heading = math.atan2(vy_filtered, vx_filtered)
    return heading

def get_inf_for_vis(nobjs):#
    # 地图信息
    scenario_path='/data/niutian/Argoverse 2 Motion Forecasting Dataset/val/raw/3c375e5e-c981-4e7d-9da0-c4d8e3a8fc24/scenario_3c375e5e-c981-4e7d-9da0-c4d8e3a8fc24.parquet'
    scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)
    static_map_path=Path('/data/niutian/Argoverse 2 Motion Forecasting Dataset/val/raw/3c375e5e-c981-4e7d-9da0-c4d8e3a8fc24/log_map_archive_3c375e5e-c981-4e7d-9da0-c4d8e3a8fc24.json')

    scenario_static_map = ArgoverseStaticMap.from_json(static_map_path)

    # 真值的导入
    his_trajectory=np.empty((nobjs,50,2))
    for i in range(nobjs):
        his_position=data['agent']['position'][i]
        for j in range(50):    
            his_trajectory[i,j,:]=his_position.numpy()[j,:2]
            for z in range(2):
                if his_trajectory[i,j,z]==0:
                    his_trajectory[i,j,z]=np.nan

    # groundtruth trajectories
    gt_trajectory=np.empty((nobjs,60,2))
    for i in range(nobjs):
        gt_position=data['agent']['position'][i]
        for j in range(60):    
            gt_trajectory[i,j,:]=gt_position.numpy()[50+j,:2]
            for z in range(2):
                if gt_trajectory[i,j,z]==0:
                    gt_trajectory[i,j,z]=np.nan

    return scenario,scenario_static_map,his_trajectory,gt_trajectory
           #ego_action_position,ego_action_velocity,ego_action_heading

def visualization(forecasted_all,nobjs):
    save_path=Path('/data/niutian/QCNet_MCTS_result/rollout_try_00_17_3_2')
    scenario,scenario_static_map,his_trajectory,gt_trajectory=get_inf_for_vis(nobjs)
    visualize_scenario(scenario,scenario_static_map,his_trajectory,gt_trajectory,
    forecasted_all,
    save_path,nobjs)

_PlotBounds = Tuple[float, float, float, float]

# Configure constants
_OBS_DURATION_TIMESTEPS = 50
_PRED_DURATION_TIMESTEPS = 60
_PLOT_BOUNDS_BUFFER_M = 30.0

_DRIVABLE_AREA_COLOR = "#1A1A1A"
_LANE_SEGMENT_COLOR= "#E0E0E0"

_DEFAULT_ACTOR_COLOR = "#D3E8EF"

_BOUNDING_BOX_ZORDER = 100  # Ensure actor bounding boxes are plotted on top of all map elements

def visualize_scenario(
    scenario: ArgoverseScenario,#
    scenario_static_map: ArgoverseStaticMap,#
    his_trajectory: NDArray[np.float64],
    gt_trajectory: NDArray[np.float64],
    forecasted_all,
    save_path: Path,#
    nobjs,
) -> None:

    plot_bounds: _PlotBounds = (0, 0, 0, 0)
    frames: List[Image] = []

    track_id = data['agent']['id']

    forecasted_trajectory_MCTS = forecasted_all
    for timestep in range(_OBS_DURATION_TIMESTEPS + _PRED_DURATION_TIMESTEPS):
        plt.rcParams['axes.facecolor'] = 'black'
        _, ax = plt.subplots()

        for track in scenario.tracks:
            for n in range(nobjs):
                if track.track_id == track_id[n]:
                    for i in range(50,110):
                        for object_state in track.object_states:
                            if object_state.timestep == i:
                                object_state.position = forecasted_trajectory_MCTS[i-50,n,:2]
                                object_state.heading = forecasted_trajectory_MCTS[i-50,n,2]
        bx = ax
        #timestep = _OBS_DURATION_TIMESTEPS - 1
        for track in scenario.tracks:
            if track.category == TrackCategory.FOCAL_TRACK or track.category == TrackCategory.SCORED_TRACK or track.track_id == 'AV':
                print(track.track_id)
                actor_timesteps: NDArrayInt = np.array(
                [
                    object_state.timestep
                    for object_state in track.object_states
                    if object_state.timestep <= timestep
                ])
                if actor_timesteps.shape[0] < 1 or actor_timesteps[-1] != timestep:
                    continue

                actor_trajectory: NDArrayFloat = np.array(
                    [
                        list(object_state.position)
                        for object_state in track.object_states
                        if object_state.timestep <= timestep
                    ]
                )
            #print(actor_trajectory.shape)
                actor_headings: NDArrayFloat = np.array(
                    [
                        object_state.heading
                        for object_state in track.object_states
                        if object_state.timestep <= timestep
                    ]
                )
                if track.track_id == 'AV':
                    track_color = 'red'
                else:
                    track_color = _DEFAULT_ACTOR_COLOR
                _ESTIMATED_VEHICLE_LENGTH_M = 3.0
                _ESTIMATED_VEHICLE_WIDTH_M = 1.5
                _ESTIMATED_CYCLIST_LENGTH_M = 1.5
                _ESTIMATED_CYCLIST_WIDTH_M = 0.7
                if track.object_type == ObjectType.VEHICLE:
                    _plot_actor_bounding_box(
                        bx,
                        actor_trajectory[-1],
                        actor_headings[-1],
                        track_color,
                        (_ESTIMATED_VEHICLE_LENGTH_M, _ESTIMATED_VEHICLE_WIDTH_M),
                    )
                elif (
                    track.object_type == ObjectType.CYCLIST
                    or track.object_type == ObjectType.MOTORCYCLIST
                ):
                    _plot_actor_bounding_box(
                        bx,
                        actor_trajectory[-1],
                        actor_headings[-1],
                        track_color,
                        (_ESTIMATED_CYCLIST_LENGTH_M, _ESTIMATED_CYCLIST_WIDTH_M),
                    )
                else:
                    bx.plot(
                        actor_trajectory[-1, 0],
                        actor_trajectory[-1, 1],
                        "o",
                        color=track_color,
                        markersize=2,
                    )
    # Plot static map elements and actor tracks
        _plot_static_map_elements(bx,scenario_static_map)#, reachable_lanes)
        plot_bounds = _plot_actor_tracks(bx, scenario, timestep) # 很关键
    
    # Plot history
        for i in range(nobjs):
    
            _plot_polylines_arrow_future(gt_trajectory[i], bx,color="#836FFF")
            _plot_polylines(bx,[gt_trajectory[i]], color="#836FFF", line_width=1, style= '--')
            #plt.plot(gt_trajectory[i,-1, 0],gt_trajectory[i,-1, 1],"*",color="#4cd33e",markersize=2, )
        
            _plot_polylines_arrow(his_trajectory[i],bx, color="#836FFF")
            _plot_polylines(bx,[his_trajectory[i]], color="#836FFF", line_width=1)
            #plt.plot(his_trajectory[i,-1, 0],his_trajectory[i,-1, 1],"*",color="#836FFF",markersize=2,)

            _plot_polylines_arrow_future(forecasted_trajectory_MCTS[:,i,:2],bx, color="#fec8f9")
            _plot_polylines(bx,[forecasted_trajectory_MCTS[:,i,:2]], color="#fec8f9", line_width=1, style= '--')

            
        plt.xlim(
            plot_bounds[0] - _PLOT_BOUNDS_BUFFER_M,
            plot_bounds[1] + _PLOT_BOUNDS_BUFFER_M,
        )
        plt.ylim(
            plot_bounds[2] - _PLOT_BOUNDS_BUFFER_M,
            plot_bounds[3] + _PLOT_BOUNDS_BUFFER_M,
        )
        plt.gca().set_aspect("equal", adjustable="box")
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        # plt_bounds_xmin = plot_bounds[0] - _PLOT_BOUNDS_BUFFER_M
        # plt_bounds_xmax = plot_bounds[1] + _PLOT_BOUNDS_BUFFER_M
        # plt_bounds_ymin = plot_bounds[2] - _PLOT_BOUNDS_BUFFER_M
        # plt_bounds_ymax = plot_bounds[3] + _PLOT_BOUNDS_BUFFER_M

        # bx.set_xlim(plt_bounds_xmin, plt_bounds_xmax)
        # bx.set_xticks([])
        # bx.set_ylim(plt_bounds_ymin, plt_bounds_ymax)
        # bx.set_yticks([])

        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        buf = io.BytesIO()
        
        plt.savefig(buf, format="png",dpi=500)
        plt.close()
        buf.seek(0)
        frame = img.open(buf)
        frames.append(frame)

        # Write buffered frames to MP4V-encoded video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v") #定义视频编码
    vid_path = str(save_path.parents[0] / f"{save_path.stem}.mp4")
    video = cv2.VideoWriter(vid_path, fourcc, fps=10, frameSize=frames[0].size) # 创建视频写入对象，帧率（这里是每秒10帧）和帧大小
    # 循环遍历frames中的每个图像帧，将每帧图像从RGB转换为BGR格式（因为OpenCV使用BGR），然后写入视频文件。
    for i in range(len(frames)):
        frame_temp = frames[i].copy() 
        video.write(cv2.cvtColor(np.array(frame_temp), cv2.COLOR_RGB2BGR))
    video.release()

def _plot_static_map_elements(
    ax,
    static_map: ArgoverseStaticMap,
    ##: Dict[int, Dict[str, float]],
    show_ped_xings: bool = False,
) -> None:
    """Plot all static map elements associated with an Argoverse scenario.
    Args:
        static_map: Static map containing elements to be plotted.
        reachable_lanes: Reachable lane segments that are within a threshold
            distance along the ground-truth lane assignment.
        show_ped_xings: Configures whether pedestrian crossings should be plotted.
    """
    # Plot drivable areas
    for drivable_area in static_map.vector_drivable_areas.values():
        _plot_polygons(ax,[drivable_area.xyz], alpha=0.5, color=_DRIVABLE_AREA_COLOR)

    # Plot lane segments with centerlines
    for lane_segment in static_map.vector_lane_segments.values():
        _plot_polylines(
            ax,
            [
                lane_segment.left_lane_boundary.xyz,
                lane_segment.right_lane_boundary.xyz,
            ],
            line_width=0.25,
            color=_LANE_SEGMENT_COLOR,
        )
        # scenario_lane_centerline, _ = interp_utils.compute_midpoint_line(
        #     left_ln_boundary=lane_segment.left_lane_boundary.xyz,
        #     right_ln_boundary=lane_segment.right_lane_boundary.xyz,
        #     num_interp_pts=interp_utils.NUM_CENTERLINE_INTERP_PTS,
        # )
        # _plot_polylines(
        #     ax,
        #     [
        #         scenario_lane_centerline,
        #     ],
        #     line_width=0.5,
        #     style="--",
        #     color=_LANE_SEGMENT_COLOR,
        # )

    # Plot pedestrian crossings
    if show_ped_xings:
        for ped_xing in static_map.vector_pedestrian_crossings.values():
            _plot_polylines(
                ax,
                [ped_xing.edge1.xyz, ped_xing.edge2.xyz],
                alpha=1.0,
                color=_LANE_SEGMENT_COLOR,
            )

def _plot_actor_tracks(
    ax: plt.Axes, scenario: ArgoverseScenario, timestep: int
) -> Optional[_PlotBounds]:
    """Plot all actor tracks (up to a particular time step) associated with an Argoverse scenario.
    Args:
        ax: Axes on which actor tracks should be plotted.
        scenario: Argoverse scenario for which to plot actor tracks.
        timestep: Tracks are plotted for all actor data up to the specified time step.
    Returns:
        track_bounds: (x_min, x_max, y_min, y_max) bounds for the extent of actor tracks.
    """
    track_bounds = None
    for track in scenario.tracks:
        # Get timesteps for which actor data is valid
        actor_timesteps: NDArrayInt = np.array(
            [
                object_state.timestep
                for object_state in track.object_states
                if object_state.timestep <= timestep
            ]
        )
        if actor_timesteps.shape[0] < 1 or actor_timesteps[-1] != timestep:
            continue

        # Get actor trajectory and heading history
        actor_trajectory: NDArrayFloat = np.array(
            [
                list(object_state.position)
                for object_state in track.object_states
                if object_state.timestep <= timestep
            ]
        )
        actor_headings: NDArrayFloat = np.array(
            [
                object_state.heading
                for object_state in track.object_states
                if object_state.timestep <= timestep
            ]
        )

        # Plot polyline for focal agent location history
        track_color = _DEFAULT_ACTOR_COLOR
        if track.track_id == 'AV':# track.category == TrackCategory.FOCAL_TRACK:
            x_min, x_max = actor_trajectory[:, 0].min(), actor_trajectory[:, 0].max()
            y_min, y_max = actor_trajectory[:, 1].min(), actor_trajectory[:, 1].max()
            track_bounds = (x_min, x_max, y_min, y_max)
            #track_color = _FOCAL_AGENT_COLOR #表示蓝色
            #_plot_polylines([actor_trajectory], color=track_color, line_width=1)
        else:
            continue

    return track_bounds

def _plot_polylines(  # original
    ax,
    polylines: Sequence[NDArrayFloat],
    *,
    style: str = "-",
    line_width: float = 1.0,
    alpha: float = 1.0,
    color: str = "r",
) -> None:
    """Plot a group of polylines with the specified config.
    Args:
        polylines: Collection of (N, 2) polylines to plot.
        style: Style of the line to plot (e.g. `-` for solid, `--` for dashed)
        line_width: Desired width for the plotted lines.
        alpha: Desired alpha for the plotted lines.
        color: Desired color for the plotted lines.
    """
    for polyline in polylines: # polylines的维度已经是[50,2]
        ax.plot(
            polyline[:, 0],
            polyline[:, 1],
            style,
            linewidth=line_width,
            color=color,
            alpha=alpha,
        )
   
def _plot_polylines_arrow(
    polylines: Sequence[NDArrayFloat],
    ax,
    color,
) -> None:
    """Plot a group of polylines with the specified config.
    Args:
        polylines: Collection of (N, 2) polylines to plot.
        style: Style of the line to plot (e.g. `-` for solid, `--` for dashed)
        line_width: Desired width for the plotted lines.
        alpha: Desired alpha for the plotted lines.
        color: Desired color for the plotted lines.
    """
   
    # fig, ax = plt.subplots()
    x = np.empty(50)
    y = np.empty(50)
    #print(polylines.shape)
    for i in range(50):
        x[i]=polylines[i,0]
        y[i]=polylines[i,1]
    # x = [1, 2, 3, 4, 5]
    # y = [1, 4, 9, 16, 25]
    ax.arrow(x[-2], y[-2],x[-1]-x[-2], y[-1]-y[-2],width=0.3,head_width=0.8,head_length=0.8,color=color)
    #ax.plot(x, y)
    #ax.annotate(" ",xy=(x[-1], y[-1]), arrowprops=dict(facecolor=color, arrowstyle='-|>', linewidth=0.025)) #shrink=0.05,

def _plot_polylines_arrow_future(
    polylines: Sequence[NDArrayFloat],
    ax,
    color,
) -> None:
    """Plot a group of polylines with the specified config.
    Args:
        polylines: Collection of (N, 2) polylines to plot.
        style: Style of the line to plot (e.g. `-` for solid, `--` for dashed)
        line_width: Desired width for the plotted lines.
        alpha: Desired alpha for the plotted lines.
        color: Desired color for the plotted lines.
    """
   
    # fig, ax = plt.subplots()
    x = np.empty(60)
    y = np.empty(60)
    #print(polylines.shape)
    for i in range(60):
        x[i]=polylines[i,0]
        y[i]=polylines[i,1]
    # x = [1, 2, 3, 4, 5]
    # y = [1, 4, 9, 16, 25]
    ax.arrow(x[-2], y[-2],x[-1]-x[-2], y[-1]-y[-2],width=0.3,head_width=0.8,head_length=0.8,color=color)
    #ax.plot(x, y)
    #ax.annotate(" ",xy=(x[-1], y[-1]), arrowprops=dict(facecolor=color, arrowstyle='-|>', linewidth=0.025)) #shrink=0.05,

def _plot_polylines_arrow_ego(
    polylines: Sequence[NDArrayFloat],
    ax,
    color,
) -> None:
    """Plot a group of polylines with the specified config.
    Args:
        polylines: Collection of (N, 2) polylines to plot.
        style: Style of the line to plot (e.g. `-` for solid, `--` for dashed)
        line_width: Desired width for the plotted lines.
        alpha: Desired alpha for the plotted lines.
        color: Desired color for the plotted lines.
    """
   
    # fig, ax = plt.subplots()
    x = np.empty(2)
    y = np.empty(2)
    #print(polylines.shape)
    for i in range(2):
        x[i]=polylines[i,0]
        y[i]=polylines[i,1]
    # x = [1, 2, 3, 4, 5]
    # y = [1, 4, 9, 16, 25]

    #ax.plot(x, y)
    #print(x[-2], y[-2],x[-1], y[-1])
    ax.arrow(x[-2], y[-2],x[-1]-x[-2], y[-1]-y[-2],width=0.1,head_width=0.5,head_length=0.5,color=color,zorder=100)
    #ax.annotate(" ", xy=(x[-1], y[-1]), arrowprops=dict(facecolor=color, arrowstyle='-|>', linewidth=0.025)) # shrink=0.05,

def _plot_polygons(
    ax, polygons: Sequence[NDArrayFloat], *, alpha: float = 1.0, color: str = "r"
) -> None:
    """Plot a group of filled polygons with the specified config.
    Args:
        polygons: Collection of polygons specified by (N,2) arrays of vertices.
        alpha: Desired alpha for the polygon fill.
        color: Desired color for the polygon.
    """
    for polygon in polygons:
        ax.fill(polygon[:, 0], polygon[:, 1], color=color, alpha=alpha)

def _plot_actor_bounding_box(
    ax: plt.Axes,
    cur_location: NDArrayFloat,
    heading: float,
    color: str,
    bbox_size: Tuple[float, float],
) -> None:
    """Plot an actor bounding box centered on the actor's current location.
    Args:
        ax: Axes on which actor bounding box should be plotted.
        cur_location: Current location of the actor (2,).
        heading: Current heading of the actor (in radians).
        color: Desired color for the bounding box.
        bbox_size: Desired size for the bounding box (length, width).
    """
    (bbox_length, bbox_width) = bbox_size

    # Compute coordinate for pivot point of bounding box
    d = np.hypot(bbox_length, bbox_width)
    theta_2 = math.atan2(bbox_width, bbox_length)
    pivot_x = cur_location[0] - (d / 2) * math.cos(heading + theta_2)
    pivot_y = cur_location[1] - (d / 2) * math.sin(heading + theta_2)

    vehicle_bounding_box = Rectangle(
        (pivot_x, pivot_y),
        bbox_length,
        bbox_width,
        np.degrees(heading),
        edgecolor="#FFFFFF",
        facecolor=color,
        zorder=_BOUNDING_BOX_ZORDER,
    )
    ax.add_patch(vehicle_bounding_box)

if __name__ == '__main__':

    forecasted_all = get_forecasted_traj()

    visualization(
        forecasted_all,
        nobjs,)
    

