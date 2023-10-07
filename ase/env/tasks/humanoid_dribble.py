# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch

import env.tasks.humanoid as humanoid
import env.tasks.humanoid_amp as humanoid_amp
import env.tasks.humanoid_amp_task as humanoid_amp_task
from utils import torch_utils

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *

class HumanoidDribble(humanoid_amp_task.HumanoidAMPTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self._tar_speed = cfg["env"]["tarSpeed"]
        self.postar = cfg["env"]["tarPos"]
        self._tar_change_steps_min = cfg["env"]["tarChangeStepsMin"]
        self._tar_change_steps_max = cfg["env"]["tarChangeStepsMax"]
        self._tar_dist_max = cfg["env"]["tarDistMax"]
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        self._tar_change_steps = torch.zeros([self.num_envs], device=self.device, dtype=torch.int64)
        self._prev_root_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self._prev_ball_pos = torch.zeros([self.num_envs, 2], device=self.device, dtype=torch.float)
        self._ball_pos = torch.zeros([self.num_envs, 2], device=self.device, dtype=torch.float)
        self._ball_vel = torch.zeros([self.num_envs, 2], device=self.device, dtype=torch.float)
        self._ball_ori = torch.zeros([self.num_envs, 4], device=self.device, dtype=torch.float)
        self._ball_ang_vel = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self._tar_pos = torch.zeros([self.num_envs, 2], device=self.device, dtype=torch.float)
        self._tar_pos[..., 0] = self.postar[0]
        self._tar_pos[..., 1] = self.postar[1]
        # if (not self.headless):
        #     self._build_ball_state_tensors()
        self._build_ball_state_tensors()
        self._build_marker_state_tensors()

        return

    def _build_ball_state_tensors(self):
        num_actors = self._root_states.shape[0] // self.num_envs
        self._ball_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., 1, :]
        self._ball_pos = self._ball_states[..., :2].clone()
        self._ball_vel = self._ball_states[..., 7:9].clone()
        self._ball_ori = self._ball_states[..., 3:7].clone()
        self._ball_ang_vel = self._ball_states[..., 10:13].clone()
        # print("-------------------------------------")
        # print(self._ball_pos)
        # print("----------------------------------------")
        # print(self.base_init_state)
        # print("-------------------------------------------")
        # print(self._humanoid_root_states[...,:3])
        # print("----------------------------------------")
        self._ball_actor_ids = self._humanoid_actor_ids + 1
        return

    def _build_marker_state_tensors(self):
        num_actors = self._root_states.shape[0] // self.num_envs
        self._marker_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., 2, :]
        self._marker_pos = self._marker_states[..., :2]
        # print("-------------------------------------")
        # print(self._ball_pos)
        # print("----------------------------------------")
        # print(self.base_init_state)
        # print("-------------------------------------------")
        # print(self._humanoid_root_states[...,:3])
        # print("----------------------------------------")
        self._marker_actor_ids = self._ball_actor_ids + 1
        return

    def _create_envs(self, num_envs, spacing, num_per_row):
        if (not self.headless):
            self._ball_handles = []
            self._marker_handles = []
            self._load_ball_asset()
            self._load_marker_asset()

        super()._create_envs(num_envs, spacing, num_per_row)
        return
    
    def _load_ball_asset(self):
        asset_root = "ase/data/assets/mjcf/"
        asset_file = "ball.urdf"
        asset_options = gymapi.AssetOptions()
        self._ball_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        return

    def _load_marker_asset(self):
        asset_root = "ase/data/assets/mjcf/"
        asset_file = "location_marker.urdf" 

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.density = 1.0
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._marker_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        return   

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        
        if (not self.headless):
            self._build_ball(env_id, env_ptr)
            self._build_marker(env_id, env_ptr)

        return

    def _build_ball(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 2
        segmentation_id = 0

        pos_ball = self.base_init_state.clone()
        pos_ball[0] += 1.0
        pos_ball[2] = 0.101
        default_pose = gymapi.Transform()
        default_pose.p = gymapi.Vec3(*pos_ball)

        ball_handle = self.gym.create_actor(env_ptr, self._ball_asset, default_pose, "ball", col_group, col_filter, segmentation_id)
        self.gym.set_rigid_body_color(env_ptr, ball_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.56,0.93,0.56))
        self._ball_handles.append(ball_handle)

        return

    def _build_marker(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 2
        segmentation_id = 0

        pos_marker = self.postar.copy()
        pos_marker.append(0.9)
        pos_marker = torch.tensor(pos_marker,dtype=torch.float,device=self.device,requires_grad=False)
        default_pose = gymapi.Transform()
        default_pose.p = gymapi.Vec3(*pos_marker)

        marker_handle = self.gym.create_actor(env_ptr, self._marker_asset, default_pose, "marker", col_group, col_filter, segmentation_id)
        self.gym.set_rigid_body_color(env_ptr, marker_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.80,0.36,0.36))
        self._marker_handles.append(marker_handle)




        return


    def _draw_task(self):
        self._update_ball()
        self._update_marker()

        cols = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)

        self.gym.clear_lines(self.viewer)

        starts = self._humanoid_root_states[..., 0:3]
        ends = self._ball_pos

        verts = torch.cat([starts, ends], dim=-1).cpu().numpy()

        for i, env_ptr in enumerate(self.envs):
            curr_verts = verts[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr, curr_verts.shape[0], curr_verts, cols)

        return
  
    def _update_ball(self):
        self._ball_pos[..., 0:2] = self._ball_pos
        self._ball_pos[..., 2] = 0.101
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(self._ball_actor_ids), len(self._ball_actor_ids))
        return

    def _update_marker(self):
        self._marker_pos[..., 0:2] = self._tar_pos
        self._marker_pos[..., 2] = 0.0

        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(self._marker_actor_ids), len(self._marker_actor_ids))
        return
   
    def _update_task(self):
        reset_task_mask = self.progress_buf >= self._tar_change_steps
        rest_env_ids = reset_task_mask.nonzero(as_tuple=False).flatten()
        if len(rest_env_ids) > 0:
            self._reset_task(rest_env_ids)
        return

    def _update_task_no_reset(self):
        reset_task_mask = self.progress_buf >= self._tar_change_steps
        rest_env_ids = reset_task_mask.nonzero(as_tuple=False).flatten()
        if len(rest_env_ids) > 0:
            self._reset_task_no_reset(rest_env_ids)
        return      

    def _reset_task_no_reset(self, env_ids):
        n = len(env_ids)

        # char_root_pos = self._humanoid_root_states[env_ids, 0:2]
        # rand_pos = self._tar_dist_max * (2.0 * torch.rand([n, 2], device=self.device) - 1.0)

        change_steps = torch.randint(low=self._tar_change_steps_min, high=self._tar_change_steps_max,
                                     size=(n,), device=self.device, dtype=torch.int64)

        self._ball_pos[env_ids] = self._humanoid_root_states[env_ids,0:2].clone()
        self._ball_pos[env_ids,0] += torch_rand_float(5,5,(len(env_ids),1),device=self.device).squeeze()
        self._ball_pos[env_ids,1] += torch_rand_float(1.5,1.5,(len(env_ids),1),device=self.device).squeeze()
        self._ball_states[env_ids,0:2] = self._ball_pos[env_ids].clone()
        self._ball_vel[env_ids] = torch.zeros_like(self._humanoid_root_states[env_ids,7:9],device=self.device,dtype=torch.float)
        self._ball_ang_vel[env_ids] = torch.zeros_like(self._humanoid_root_states[env_ids,10:13],device=self.device,dtype=torch.float)
        self._ball_states[env_ids,7:9] = self._ball_vel[env_ids].clone()
        self._ball_states[env_ids,10:13] = self._ball_ang_vel[env_ids].clone()
        self._ball_ori[env_ids] = self._ball_states[env_ids,3:7].clone()
        # self._marker_pos[env_ids] = char_root_pos + rand_pos
        self._marker_pos[env_ids] = self._tar_pos[env_ids]
        # self._ball_pos[env_ids] = self._humanoid_root_states[env_ids,0:2]
        # self._ball_pos[env_ids,0] += 1
        self._tar_change_steps[env_ids] = self.progress_buf[env_ids] + change_steps
        return

    def _reset_task(self, env_ids):
        n = len(env_ids)

        # char_root_pos = self._humanoid_root_states[env_ids, 0:2]
        # rand_pos = self._tar_dist_max * (2.0 * torch.rand([n, 2], device=self.device) - 1.0)

        change_steps = torch.randint(low=self._tar_change_steps_min, high=self._tar_change_steps_max,
                                     size=(n,), device=self.device, dtype=torch.int64)

        self._ball_pos[env_ids] = self._humanoid_root_states[env_ids,0:2].clone()
        self._ball_pos[env_ids,0] += torch_rand_float(5,5,(len(env_ids),1),device=self.device).squeeze()
        self._ball_pos[env_ids,1] += torch_rand_float(1.5,1.5,(len(env_ids),1),device=self.device).squeeze()
        self._ball_states[env_ids,0:2] = self._ball_pos[env_ids].clone()
        self._ball_vel[env_ids] = torch.zeros_like(self._humanoid_root_states[env_ids,7:9],device=self.device,dtype=torch.float)
        self._ball_ang_vel[env_ids] = torch.zeros_like(self._humanoid_root_states[env_ids,10:13],device=self.device,dtype=torch.float)
        self._ball_states[env_ids,7:9] = self._ball_vel[env_ids].clone()
        self._ball_states[env_ids,10:13] = self._ball_ang_vel[env_ids].clone()
        self._ball_ori[env_ids] = self._ball_states[env_ids,3:7].clone()
        # self._marker_pos[env_ids] = char_root_pos + rand_pos
        self._marker_pos[env_ids] = self._humanoid_root_states[env_ids,0:2] + self._tar_pos[env_ids]
        self._reset_ball_tensors(env_ids)
        self._refresh_sim_tensors()
        # self._ball_pos[env_ids] = self._humanoid_root_states[env_ids,0:2]
        # self._ball_pos[env_ids,0] += 1
        self._tar_change_steps[env_ids] = self.progress_buf[env_ids] + change_steps
        return

    def _reset_ball_tensors(self, env_ids):
        env_ids_int32 = self._ball_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))


    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._prev_root_pos[:] = self._humanoid_root_states[..., 0:3]
        self._prev_ball_pos[:] = self._ball_states[..., 0:2].clone()
        return
  
    def get_task_obs_size(self):
        obs_size = 0
        if (self._enable_task_obs):
            # obs_size = 2 + 2 + 4 + 3 #ball_pos + ball_lin_vel + ball_orientation + ball_ang_vel
            obs_size = 2 + 2 + 3
        return obs_size

    def _compute_task_obs(self, env_ids=None):
        if (env_ids is None):
            root_states = self._humanoid_root_states
            ball_pos = self._ball_pos
            ball_vel = self._ball_vel
            ball_ori = self._ball_ori
            ball_ang_vel = self._ball_ang_vel
            # tar_pos = self._marker_pos
        else:
            root_states = self._humanoid_root_states[env_ids]
            ball_pos = self._ball_pos[env_ids]
            ball_vel = self._ball_vel[env_ids]
            ball_ori = self._ball_ori[env_ids]
            ball_ang_vel = self._ball_ang_vel[env_ids]
        
        obs = compute_location_observations(root_states, ball_pos, ball_vel, ball_ori, ball_ang_vel)
        return obs

    def _compute_reward(self, actions):
        root_pos = self._humanoid_root_states[..., 0:3]
        # root_rot = self._humanoid_root_states[..., 3:7]
        self.rew_buf[:] = compute_location_reward(root_pos, self._prev_root_pos, self._tar_pos,
                                                 self._ball_pos, self._tar_speed,
                                                 self.dt, self._prev_ball_pos)
        return


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_location_observations(root_states, ball_pos, ball_vel, ball_ori, ball_ang_vel):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]

    ball_pos3d = torch.cat([ball_pos, torch.zeros_like(ball_pos[..., 0:1])], dim=-1)
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot) 
    local_ball_pos = quat_rotate(heading_rot, ball_pos3d - root_pos)
    local_ball_pos = local_ball_pos[..., 0:2]

    ball_vel3d = torch.cat([ball_vel, torch.zeros_like(ball_vel[..., 0:1])], dim=-1)
    ball_lin_vel = quat_rotate_inverse(root_rot, ball_vel3d)
    ball_lin_vel = ball_lin_vel[..., 0:2]

    _ball_ang_vel = quat_rotate_inverse(root_rot, ball_ang_vel)

    

    obs = torch.cat([local_ball_pos, ball_lin_vel, _ball_ang_vel], dim=-1)
    return obs

@torch.jit.script
def compute_location_reward(root_pos, prev_root_pos, tar_pos, ball_pos, tar_speed, dt, prev_ball_pos):
    # type: (Tensor, Tensor, Tensor, Tensor, float, float, Tensor) -> Tensor
    dist_threshold = 0.1
    tar_dis_threshold = 0.75

    root_vel_err_scale = 2.5
    root_pos_err_scale = 0.5
    ball_vel_err_scale = 1
    tar_pos_err_scale = 0.5
    cv_w = 0.1
    cp_w = 0.1
    bv_w = 0.3
    bp_w = 0.5
    # cv_w = 0.3
    # cp_w = 0.3
    # bv_w = 0.2
    # bp_w = 0.2
    # cv_w = 0.2
    # cp_w = 0.2
    # bv_w = 0.2
    # bp_w = 0.4

    #----------cv reward--------------
    ball_dir = ball_pos - root_pos[..., 0:2]  
    ball_dir = torch.nn.functional.normalize(ball_dir, dim=-1)  #d_t^{ball}
    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt   #v_t^{com}
    ball_dir_speed = torch.sum(ball_dir * root_vel[..., :2], dim=-1)
    # v_target = torch.ones(tar_dir_speed.shape, device=self.device, dtype=torch.float) #v^*
    v_target = tar_speed
    cv_err = v_target - ball_dir_speed
    cv_err = torch.clamp_min(cv_err,0.0)
    cv_reward = torch.exp(-root_vel_err_scale * (cv_err * cv_err))
    speed_mask = ball_dir_speed <= 0
    cv_reward[speed_mask] = 0

    #----------cp reward--------------
    pos_diff = ball_pos - root_pos[..., 0:2]
    pos_err = torch.sum(pos_diff * pos_diff, dim=-1)
    cp_reward = torch.exp(-root_pos_err_scale * pos_err)

    #----------bv reward--------------
    pos_target = tar_pos
    ball_dir = pos_target - ball_pos
    ball_dir = torch.nn.functional.normalize(ball_dir, dim=-1)
    delta_ball_pos = ball_pos - prev_ball_pos
    ball_vel = delta_ball_pos / dt
    ball_dir_speed = torch.sum(ball_dir * ball_vel, dim=-1)
    # bv_err = v_target - ball_dir_speed
    bv_err = v_target - ball_dir_speed
    bv_err = torch.clamp_min(bv_err,0.0)
    bv_reward = torch.exp(-ball_vel_err_scale * (bv_err * bv_err))
    speed_mask = ball_dir_speed <= 0
    bv_reward[speed_mask] = 0
    speed_mask = ball_dir_speed >= 2
    bv_reward[speed_mask] = 0

    #----------bp reward--------------
    tar_pos_diff = pos_target - root_pos[..., 0:2]
    tar_pos_err = torch.sum(tar_pos_diff * tar_pos_diff, dim=-1)
    bp_reward = torch.exp(-tar_pos_err_scale * tar_pos_err)

    # heading_rot = torch_utils.calc_heading_quat(root_rot)
    # facing_dir = torch.zeros_like(root_pos)
    # facing_dir[..., 0] = 1.0
    # facing_dir = quat_rotate(heading_rot, facing_dir)
    # facing_err = torch.sum(tar_dir * facing_dir[..., 0:2], dim=-1)
    # facing_reward = torch.clamp_min(facing_err, 0.0)


    # dist_mask = pos_err < dist_threshold
    # # facing_reward[dist_mask] = 1.0
    # cv_reward[dist_mask] = 1.0

    tar_mask = tar_pos_err < tar_dis_threshold
    bv_reward[tar_mask] = 1.0

    #----------penalize -------------- 

    reward = cv_w * cv_reward + cp_w * cp_reward + bv_w * bv_reward + bp_w * bp_reward

    return reward