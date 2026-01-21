import os
import numpy as np
from omegaconf import OmegaConf
import torch
import hydra
import sys
import gym
import gymnasium
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnvWrapper
import json

from dppo.env.gym_utils.wrapper import wrapper_dict
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils


def make_robomimic_env(
	render=False, 
	env='square', 
	normalization_path=None, 
	low_dim_keys=None, 
	dppo_path=None,
	impedance_mode='fixed',
	control_obs=False,
):
	wrappers = OmegaConf.create({
		'robomimic_lowdim': {
			'normalization_path': normalization_path,
			'low_dim_keys': low_dim_keys,
			'impedance_mode': impedance_mode,
			'control_obs': control_obs,
		},
	})
	obs_modality_dict = {
		"low_dim": (
			wrappers.robomimic_image.low_dim_keys
			if "robomimic_image" in wrappers
			else wrappers.robomimic_lowdim.low_dim_keys
		),
		"rgb": (
			wrappers.robomimic_image.image_keys
			if "robomimic_image" in wrappers
			else None
		),
	}
	if obs_modality_dict["rgb"] is None:
		obs_modality_dict.pop("rgb")
	ObsUtils.initialize_obs_modality_mapping_from_dict(obs_modality_dict)
	robomimic_env_cfg_path = f'{dppo_path}/cfg/robomimic/env_meta/{env}.json'
	with open(robomimic_env_cfg_path, "r") as f:
		env_meta = json.load(f)
	env_meta["reward_shaping"] = False

	# TODO: manually setting controller impedance mode for now
	env_meta["env_kwargs"]["controller_configs"]["impedance_mode"] = impedance_mode
	env_meta["env_kwargs"]["controller_configs"]["kp_limits"] = [0, 1500]
	# TODO: Exposing controller config parameters to robomimic env for normalization purposes.
	wrappers.robomimic_lowdim.controller_configs = env_meta["env_kwargs"]["controller_configs"]

	env = EnvUtils.create_env_from_metadata(
		env_meta=env_meta,
		render=False,
		render_offscreen=render,
		use_image_obs=False,
	)
	env.env.hard_reset = False
	for wrapper, args in wrappers.items():
		env = wrapper_dict[wrapper](env, **args)
	return env

class ResidualPolicyWrapper(gym.Env):
	def __init__(
		self,
		env,
		cfg,
		full_render=False,
	):
		self.env = env
		self.cfg = cfg
		self.full_render = full_render
		self.action_dim = cfg.action_dim
		self.policy_type = cfg.policy.type
		self.impedance_mode = cfg.policy.impedance_mode

		# Updating action space based on impedance mode.
		action_space_low = np.array([-1.0] * self.action_dim)
		action_space_high = np.array([1.0] * self.action_dim)

		# Add action dimension for scale factor, if necessary.
		if self.impedance_mode == "variable":
			action_space_low = np.concatenate(
				[action_space_low, np.array([-1.0] * 2)]
			)
			action_space_high = np.concatenate(
				[action_space_high, np.array([1.0] * 2)]
			)
		elif self.impedance_mode == "variable_kp":
			action_space_low = np.concatenate(
				[action_space_low, np.array([-1.0] * 1)]
			)
			action_space_high = np.concatenate(
				[action_space_high, np.array([1.0] * 1)]
			)

		# Creating action and observation spaces
		self.action_space = spaces.Box(
			low=action_space_low,
			high=action_space_high,
			dtype=np.float32
		 )
		self.observation_space = env.observation_space
	
	def seed(self, seed=None):
		if seed is not None:
			np.random.seed(seed=seed)
		else:
			np.random.seed()
	
	def reset(self, **kwargs):
		options = kwargs.get("options", {})
		new_seed = options.get("seed", None)
		if new_seed is not None:
			self.seed(seed=new_seed)
		return self.env.reset()

	def step(self, action):
		# Expand control parameters to robot joint dimensions, if necessary.
		# TODO: eventually, this may need to read from a config rather than be hardcoded.
		if self.impedance_mode == "variable":
			damping, stiffness, delta = action[0], action[1], action[2:]
			action = np.concatenate([
				np.repeat(damping, 6),
				np.repeat(stiffness, 6),
				delta
			], axis=0)
		elif self.impedance_mode == "variable_kp":
			stiffness, delta = action[0], action[1:]
			action = np.concatenate([
				np.repeat(stiffness, 6),
				delta
			], axis=0)
			
		obs, reward, done, info = self.env.step(action)
		if self.full_render:
			info['render'] = self.env.render()
		return obs, reward, done, info
	
	def render(self, **kwargs):
		return self.env.render()

class ObservationWrapperRobomimic(gym.Env):
	def __init__(
		self,
		env,
		reward_offset=1,
	):
		self.env = env
		self.action_space = env.action_space
		self.observation_space = env.observation_space
		self.reward_offset = reward_offset

	def seed(self, seed=None):
		if seed is not None:
			np.random.seed(seed=seed)
		else:
			np.random.seed()

	def reset(self, **kwargs):
		options = kwargs.get("options", {})
		new_seed = options.get("seed", None)
		if new_seed is not None:
			self.seed(seed=new_seed)
		raw_obs = self.env.reset()
		obs = raw_obs['state'].flatten()
		return obs

	def step(self, action):
		raw_obs, reward, done, info = self.env.step(action)
		reward = (reward - self.reward_offset)
		obs = raw_obs['state'].flatten()
		return obs, reward, done, info

	def render(self, **kwargs):
		return self.env.render()
	

class ObservationWrapperGym(gym.Env):
	def __init__(
		self,
		env,
		normalization_path,
	):
		self.env = env
		self.action_space = env.action_space
		self.observation_space = env.observation_space
		normalization = np.load(normalization_path)
		self.obs_min = normalization["obs_min"]
		self.obs_max = normalization["obs_max"]
		self.action_min = normalization["action_min"]
		self.action_max = normalization["action_max"]

	def seed(self, seed=None):
		if seed is not None:
			np.random.seed(seed=seed)
		else:
			np.random.seed()

	def reset(self, **kwargs):
		options = kwargs.get("options", {})
		new_seed = options.get("seed", None)
		if new_seed is not None:
			self.seed(seed=new_seed)
		raw_obs = self.env.reset()
		obs = self.normalize_obs(raw_obs)
		return obs

	def step(self, action):
		raw_action = self.unnormalize_action(action)
		raw_obs, reward, done, info = self.env.step(raw_action)
		obs = self.normalize_obs(raw_obs)
		return obs, reward, done, info

	def render(self, **kwargs):
		return self.env.render()
	
	def normalize_obs(self, obs):
		return 2 * ((obs - self.obs_min) / (self.obs_max - self.obs_min + 1e-6) - 0.5)

	def unnormalize_action(self, action):
		action = (action + 1) / 2
		return action * (self.action_max - self.action_min) + self.action_min
	

class ActionChunkWrapper(gymnasium.Env):
	def __init__(self, env, cfg, max_episode_steps=300):
		self.max_episode_steps = max_episode_steps
		self.env = env
		self.act_steps = cfg.act_steps
		self.action_space = spaces.Box(
			low=np.tile(env.action_space.low, cfg.act_steps),
			high=np.tile(env.action_space.high, cfg.act_steps),
			dtype=np.float32
		)
		# self.observation_space = spaces.Box(
		# 	low=-np.ones(cfg.obs_dim),
		# 	high=np.ones(cfg.obs_dim),
		# 	dtype=np.float32
		# )
		# TODO: need to manually make this, becuase sb3 buffer requries np.float32
		# self.observation_space = self.env.observation_space["state"]
		self.observation_space = spaces.Box(
			low=self.env.observation_space["state"].low,
			high=self.env.observation_space["state"].high,
			dtype=np.float32
		)
		self.count = 0

	def reset(self, seed=None):
		obs = self.env.reset(seed=seed)
		self.count = 0
		return obs, {}
	
	def step(self, action):
		if len(action.shape) == 1:
			action = action.reshape(self.act_steps, -1)
		obs_ = []
		reward_ = []
		done_ = []
		info_ = []
		done_i = False
		for i in range(action.shape[0]):
			self.count += 1
			obs_i, reward_i, done_i, info_i = self.env.step(action[i])
			obs_.append(obs_i)
			reward_.append(reward_i)
			done_.append(done_i)
			info_.append(info_i)
		obs = obs_[-1]
		reward = sum(reward_)
		done = np.max(done_)
		info = info_[-1].copy()
		# Also adding entire chunk info history
		info["chunk_info"] = info_.copy()
		if self.count >= self.max_episode_steps:
			done = True
		if done:
			info['terminal_observation'] = obs
		return obs, reward, done, False, info

	def render(self):
		return self.env.render()
	
	def close(self):
		return
	

class DiffusionPolicyEnvWrapper(VecEnvWrapper):
	def __init__(self, env, cfg, base_policy):
		super().__init__(env)
		self.action_horizon = cfg.act_steps
		self.action_dim = cfg.action_dim
		self.action_space = spaces.Box(
			low=-cfg.train.action_magnitude*np.ones(self.action_dim*self.action_horizon),
			high=cfg.train.action_magnitude*np.ones(self.action_dim*self.action_horizon),
			dtype=np.float32
		)
		self.obs_dim = cfg.obs_dim
		self.observation_space = spaces.Box(
			low=-np.ones(self.obs_dim),
			high=np.ones(self.obs_dim),
			dtype=np.float32
		)
		self.env = env
		self.device = cfg.model.device
		self.base_policy = base_policy
		self.obs = None

	def step_async(self, actions):
		actions = torch.tensor(actions, device=self.device, dtype=torch.float32)
		actions = actions.view(-1, self.action_horizon, self.action_dim)
		diffused_actions = self.base_policy(self.obs, actions)
		self.venv.step_async(diffused_actions)

	def step_wait(self):
		obs, rewards, dones, infos = self.venv.step_wait()
		self.obs = torch.tensor(obs, device=self.device, dtype=torch.float32)
		obs_out = self.obs
		return obs_out.detach().cpu().numpy(), rewards, dones, infos

	def reset(self):
		obs = self.venv.reset()
		self.obs = torch.tensor(obs, device=self.device, dtype=torch.float32)
		obs_out = self.obs
		return obs_out.detach().cpu().numpy()

class LiftEvalWrapper(ObservationWrapperRobomimic):
	def __init__(self, env, reward_offset=1):
		super().__init__(env, reward_offset=reward_offset)
		self.subgoals = ['reach', 'grasp', 'success']
    
    # only overriding step to extract subgoal information
	def step(self, action):
		raw_obs, reward, done, info = self.env.step(action)
		reward = (reward - self.reward_offset)
		obs = raw_obs['state'].flatten()

		lift_env = self.env.env.env
		# Check reach
		cube_pos = lift_env.sim.data.body_xpos[lift_env.cube_body_id]
		gripper_site_pos = lift_env.sim.data.site_xpos[lift_env.robots[0].eef_site_id]
		dist = np.linalg.norm(cube_pos - gripper_site_pos)
		reach = dist < 0.05
		# Check grasp
		grasp = lift_env._check_grasp(
			gripper=lift_env.robots[0].gripper, object_geoms=lift_env.cube
		)
		# Check success
		success = lift_env._check_success()

		info['reach'] = reach
		info['grasp'] = grasp
		info['success'] = success
			
		return obs, reward, done, info

class CanEvalWrapper(ObservationWrapperRobomimic):
	def __init__(self, env, reward_offset=1):
		super().__init__(env, reward_offset=reward_offset)
		self.subgoals = ['reach', 'grasp', 'hover', 'success']

	# only overriding step to extract subgoal information
	def step(self, action):
		raw_obs, reward, done, info = self.env.step(action)
		reward = (reward - self.reward_offset)
		obs = raw_obs['state'].flatten()

		can_env = self.env.env.env
		can_obj = can_env.objects[can_env.object_id]
		can_obj_id = can_env.obj_body_id["Can"]

		# Check reach
		can_pos = can_env.sim.data.body_xpos[can_obj_id]
		gripper_site_pos = can_env.sim.data.site_xpos[can_env.robots[0].eef_site_id]
		dist = np.linalg.norm(can_pos - gripper_site_pos)
		reach = dist < 0.05

		# Check grasp
		grasp = can_env._check_grasp(
			gripper=can_env.robots[0].gripper, object_geoms=can_obj
		)

		# Check hover
		can_target_bin = can_env.target_bin_placements[can_env.object_to_id["can"]]
		y_check = np.abs(can_pos[1] - can_target_bin[1]) < can_env.bin_size[1] / 4.0
		x_check = np.abs(can_pos[0] - can_target_bin[0]) < can_env.bin_size[0] / 4.0
		hover = y_check and x_check

		# Check success
		success = can_env._check_success()

		info['reach'] = reach
		info['grasp'] = grasp
		info['hover'] = hover
		info['success'] = success
			
		return obs, reward, done, info


eval_wrapper_dict = {
	'lift': LiftEvalWrapper,
	'can': CanEvalWrapper,
}

subgoal_list_dict = {
	'lift': ['reach', 'grasp', 'success'],
	'can': ['reach', 'grasp', 'hover', 'success'],
}