import torch
import wandb
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import hydra
from tqdm import tqdm

import matplotlib
# Use 'Agg' backend for headless environments
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from PIL import Image
import os
import io
import warnings

# TODO: clean up.

class DPPOBasePolicyWrapper:
	def __init__(self, base_policy):
		self.base_policy = base_policy
		
	def __call__(self, obs, initial_noise, return_numpy=True):
		cond = {
			"state": obs,
			"noise_action": initial_noise,
		}
		with torch.no_grad():
			samples = self.base_policy(cond=cond, deterministic=True)
		diffused_actions = (samples.trajectories.detach())
		if return_numpy:
			diffused_actions = diffused_actions.cpu().numpy()
		return diffused_actions	


def load_base_policy(cfg):
	base_policy = hydra.utils.instantiate(cfg.model)
	base_policy = base_policy.eval()
	return DPPOBasePolicyWrapper(base_policy)


class LoggingCallback(BaseCallback):
	def __init__(self,
		cfg,
		action_chunk=4, 
		log_freq=1000,
		use_wandb=True, 
		eval_env=None, 
		eval_freq=70, 
		eval_episodes=2, 
		verbose=1, 
		rew_offset=0, 
		num_train_env=1,
		num_eval_env=1,
		algorithm='dsrl_sac',
		max_steps=-1,
		deterministic_eval=False,
		subgoal_list=["success"],
	):
		super().__init__(verbose)
		self.cfg = cfg
		self.action_chunk = action_chunk
		self.log_freq = log_freq
		self.episode_rewards = []
		self.episode_lengths = []
		self.use_wandb = use_wandb
		self.eval_env = eval_env
		self.eval_episodes = eval_episodes
		self.eval_freq = eval_freq
		self.log_count = 0
		self.total_reward = 0
		self.rew_offset = rew_offset
		self.total_timesteps = 0
		self.num_train_env = num_train_env
		self.num_eval_env = num_eval_env
		self.episode_success = np.zeros(self.num_train_env)
		self.episode_completed = np.zeros(self.num_train_env)
		self.algorithm = algorithm
		self.max_steps = max_steps
		self.deterministic_eval = deterministic_eval
		self.subgoal_list = subgoal_list

		assert "success" in self.subgoal_list, "Success check must be in subgoal list for logging."

	def _on_step(self):
		for info in self.locals['infos']:
			if 'episode' in info:
				self.episode_rewards.append(info['episode']['r'])
				self.episode_lengths.append(info['episode']['l'])
		rew = self.locals['rewards']
		self.total_reward += np.mean(rew)
		self.episode_success[rew > -self.rew_offset * self.action_chunk] = 1
		self.episode_completed[self.locals['dones']] = 1
		self.total_timesteps += self.action_chunk * self.model.n_envs
		if self.n_calls % self.log_freq == 0:
			if len(self.episode_rewards) > 0:
				if self.use_wandb:
					self.log_count += 1
					wandb.log({
						"train/ep_len_mean": np.mean(self.episode_lengths),
						"train/success_rate": np.sum(self.episode_success) / np.sum(self.episode_completed),
						"train/ep_rew_mean": np.mean(self.episode_rewards),
						"train/rew_mean": np.mean(self.total_reward),
						"train/timesteps": self.total_timesteps,
						"train/ent_coef": self.locals['self'].logger.name_to_value['train/ent_coef'],
						"train/actor_loss": self.locals['self'].logger.name_to_value['train/actor_loss'],
						"train/critic_loss": self.locals['self'].logger.name_to_value['train/critic_loss'],
						"train/ent_coef_loss": self.locals['self'].logger.name_to_value['train/ent_coef_loss'],
					}, step=self.num_timesteps)
					# Logging gain losses, if necessary.
					if self.cfg.policy.impedance_mode != "fixed" and self.cfg.policy.smooth_gain_lambda > 0:
						wandb.log({
							"train/smooth_gain_loss": self.locals['self'].logger.name_to_value['train/smooth_gain_loss'],
						})
					# Logging gradient norms for debugging, if necessary.
					if 'debug/actor_grad_norm' in self.locals['self'].logger.name_to_value:
						wandb.log({
							"train/actor_grad_norm": self.locals['self'].logger.name_to_value['debug/actor_grad_norm'],
						})
					if 'debug/smooth_gain_grad_norm' in self.locals['self'].logger.name_to_value:
						wandb.log({
							"train/smooth_gain_grad_norm": self.locals['self'].logger.name_to_value['debug/smooth_gain_grad_norm'],
						})

					if np.sum(self.episode_completed) > 0:
						wandb.log({
							"train/success_rate": np.sum(self.episode_success) / np.sum(self.episode_completed),
						}, step=self.num_timesteps)
					if self.algorithm == 'dsrl_na':
						wandb.log({
							"train/noise_critic_loss": self.locals['self'].logger.name_to_value['train/noise_critic_loss'],
						}, step=self.num_timesteps)
				self.episode_rewards = []
				self.episode_lengths = []
				self.total_reward = 0
				self.episode_success = np.zeros(self.num_train_env)
				self.episode_completed = np.zeros(self.num_train_env)

		# NOTE: this might behave weirdly with checkpoint resuming if save_freq for checkpoint...
		# ...is not divisible by eval_freq.
		if self.n_calls % self.eval_freq == 0:
			self.evaluate(self.locals['self'], deterministic=False)
			if self.deterministic_eval:
				self.evaluate(self.locals['self'], deterministic=True)
		return True
	
	def evaluate(self, agent, deterministic=False, evaluate_base=False):
		if self.eval_episodes > 0:
			env = self.eval_env

			# Rollout izualization arrays.
			rollout_vid = []
			obs_arr = []
			action_arr = []
			scale_viz_arr = []


			with torch.no_grad():
				# Initializing rollout metrics.
				rews = []
				rew_total, shaped_rew_total, total_ep = 0, 0, 0
				rew_ep = np.zeros(self.num_eval_env)
				shaped_rew_ep = np.zeros(self.num_eval_env)
				delta_action_norms = []
				subgoal_rate_arrs = {subgoal: [] for subgoal in self.subgoal_list}
				subgoal_time_arrs = {subgoal: [] for subgoal in self.subgoal_list}
				subgoal_success_time_arrs = {subgoal: [] for subgoal in self.subgoal_list}

				for i in range(self.eval_episodes):
					obs = env.reset()
					action_arr_i = [] 
					obs_arr_i = []
					r = []
					delta_action_norms_i = []

					subgoal_rate_arrs_i = {subgoal: np.zeros(obs.shape[0]) for subgoal in self.subgoal_list}
					subgoal_time_arrs_i = {subgoal: np.zeros(obs.shape[0]) + self.max_steps for subgoal in self.subgoal_list}

					for step_i in range(self.max_steps):
						# Sample action and step environment.
						action, predict_second_return = agent.predict_diffused(obs, deterministic=deterministic, sample_base=evaluate_base)
						next_obs, reward, done, info = env.step(action)

						action_arr_i.append(action)
						obs_arr_i.append(obs)
						# Logging, if necessary
						if i == 0:
							obs_arr.append(obs[0])
							action_arr.append(action[0])
							rollout_vid.append(env.env_method('render')[0])
							if predict_second_return is not None:
								scale_viz_arr.append(predict_second_return[0].mean())

						# Ugly manual check for subgoal success info.
						chunk_info = [info_dict["chunk_info"] for info_dict in info]
						for env_i in range(obs.shape[0]):
							for chunk_step_i in range(self.action_chunk):
								step_info = chunk_info[env_i][chunk_step_i]
								for subgoal in self.subgoal_list:
									if step_info[subgoal] and subgoal_rate_arrs_i[subgoal][env_i] == 0:
										subgoal_rate_arrs_i[subgoal][env_i] = 1
										subgoal_time_arrs_i[subgoal][env_i] = step_i + chunk_step_i / self.action_chunk
						
						# Post-processing environment step.
						obs = next_obs
						rew_ep += reward
						rew_total += sum(rew_ep[done])
						rew_ep[done] = 0 
						total_ep += np.sum(done)
						delta_action_norms_i.append(np.linalg.norm(action, axis=-1))
						r.append(reward)

					# ------------------------- EPISODE POST-PROCESSING--------------------------

					# Updating rollout metrics.
					rews.append(np.mean(np.array(r)))
					delta_action_norms.append(np.array(delta_action_norms_i).mean())

					# Updating subgoal metrics.
					for subgoal in self.subgoal_list:
						subgoal_rate_arrs[subgoal].append(subgoal_rate_arrs_i[subgoal].mean())
						subgoal_time_arrs[subgoal].append(subgoal_time_arrs_i[subgoal].mean())
						success_i = subgoal_rate_arrs_i[subgoal] == 1
						subgoal_success_time_arrs[subgoal].append(
							subgoal_time_arrs_i[subgoal][success_i].mean()
							if np.sum(success_i) > 0 else self.max_steps
						)
					print(f'eval episode {i} at timestep {self.total_timesteps}')

					# Computing shaped rewards with obs - next_obs pairs.
					# TODO: WARNING: this is technically diffeent from how environment returns are aggregated.
					# TODO: WARNING: if this discrepancy emerges later, need to also track 'dones' to ensure consistency.
					action_arr_i = np.array(action_arr_i)
					obs_arr_i = np.array(obs_arr_i)
					# for o, o_next in zip(obs_arr_i[:-1], obs_arr_i[1:]):
					# 	# shaped_reward = agent.get_shaped_rewards(
					# 	# 	torch.tensor(o, device=agent.device, dtype=torch.float32),
					# 	# 	torch.tensor(o_next, device=agent.device, dtype=torch.float32),
					# 	# )
					# 	# shaped_rew_ep += shaped_reward.cpu().numpy().reshape(-1)
					# 	shaped_rew_ep = [0]
					for a, o in zip(action_arr_i, obs_arr_i):
						shaped_reward = agent.get_shaped_rewards(
							torch.tensor(a, device=agent.device, dtype=torch.float32),
							torch.tensor(o, device=agent.device, dtype=torch.float32),
						)
						shaped_rew_ep += shaped_reward.cpu().numpy().reshape(-1)
					shaped_rew_total += sum(shaped_rew_ep)				
				
				# ------------------------- MULTI-EPISODE EVALUATION POST-PROCESSING--------------------------

				# Computing evaluation and subgoal metrics - this will include success rate.
				delta_action_norms = np.array(delta_action_norms).mean()
				for subgoal in self.subgoal_list:
					subgoal_rate_arrs[subgoal] = np.array(subgoal_rate_arrs[subgoal]).mean()
					subgoal_time_arrs[subgoal] = np.array(subgoal_time_arrs[subgoal]).mean()
					subgoal_success_time_arrs[subgoal] = np.array(subgoal_success_time_arrs[subgoal]).mean()
				throughput = subgoal_rate_arrs["success"] / subgoal_time_arrs["success"]

				if total_ep > 0:
					avg_rew = rew_total / total_ep
					avg_shaped_rew = shaped_rew_total / total_ep
				else:
					avg_rew = 0
					avg_shaped_rew = 0

				# -------------------------- ROLLOUT VISUALIZATION --------------------------
				rollout_vid = np.array(rollout_vid)
				rollout_vid = rollout_vid.transpose(0, 3, 1, 2)  # T, C, H, W
				# rollout_vid_frames = [Image.fromarray(f) for f in rollout_vid]

				# # Computing predicted Q and V values for logged rollout.
				# obs_arr = np.array(obs_arr)
				# action_arr = np.array(action_arr)
				# # NOTE: this will treat rollout length as batch size.
				# pred_mean_qs = torch.cat(
				# 	agent.base_critic_value.forward_q(
				# 		torch.tensor(obs_arr, device=agent.device, dtype=torch.float32),
				# 		torch.tensor(action_arr, device=agent.device, dtype=torch.float32),
				# 	), dim=1
				# ).mean(dim=1, keepdim=True).cpu().numpy()
				# pred_vs = agent.base_critic_value.forward_v(
				# 	torch.tensor(obs_arr, device=agent.device, dtype=torch.float32)
				# ).cpu().numpy()
				# combined_frames = plot_data_with_frames(
				# 	rollout_vid_frames,
				# 	{"pred mean Q": pred_mean_qs, "pred V": pred_vs},
				# 	"Base Value Function Predictions",
				# )
				# combined_frames = np.stack([np.asarray(f) for f in combined_frames], axis=0)
				# combined_frames = combined_frames.transpose(0, 3, 1, 2)
				
				# -------------------------- WANDB LOGGING --------------------------
				if self.use_wandb:
					name = 'eval_base' if evaluate_base else 'eval'
					wandb.log({
						# f"{name}/success_rate": success_rate,
						f"{name}/reward": avg_rew,
						f"{name}/timesteps": self.total_timesteps,
					}, step=self.num_timesteps)
					
					# Log additional throughput metrics.
					wandb.log({
						f"{name}/shaped_reward": avg_shaped_rew,
						f"{name}/throughput": throughput,
						f"{name}/delta_action_norm": delta_action_norms,
					}, step=self.num_timesteps)

					# Log subgoal metrics.
					subgoal_log_dict = {
						f"{name}/{subgoal}_rate": subgoal_rate_arrs[subgoal] for subgoal in self.subgoal_list
					}
					subgoal_log_dict.update({
						f"{name}/{subgoal}_time": subgoal_success_time_arrs[subgoal] for subgoal in self.subgoal_list
					})
					wandb.log(subgoal_log_dict, step=self.num_timesteps)

					# Log rollout video.
					with warnings.catch_warnings():
						# NOTE: Suppressing warnings due to inconsistency between WandB and PIL.
						warnings.simplefilter("ignore")
						wandb.log({
							f"{name}/rollout_vid": wandb.Video(rollout_vid, fps=10, format="gif")
						}, step=self.num_timesteps)

						# if len(scale_viz_arr) > 0:
						# 	wandb.log({
						# 		f"{name}/scale_viz_vid": wandb.Video(scale_viz_frames, fps=10, format="gif")
						# 	}, step=self.num_timesteps)

	def set_timesteps(self, timesteps):
		self.total_timesteps = timesteps



def collect_initial_rollouts(model, env, num_steps, base_policy, cfg):
	obs = env.reset()
	for i in tqdm(range(num_steps)):
		# noise = torch.randn(cfg.env.n_envs, cfg.act_steps, cfg.action_dim).to(device=cfg.device)
		# action = base_policy(torch.tensor(obs, device=cfg.device, dtype=torch.float32), noise)
		# action = action.reshape(-1, cfg.act_steps * cfg.action_dim)
		# if model.policy_impedance_mode != "fixed":
		# 	action = model.augment_controller_action(action)
		action = model.sample_base_policy(obs, return_numpy=True)

		# Add initial rollout noise for better coverage, if necessary.
		# NOTE: consider adding noise before augmenting controller action?
		if cfg.train.init_rollout_noise_magnitude > 0:
			action += np.random.rand(*action.shape) * 2 * cfg.train.init_rollout_noise_magnitude - cfg.train.init_rollout_noise_magnitude

		next_obs, reward, done, info = env.step(action)
		if cfg.algorithm == 'fast':
			action_store = action

		model.replay_buffer.add(
				obs=obs,
				next_obs=next_obs,
				action=action_store,
				reward=reward,
				done=done,
				infos=info,
			)
		obs = next_obs
	model.replay_buffer.final_offline_step()
	
def load_offline_data(model, offline_data_path, n_env, chunk_size, reward_offset):
	offline_data = np.load(offline_data_path)
	
	# Check if data needs to be pre-processed or not.
	if 'traj_lengths' in offline_data:
		processed_data = preprocess_offline_data(offline_data, chunk_size, reward_offset)
		obs = processed_data['states']
		next_obs = processed_data['states_next']
		actions = processed_data['actions']
		rewards = processed_data['rewards']
		terminals = processed_data['terminals']
	else:
		obs = offline_data['states']
		next_obs = offline_data['states_next']
		actions = offline_data['actions']
		rewards = offline_data['rewards']
		terminals = offline_data['terminals']

	# Depending on policy/env impedance mode, augment actions based on controller config.
	if model.policy_impedance_mode != "fixed":
		actions = model.augment_controller_action(actions)

	# Adding default controller params to observations, if necessary.
	if model.control_obs:
		control_obs = np.zeros((obs.shape[0], 2))
		obs = np.concatenate([obs, control_obs], axis=-1)
		next_obs = np.concatenate([next_obs, control_obs], axis=-1)

	for i in range(int(obs.shape[0]/n_env)):
		model.replay_buffer.add(
					obs=obs[n_env*i:n_env*i+n_env],
					next_obs=next_obs[n_env*i:n_env*i+n_env],
					action=actions[n_env*i:n_env*i+n_env],
					reward=rewards[n_env*i:n_env*i+n_env],
					done=terminals[n_env*i:n_env*i+n_env],
					infos=[{}] * n_env,
				)
	model.replay_buffer.final_offline_step()

def preprocess_offline_data(offline_data, chunk_size, reward_offset):
	"""
	Converts from (states, actions, traj_lengths) to action-chunked
	(states, states_next, actions, rewards, terminals).
	"""
	states = offline_data['states']
	actions = offline_data['actions']
	traj_lengths = offline_data['traj_lengths']

	# Initializing arrays
	processed_states = []
	processed_states_next = []
	processed_actions = []
	processed_rewards = []
	processed_terminals = []
	
	# TODO: Consider vectorizing this.
	idx = 0
	for length in traj_lengths:
		for t in range(length - chunk_size + 1):
			# Grabbing states.
			processed_states.append(states[idx])
			processed_states_next.append(states[idx + 1] if t < length - 1 else states[idx])
			
			# Grabbing actions.
			end_idx = idx + chunk_size
			processed_actions.append(actions[idx:end_idx].reshape(-1))

			# Grabbing rewards.
			# processed_rewards.append(1.0 - reward_offset * chunk_size if t + chunk_size == length else -reward_offset * chunk_size)
			processed_rewards.append(-reward_offset * chunk_size)

			# Grabbing terminals.
			processed_terminals.append(t + chunk_size == length)
			idx += 1
		# Drop the last few samples that don't fit into a full chunk, and skip to next trajectory.
		idx += (chunk_size - 1)

	processed_actions = np.array(processed_actions)
	processed_states = np.array(processed_states)
	processed_states_next = np.array(processed_states_next)
	processed_rewards = np.array(processed_rewards)
	processed_terminals = np.array(processed_terminals, dtype=bool)

	return {
		'states': processed_states,
		'states_next': processed_states_next,
		'actions': processed_actions,
		'rewards': processed_rewards,
		'terminals': processed_terminals,
	}

def visualize_base_value(model, env, max_steps, cfg):
	"""
	For now, assume FAST environment and model.
	"""
	scale = 1.0
	log_dir = f"debug/fast/{cfg.env.name}"
	log_dir += f"/scale={scale}/seed={cfg.seed}"
	# log_dir += f"offset={cfg.env.reward_offset}"
	# log_dir += f"_fqe={cfg.base.fqe_steps}_vd={cfg.base.vd_steps}"
	# log_dir += f"_init_steps={cfg.train.init_rollout_steps}"
	os.makedirs(log_dir, exist_ok=True)

	rollout_vid = []
	obs_arr = []
	action_arr = []
	done_arr = []
	time_to_goal_arr = np.zeros(cfg.env.n_eval_envs)
	success_arr = np.zeros(cfg.env.n_eval_envs)
	chunk_size = model.diffusion_act_chunk

	with torch.no_grad():
		obs = env.reset()
		for _ in tqdm(range(max_steps)):
			action, _ = model.predict_diffused(obs, deterministic=True, sample_base=True)
			# Manually scaling actions.
			action = action.reshape(-1, cfg.act_steps, cfg.action_dim)
			action[:, :, 0:3] *= np.power(10.0, scale)
			action = action.reshape(-1, cfg.act_steps * cfg.action_dim)

			next_obs, reward, done, info = env.step(action)
			# TODO: need to extract reaching check, and grasping check
			# TODO: also compute time-to-reach and time-to-grasp
			breakpoint()

			obs_arr.append(obs)
			action_arr.append(action)
			done_arr.append(done)
			is_success_i = reward > -cfg.env.reward_offset * chunk_size
			# success_arr[reward > -cfg.env.reward_offset * chunk_size] = 1
			success_arr[is_success_i] = 1
			time_to_goal_arr[~is_success_i] += 1

			obs = next_obs
			rollout_vid.append(env.env_method('render'))

	# Converting trajectory to arrays
	rollout_vid = np.array(rollout_vid)
	obs_arr = np.array(obs_arr)
	action_arr = np.array(action_arr)

	pred_mean_q_arr = []
	pred_v_arr = []

	with torch.no_grad():
		for i in tqdm(range(max_steps)):
			obs_i = torch.tensor(obs_arr[i], device=model.device, dtype=torch.float32)
			action_i = torch.tensor(action_arr[i], device=model.device, dtype=torch.float32)
			pred_mean_qs = torch.cat(model.base_critic_value.forward_q(obs_i, action_i), dim=1).mean(dim=1, keepdim=True)
			pred_vs = model.base_critic_value.forward_v(obs_i)
			pred_mean_q_arr.append(pred_mean_qs.cpu().numpy())
			pred_v_arr.append(pred_vs.cpu().numpy())

	pred_mean_q_arr = np.array(pred_mean_q_arr)
	pred_v_arr = np.array(pred_v_arr)

	# Logging stuff.
	num_envs = obs_arr.shape[1]
	print("Total success rate: ", np.sum(success_arr) / num_envs)
	avg_time_to_goal = np.mean(time_to_goal_arr)
	avg_time_to_goal_success = np.mean(time_to_goal_arr[success_arr == 1]) if np.sum(success_arr) > 0 else max_steps
	print("Average time to goal: ", avg_time_to_goal)
	print("Average time to goal (successful episodes): ", avg_time_to_goal_success)
	return
	
	for env_i in tqdm(range(num_envs)):
		rollout_vid_i = rollout_vid[:, env_i, ...]
		pred_mean_qs_i = pred_mean_q_arr[:, env_i, 0]
		pred_vs_i = pred_v_arr[:, env_i, 0]
		success_tag = "success" if success_arr[env_i] == 1 else "fail"
		tag = f"{env_i}_{success_tag}"

		# Convert rollout vid to video.
		rollout_vid_frames_i = [Image.fromarray(f) for f in rollout_vid_i]

		# Plot predicted Q vs V
		plt.figure()
		plt.plot(pred_mean_qs_i, label='Predicted Mean Q')
		plt.plot(pred_vs_i, label='Predicted V')
		plt.xlabel('Timestep')
		plt.ylabel('Value')
		plt.title('Base Value Function Predictions')
		plt.legend()
		plt.savefig(f"{log_dir}/value_plot_{tag}.png")
		plt.close()

		combined_frames = plot_data_with_frames(
			rollout_vid_frames_i, 
			{
				"pred mean Q": pred_mean_qs_i, 
				"pred V": pred_vs_i,
			},
			"Base Value Function Predictions",
			)
		combined_frames[0].save(
			f"{log_dir}/rollout_{tag}.gif",
			save_all=True,
			append_images=combined_frames[1:],
			loop=0,
		)

def plot_data_with_frames(frames, data_dict, title):
	num_frames = len(frames)
	h = frames[0].height
	w = frames[0].width

	buf = io.BytesIO()
	combined_frames = []

	for i, frame in enumerate(frames):
		buf.truncate(0)
		buf.seek(0)
		plt.figure(figsize=(w / 100, h / 100), dpi=100)
		plt.xlim(0, num_frames)

		y_min = min([min(v[:i+1]) for v in data_dict.values()])
		y_max = max([max(v[:i+1]) for v in data_dict.values()])
		# Additionally filtering bounds to be at least -1 to 1.
		plt.ylim(y_min - 0.1, y_max + 0.1)

		for label, data in data_dict.items():
			plt.plot(data[:i+1], label=label)
		plt.xlabel('Timestep')
		plt.ylabel('Value')
		plt.title(title)
		plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
		plt.tight_layout()
		plt.legend()

		plt.savefig(buf, format='png')
		plt.close()
		buf.seek(0)

		plt_img = Image.open(buf).copy().convert('RGB')

		# Creating new Image, and pasting both frame and plot side by side.
		combined_img = Image.new('RGB', (w * 2, h))
		combined_img.paste(frame, (0, 0))
		combined_img.paste(plt_img, (w, 0))
		combined_frames.append(combined_img)

	return combined_frames

def flatten_wandb_cfg(wandb_cfg):
	"""
	Helper function to parse wandb config.
	"""
	if isinstance(wandb_cfg, dict):
		if "value" in wandb_cfg.keys():
			return wandb_cfg["value"]
		else:
			return {k: flatten_wandb_cfg(v) for k, v in wandb_cfg.items()}
	return wandb_cfg

def plot_metric_frames(data_dict, subgoal_dict, title, xlabel='Timestep', ylabel='Value', h=256, w=256):
	"""
	Plots a metric dictionary as an image and returns as PIL frames.
	"""
	num_frames = len(next(iter(data_dict.values())))
	final_subgoal_time = max(subgoal_dict.values())
	# Cutting off frames after success.
	num_frames = int(min(num_frames, final_subgoal_time + 10))
	buf = io.BytesIO()
	frames = []

	for i in range(num_frames):
		buf.truncate(0)
		buf.seek(0)
		plt.figure(figsize=(w / 100, h / 100), dpi=100)
		plt.xlim(0, num_frames)

		y_min = min([min(v[:i+1]) for v in data_dict.values()])
		y_max = max([max(v[:i+1]) for v in data_dict.values()])
		# Additionally filtering bounds to be at least -1 to 1.
		plt.ylim(y_min - 0.1, y_max + 0.1)
		plt.xlim(0, max(10, i + 1))

		for label, data in data_dict.items():
			plt.plot(data[:i+1], label=label)

		# Plotting vertical subgoal lines.
		# if i >= vert_line_x:
		# 	plt.axvline(x=vert_line_x, color='red', linestyle='--', linewidth=0.5)
		for subgoal, time in subgoal_dict.items():
			if i >= time:
				ax = plt.gca()
				ax.axvline(x=time, linestyle='--', linewidth=1.5, color='red')
				# ax.set_xticks([time], minor=True)
				# ax.set_xticklabels([subgoal], minor=True)
				ax.text(
					time + 0.5, 
					# y_max - 0.1 * (y_max - y_min), 
					y_max,
					subgoal[0].upper(), 
					rotation=0, 
					color='red', 
					fontsize=12
				)

		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.title(title)
		plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
		plt.tight_layout()
		# plt.legend()

		plt.savefig(buf, format='png')
		plt.close()
		buf.seek(0)

		plt_img = Image.open(buf).copy().convert('RGB')
		frames.append(plt_img)

	return frames

def plot_rollout_with_metrics(frames, metric_frames):
	# TODO: THIS ALSO NEEDS TO GRAB MAX LENGTH FRO MMETRIC FRAMES!
	num_frames = len(metric_frames[0])
	# TODO: ASSERT ALL METRIC FRAMES HAVE SAME LENGTH.
	total_subplots = 1 + len(metric_frames)

	# plots per row is square root, ceiling-ed.
	plots_per_row = int(np.ceil(np.sqrt(total_subplots)))
	plots_per_col = int(np.ceil(total_subplots / plots_per_row))

	combined_frames = []
	for i in range(num_frames):
		h = frames[0].height
		w = frames[0].width

		# Create a new image to hold the combined frame.
		combined_img = Image.new('RGB', (w * plots_per_row, h * plots_per_col))

		# Paste the main frame.
		combined_img.paste(frames[i], (0, 0))

		# Paste metric frames.
		for j, metric_frame in enumerate(metric_frames):
			row = (j + 1) // plots_per_row
			col = (j + 1) % plots_per_row
			combined_img.paste(metric_frame[i], (w * col, h * row))

		combined_frames.append(combined_img)
	
	# Add the last frame 20 times for better visibility.
	for _ in range(40):
		combined_frames.append(combined_frames[-1])
	return combined_frames