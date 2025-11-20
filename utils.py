import torch
import wandb
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import hydra
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import os
import io

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
	):
		super().__init__(verbose)
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
					}, step=self.log_count)
					if np.sum(self.episode_completed) > 0:
						wandb.log({
							"train/success_rate": np.sum(self.episode_success) / np.sum(self.episode_completed),
						}, step=self.log_count)
					if self.algorithm == 'dsrl_na':
						wandb.log({
							"train/noise_critic_loss": self.locals['self'].logger.name_to_value['train/noise_critic_loss'],
						}, step=self.log_count)
				self.episode_rewards = []
				self.episode_lengths = []
				self.total_reward = 0
				self.episode_success = np.zeros(self.num_train_env)
				self.episode_completed = np.zeros(self.num_train_env)

		if self.n_calls % self.eval_freq == 0:
			self.evaluate(self.locals['self'], deterministic=False)
			if self.deterministic_eval:
				self.evaluate(self.locals['self'], deterministic=True)
		return True
	
	def evaluate(self, agent, deterministic=False, evaluate_base=False):
		if self.eval_episodes > 0:
			env = self.eval_env

			# Logging arrays
			rollout_vid = []
			obs_arr = []
			action_arr = []

			with torch.no_grad():
				success, rews = [], []
				rew_total, total_ep = 0, 0
				rew_ep = np.zeros(self.num_eval_env)
				for i in range(self.eval_episodes):
					obs = env.reset()

					# # log rollout vid, if necessary
					# if i == 0:
					# 	rollout_vid.append(env.env_method('render', indices=rollout_vid_index)[0])

					success_i = np.zeros(obs.shape[0])
					r = []
					for _ in range(self.max_steps):
						if self.algorithm == 'dsrl_sac':
							action, _ = agent.predict(obs, deterministic=deterministic)
						elif self.algorithm == 'dsrl_na':
							action, _ = agent.predict_diffused(obs, deterministic=deterministic)
						elif self.algorithm == 'fast':
							action, _ = agent.predict_diffused(obs, deterministic=deterministic, sample_base=evaluate_base)
						next_obs, reward, done, info = env.step(action)

						# Logging, if necessary
						if i == 0:
							obs_arr.append(obs[0])
							action_arr.append(action[0])
							rollout_vid.append(env.env_method('render')[0])
						
						obs = next_obs
						rew_ep += reward
						rew_total += sum(rew_ep[done])
						rew_ep[done] = 0 
						total_ep += np.sum(done)
						success_i[reward > -self.rew_offset * self.action_chunk] = 1
						r.append(reward)

						# # log rollout vid, if necessary
						# if i == 0:
						# 	rollout_vid.append(env.env_method('render', indices=rollout_vid_index)[0])

					success.append(success_i.mean())
					rews.append(np.mean(np.array(r)))
					print(f'eval episode {i} at timestep {self.total_timesteps}')
				success_rate = np.mean(success)
				if total_ep > 0:
					avg_rew = rew_total / total_ep
				else:
					avg_rew = 0

				# Computing predicted Q and V-values for logged rollout.
				rollout_vid = np.array(rollout_vid)
				obs_arr = np.array(obs_arr)
				action_arr = np.array(action_arr)
				# NOTE: this will treat rollout length as batch size.
				pred_mean_qs = torch.cat(
					agent.base_critic(
						torch.tensor(obs_arr, device=agent.device, dtype=torch.float32),
						torch.tensor(action_arr, device=agent.device, dtype=torch.float32),
					), dim=1
				).mean(dim=1, keepdim=True).cpu().numpy()
				pred_vs = agent.value_net(
					torch.tensor(obs_arr, device=agent.device, dtype=torch.float32)
				).cpu().numpy()
				rollout_vid_frames = [Image.fromarray(f) for f in rollout_vid]
				combined_frames = plot_base_value(rollout_vid_frames, pred_mean_qs, pred_vs)
				combined_frames = np.stack([np.asarray(f) for f in combined_frames], axis=0)
				combined_frames = combined_frames.transpose(0, 3, 1, 2)
				
				if self.use_wandb:
					name = 'eval_base' if evaluate_base else 'eval'
					if deterministic:
						wandb.log({
							f"{name}/success_rate_deterministic": success_rate,
							f"{name}/reward_deterministic": avg_rew,
						}, step=self.log_count)
					else:
						wandb.log({
							f"{name}/success_rate": success_rate,
							f"{name}/reward": avg_rew,
							f"{name}/timesteps": self.total_timesteps,
						}, step=self.log_count)
					# Log rollout video
					# rollout_vid = np.array(rollout_vid).transpose((0, 3, 1, 2))
					wandb.log({
						f"{name}/rollout_vid": wandb.Video(combined_frames, fps=10, format="gif")
					}, step=self.log_count)

	def set_timesteps(self, timesteps):
		self.total_timesteps = timesteps



def collect_rollouts(model, env, num_steps, base_policy, cfg):
	obs = env.reset()
	for i in tqdm(range(num_steps)):
		noise = torch.randn(cfg.env.n_envs, cfg.act_steps, cfg.action_dim).to(device=cfg.device)
		if cfg.algorithm == 'dsrl_sac':
			noise[noise < -cfg.train.action_magnitude] = -cfg.train.action_magnitude
			noise[noise > cfg.train.action_magnitude] = cfg.train.action_magnitude
		action = base_policy(torch.tensor(obs, device=cfg.device, dtype=torch.float32), noise)
		next_obs, reward, done, info = env.step(action)
		if cfg.algorithm == 'dsrl_na':
			action_store = action
		elif cfg.algorithm == 'dsrl_sac':
			action_store = noise.detach().cpu().numpy()
		elif cfg.algorithm == 'fast':
			action_store = action
		action_store = action_store.reshape(-1, action_store.shape[1] * action_store.shape[2])
		if cfg.algorithm == 'dsrl_sac':
			action_store = model.policy.scale_action(action_store)
		# if cfg.algorithm == 'fast':
		# 	action_store = model.policy.scale_action(action_store)
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
	# this function should only be applied with dsrl_na
	offline_data = np.load(offline_data_path)
	# obs = offline_data['states']
	# next_obs = offline_data['states_next']
	# actions = offline_data['actions']
	# rewards = offline_data['rewards']
	# terminals = offline_data['terminals']
	
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
	log_dir = f"/home/ecai/debug/" # offset={cfg.env.reward_offset}_fqe={cfg.base.fqe_steps}_vd={cfg.base.vd_steps}"
	log_dir += f"offset={cfg.env.reward_offset}"
	log_dir += f"_fqe={cfg.base.fqe_steps}_vd={cfg.base.vd_steps}"
	log_dir += f"_init_steps={cfg.train.init_rollout_steps}"
	os.makedirs(log_dir, exist_ok=True)

	rollout_vid = []
	obs_arr = []
	action_arr = []
	done_arr = []
	success_arr = np.zeros(cfg.env.n_eval_envs)
	chunk_size = model.diffusion_act_chunk

	with torch.no_grad():
		obs = env.reset()
		for _ in tqdm(range(max_steps)):
			action, _ = model.predict_diffused(obs, deterministic=True, sample_base=True)
			next_obs, reward, done, info = env.step(action)

			obs_arr.append(obs)
			action_arr.append(action)
			done_arr.append(done)
			success_arr[reward > -cfg.env.reward_offset * chunk_size] = 1

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
			obs_i = torch.tensor(obs_arr[i], device=model.device)
			action_i = torch.tensor(action_arr[i], device=model.device)
			pred_mean_qs = torch.cat(model.base_critic(obs_i, action_i), dim=1).mean(dim=1, keepdim=True)
			pred_vs = model.value_net(obs_i)
			pred_mean_q_arr.append(pred_mean_qs.cpu().numpy())
			pred_v_arr.append(pred_vs.cpu().numpy())

	pred_mean_q_arr = np.array(pred_mean_q_arr)
	pred_v_arr = np.array(pred_v_arr)

	# Logging stuff.
	num_envs = obs_arr.shape[1]
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

		combined_frames = plot_base_value(rollout_vid_frames_i, pred_mean_qs_i, pred_vs_i, log_dir, tag)
		combined_frames[0].save(
			f"{log_dir}/rollout_{tag}.gif",
			save_all=True,
			append_images=combined_frames[1:],
			loop=0,
		)


def plot_base_value(frames, qs, vs):
	num_frames = len(frames)
	y_min = min(min(qs), min(vs))
	y_max = max(max(qs), max(vs))
	h = frames[0].height
	w = frames[0].width

	buf = io.BytesIO()
	combined_frames = []

	for i, frame in enumerate(frames):
		buf.truncate(0)
		buf.seek(0)
		plt.figure(figsize=(w / 100, h / 100), dpi=100)
		plt.xlim(0, num_frames)
		plt.ylim(y_min - 0.1, y_max + 0.1)
		plt.plot(qs[:i+1], label='Predicted Mean Q')
		plt.plot(vs[:i+1], label='Predicted V')
		plt.xlabel('Timestep')
		plt.ylabel('Value')
		plt.title('Base Value Function Predictions')
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