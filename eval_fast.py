import os
import warnings
warnings.filterwarnings("ignore")
import math
import torch
import random
import wandb
import uuid
import re
import numpy as np
import hydra
from omegaconf import OmegaConf
import gym, d4rl
import d4rl.gym_mujoco
import sys
sys.path.append('./dppo')
 
from stable_baselines3 import FAST
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from env_utils import DiffusionPolicyEnvWrapper, ObservationWrapperRobomimic, ObservationWrapperGym, ActionChunkWrapper, make_robomimic_env, eval_wrapper_dict, subgoal_list_dict
from utils import load_base_policy, load_offline_data, collect_initial_rollouts, LoggingCallback, visualize_base_value, plot_data_with_frames
from PIL import Image

OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil)
OmegaConf.register_new_resolver("round_down", math.floor)

base_path = os.path.dirname(os.path.abspath(__file__))

@hydra.main(
	config_path=os.path.join(base_path, "cfg/robomimic"), config_name="fast_can.yaml", version_base=None
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    assert cfg.use_wandb, "WandB logging must be enabled."
    # If resuming from a previous run...
    if cfg.resume:
        # ...run id must already be specified.
        assert cfg.wandb.id is not None, "Must provide wandb run id to resume from."
        # Manually set run_dir to match previous run.
        cfg.run_dir = os.path.join(cfg.log_dir, cfg.wandb.id)

        # Ensure that run_dir exists.
        if not os.path.exists(cfg.run_dir):
            raise ValueError(f"Provided run_dir {cfg.run_dir} does not exist!")
        # Grab most recent checkpoint.
        recent_steps = 0
        for file in os.listdir(os.path.join(cfg.run_dir, "checkpoint")):
            # Grabbing model checkpoint by filename pattern.
            if re.match(r"^ft_policy_\d+_steps\.zip$", file):
                steps = file.split("_")[2]
                # Ensuring that replay buffer is also saved for checkpoint.
                # if os.path.exists(os.path.join(cfg.run_dir, "checkpoint", f"ft_policy_replay_buffer_{steps}_steps.pkl")):
                if int(steps) > recent_steps:
                    recent_steps = int(steps)
        assert recent_steps > 0, "No valid checkpoint found to resume from."
        print(f"Resuming from checkpoint at step {recent_steps}.")
        model_load_path = os.path.join(cfg.run_dir, "checkpoint", f"ft_policy_{recent_steps}_steps.zip")
        buffer_load_path = os.path.join(cfg.run_dir, "checkpoint", f"ft_policy_replay_buffer_{recent_steps}_steps.pkl")

        # Initialize wandb in resume mode.
        # resume_kwargs = {"resume_from": f"{cfg.wandb.id}?_step={recent_steps}"}
        resume_kwargs = {"resume": "must"}
    # Otherwise, train from scratch.
    else:
        # Manually generating run id to handle customized directory structure.
        run_id = uuid.uuid4().hex[:8]
        cfg.wandb.id = run_id
        # Creating persistent run directory for checkpoints, etc.
        cfg.run_dir = os.path.join(cfg.log_dir, cfg.wandb.id)
        os.makedirs(cfg.run_dir, exist_ok=False)
        # Ensure that wandb initializes a new run.
        resume_kwargs = {"resume": False}

    # Initializing wandb run.
    wandb.init(
        id=cfg.wandb.id,
        dir=cfg.run_dir,
        project=cfg.wandb.project,
        name=cfg.wandb.id,
        group=cfg.wandb.group,
        monitor_gym=True,
        save_code=True,
        config=OmegaConf.to_container(cfg, resolve=True),
        **resume_kwargs,
    )
    # If specified, save wandb run id to local path for future resuming (e.g., in slurm jobs).
    if cfg.wandb.save_id_local_path is not None:
        with open(os.path.join(cfg.log_dir, cfg.wandb.save_id_local_path), 'w') as f:
            f.write(cfg.wandb.id)

    MAX_STEPS = int(cfg.env.max_episode_steps / cfg.act_steps)

    num_env = cfg.env.n_envs
    assert num_env == 1, "Evaluation doesn't need training env - set to 1."

    # TODO: manually setting impedance mode for now
    impedance_mode = "variable"
    def make_env():
        if cfg.env_name in ['halfcheetah-medium-v2', 'hopper-medium-v2', 'walker2d-medium-v2']:
            env = gym.make(cfg.env_name)
            env = ObservationWrapperGym(env, cfg.normalization_path)
        elif cfg.env_name in ['lift', 'can', 'square', 'transport']:
            env = make_robomimic_env(
                render=True, 
                env=cfg.env_name, 
                normalization_path=cfg.normalization_path, 
                low_dim_keys=cfg.env.wrappers.robomimic_lowdim.low_dim_keys, 
                dppo_path=cfg.dppo_path,
                impedance_mode=impedance_mode,
            )
            # env = ObservationWrapperRobomimic(env, reward_offset=cfg.env.reward_offset)
            env = eval_wrapper_dict[cfg.env_name](env, reward_offset=cfg.env.reward_offset)
        env = ActionChunkWrapper(env, cfg, max_episode_steps=cfg.env.max_episode_steps)
        return env

    # env = make_env()
    # breakpoint()

    env = make_vec_env(make_env, n_envs=num_env, vec_env_cls=SubprocVecEnv)
    env.seed(cfg.seed + 1)

    assert cfg.algorithm == 'fast', "Only FAST algorithm is supported in this training script."

    # If resuming, load checkpoint parameters.
    if cfg.resume:
        model = FAST.load(path=model_load_path, env=env)
        # model.load_replay_buffer(path=buffer_load_path)
    # Otherwise, train from scratch.
    else:
        post_linear_modules = None
        if cfg.train.use_layer_norm:
            post_linear_modules = [torch.nn.LayerNorm]

        net_arch = []
        for _ in range(cfg.train.num_layers):
            net_arch.append(cfg.train.layer_size)
        policy_kwargs = dict(
            net_arch=dict(pi=net_arch, qf=net_arch),
            activation_fn=torch.nn.Tanh,
            log_std_init=0.0,
            post_linear_modules=post_linear_modules,
            n_critics=cfg.train.n_critics,
        )

        # TODO: clean up; this code block is a little redundant with above, refactor later
        base_post_linear_modules = None
        if cfg.base.use_layer_norm:
            base_post_linear_modules = [torch.nn.LayerNorm]
        base_net_arch = []
        for _ in range(cfg.base.num_layers):
            base_net_arch.append(cfg.base.layer_size)
        base_kwargs = dict(
            net_arch=base_net_arch,
            activation_fn=torch.nn.Tanh,
            post_linear_modules=base_post_linear_modules,
            n_critics=cfg.base.n_critics,
        )
        model = FAST(
            "MlpPolicy",
            env,
            base_kwargs,
            learning_rate=cfg.train.actor_lr,
            buffer_size=20000000,      # Replay buffer size
            learning_starts=1,    # How many steps before learning starts (total steps for all env combined)
            batch_size=cfg.train.batch_size,
            tau=cfg.train.tau,                # Target network update rate
            gamma=cfg.train.discount,               # Discount factor
            train_freq=cfg.train.train_freq,             # Update the model every train_freq steps
            gradient_steps=cfg.train.utd,         # How many gradient steps to do at each update
            action_noise=None,        # No additional action noise
            optimize_memory_usage=False,
            ent_coef="auto" if cfg.train.ent_coef == -1 else cfg.train.ent_coef,          # Automatic entropy tuning
            target_update_interval=1, # Update target network every interval
            target_entropy="auto" if cfg.train.target_ent == -1 else cfg.train.target_ent,    # Automatic target entropy
            use_sde=False,
            sde_sample_freq=-1,
            # tensorboard_log=cfg.logdir,
            tensorboard_log=None, # Disabling tensorboard logging, since we use WandB.
            verbose=1,
            policy_kwargs=policy_kwargs,
            diffusion_act_dim=(cfg.act_steps, cfg.action_dim),
            critic_backup_combine_type=cfg.train.critic_backup_combine_type,
            base_gamma=cfg.base.discount,
            base_gradient_steps=cfg.policy.base_gradient_steps,
            policy_action_condition=cfg.policy.action_condition,
            shape_rewards=cfg.policy.shape_rewards,
            cfg=cfg,
        )

    # Manually setting base policy - this is not saved with model.
    base_policy = load_base_policy(cfg)
    model.diffusion_policy = base_policy

    # Creating evaluation environment.
    num_env_eval = cfg.env.n_eval_envs
    eval_env = make_vec_env(make_env, n_envs=num_env_eval, vec_env_cls=SubprocVecEnv)
    eval_env.seed(cfg.seed + num_env + 1)

    # Visualize base policy value function.
    from tqdm import tqdm
    scale = 0.0
    damping = 0.1
    kp = 150
    save_video = True
    sample_base = False if cfg.resume else True
    plot_values = False

    # Only save video when seed = 1
    save_video = save_video and (cfg.seed == 1)
    log_dir = f"debug/fast/{cfg.env.name}"
    # log_dir += f"/scale={scale}"
    log_dir += f"/kp={kp}_damping={damping}"
    if cfg.resume:
        log_dir += f"_resume={cfg.wandb.id}"
    os.makedirs(log_dir, exist_ok=True)

    rollout_vid = []
    obs_arr = []
    action_arr = []
    done_arr = []
    time_to_goal_arr = np.zeros(cfg.env.n_eval_envs)
    success_arr = np.zeros(cfg.env.n_eval_envs)
    chunk_size = model.diffusion_act_chunk

    reached_arr = np.zeros(cfg.env.n_eval_envs)
    grasped_arr = np.zeros(cfg.env.n_eval_envs)
    lifted_arr = np.zeros(cfg.env.n_eval_envs)

    reached_time_arr = np.zeros(cfg.env.n_eval_envs) + MAX_STEPS
    grasped_time_arr = np.zeros(cfg.env.n_eval_envs) + MAX_STEPS
    lifted_time_arr = np.zeros(cfg.env.n_eval_envs) + MAX_STEPS

    subgoal_list = subgoal_list_dict[cfg.env_name]
    subgoal_arrs = {subgoal: np.zeros(cfg.env.n_eval_envs) for subgoal in subgoal_list}
    subgoal_time_arrs = {subgoal: np.zeros(cfg.env.n_eval_envs) + MAX_STEPS for subgoal in subgoal_list}

    with torch.no_grad():
        obs = eval_env.reset()
        for step_i in tqdm(range(MAX_STEPS)):
            action, _ = model.predict_diffused(obs, deterministic=True, sample_base=sample_base)
            # Manually scaling actions.
            # action = action.reshape(-1, cfg.act_steps, cfg.action_dim)
            # action[:, :, 0:3] *= np.power(10.0, scale)
            # action = action.reshape(-1, cfg.act_steps * cfg.action_dim)

            # TODO: manually setting kp
            action = action.reshape(-1, cfg.act_steps, cfg.action_dim)
            damping_action = np.ones((action.shape[0], action.shape[1], 6), dtype=np.float32) * damping
            kp_action = np.ones((action.shape[0], action.shape[1], 6), dtype=np.float32) * kp
            action = np.concatenate([damping_action, kp_action, action], axis=-1)
            action = action.reshape(-1, cfg.act_steps * (cfg.action_dim + 12))

            # Step env.
            next_obs, reward, done, info = eval_env.step(action)

            # Ugly manual check for grasp and success from chunk info.
            chunk_info = [info_dict["chunk_info"] for info_dict in info]
            for env_i in range(cfg.env.n_eval_envs):
                for chunk_step_i in range(chunk_size):
                    step_info = chunk_info[env_i][chunk_step_i]
                    for subgoal in subgoal_list:
                        if step_info[subgoal] and subgoal_arrs[subgoal][env_i] == 0:
                            subgoal_arrs[subgoal][env_i] = 1
                            subgoal_time_arrs[subgoal][env_i] = step_i + chunk_step_i / chunk_size
                    
            obs_arr.append(obs)
            action_arr.append(action)
            done_arr.append(done)
            is_success_i = reward > -cfg.env.reward_offset * chunk_size
            success_arr[is_success_i] = 1
            time_to_goal_arr[~is_success_i] += 1

            obs = next_obs
            rollout_vid.append(eval_env.env_method('render'))
    
    # Converting trajectory to arrays
    rollout_vid = np.array(rollout_vid)
    obs_arr = np.array(obs_arr)
    action_arr = np.array(action_arr)

    # Logging stuff.
    subgoal_rates = {subgoal: np.sum(subgoal_arrs[subgoal]) / cfg.env.n_eval_envs for subgoal in subgoal_list}
    subgoal_avg_success_times = {
        subgoal: np.mean(subgoal_time_arrs[subgoal][subgoal_arrs[subgoal] == 1]) 
        if np.sum(subgoal_arrs[subgoal]) > 0 else MAX_STEPS 
        for subgoal in subgoal_list
    }

    print(f"Subgoal {'/'.join(subgoal_list)} rates: {' / '.join([f'{subgoal_rates[subgoal]:.2f}' for subgoal in subgoal_list])}")
    print(f"Subgoal {'/'.join(subgoal_list)} avg success times: {' / '.join([f'{subgoal_avg_success_times[subgoal]:.2f}' for subgoal in subgoal_list])}")
    print("Success rate:", np.sum(success_arr) / cfg.env.n_eval_envs)
    
    if save_video:
        # Converting trajectory to arrays
        rollout_vid = np.array(rollout_vid)
        obs_arr = np.array(obs_arr)
        action_arr = np.array(action_arr)

        pred_mean_q_arr = []
        pred_v_arr = []

        with torch.no_grad():
            for i in tqdm(range(MAX_STEPS)):
                obs_i = torch.tensor(obs_arr[i], device=model.device, dtype=torch.float32)
                action_i = torch.tensor(action_arr[i], device=model.device, dtype=torch.float32)
                pred_mean_qs = torch.cat(model.base_critic_value.forward_q(obs_i, action_i), dim=1).mean(dim=1, keepdim=True)
                pred_vs = model.base_critic_value.forward_v(obs_i)
                pred_mean_q_arr.append(pred_mean_qs.cpu().numpy())
                pred_v_arr.append(pred_vs.cpu().numpy())

        pred_mean_q_arr = np.array(pred_mean_q_arr)
        pred_v_arr = np.array(pred_v_arr)

        for env_i in tqdm(range(num_env_eval)):
            rollout_vid_i = rollout_vid[:, env_i, ...]
            pred_mean_qs_i = pred_mean_q_arr[:, env_i, 0]
            pred_vs_i = pred_v_arr[:, env_i, 0]
            success_tag = "success" if success_arr[env_i] == 1 else "fail"
            tag = f"{env_i}_{success_tag}"

            # Convert rollout vid to video.
            rollout_vid_frames_i = [Image.fromarray(f) for f in rollout_vid_i]

            if plot_values:
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
            else:
                rollout_vid_frames_i[0].save(
                    f"{log_dir}/rollout_{tag}.gif",
                    save_all=True,
                    append_images=rollout_vid_frames_i[1:],
                    loop=0,
                )



if __name__ == "__main__":
    main()