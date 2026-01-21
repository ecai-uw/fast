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
from hydra.core.hydra_config import HydraConfig # used to parse command overrides
from omegaconf import OmegaConf
import gym, d4rl
import d4rl.gym_mujoco
import sys
sys.path.append('./dppo')
 
from stable_baselines3 import FAST
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from env_utils import (
    DiffusionPolicyEnvWrapper,
    ResidualPolicyWrapper,
    ObservationWrapperRobomimic, 
    ObservationWrapperGym, 
    ActionChunkWrapper, 
    make_robomimic_env, 
    eval_wrapper_dict, 
    subgoal_list_dict
)
from utils import (
    load_base_policy, 
    load_offline_data, 
    collect_initial_rollouts, 
    LoggingCallback, 
    visualize_base_value, 
    plot_data_with_frames, 
    flatten_wandb_cfg,
    plot_metric_frames,
    plot_rollout_with_metrics,
)
from PIL import Image
from tqdm import tqdm

OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil)
OmegaConf.register_new_resolver("round_down", math.floor)

base_path = os.path.dirname(os.path.abspath(__file__))

@hydra.main(
	config_path=os.path.join(base_path, "cfg/robomimic"), config_name="fast_can.yaml", version_base=None
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    eval_cfg = cfg.eval.copy()

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

        # Grab final checkpoint.
        model_load_path = os.path.join(cfg.run_dir, "checkpoint", "final.zip")

        # Loading previous wandb config to ensure consistency.
        wandb_cfg_path = os.path.join(cfg.run_dir, "wandb", "latest-run", "files", "config.yaml")
        prev_wandb_cfg = OmegaConf.load(wandb_cfg_path)
        prev_wandb_cfg = OmegaConf.create(
            flatten_wandb_cfg(OmegaConf.to_container(prev_wandb_cfg, resolve=True))
        )

        # Grabbing command line overrides for current evaluation run.
        overrides = OmegaConf.from_dotlist(list(HydraConfig.get().overrides.task))

        # Merging previous wandb config with current command overrides.
        cfg = OmegaConf.merge(prev_wandb_cfg, overrides)

    # If evaluating base policy, update run dir.
    if eval_cfg.eval_base:
        cfg.run_dir = f"./logs/{cfg.env_name}_base_eval"

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    MAX_STEPS = int(cfg.env.max_episode_steps / cfg.act_steps)

    num_env = cfg.env.n_envs
    assert num_env == 1, "Evaluation doesn't need training env - set to 1."
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
                impedance_mode=cfg.policy.impedance_mode,
                control_obs=cfg.env.control_obs,
            )
            env = eval_wrapper_dict[cfg.env_name](env, reward_offset=cfg.env.reward_offset)
        env = ResidualPolicyWrapper(env, cfg, full_render=True)
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

    # Run evaluation rollouts.
    # scale = 0.0
    # damping = 0.1
    # kp = 150
    save_video = eval_cfg.save_video and (cfg.seed == 1)
    sample_base = eval_cfg.eval_base
    subgoal_list = subgoal_list_dict[cfg.env_name]
    eval_episodes = int(cfg.num_evals / num_env_eval)

    # Only save video when seed = 1
    log_dir = os.path.join(cfg.run_dir, "videos")
    # log_dir = f"debug/fast/{cfg.env.name}"
    # log_dir += f"/scale={scale}"
    # log_dir += f"/kp={kp}_damping={damping}"
    # if cfg.resume:
    #     log_dir += f"_resume={cfg.wandb.id}"
    os.makedirs(log_dir, exist_ok=True)


    with torch.no_grad():
        # Initializing rollout visualization data.
        rollout_frames_unchunked = []
        rollout_deltas_unchunked = []
        rollout_stiffness_unchunked = []
        rollout_damping_unchunked = []

        # Initializing aggregated evaluation metrics.
        delta_action_norms = np.zeros((eval_episodes, num_env_eval))
        subgoal_rate_arrs = {subgoal: np.zeros((eval_episodes, num_env_eval)) for subgoal in subgoal_list}
        subgoal_time_arrs = {subgoal: np.zeros((eval_episodes, num_env_eval)) for subgoal in subgoal_list}

        # Running evaluation episodes.
        for i in range(eval_episodes):
            obs = eval_env.reset()

            # Initializing per-episode rollout metrics.
            delta_action_norms_i = []
            subgoal_rate_arrs_i = {subgoal: np.zeros(num_env_eval) for subgoal in subgoal_list}
            subgoal_time_arrs_i = {subgoal: np.zeros(num_env_eval) + MAX_STEPS for subgoal in subgoal_list}

            for step_i in range(MAX_STEPS):
                # Sample action and step environment.
                action, predict_second_return = model.predict_diffused(
                    obs, deterministic=cfg.deterministic_eval, sample_base=sample_base
                )
                next_obs, reward, done, info = eval_env.step(action)
                # TODO: PICK UP FROM HERE, PARSE FULL RENDERS TO GET CHUNK-UNROLLED VIDEOS

                if i == 0:
                    # Saving rollout profile metrics for visualization.
                    rollout_frames_unchunked.append([
                        np.array([info[env_i]["chunk_info"][t]["render"] for t in range(model.diffusion_act_chunk)])
                        for env_i in range(num_env_eval)
                    ])
                    rollout_deltas_unchunked.append([
                        np.array([info[env_i]["chunk_info"][t]["delta"] for t in range(model.diffusion_act_chunk)])
                        for env_i in range(num_env_eval)
                    ])
                    rollout_damping_unchunked.append([
                        np.array([info[env_i]["chunk_info"][t]["damping"] for t in range(model.diffusion_act_chunk)])
                        for env_i in range(num_env_eval)
                    ])
                    rollout_stiffness_unchunked.append([
                        np.array([info[env_i]["chunk_info"][t]["stiffness"] for t in range(model.diffusion_act_chunk)])
                        for env_i in range(num_env_eval)
                    ])

                # Ugly manual check for subgoal success info.
                chunk_info = [info_dict["chunk_info"] for info_dict in info]
                for env_i in range(num_env_eval):
                    for chunk_step_i in range(model.diffusion_act_chunk):
                        step_info = chunk_info[env_i][chunk_step_i]
                        for subgoal in subgoal_list:
                            if step_info[subgoal] and subgoal_rate_arrs_i[subgoal][env_i] == 0:
                                subgoal_rate_arrs_i[subgoal][env_i] = 1
                                subgoal_time_arrs_i[subgoal][env_i] = step_i + chunk_step_i / model.diffusion_act_chunk

                # Post-processing environment step.
                obs = next_obs
                # TODO: rew stuff here?
                # TODO: DELTA IS MEANINGLESS HERE, ACCOUNT FOR RESIDUAL/CONTROL PARAMS
                delta_action_norms_i.append(np.linalg.norm(action, axis=-1))

            # ------- EPISODE POST-PROCESSING -------

            # Updating aggregated rollout metrics.
            # TODO: SPLIT THIS UP IF USING CONTROL PARAMS
            # TODO: besides action norms, velocity profile?
            delta_action_norms[i] = np.array(delta_action_norms_i).mean(axis=0)

            # Updating subgoal metrics.
            for subgoal in subgoal_list:
                subgoal_rate_arrs[subgoal][i] = subgoal_rate_arrs_i[subgoal]
                subgoal_time_arrs[subgoal][i] = subgoal_time_arrs_i[subgoal]
            print(f"Eval episode {i+1}/{eval_episodes} completed.")

        # ------- EVALUATION POST-PROCESSING -------
        # Computing evaluation and subgoal metrics - this will include success rate.
        delta_action_norms = delta_action_norms.mean()
        subgoal_rates = {subgoal: subgoal_rate_arrs[subgoal].mean() for subgoal in subgoal_list}
        subgoal_times = {subgoal: subgoal_time_arrs[subgoal].mean() for subgoal in subgoal_list}
        subgoal_success_times = {
            subgoal: subgoal_time_arrs[subgoal][subgoal_rate_arrs[subgoal] == 1].mean() 
            if np.sum(subgoal_rate_arrs[subgoal]) > 0 else MAX_STEPS
            for subgoal in subgoal_list
        }
        throughputs = {
            subgoal: subgoal_rates[subgoal] / subgoal_times[subgoal]
            if subgoal_times[subgoal] > 0 else 0.0 # this is kinda unintuitive?
            for subgoal in subgoal_list
        }

        # Printing out evaluation metrics.
        print(f"Average delta action norm: {delta_action_norms:.4f}")
        print(f"Subgoal {'/'.join(subgoal_list)} rates: {' / '.join([f'{subgoal_rates[subgoal]:.2f}' for subgoal in subgoal_list])}")
        print(f"Subgoal {'/'.join(subgoal_list)} avg success times: {' / '.join([f'{subgoal_success_times[subgoal]:.2f}' for subgoal in subgoal_list])}")
        print(f"Subgoal {'/'.join(subgoal_list)} throughputs: {' / '.join([f'{throughputs[subgoal]:.4f}' for subgoal in subgoal_list])}")

        # Raw output for spreadsheet is success rate, success time, success throughput, then rates for all other subgoals, then times for all other subgoals
        raw_output = f"{subgoal_rates[subgoal_list[-1]]:.2f}, {subgoal_success_times[subgoal_list[-1]]:.2f}, {throughputs[subgoal_list[-1]]:.4f}"
        for subgoal in subgoal_list[:-1]:
            raw_output += f", {subgoal_rates[subgoal]:.2f}"
        for subgoal in subgoal_list[:-1]:
            raw_output += f", {subgoal_success_times[subgoal]:.2f}"
        print("Raw output for spreadsheet:")
        print(raw_output)

        # ------ ROLLOUT VISUALIZATION --------
        if save_video:
            # rollout_frames = np.array(rollout_frames)
            rollout_frames_unchunked = np.concatenate(
                rollout_frames_unchunked, axis=1
            )
            rollout_damping_unchunked = np.concatenate(
                rollout_damping_unchunked, axis=1
            ).mean(axis=-1)
            rollout_stiffness_unchunked = np.concatenate(
                rollout_stiffness_unchunked, axis=1
            ).mean(axis=-1)
            # Processing rollout deltas - seperate delta positions from delta orientations.
            rollout_deltas_unchunked = np.concatenate(rollout_deltas_unchunked, axis=1)
            rollout_delta_pos_norms_unchunked = np.linalg.norm(
                rollout_deltas_unchunked[..., :3], axis=-1
            )

            for env_i in tqdm(range(num_env_eval)):
                # rollout_frames_i = rollout_frames[:, env_i, ...]
                rollout_frames_unchunked_i = rollout_frames_unchunked[env_i, ...]
                success_tag = "success" if subgoal_rate_arrs[subgoal_list[-1]][0, env_i].sum() == 1 else "fail"
                tag = f"{env_i}_{success_tag}"

                # Convert rollout vid to video.
                rollout_vid_frames_i = [Image.fromarray(f) for f in rollout_frames_unchunked_i]
                rollout_vid_frames_i[0].save(
                    f"{log_dir}/rollout_{tag}.gif",
                    save_all=True,
                    append_images=rollout_vid_frames_i[model.diffusion_act_chunk::model.diffusion_act_chunk],
                    loop=0,
                    duration=25 * model.diffusion_act_chunk,
                )

                # Only plot metrics for first few envs to save time.
                if env_i >= 2:
                    continue
                
                rollout_vid_frames_unchunked_i = [Image.fromarray(f) for f in rollout_frames_unchunked[env_i, ...]]
                # Plotting metrics across rollouts.
                delta_pos_i_plots = plot_metric_frames(
                    {"delta": rollout_delta_pos_norms_unchunked[env_i, ...]},
                    title=f"Rollout Delta Positions",
                )
                damping_i_plots = plot_metric_frames(
                    {"damping": rollout_damping_unchunked[env_i, ...]},
                    title=f"Rollout Damping",
                )
                stiffness_i_plots = plot_metric_frames(
                    {"stiffness": rollout_stiffness_unchunked[env_i, ...]},
                    title=f"Rollout Stiffness",
                )

                rollout_metric_frames_i = plot_rollout_with_metrics(rollout_vid_frames_unchunked_i, [delta_pos_i_plots, damping_i_plots, stiffness_i_plots])
                rollout_metric_frames_i[0].save(
                    f"{log_dir}/rollout_{tag}_metrics.gif",
                    save_all=True,
                    append_images=rollout_metric_frames_i[1:],
                    loop=0,
                    duration=25,
                )
                

if __name__ == "__main__":
    main()