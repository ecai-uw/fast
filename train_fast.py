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
 
from stable_baselines3 import SAC, DSRL, FAST
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from env_utils import DiffusionPolicyEnvWrapper, ObservationWrapperRobomimic, ObservationWrapperGym, ActionChunkWrapper, make_robomimic_env
from utils import load_base_policy, load_offline_data, collect_initial_rollouts, LoggingCallback, visualize_base_value

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
                if os.path.exists(os.path.join(cfg.run_dir, "checkpoint", f"ft_policy_replay_buffer_{steps}_steps.pkl")):
                    if int(steps) > recent_steps:
                        recent_steps = int(steps)
        assert recent_steps > 0, "No valid checkpoint found to resume from."
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
        name=cfg.wandb.name,
        group=cfg.wandb.group,
        monitor_gym=True,
        save_code=True,
        config=OmegaConf.to_container(cfg, resolve=True),
        **resume_kwargs,
    )

    MAX_STEPS = int(cfg.env.max_episode_steps / cfg.act_steps)

    num_env = cfg.env.n_envs
    def make_env():
        if cfg.env_name in ['halfcheetah-medium-v2', 'hopper-medium-v2', 'walker2d-medium-v2']:
            env = gym.make(cfg.env_name)
            env = ObservationWrapperGym(env, cfg.normalization_path)
        elif cfg.env_name in ['lift', 'can', 'square', 'transport']:
            env = make_robomimic_env(render=True, env=cfg.env_name, normalization_path=cfg.normalization_path, low_dim_keys=cfg.env.wrappers.robomimic_lowdim.low_dim_keys, dppo_path=cfg.dppo_path)
            env = ObservationWrapperRobomimic(env, reward_offset=cfg.env.reward_offset)
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
        model.load_replay_buffer(path=buffer_load_path)
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
            # diffusion_policy=base_policy,
            diffusion_act_dim=(cfg.act_steps, cfg.action_dim),
            critic_backup_combine_type=cfg.train.critic_backup_combine_type,
            base_gamma=cfg.base.discount,
            policy_action_condition=cfg.policy.action_condition,
            shape_rewards=cfg.policy.shape_rewards,
            cfg=cfg,
        )

    # Manually setting base policy - this is not saved with model.
    base_policy = load_base_policy(cfg)
    model.diffusion_policy = base_policy

    # Creating checkpoint callback.
    # NOTE: CheckpointCallback inherits from BaseCallback, which automatically updates num_timesteps based on model.
    # NOTE: save_freq is based on self.n_calls, not self.num_timesteps, but the path is saved to num_timesteps...
    # NOTE: ...this could lead to unexpected behavior, but should be fine for now.
    checkpoint_callback = CheckpointCallback(
        save_freq=cfg.save_model_interval, 
        save_path=cfg.run_dir+'/checkpoint/',
        name_prefix='ft_policy',
        save_replay_buffer=cfg.save_replay_buffer, 
        save_vecnormalize=True,
    )

    # Creating evaluation environment and logging callback.
    num_env_eval = cfg.env.n_eval_envs
    eval_env = make_vec_env(make_env, n_envs=num_env_eval, vec_env_cls=SubprocVecEnv)
    eval_env.seed(cfg.seed + num_env + 1) 

    logging_callback = LoggingCallback(
        action_chunk = cfg.act_steps, 
        eval_episodes = int(cfg.num_evals / num_env_eval), 
        log_freq=MAX_STEPS, 
        use_wandb=cfg.use_wandb, 
        eval_env=eval_env,
        eval_freq=cfg.eval_interval,
        num_train_env=num_env,
        num_eval_env=num_env_eval,
        rew_offset=cfg.env.reward_offset,
        algorithm=cfg.algorithm,
        max_steps=MAX_STEPS,
        deterministic_eval=cfg.deterministic_eval,
    )
    # TODO: If training from scratch, evaluate base policy.
    # TODO: doing this for now to sanity check that run resuming works as intended.
    if not cfg.resume:
        logging_callback.evaluate(model, deterministic=False, evaluate_base=True)
        if cfg.deterministic_eval:
            logging_callback.evaluate(model, deterministic=True)
        logging_callback.log_count += 1


    # If training from scratch, load offline data and collect initial rollouts.
    if not cfg.resume:
        if cfg.load_offline_data:
            load_offline_data(model, cfg.offline_data_path, num_env, cfg.act_steps, cfg.env.reward_offset)
        if cfg.train.init_rollout_steps > 0:
            collect_initial_rollouts(model, env, cfg.train.init_rollout_steps, base_policy, cfg)	
            logging_callback.set_timesteps(cfg.train.init_rollout_steps * num_env)

        # run value distillation
        # TODO: Add logic to save/load trained base value function
        model.train_base_value(
            fqe_steps=cfg.base.fqe_steps,
            vd_steps=cfg.base.vd_steps,
            batch_size=cfg.base.batch_size,
            vd_samples=cfg.base.vd_samples,
            # lr_scheduler=cfg.base.lr_scheduler,
        )
    # Debugging step: evaluate and visualize base Q and V and demo trajectories, see if they make sense.
    # visualize_base_value(model, eval_env, MAX_STEPS, cfg)
    # quit()
    # breakpoint()

    # Train the agent.
    callbacks = [checkpoint_callback, logging_callback]
    total_timesteps = cfg.total_timesteps - model.num_timesteps if cfg.resume else cfg.total_timesteps
    model.learn(
        total_timesteps = total_timesteps, # 20000000,
        callback = callbacks,
        progress_bar = True,
        reset_num_timesteps = not cfg.resume,
    )

    # Save the final model
    if len(cfg.name) > 0:
        model.save(cfg.run_dir+"/checkpoint/final")

    # Close environment and wandb
    env.close()
    if cfg.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()