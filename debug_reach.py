import robosuite as suite
from envs.reach import Reach
import numpy as np
from robosuite.controllers import load_controller_config

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
from env_utils import make_robomimic_env
from omegaconf import OmegaConf
import json
from dppo.env.gym_utils.wrapper import wrapper_dict

from PIL import Image

controller_config = load_controller_config(default_controller="OSC_POSITION")

# env = suite.make(
#     env_name="Reach",
#     robots="Panda",
#     has_renderer=True,
#     use_camera_obs=False,
#     control_freq=20,
#     controller_configs=controller_config,
# )


# NEEDED ITEMS FOR ROBOMIMIC TEMPLATE:
# base policy path
# offline data path
# normalization path


env = "square"
normalization_path = "./dppo/log/robomimic/square/normalization.npz"
low_dim_keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
dppo_path = "./dppo"


wrappers = OmegaConf.create({
    'robomimic_lowdim': {
        'normalization_path': normalization_path,
        'low_dim_keys': low_dim_keys,
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

# Manually changing env name for debugging
env_meta["env_name"] = "Reach"

env = EnvUtils.create_env_from_metadata(
    env_meta=env_meta,
    render=False,
    render_offscreen=True,
    use_image_obs=False,
)
env.env.hard_reset = False
for wrapper, args in wrappers.items():
    env = wrapper_dict[wrapper](env, **args)

# img = env.render()
# # save image for debugging
# im = Image.fromarray(img)
# im.save("debug_reach_env.png")


# For debugging, reduce env.obs_min and env.obs_max to only the keys we care about
env.obs_min = env.obs_min[0:9]
env.obs_max = env.obs_max[0:9]



# Save gifs for a couple of rollouts:
for i in range(5):

    # TODO: do a simple debugging rollout here, and save the video
    imgs = []

    obs = env.reset()
    print(f"Iteration {i}, Goal: {env.env.env.goal}")
    done = False
    step = 0
    while not done and step < 50:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        img = env.render()
        imgs.append(Image.fromarray(img))
        step += 1

    imgs[0].save(
        f"debug/debug_{i}.gif",
        save_all=True,
        append_images=imgs[1:],
        loop=0,
    )