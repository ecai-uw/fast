# Simple test environment for a trivial goal-reaching task.
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import Task
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.transform_utils import convert_quat
from robosuite.models.objects import BoxObject
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.mjcf_utils import new_site, new_body, new_joint, new_geom
import numpy as np


class Reach(SingleArmEnv):
    """
    This class corresponds to a simple goal reaching task for a single robot arm.
    """
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # Settings for table top goal distribution
        self.goal_xy_range = 0.4      # 40% of half-extent
        self.goal_z_range = (0.05, 0.30)  # meters above tabletop

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # object observation flag
        self.use_object_obs = use_object_obs

        # Defining goal distribution - these are properly initialized later.
        self.goal_initializer = None
        self.goal = None

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    def reward(self, action=None):
        reward = 0.0

        # sparse completion reward
        if self._check_success():
            reward = 2.25

        # reaching reward
        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        goal_pos = self.goal
        distance = np.linalg.norm(gripper_site_pos - goal_pos)
        reaching_reward = 1 - np.tanh(10.0 * distance)
        reward += reaching_reward

        # Scale reward if requested
        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.25

    def _load_model(self):
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.cube = BoxObject(
            name="cube",
            size_min=[0.020, 0.020, 0.020],  # [0.015, 0.015, 0.015],
            size_max=[0.022, 0.022, 0.022],  # [0.018, 0.018, 0.018])
            rgba=[1, 0, 0, 1],
            material=redwood,
        )

        goal_site = new_site(
            name="goal_site",
            pos=[0, 0, 1.0],
            size=[0.02],
            rgba=[1, 0, 0, 1],
        )
        goal_site_table_plane = new_site(
            name="goal_site_table_plane",
            pos=[0, 0, 1.0],
            size=[0.02],
            rgba=[0, 1, 0, 1],
        )
        mujoco_arena.worldbody.append(goal_site)
        mujoco_arena.worldbody.append(goal_site_table_plane)

        # task includes arena, robot, but no objects.
        self.model = Task(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.cube,
        )
           
    def _setup_references(self):
        # No objects, so nothing additional to set up.
        super()._setup_references()

        # Additional object references from this env
        self.cube_body_id = self.sim.model.body_name2id(self.cube.root_body)
        self.goal_site_id = self.sim.model.site_name2id("goal_site")
        self.goal_site_table_plane_id = self.sim.model.site_name2id("goal_site_table_plane")

    def _setup_observables(self):
        observables = super()._setup_observables()
        
        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"

            # cube-related observables
            @sensor(modality=modality)
            def cube_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cube_body_id])

            @sensor(modality=modality)
            def cube_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.cube_body_id]), to="xyzw")

            @sensor(modality=modality)
            def gripper_to_cube_pos(obs_cache):
                return (
                    obs_cache[f"{pf}eef_pos"] - obs_cache["cube_pos"]
                    if f"{pf}eef_pos" in obs_cache and "cube_pos" in obs_cache
                    else np.zeros(3)
                )

            sensors = [cube_pos, cube_quat, gripper_to_cube_pos]
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )
        return observables
        
    def _reset_internal(self):
        super()._reset_internal()

        table_x, table_y, table_z_offset = self.table_offset
        table_x_size, table_y_size, table_z_size = self.table_full_size

        table_top_z = table_z_offset + 0.5 * table_z_size

        # XY bounds
        x_half = 0.5 * table_x_size * self.goal_xy_range
        y_half = 0.5 * table_y_size * self.goal_xy_range

        # Sample goal position on table top within defined ranges
        goal_x = np.random.uniform(table_x - x_half, table_x + x_half)
        goal_y = np.random.uniform(table_y - y_half, table_y + y_half)
        goal_z = np.random.uniform(
            table_top_z + self.goal_z_range[0],
            table_top_z + self.goal_z_range[1],
        )
        self.goal = np.array([goal_x, goal_y, goal_z])
        goal_table_plane = np.array([goal_x, goal_y, table_top_z - 0.03])

        self.sim.model.site_pos[self.goal_site_id] = self.goal
        self.sim.model.site_pos[self.goal_site_table_plane_id] = goal_table_plane
        self.sim.forward()

    def visualize(self, vis_settings):
        # Run superclass method first
        # NOTE: maybe look into _visualize_gripper_to_target?
        super().visualize(vis_settings=vis_settings)

    def _check_success(self):
        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        goal_pos = self.goal
        distance = np.linalg.norm(gripper_site_pos - goal_pos)
        return distance < 0.05