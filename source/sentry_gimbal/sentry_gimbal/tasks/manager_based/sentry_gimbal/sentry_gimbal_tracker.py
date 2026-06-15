"""Manager-based RL task for CSV-driven two-axis gimbal tracking."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import wrap_to_pi
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from sentry_gimbal.assets.robot.gimbal import GIMBAL_CFG, GIMBAL_JOINT_LIMITS, GIMBAL_JOINT_NAMES

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


DEFAULT_TRAJECTORY_DIR = Path(__file__).resolve().parents[3] / "data" / "trajectories"


@configclass
class CsvGimbalTrajectoryCommandCfg(CommandTermCfg):
    """CSV target trajectory command for yaw/pitch tracking."""

    class_type: type = None

    asset_name: str = "robot"
    trajectory_dir: str | None = str(DEFAULT_TRAJECTORY_DIR)
    trajectory_files: tuple[str, ...] = ()
    time_column: str = "time"
    pitch_column: str = "target_pitch"
    yaw_column: str = "target_yaw"
    angles_in_degrees: bool = False
    random_start: bool = True
    loop: bool = False
    segment_duration_s: float = 0.0
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

    def __post_init__(self):
        self.class_type = CsvGimbalTrajectoryCommand


class CsvGimbalTrajectoryCommand(CommandTerm):
    """Plays target yaw/pitch trajectories loaded from CSV files.

    The command tensor is ordered as
    ``[target_yaw, target_pitch, target_yaw_vel, target_pitch_vel]``.
    CSV files are expected to contain ``time,target_pitch,target_yaw`` by default.
    """

    cfg: CsvGimbalTrajectoryCommandCfg

    def __init__(self, cfg: CsvGimbalTrajectoryCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self._gimbal_joint_ids, _ = self.robot.find_joints(GIMBAL_JOINT_NAMES, preserve_order=True)
        self._traj_time: list[torch.Tensor] = []
        self._traj_pos: list[torch.Tensor] = []
        self._traj_duration: list[float] = []
        self._load_trajectories()

        self._command = torch.zeros(self.num_envs, 4, device=self.device)
        self._traj_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._start_time = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_yaw"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_pitch"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _load_trajectories(self) -> None:
        paths = self._resolve_csv_paths()
        if not paths:
            raise FileNotFoundError(
                "No gimbal trajectory CSV files were found. Put files under "
                f"'{self.cfg.trajectory_dir}' or set commands.target_trajectory.trajectory_files."
            )

        for path in paths:
            time_values: list[float] = []
            pitch_values: list[float] = []
            yaw_values: list[float] = []
            with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    row = {key.strip(): value for key, value in row.items() if key is not None}
                    time_values.append(float(row[self.cfg.time_column]))
                    pitch_values.append(float(row[self.cfg.pitch_column]))
                    yaw_values.append(float(row[self.cfg.yaw_column]))

            if not time_values:
                raise ValueError(f"Trajectory CSV '{path}' is empty.")

            rows = sorted(zip(time_values, yaw_values, pitch_values), key=lambda item: item[0])
            filtered_rows = []
            last_time = None
            for item in rows:
                if last_time is None or item[0] > last_time:
                    filtered_rows.append(item)
                    last_time = item[0]

            if not filtered_rows:
                raise ValueError(f"Trajectory CSV '{path}' does not contain valid time samples.")

            time_tensor = torch.tensor([row[0] for row in filtered_rows], dtype=torch.float32, device=self.device)
            pos_tensor = torch.tensor(
                [[row[1], row[2]] for row in filtered_rows], dtype=torch.float32, device=self.device
            )
            time_tensor = time_tensor - time_tensor[0]
            if self.cfg.angles_in_degrees:
                pos_tensor = torch.deg2rad(pos_tensor)

            self._traj_time.append(time_tensor)
            self._traj_pos.append(pos_tensor)
            self._traj_duration.append(float(time_tensor[-1].item()))

    def _resolve_csv_paths(self) -> list[Path]:
        if self.cfg.trajectory_files:
            paths = [Path(path) for path in self.cfg.trajectory_files]
        elif self.cfg.trajectory_dir:
            paths = sorted(Path(self.cfg.trajectory_dir).glob("*.csv"))
        else:
            paths = []

        missing_paths = [str(path) for path in paths if not path.is_file()]
        if missing_paths:
            raise FileNotFoundError(f"Missing trajectory CSV files: {missing_paths}")
        return paths

    def _update_metrics(self) -> None:
        target_pos = self._command[:, :2]
        actual_pos = self.robot.data.joint_pos[:, self._gimbal_joint_ids]
        error = wrap_to_pi(target_pos - actual_pos)
        self.metrics["error_yaw"][:] = torch.abs(error[:, 0])
        self.metrics["error_pitch"][:] = torch.abs(error[:, 1])

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return

        traj_ids = torch.randint(len(self._traj_time), (env_ids.numel(),), device=self.device)
        self._traj_ids[env_ids] = traj_ids

        if not self.cfg.random_start:
            self._start_time[env_ids] = 0.0
            self._update_command()
            return

        episode_s = self.cfg.segment_duration_s
        if episode_s <= 0.0:
            episode_s = getattr(self._env, "max_episode_length_s", 0.0)

        for traj_id in torch.unique(traj_ids).tolist():
            selected = env_ids[traj_ids == traj_id]
            duration = self._traj_duration[traj_id]
            max_start = duration if self.cfg.loop else max(0.0, duration - episode_s)
            if max_start <= 0.0:
                self._start_time[selected] = 0.0
            else:
                self._start_time[selected] = torch.rand(selected.numel(), device=self.device) * max_start

        self._update_command()

    def _update_command(self) -> None:
        elapsed = self._env.episode_length_buf.to(device=self.device, dtype=torch.float32) * self._env.step_dt
        query_time = self._start_time + elapsed

        for traj_id in range(len(self._traj_time)):
            env_ids = (self._traj_ids == traj_id).nonzero(as_tuple=False).flatten()
            if env_ids.numel() == 0:
                continue
            pos, vel = self._sample_trajectory(traj_id, query_time[env_ids])
            self._command[env_ids, :2] = pos
            self._command[env_ids, 2:] = vel

    def _sample_trajectory(self, traj_id: int, query_time: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        time = self._traj_time[traj_id]
        pos = self._traj_pos[traj_id]

        if time.numel() == 1:
            return pos[0].expand(query_time.numel(), -1), torch.zeros(query_time.numel(), 2, device=self.device)

        duration = max(self._traj_duration[traj_id], 1.0e-6)
        if self.cfg.loop:
            query_time = torch.remainder(query_time, duration)
        else:
            query_time = query_time.clamp(0.0, float(time[-1].item()))

        idx1 = torch.searchsorted(time, query_time, right=False).clamp(1, time.numel() - 1)
        idx0 = idx1 - 1
        t0 = time[idx0]
        t1 = time[idx1]
        alpha = ((query_time - t0) / (t1 - t0).clamp_min(1.0e-6)).unsqueeze(-1)

        p0 = pos[idx0]
        p1 = pos[idx1]
        sampled_pos = p0 + alpha * (p1 - p0)
        sampled_vel = (p1 - p0) / (t1 - t0).clamp_min(1.0e-6).unsqueeze(-1)
        return sampled_pos, sampled_vel

CsvGimbalTrajectoryCommandCfg.class_type = CsvGimbalTrajectoryCommand


def gimbal_tracking_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=GIMBAL_JOINT_NAMES, preserve_order=True),
) -> torch.Tensor:
    """Target yaw/pitch minus simulated yaw/pitch."""
    asset: Articulation = env.scene[asset_cfg.name]
    target_pos = env.command_manager.get_command(command_name)[:, :2]
    actual_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return wrap_to_pi(target_pos - actual_pos)


def gimbal_target_velocity(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Target yaw/pitch velocity from the active CSV trajectory."""
    return env.command_manager.get_command(command_name)[:, 2:]


def track_gimbal_angle_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=GIMBAL_JOINT_NAMES, preserve_order=True),
) -> torch.Tensor:
    """Reward accurate yaw/pitch angle tracking with an exponential kernel."""
    error = gimbal_tracking_error(env, command_name, asset_cfg)
    return torch.exp(-torch.sum(torch.square(error), dim=1) / std**2)


def gimbal_angle_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=GIMBAL_JOINT_NAMES, preserve_order=True),
) -> torch.Tensor:
    """Penalize squared yaw/pitch tracking error."""
    error = gimbal_tracking_error(env, command_name, asset_cfg)
    return torch.sum(torch.square(error), dim=1)


def gimbal_velocity_error_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=GIMBAL_JOINT_NAMES, preserve_order=True),
) -> torch.Tensor:
    """Penalize mismatch between target and simulated yaw/pitch velocities."""
    asset: Articulation = env.scene[asset_cfg.name]
    target_vel = env.command_manager.get_command(command_name)[:, 2:]
    actual_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(target_vel - actual_vel), dim=1)


def gimbal_error_out_of_bound(
    env: ManagerBasedRLEnv,
    command_name: str,
    max_error: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=GIMBAL_JOINT_NAMES, preserve_order=True),
) -> torch.Tensor:
    """Terminate when yaw or pitch tracking error is too large."""
    error = torch.abs(gimbal_tracking_error(env, command_name, asset_cfg))
    return torch.any(error > max_error, dim=1)


def _configured_gimbal_joint_limits(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    limits = torch.tensor([GIMBAL_JOINT_LIMITS[name] for name in GIMBAL_JOINT_NAMES], device=device, dtype=dtype)
    return limits[:, 0], limits[:, 1]


def gimbal_joint_pos_limits(
    env: ManagerBasedRLEnv,
    soft_ratio: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=GIMBAL_JOINT_NAMES, preserve_order=True),
) -> torch.Tensor:
    """Penalize yaw/pitch positions outside the configured soft joint range."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    lower, upper = _configured_gimbal_joint_limits(joint_pos.device, joint_pos.dtype)
    center = 0.5 * (lower + upper)
    half_width = 0.5 * (upper - lower) * soft_ratio
    soft_lower = center - half_width
    soft_upper = center + half_width

    out_of_limits = -(joint_pos - soft_lower).clip(max=0.0)
    out_of_limits += (joint_pos - soft_upper).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def gimbal_joint_pos_out_of_bound(
    env: ManagerBasedRLEnv,
    margin: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=GIMBAL_JOINT_NAMES, preserve_order=True),
) -> torch.Tensor:
    """Terminate when yaw/pitch leave the configured mechanical joint limits."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    lower, upper = _configured_gimbal_joint_limits(joint_pos.device, joint_pos.dtype)
    return torch.any((joint_pos < lower - margin) | (joint_pos > upper + margin), dim=1)


@configclass
class SentryGimbalTrackerSceneCfg(InteractiveSceneCfg):
    """Scene with a fixed-base two-axis RM gimbal."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(20.0, 20.0)),
    )

    robot: ArticulationCfg = GIMBAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=800.0),
    )


@configclass
class CommandsCfg:
    """Command terms for the target pitch/yaw trajectory."""

    target_trajectory = CsvGimbalTrajectoryCommandCfg()


@configclass
class ActionsCfg:
    """Two normalized actions mapped to bounded yaw/pitch position targets."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=GIMBAL_JOINT_NAMES,
        preserve_order=True,
        scale={
            "yaw_joint": 0.6,
            "pitch_joint": 0.25,
        },
        use_default_offset=True,
        clip={
            "yaw_joint": (-0.8, 0.8),
            "pitch_joint": (-0.25, 0.35),
        },
    )


@configclass
class ObservationsCfg:
    """Policy observations for trajectory tracking."""

    @configclass
    class PolicyCfg(ObsGroup):
        tracking_error = ObsTerm(
            func=gimbal_tracking_error,
            params={"command_name": "target_trajectory"},
            noise=Unoise(n_min=-0.001, n_max=0.001),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=GIMBAL_JOINT_NAMES, preserve_order=True),
            },
            noise=Unoise(n_min=-0.001, n_max=0.001),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=GIMBAL_JOINT_NAMES, preserve_order=True),
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        target_vel = ObsTerm(func=gimbal_target_velocity, params={"command_name": "target_trajectory"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for precise, smooth yaw/pitch tracking."""

    track_angle = RewTerm(
        func=track_gimbal_angle_exp,
        weight=5.0,
        params={"command_name": "target_trajectory", "std": 0.2},
    )
    angle_error = RewTerm(
        func=gimbal_angle_l2,
        weight=-8.0,
        params={"command_name": "target_trajectory"},
    )
    velocity_error = RewTerm(
        func=gimbal_velocity_error_l2,
        weight=-0.02,
        params={"command_name": "target_trajectory"},
    )
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.002,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=GIMBAL_JOINT_NAMES, preserve_order=True)},
    )
    joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=GIMBAL_JOINT_NAMES, preserve_order=True)},
    )
    joint_limits = RewTerm(
        func=gimbal_joint_pos_limits,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=GIMBAL_JOINT_NAMES, preserve_order=True),
            "soft_ratio": 0.9,
        },
    )
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.002)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-0.0005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=GIMBAL_JOINT_NAMES, preserve_order=True)},
    )


@configclass
class TerminationsCfg:
    """Episode termination conditions."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    joint_out_of_limit = DoneTerm(
        func=gimbal_joint_pos_out_of_bound,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=GIMBAL_JOINT_NAMES, preserve_order=True),
            "margin": 0.02,
        },
    )
    tracking_error_too_large = DoneTerm(
        func=gimbal_error_out_of_bound,
        params={"command_name": "target_trajectory", "max_error": 0.75},
    )


@configclass
class EventCfg:
    """Reset events."""

    reset_gimbal_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=GIMBAL_JOINT_NAMES, preserve_order=True),
            "position_range": (-0.02, 0.02),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class SentryGimbalTrackerEnvCfg(ManagerBasedRLEnvCfg):
    """CSV target trajectory tracking environment for the RM sentry gimbal."""

    scene: SentryGimbalTrackerSceneCfg = SentryGimbalTrackerSceneCfg(num_envs=4096, env_spacing=2.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 5.0
        self.viewer.eye = (1.8, 1.8, 1.2)
        self.viewer.lookat = (0.0, 0.0, 0.05)
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation

        self.scene.robot.spawn.rigid_props.disable_gravity = False
        self.scene.robot.spawn.articulation_props.fix_root_link = True

        yaw_limits = GIMBAL_JOINT_LIMITS["yaw_joint"]
        pitch_limits = GIMBAL_JOINT_LIMITS["pitch_joint"]
        self.terminations.tracking_error_too_large.params["max_error"] = max(
            0.75,
            0.5 * max(yaw_limits[1] - yaw_limits[0], pitch_limits[1] - pitch_limits[0]),
        )
