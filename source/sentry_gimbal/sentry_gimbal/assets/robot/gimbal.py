"""Configuration for the fixed-base two-axis RM sentry gimbal."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg

GIMBAL_MODEL_DIR = Path(__file__).resolve().parent / "gimbal"
GIMBAL_USD_PATH = GIMBAL_MODEL_DIR / "gimbal.usd"

GIMBAL_JOINT_NAMES = ["yaw_joint", "pitch_joint"]

GIMBAL_JOINT_LIMITS = {
    "yaw_joint": (-1.5708, 1.5708),
    "pitch_joint": (-0.34907, 0.5236),
}

GIMBAL_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(GIMBAL_USD_PATH),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.02,
            angular_damping=0.05,
            max_linear_velocity=100.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            fix_root_link=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "yaw_joint": 0.0,
            "pitch_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "yaw_motor": DCMotorCfg(
            joint_names_expr=["yaw_joint"],
            effort_limit=1.2,
            effort_limit_sim=1.2,
            saturation_effort=1.2,
            velocity_limit=13.82,
            velocity_limit_sim=13.82,
            stiffness=8.0,
            damping=0.5,
            armature=0.01,
            friction=0.01,
        ),
        "pitch_motor": DCMotorCfg(
            joint_names_expr=["pitch_joint"],
            effort_limit=3.0,
            effort_limit_sim=3.0,
            saturation_effort=3.0,
            velocity_limit=12.556,
            velocity_limit_sim=12.556,
            stiffness=14.0,
            damping=0.8,
            armature=0.01,
            friction=0.02,
        ),
    },
)
