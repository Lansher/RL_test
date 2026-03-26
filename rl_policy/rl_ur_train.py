import argparse
import os
import warnings
from collections import deque
from typing import List, Tuple

import gym
import mujoco
import numpy as np
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
import torch.nn as nn


class CurriculumCallback(BaseCallback):
    """
    自动课程「考官」：按最近若干回合的成功率监控进度，达标后通过 env_method
    调用各子环境的 set_curriculum_stage，无需停训即可换挡。
    """

    def __init__(
        self,
        stages: List[dict],
        upgrade_threshold: float = 0.85,
        window: int = 100,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.stages = stages
        self.current_stage = 0
        self.recent_successes: deque = deque(maxlen=window)
        self.upgrade_threshold = float(upgrade_threshold)
        self.window = int(window)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        dones = self.locals.get("dones")
        if infos is None or dones is None:
            return True
        dones = np.asarray(dones).reshape(-1)
        n = min(len(dones), len(infos))
        for i in range(n):
            if bool(dones[i]):
                info_i = infos[i]
                if isinstance(info_i, (list, tuple)) and len(info_i) > 0:
                    info_i = info_i[0]
                if not isinstance(info_i, dict):
                    continue
                is_success = bool(info_i.get("is_success", False))
                self.recent_successes.append(float(is_success))
        return True

    def _on_rollout_end(self) -> None:
        super()._on_rollout_end()
        if self.training_env is None:
            return
        if self.current_stage >= len(self.stages) - 1:
            return
        if len(self.recent_successes) < self.window:
            return

        current_success_rate = float(np.mean(self.recent_successes))
        if self.logger is not None:
            try:
                self.logger.record("curriculum/stage", float(self.current_stage))
                self.logger.record("curriculum/recent_success_rate", current_success_rate)
            except Exception:
                pass

        if current_success_rate < self.upgrade_threshold:
            return

        self.current_stage += 1
        new_params = self.stages[self.current_stage]
        print(
            f"\n{'=' * 50}\n"
            f"*** 达成晋升条件！近期成功率 {current_success_rate * 100:.1f}% ***\n"
            f"*** 进入难度阶段 {self.current_stage} / {len(self.stages) - 1} ***\n"
            f"    控制臂: {new_params['single_arm']}, 固定目标: {new_params['fixed_target']}\n"
            f"{'=' * 50}\n",
            flush=True,
        )
        self.recent_successes.clear()

        self.training_env.env_method(
            "set_curriculum_stage",
            single_arm=new_params["single_arm"],
            fixed_target=new_params["fixed_target"],
        )


# ---------------------------------------------------------------------------
# 项目默认超参（与 rl_ur_test.py 中同名常量保持一致）。日常训练/验证只需改
# --log_dir / --model_path / --n_env / --single_arm 等少量参数，其余用这里默认值。
# ---------------------------------------------------------------------------
DEFAULT_MAX_DELTA_CTRL = 0.08
DEFAULT_ACTION_SMOOTHING_COEF = 0.75
DEFAULT_ACTION_DEADZONE = 0.01
DEFAULT_ACTION_RAW_LPF_COEF = 0.90
DEFAULT_ACTION_BOUND = 1.0
DEFAULT_JOINT_LIMIT_EPSILON = 0.02
# 6DOF 课程：episode 增加时阈值从 start → end 线性收紧（end 更严 = 更小）
DEFAULT_DIST_SUCCESS_START = 0.30  # m
DEFAULT_DIST_SUCCESS_END = 0.12  # m（更紧位置；原 0.18）
DEFAULT_ORI_SUCCESS_START = 2.0  # rad，初期易
DEFAULT_ORI_SUCCESS_END = 0.85  # rad（更紧姿态；原 1.3）
DEFAULT_CURRICULUM_EPISODES = 8000  # 拉长过渡，慢慢对齐
DEFAULT_TOUCH_STOP = False  # False：触碰不冻结，利于 6DOF 成功
# both：双臂同时按策略动；nearest_active：方块离谁近谁动，另一侧保持 home
DEFAULT_DUAL_ARM_MODE = "nearest_active"
# 腕部（wrist_1/2/3 连杆）与桌面 worktable_top 接触惩罚（每决策步至多扣一次）；0 关闭
DEFAULT_WRIST_TABLE_PENALTY_SCALE = 10.0
DEFAULT_LOG_DIR = "./logs/ppo_ur_task1_run_v4/"
DEFAULT_MODEL_PATH = "./models/ppo_ur_task1_run_v4_model.zip"


def _quat_dot(q1_wxyz: np.ndarray, q2_wxyz: np.ndarray) -> float:
    # Quaternions are (w, x, y, z)
    return float(np.dot(q1_wxyz, q2_wxyz))


def _quat_angle(q1_wxyz: np.ndarray, q2_wxyz: np.ndarray) -> float:
    # Small helper to compute the rotation angle between two quaternions.
    # angle = 2 * arccos(|dot(q1,q2)|)
    q1 = q1_wxyz / (np.linalg.norm(q1_wxyz) + 1e-12)
    q2 = q2_wxyz / (np.linalg.norm(q2_wxyz) + 1e-12)
    d = abs(_quat_dot(q1, q2))
    d = float(np.clip(d, -1.0, 1.0))
    return float(2.0 * np.arccos(d))


class URDualArmTask1Env(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        stage: str = "static",
        render: bool = False,
        frame_skip: int = 5,
        episode_len: int = 256,
        seed: int = 0,
        single_arm: str = "none",
        fixed_target: bool = False,
        success_bonus: float = 100.0,
        collision_penalty_scale: float = 0.0,
        wrist_table_penalty_scale: float = DEFAULT_WRIST_TABLE_PENALTY_SCALE,
        action_penalty_scale: float = 0.0,
        max_delta_ctrl: float = DEFAULT_MAX_DELTA_CTRL,
        joint_limit_epsilon: float = DEFAULT_JOINT_LIMIT_EPSILON,
        action_smoothing_coef: float = DEFAULT_ACTION_SMOOTHING_COEF,
        action_deadzone: float = DEFAULT_ACTION_DEADZONE,
        action_raw_lpf_coef: float = DEFAULT_ACTION_RAW_LPF_COEF,
        action_bound: float = DEFAULT_ACTION_BOUND,
        curriculum_episodes: int = DEFAULT_CURRICULUM_EPISODES,
        dist_success_start: float = DEFAULT_DIST_SUCCESS_START,
        dist_success_end: float = DEFAULT_DIST_SUCCESS_END,
        ori_success_start: float = DEFAULT_ORI_SUCCESS_START,
        ori_success_end: float = DEFAULT_ORI_SUCCESS_END,
        touch_stop: bool = DEFAULT_TOUCH_STOP,
        dual_arm_mode: str = DEFAULT_DUAL_ARM_MODE,
    ):
        super().__init__()
        self.stage = stage
        self.render_mode = render
        self.frame_skip = int(frame_skip)
        self.episode_len = int(episode_len)
        self.np_random = np.random.default_rng(seed)

        if single_arm not in {"none", "left", "right"}:
            raise ValueError("single_arm must be one of: none, left, right")
        if dual_arm_mode not in ("both", "nearest_active"):
            raise ValueError("dual_arm_mode must be 'both' or 'nearest_active'")
        self.single_arm = single_arm
        self.dual_arm_mode = dual_arm_mode
        self.fixed_target = bool(fixed_target)
        self.success_bonus = float(success_bonus)
        self.collision_penalty_scale = float(collision_penalty_scale)
        self.wrist_table_penalty_scale = float(wrist_table_penalty_scale)
        self.action_penalty_scale = float(action_penalty_scale)

        script_dir = os.path.dirname(os.path.realpath(__file__))
        xml_path = os.path.join(
            script_dir, "..", "mujoco_asserts", "universal_robots_ur5e", "scene_dual_arm.xml"
        )

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Left/right end-effector sites
        self.left_site_name = "left_tool0_site"
        self.right_site_name = "right_tool0_site"
        self.left_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, self.left_site_name
        )
        self.right_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, self.right_site_name
        )
        if self.left_site_id < 0 or self.right_site_id < 0:
            raise ValueError("Tool sites not found in UR model.")

        # Target object (freejoint body)
        self.obj_body_name = "target_object"
        self.obj_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.obj_body_name
        )
        if self.obj_body_id < 0:
            raise ValueError("target_object body not found in scene.")

        # Get freejoint qpos/qvel addresses for the target object
        self.obj_jnt_id = int(self.model.body_jntadr[self.obj_body_id])
        if self.obj_jnt_id < 0:
            raise ValueError("target_object freejoint not found.")
        self.obj_qposadr = int(self.model.jnt_qposadr[self.obj_jnt_id])
        self.obj_qveladr = int(self.model.jnt_dofadr[self.obj_jnt_id])

        self.target_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "target_object_geom"
        )
        self.table_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "worktable_top"
        )
        # Base mesh collision geometry (for avoiding/eliminating interpenetration).
        self.base_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "base_col"
        )

        # Cache robot collision geom ids by body name prefix.
        # (Most robot collision geoms live in bodies named left_*/right_*.)
        self.robot_geom_ids = set()
        for gid in range(int(self.model.ngeom)):
            bodyid = int(self.model.geom_bodyid[gid])
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bodyid)
            if body_name and (body_name.startswith("left_") or body_name.startswith("right_")):
                self.robot_geom_ids.add(int(gid))

        # Wrist link geoms (wrist_1 / wrist_2 / wrist_3) — penalize contact with worktable_top
        self.wrist_geom_ids = set()
        for gid in range(int(self.model.ngeom)):
            bodyid = int(self.model.geom_bodyid[gid])
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bodyid)
            if body_name and (
                "_wrist_1_link" in body_name
                or "_wrist_2_link" in body_name
                or "_wrist_3_link" in body_name
            ):
                self.wrist_geom_ids.add(int(gid))

        # Joint names (12 DOF)
        self.joint_names = [
            "left_shoulder_pan_joint",
            "left_shoulder_lift_joint",
            "left_elbow_joint",
            "left_wrist_1_joint",
            "left_wrist_2_joint",
            "left_wrist_3_joint",
            "right_shoulder_pan_joint",
            "right_shoulder_lift_joint",
            "right_elbow_joint",
            "right_wrist_1_joint",
            "right_wrist_2_joint",
            "right_wrist_3_joint",
        ]

        # Actuator names match joint names in dual_ur5e.xml
        self.actuator_names = list(self.joint_names)
        self.actuator_ids = []
        for name in self.actuator_names:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid < 0:
                raise ValueError(f"Actuator '{name}' not found.")
            self.actuator_ids.append(int(aid))

        # Build joint limits (for clipping ctrl); policy action lives in [-action_bound, action_bound].
        limits = []
        # Cache joint qpos/qvel addresses for proprioception in the observation.
        # (Do not assume qpos/qvel ordering; MuJoCo layout includes the object's freejoint.)
        self.robot_qpos_adrs = []
        self.robot_qvel_adrs = []
        for jname in self.joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                raise ValueError(f"Joint '{jname}' not found.")
            self.robot_qpos_adrs.append(int(self.model.jnt_qposadr[jid]))
            self.robot_qvel_adrs.append(int(self.model.jnt_dofadr[jid]))
            lo, hi = self.model.jnt_range[jid, 0], self.model.jnt_range[jid, 1]
            if not bool(self.model.jnt_limited[jid]):
                lo, hi = -np.pi, np.pi
            limits.append((float(lo), float(hi)))
        self.joint_limits = np.asarray(limits, dtype=np.float32)  # (12, 2)
        self.robot_qpos_adrs = np.asarray(self.robot_qpos_adrs, dtype=np.int64)  # (12,)
        self.robot_qvel_adrs = np.asarray(self.robot_qvel_adrs, dtype=np.int64)  # (12,)

        # Action: normalized delta-joint command in [-action_bound, action_bound] (default 1.0).
        # Smaller bounds (e.g. 0.5) shrink the policy's output range and exploration near home.
        if action_bound <= 0.0:
            raise ValueError("action_bound must be positive.")
        self.action_bound = float(action_bound)
        self.action_space = spaces.Box(
            low=-self.action_bound,
            high=self.action_bound,
            shape=(12,),
            dtype=np.float32,
        )

        # Incremental control rate limit: action is scaled to delta-radians per step.
        self.max_delta_ctrl = float(max_delta_ctrl)
        # If a joint is already near a limit, suppress delta that would push it further.
        self.joint_limit_epsilon = float(joint_limit_epsilon)
        # Exponential smoothing for delta_q to reduce high-frequency chatter.
        # 0.0 = no smoothing (original behavior).
        # 1.0 = fully rely on previous delta_q.
        self.action_smoothing_coef = float(action_smoothing_coef)
        # |a| < deadzone -> 0 (suppress noisy policy output near zero).
        self.action_deadzone = float(action_deadzone)
        # Low-pass filter on *raw* action a before delta_q = a * max_delta_ctrl
        # (stronger against +/-1 sign flips than smoothing delta_q alone).
        self.action_raw_lpf_coef = float(action_raw_lpf_coef)
        self.prev_ctrl = np.zeros(12, dtype=np.float32)
        self.prev_a_filtered = np.zeros(12, dtype=np.float32)
        # In single-arm debug modes we treat the other arm as "truly fixed".
        # Position actuators are compliant under contact, so "fixed ctrl target"
        # can still drift. We will hard-hold the fixed arm's qpos/qvel during rollout.
        self.fixed_hold_qpos = None
        self.fixed_hold_qvel = None

        # Touch-stop: once the robot makes contact with the target object,
        # we hard-freeze the robot to prevent post-contact bouncing/overshoot.
        # If True while training RL, first contact before full 6DOF success can permanently
        # prevent reaching success — use touch_stop=False during training.
        self.touch_stop_enabled = bool(touch_stop)
        self.has_touched_object = False
        # Split home qpos for nearest_active dual-arm mode (set in reset()).
        self.home_left_qpos = None
        self.home_right_qpos = None
        self.touch_hold_qpos = None
        # Delta smoothing memory (for low-pass filtering).
        self.prev_delta_q = np.zeros(12, dtype=np.float32)

        # Observation:
        # robot_qpos(12) + robot_qvel(12) + lpos(3)+lquat(4) + rpos(3)+rquat(4) +
        # rel_lpos(3)+rel_rpos(3) + object_pos(3)+object_quat(4)+object_vel(6)
        obs_dim = 57
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Render (optional)
        self.handle = None
        if self.render_mode:
            import mujoco.viewer as mj_viewer

            self.handle = mj_viewer.launch_passive(self.model, self.data)
            self.handle.cam.distance = 3
            self.handle.cam.azimuth = 0
            self.handle.cam.elevation = -30

        self.step_number = 0
        self.goal_reached = False
        self.success_streak = 0
        self.episode_count = 0

        # keyframe id (mj_resetDataKeyframe needs an integer id)
        self.home_key_id = None
        for k in range(int(self.model.nkey)):
            if self.model.key(k).name == "home":
                self.home_key_id = int(k)
                break
        if self.home_key_id is None:
            raise ValueError("Keyframe 'home' not found in scene_dual_arm.xml.")

        # Table/object placement (world coordinates)
        # x/y is taken from geom_xpos[worktable_top] at reset time to guarantee
        # the object is placed at the current table center.
        self.table_center_xy = np.array([0.0, 0.0], dtype=np.float32)
        self.table_half_extents_xy = np.array([0.15, 0.30], dtype=np.float32)
        # Box object half-extents: (x, y, z) in meters
        self.obj_half_extents = np.array([0.02, 0.02, 0.02], dtype=np.float32)
        self.table_top_z = 0.73  # legacy (world z is read from geom at reset)
        self.obj_clearance = 0.016  # small lift above the table surface

        # Curriculum + success gating (6DOF: distance + orientation both must satisfy).
        if dist_success_end <= 0.0 or dist_success_end > dist_success_start:
            raise ValueError("Need 0 < dist_success_end <= dist_success_start (tighter = smaller).")
        if ori_success_end <= 0.0 or ori_success_end >= ori_success_start:
            raise ValueError("Need 0 < ori_success_end < ori_success_start (tighter = smaller angle).")
        self.dist_success_start = float(dist_success_start)
        self.dist_success_end = float(dist_success_end)
        self.ori_success_start = float(ori_success_start)
        self.ori_success_end = float(ori_success_end)
        # Episodes over which success thresholds linearly tighten from *_start to *_end.
        self.curriculum_episodes = max(1, int(curriculum_episodes))
        # Match troubleshooting setting: only 1-step hold for success.
        # This makes success signals more frequent and helps the policy learn.
        self.success_hold_steps = 1

        # Small x/y randomization around worktable top center (no table height change).
        self.xy_randomize = True
        self.xy_rand_max = np.array([0.06, 0.08], dtype=np.float32)  # max offset (m)
        self.xy_safe_margin = 0.04  # keep sphere away from edges (m)

    def _current_success_thresholds(self) -> Tuple[float, float]:
        """Return (dist_success, ori_success) for current curriculum episode."""
        t = float(min(1.0, self.episode_count / max(1, self.curriculum_episodes)))
        dist_success = float(self.dist_success_start + t * (self.dist_success_end - self.dist_success_start))
        ori_success = float(self.ori_success_start + t * (self.ori_success_end - self.ori_success_start))
        return dist_success, ori_success

    def _get_site_pos_quat(self, site_id: int) -> Tuple[np.ndarray, np.ndarray]:
        pos = np.array(self.data.site(site_id).xpos, dtype=np.float32)
        # MuJoCo helper expects float64 arrays
        xmat = np.array(self.data.site(site_id).xmat, dtype=np.float64)  # 9 elems
        quat64 = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat64, xmat)  # returns (w, x, y, z)
        return pos, quat64.astype(np.float32)

    def _get_object_pos_quat_vel(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        body = self.data.body(self.obj_body_id)
        obj_pos = np.array(body.xpos, dtype=np.float32)
        obj_quat = np.array(body.xquat, dtype=np.float32)  # (w,x,y,z)
        # cvel: [vx,vy,vz, wx,wy,wz]
        cvel = np.array(body.cvel, dtype=np.float32)
        obj_linvel = cvel[:3]
        obj_angvel = cvel[3:6]
        return obj_pos, obj_quat, obj_linvel, obj_angvel

    def _map_action_to_ctrl(self, action: np.ndarray) -> np.ndarray:
        b = float(self.action_bound)
        a = np.clip(np.asarray(action, dtype=np.float32), -b, b)
        normalized = (a + b) / (2.0 * b)  # [-b,b] -> [0,1]
        lo = self.joint_limits[:, 0]
        hi = self.joint_limits[:, 1]
        ctrl = lo + normalized * (hi - lo)
        return ctrl.astype(np.float32)

    def set_curriculum_stage(self, single_arm: str, fixed_target: bool):
        """动态调整课程难度（由 CurriculumCallback 通过 VecEnv.env_method 调用）。"""
        if single_arm not in {"none", "left", "right"}:
            raise ValueError("single_arm must be one of: none, left, right")
        self.single_arm = single_arm
        self.fixed_target = bool(fixed_target)
        print(
            f"\n[环境难度调整] 控制臂: {self.single_arm}, 目标固定: {self.fixed_target}\n",
            flush=True,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.episode_count += 1
        self.success_streak = 0
        self.dist_success_current, self.ori_success_current = self._current_success_thresholds()

        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key_id)
        # Update derived quantities (e.g., geom_xpos) before reading them
        mujoco.mj_forward(self.model, self.data)
        self.step_number = 0
        self.goal_reached = False

        # Place object at the current worktable_top center (x/y).
        table_top_world_xy = np.array(self.data.geom_xpos[self.table_geom_id][:2], dtype=np.float32)
        table_top_world_z = float(self.data.geom_xpos[self.table_geom_id][2])
        obj_z = table_top_world_z + float(self.obj_half_extents[2]) + float(self.obj_clearance)

        obj_x = float(table_top_world_xy[0])
        obj_y = float(table_top_world_xy[1])

        yaw = 0.0 if self.fixed_target else float(self.np_random.uniform(-np.pi, np.pi))

        # For random x/y, avoid spawning unreachable targets by using a simple
        # distance-from-home reachability rejection sampler.
        if self.xy_randomize and not self.fixed_target:
            # Sample x/y within the real worktable_top collision box.
            # (model.geom_size for box are half-extents)
            table_half = np.array(self.model.geom_size[self.table_geom_id][:2], dtype=np.float32)
            avail = np.maximum(table_half - self.xy_safe_margin, 0.0)
            max_off = avail

            # Reachability threshold (meters): related to the current curriculum distance threshold.
            reach_dist = float(self.dist_success_current) * 1.6
            reach_dist = float(np.clip(reach_dist, 0.2, 0.6))

            max_attempts = 25
            accepted = False
            for _ in range(max_attempts):
                cand_x = obj_x + float(self.np_random.uniform(-max_off[0], max_off[0]))
                cand_y = obj_y + float(self.np_random.uniform(-max_off[1], max_off[1]))
                cand_yaw = float(self.np_random.uniform(-np.pi, np.pi))

                # Candidate placement.
                self.data.qpos[self.obj_qposadr : self.obj_qposadr + 3] = np.array(
                    [cand_x, cand_y, obj_z], dtype=np.float32
                )
                self.data.qpos[self.obj_qposadr + 3 : self.obj_qposadr + 7] = np.array(
                    [0.0, np.cos(cand_yaw / 2.0), np.sin(cand_yaw / 2.0), 0.0], dtype=np.float32
                )

                mujoco.mj_forward(self.model, self.data)

                opos = np.array(self.data.body(self.obj_body_id).xpos, dtype=np.float32)
                lpos = np.array(self.data.site(self.left_site_id).xpos, dtype=np.float32)
                rpos = np.array(self.data.site(self.right_site_id).xpos, dtype=np.float32)
                dL = float(np.linalg.norm(lpos - opos))
                dR = float(np.linalg.norm(rpos - opos))

                if self.single_arm == "left":
                    ok = dL < reach_dist
                elif self.single_arm == "right":
                    ok = dR < reach_dist
                else:
                    ok = min(dL, dR) < reach_dist

                if ok:
                    obj_x = cand_x
                    obj_y = cand_y
                    yaw = cand_yaw
                    accepted = True
                    break

            # If not accepted, keep the last sampled obj_x/obj_y/yaw.
            if not accepted:
                # Ensure qpos/yaw reflect the last attempt.
                self.data.qpos[self.obj_qposadr : self.obj_qposadr + 3] = np.array(
                    [obj_x, obj_y, obj_z], dtype=np.float32
                )

        # freejoint qpos: (x,y,z, qw,qx,qy,qz)
        self.data.qpos[self.obj_qposadr : self.obj_qposadr + 3] = np.array(
            [obj_x, obj_y, obj_z], dtype=np.float32
        )
        # Requirement: local +Z should always point toward the ground (world -Z).
        # Keep +Z down by applying a fixed 180deg flip about X, then randomize yaw around world Z.
        # With MuJoCo qpos order [w, x, y, z], resulting quaternion is:
        #   q = [0, cos(yaw/2), sin(yaw/2), 0]
        self.data.qpos[self.obj_qposadr + 3 : self.obj_qposadr + 7] = np.array(
            [0.0, np.cos(yaw / 2.0), np.sin(yaw / 2.0), 0.0], dtype=np.float32
        )

        # Velocities (world frame)
        if self.fixed_target or self.stage == "static":
            lin_v = np.zeros(3, dtype=np.float32)
            ang_v = np.zeros(3, dtype=np.float32)
        elif self.stage == "slow_dynamic":
            speed = 0.05  # m/s (课程笔记建议量级)
            theta = self.np_random.uniform(0.0, 2.0 * np.pi)
            lin_v = np.array([speed * np.cos(theta), speed * np.sin(theta), 0.0], dtype=np.float32)
            # small angular velocity
            ang_speed = 0.5  # rad/s (tunable)
            axis = self.np_random.normal(size=3).astype(np.float32)
            axis = axis / (np.linalg.norm(axis) + 1e-12)
            ang_v = ang_speed * axis
        else:
            # fallback: mild dynamics
            speed = 0.02
            theta = self.np_random.uniform(0.0, 2.0 * np.pi)
            lin_v = np.array([speed * np.cos(theta), speed * np.sin(theta), 0.0], dtype=np.float32)
            ang_v = np.zeros(3, dtype=np.float32)

        self.data.qvel[self.obj_qveladr : self.obj_qveladr + 3] = lin_v
        self.data.qvel[self.obj_qveladr + 3 : self.obj_qveladr + 6] = ang_v

        mujoco.mj_forward(self.model, self.data)

        # Initialize previous ctrl to the current joint positions at `home`.
        # This reduces initial control mismatch (data.ctrl may not match qpos).
        robot_qpos = np.asarray(self.data.qpos[self.robot_qpos_adrs], dtype=np.float32)
        robot_qvel = np.asarray(self.data.qvel[self.robot_qvel_adrs], dtype=np.float32)
        self.prev_ctrl = robot_qpos.copy()
        self.fixed_hold_qpos = robot_qpos.copy()
        self.fixed_hold_qvel = robot_qvel.copy()
        self.home_left_qpos = robot_qpos[0:6].astype(np.float32, copy=True)
        self.home_right_qpos = robot_qpos[6:12].astype(np.float32, copy=True)
        for i, aid in enumerate(self.actuator_ids):
            self.data.ctrl[aid] = float(robot_qpos[i])

        mujoco.mj_forward(self.model, self.data)

        obs = self._get_observation()

        # Reset touch-stop state for this episode.
        self.has_touched_object = False
        self.touch_hold_qpos = None
        self.prev_delta_q = np.zeros(12, dtype=np.float32)
        self.prev_a_filtered[:] = 0.0
        return obs, {}

    def _get_observation(self) -> np.ndarray:
        lpos, lquat = self._get_site_pos_quat(self.left_site_id)
        rpos, rquat = self._get_site_pos_quat(self.right_site_id)
        opos, oquat, olinvel, oangvel = self._get_object_pos_quat_vel()

        # Proprioception: robot joint states (qpos/qvel) for learning kinematics.
        robot_qpos = np.asarray(self.data.qpos[self.robot_qpos_adrs], dtype=np.float32)  # (12,)
        robot_qvel = np.asarray(self.data.qvel[self.robot_qvel_adrs], dtype=np.float32)  # (12,)

        # Relative object position w.r.t. each end-effector.
        rel_lpos = (opos - lpos).astype(np.float32)
        rel_rpos = (opos - rpos).astype(np.float32)

        obs = np.concatenate(
            [
                robot_qpos,
                robot_qvel,
                lpos,
                lquat,
                rpos,
                rquat,
                rel_lpos,
                rel_rpos,
                opos,
                oquat,
                olinvel,
                oangvel,
            ]
        ).astype(np.float32)
        return obs

    def _apply_fixed_arm_hold(self) -> None:
        """Hard-hold the non-controlled arm in single-arm debug modes."""
        if self.single_arm not in {"left", "right"}:
            return
        if self.fixed_hold_qpos is None or self.fixed_hold_qvel is None:
            return

        if self.single_arm == "left":
            j_from, j_to = 6, 12  # keep right arm fixed
        else:
            j_from, j_to = 0, 6  # keep left arm fixed

        qpos_ids = self.robot_qpos_adrs[j_from:j_to]
        qvel_ids = self.robot_qvel_adrs[j_from:j_to]
        self.data.qpos[qpos_ids] = self.fixed_hold_qpos[j_from:j_to]
        self.data.qvel[qvel_ids] = 0.0

        # Refresh fixed-arm ctrl so the position servo keeps the same target.
        for i in range(j_from, j_to):
            aid = self.actuator_ids[i]
            self.data.ctrl[aid] = float(self.fixed_hold_qpos[i])

    def _apply_nearest_inactive_arm_home_hold(self) -> None:
        """双臂 nearest_active：离方块较远的一侧保持 keyframe home（硬约束）。"""
        if self.single_arm != "none":
            return
        if self.dual_arm_mode != "nearest_active":
            return
        if self.home_left_qpos is None or self.home_right_qpos is None:
            return
        lpos, _ = self._get_site_pos_quat(self.left_site_id)
        rpos, _ = self._get_site_pos_quat(self.right_site_id)
        opos, _, _, _ = self._get_object_pos_quat_vel()
        dL = float(np.linalg.norm(lpos - opos))
        dR = float(np.linalg.norm(rpos - opos))
        if dL <= dR:
            j_from, j_to = 6, 12
            hold = self.home_right_qpos
        else:
            j_from, j_to = 0, 6
            hold = self.home_left_qpos
        qpos_ids = self.robot_qpos_adrs[j_from:j_to]
        qvel_ids = self.robot_qvel_adrs[j_from:j_to]
        self.data.qpos[qpos_ids] = hold
        self.data.qvel[qvel_ids] = 0.0
        for i in range(j_from, j_to):
            self.data.ctrl[self.actuator_ids[i]] = float(hold[i - j_from])

    def _object_touched_by_robot(self) -> bool:
        """Return True when target_object geom is in contact with robot/base (not just the table)."""
        if self.target_geom_id < 0:
            return False
        if self.table_geom_id < 0:
            table_id = -1
        else:
            table_id = int(self.table_geom_id)

        for c in self.data.contact:
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 == self.target_geom_id and g2 != table_id:
                return True
            if g2 == self.target_geom_id and g1 != table_id:
                return True
        return False

    def _robot_self_or_base_collision(self) -> bool:
        """Detect collisions that usually cause whipping: robot-robot or robot-base_col.

        We explicitly ignore contacts involving the target object because those are handled
        by the touch-stop logic.
        """
        if self.base_geom_id < 0 and not self.robot_geom_ids:
            return False

        for c in self.data.contact:
            g1, g2 = int(c.geom1), int(c.geom2)

            # Ignore any contact involving the target object.
            if g1 == int(self.target_geom_id) or g2 == int(self.target_geom_id):
                continue

            # base_col collision
            if self.base_geom_id >= 0 and (g1 == int(self.base_geom_id) or g2 == int(self.base_geom_id)):
                return True

            # robot self collision (including left-right)
            if (g1 in self.robot_geom_ids) and (g2 in self.robot_geom_ids):
                return True

        return False

    def _is_robot_geom_contact_pair(self, g1: int, g2: int) -> bool:
        """True if contact is between two robot geoms and not involving the target."""
        if g1 in self.robot_geom_ids and g2 in self.robot_geom_ids:
            if g1 == int(self.target_geom_id) or g2 == int(self.target_geom_id):
                return False
            return True
        return False

    def _apply_touch_hold(self) -> None:
        """Freeze all robot joints at the stored touch pose."""
        if not self.touch_stop_enabled:
            return
        if not self.has_touched_object:
            return
        if self.touch_hold_qpos is None:
            return

        qpos_ids = self.robot_qpos_adrs
        qvel_ids = self.robot_qvel_adrs
        self.data.qpos[qpos_ids] = self.touch_hold_qpos
        self.data.qvel[qvel_ids] = 0.0

        for i, aid in enumerate(self.actuator_ids):
            self.data.ctrl[aid] = float(self.touch_hold_qpos[i])

    def _success_now(self) -> bool:
        """Check success criteria at the current simulator state."""
        lpos, lquat = self._get_site_pos_quat(self.left_site_id)
        rpos, rquat = self._get_site_pos_quat(self.right_site_id)
        opos, oquat, _, _ = self._get_object_pos_quat_vel()

        dL = float(np.linalg.norm(lpos - opos))
        dR = float(np.linalg.norm(rpos - opos))
        ori_angle_L = _quat_angle(lquat, oquat)
        ori_angle_R = _quat_angle(rquat, oquat)

        dist_success = float(self.dist_success_current)
        ori_success = float(self.ori_success_current)

        success_left = (dL < dist_success) and (ori_angle_L < ori_success)
        success_right = (dR < dist_success) and (ori_angle_R < ori_success)

        if self.single_arm == "left":
            return bool(success_left)
        if self.single_arm == "right":
            return bool(success_right)
        return bool(success_left or success_right)

    def _compute_reward_done(self) -> Tuple[float, bool]:
        lpos, lquat = self._get_site_pos_quat(self.left_site_id)
        rpos, rquat = self._get_site_pos_quat(self.right_site_id)
        opos, oquat, _, _ = self._get_object_pos_quat_vel()

        dL = float(np.linalg.norm(lpos - opos))
        dR = float(np.linalg.norm(rpos - opos))

        ori_angle_L = _quat_angle(lquat, oquat)
        ori_angle_R = _quat_angle(rquat, oquat)

        if self.single_arm == "left":
            d = dL
            ori_angle = ori_angle_L
        elif self.single_arm == "right":
            d = dR
            ori_angle = ori_angle_R
        else:
            # Soft-min distance: mainly reward the closer arm, while keeping gradients smooth.
            # softmin(dL, dR) = -(1/k) * log(0.5*exp(-k*dL) + 0.5*exp(-k*dR))
            # Larger k -> closer to hard min.
            k = 10.0
            x1 = -k * dL
            x2 = -k * dR
            m = float(max(x1, x2))
            # log-sum-exp for numerical stability.
            log_sum = np.log(0.5 * np.exp(x1 - m) + 0.5 * np.exp(x2 - m)) + m
            d = -(1.0 / k) * float(log_sum)

            # For ori_angle, use the same softmin weights (smoothly favor the closer arm).
            # wL = exp(-k*dL) / (exp(-k*dL) + exp(-k*dR))
            wL = float(np.exp(x1 - m) / (np.exp(x1 - m) + np.exp(x2 - m) + 1e-12))
            ori_angle = wL * ori_angle_L + (1.0 - wL) * ori_angle_R

        # Positive shaping: closer -> larger reward in (0, 1], far -> ~0 (no endless negative drift from distance).
        pos_reward = float(1.0 / (1.0 + 5.0 * d))

        # Orientation shaping
        ori_reward = -0.4 * np.arctan(float(ori_angle))

        # Collision penalty (off by default)
        collision_penalty = 0.0
        if self.collision_penalty_scale != 0.0:
            for c in self.data.contact:
                # 1) Original collision: table <-> target
                if self.table_geom_id >= 0 and self.target_geom_id >= 0:
                    if (c.geom1 == self.table_geom_id and c.geom2 == self.target_geom_id) or (
                        c.geom2 == self.table_geom_id and c.geom1 == self.target_geom_id
                    ):
                        collision_penalty -= float(self.collision_penalty_scale)
                        break

                # 2) Additional collision: robot <-> base mesh (base_col)
                # Penalize any contact involving base_col so the policy avoids interpenetration.
                if self.base_geom_id >= 0:
                    if c.geom1 == self.base_geom_id or c.geom2 == self.base_geom_id:
                        collision_penalty -= float(self.collision_penalty_scale)
                        break

                # 3) Additional collision: robot self collision (including left-right)
                # Penalize robot-robot contacts unless the target object is involved.
                if self._is_robot_geom_contact_pair(c.geom1, c.geom2):
                    collision_penalty -= float(self.collision_penalty_scale)
                    break

        # Wrist (three joints links) vs worktable_top — always discouraged when scale > 0
        wrist_table_penalty = 0.0
        if self.wrist_table_penalty_scale != 0.0 and self.table_geom_id >= 0 and self.wrist_geom_ids:
            tid = int(self.table_geom_id)
            for c in self.data.contact:
                g1, g2 = int(c.geom1), int(c.geom2)
                if g1 == tid and g2 in self.wrist_geom_ids:
                    wrist_table_penalty -= float(self.wrist_table_penalty_scale)
                    break
                if g2 == tid and g1 in self.wrist_geom_ids:
                    wrist_table_penalty -= float(self.wrist_table_penalty_scale)
                    break

        # Success criteria: "任一臂到 + 姿态角对齐"
        # Use OR logic to avoid "更近那只臂切换"导致的姿态判断不一致。
        dist_success = float(self.dist_success_current)
        ori_success = float(self.ori_success_current)

        success_left = (dL < dist_success) and (ori_angle_L < ori_success)
        success_right = (dR < dist_success) and (ori_angle_R < ori_success)
        if self.single_arm == "left":
            success_now = bool(success_left)
        elif self.single_arm == "right":
            success_now = bool(success_right)
        else:
            success_now = bool(success_left or success_right)

        # Terminal bonus is added in `step()` when the success streak reaches `success_hold_steps`.
        success_bonus_now = self.success_bonus if success_now else 0.0
        reward = pos_reward + ori_reward + collision_penalty + wrist_table_penalty + success_bonus_now
        return float(reward), bool(success_now)

    def step(self, action):
        # Touch-stop: if we already touched the object, keep ctrl frozen.
        if self.touch_stop_enabled and self.has_touched_object:
            ctrl_target = self.prev_ctrl
        else:
            # Incremental delta control: a in [-action_bound, action_bound] -> delta joint radians.
            a = np.clip(
                np.asarray(action, dtype=np.float32),
                -self.action_bound,
                self.action_bound,
            )  # (12,)
            if self.action_deadzone > 0.0:
                a = np.where(np.abs(a) < self.action_deadzone, 0.0, a)
            if self.action_raw_lpf_coef > 0.0:
                a = (1.0 - self.action_raw_lpf_coef) * a + self.action_raw_lpf_coef * self.prev_a_filtered
            self.prev_a_filtered = a.astype(np.float32, copy=True)

            delta_q = a * self.max_delta_ctrl  # (12,)

            # Boundary suppression to reduce high-frequency oscillation at joint limits.
            q_now = np.asarray(self.data.qpos[self.robot_qpos_adrs], dtype=np.float32)
            lo = self.joint_limits[:, 0]
            hi = self.joint_limits[:, 1]
            at_hi = q_now >= (hi - self.joint_limit_epsilon)
            at_lo = q_now <= (lo + self.joint_limit_epsilon)
            delta_q = np.where(at_hi & (delta_q > 0.0), 0.0, delta_q)
            delta_q = np.where(at_lo & (delta_q < 0.0), 0.0, delta_q)

            # Low-pass filter delta_q to reduce high-frequency chatter.
            if self.action_smoothing_coef != 0.0:
                delta_q = (1.0 - self.action_smoothing_coef) * delta_q + self.action_smoothing_coef * self.prev_delta_q
            self.prev_delta_q = delta_q.astype(np.float32, copy=True)
            ctrl_target = self.prev_ctrl + delta_q

        # Single-arm mode: keep the other arm fixed.
        if self.single_arm == "left":
            ctrl_target[6:] = self.prev_ctrl[6:]
        elif self.single_arm == "right":
            ctrl_target[:6] = self.prev_ctrl[:6]

        # Dual-arm nearest_active: only the arm closer to the object tracks policy; other -> home.
        if self.single_arm == "none" and self.dual_arm_mode == "nearest_active":
            lpos, _ = self._get_site_pos_quat(self.left_site_id)
            rpos, _ = self._get_site_pos_quat(self.right_site_id)
            opos, _, _, _ = self._get_object_pos_quat_vel()
            dL = float(np.linalg.norm(lpos - opos))
            dR = float(np.linalg.norm(rpos - opos))
            if dL <= dR:
                ctrl_target[6:12] = self.home_right_qpos
                self.prev_delta_q[6:12] = 0.0
                self.prev_a_filtered[6:12] = 0.0
            else:
                ctrl_target[0:6] = self.home_left_qpos
                self.prev_delta_q[0:6] = 0.0
                self.prev_a_filtered[0:6] = 0.0

        # Clip to joint limits for safety.
        lo = self.joint_limits[:, 0]
        hi = self.joint_limits[:, 1]
        ctrl = np.clip(ctrl_target, lo, hi).astype(np.float32)
        self.prev_ctrl = ctrl.copy()

        for i, aid in enumerate(self.actuator_ids):
            self.data.ctrl[aid] = float(ctrl[i])

        unstable = False
        # Enforce hard hold on the fixed arm before stepping physics.
        self._apply_fixed_arm_hold()
        self._apply_nearest_inactive_arm_home_hold()
        self._apply_touch_hold()
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            if self.handle is not None:
                self.handle.sync()
            # Re-apply hold after physics integration to prevent drift.
            self._apply_fixed_arm_hold()
            self._apply_nearest_inactive_arm_home_hold()
            self._apply_touch_hold()

            # First touch detection: stop immediately to avoid post-contact bouncing.
            if (
                self.touch_stop_enabled
                and (not self.has_touched_object)
                and self._object_touched_by_robot()
            ):
                self.has_touched_object = True
                self.touch_hold_qpos = np.asarray(
                    self.data.qpos[self.robot_qpos_adrs], dtype=np.float32
                ).copy()
                self._apply_touch_hold()
                break

            # If we haven't touched the object yet, but the robot hits itself or the base,
            # stop integrating further in this decision step to avoid visible whipping.
            if (not self.has_touched_object) and self._robot_self_or_base_collision():
                break
            # Early stop within the same decision step:
            # once success is achieved (when hold=1), we stop integrating further
            # to prevent post-contact "bouncing" / overshoot.
            if self.success_hold_steps == 1 and self._success_now():
                break
            if not np.isfinite(self.data.qacc).all():
                unstable = True
                break

        self.step_number += 1
        obs = self._get_observation()

        # Stop the episode early if MuJoCo becomes numerically unstable.
        if unstable or (not np.isfinite(obs).all()) or (not np.isfinite(self.data.qacc).all()):
            return obs, -1000.0, True, True, {"is_success": False, "unstable": True}

        reward, success_now = self._compute_reward_done()
        self.goal_reached = success_now

        if success_now:
            self.success_streak += 1
        else:
            self.success_streak = 0

        done = self.success_streak >= self.success_hold_steps
        truncated = self.step_number >= self.episode_len

        if done:
            reward += 20.0

        # Optional action penalty: disabled by default to make exploration easier.
        if self.action_penalty_scale != 0.0:
            a = np.clip(
                np.asarray(action, dtype=np.float32),
                -self.action_bound,
                self.action_bound,
            )
            reward += -float(self.action_penalty_scale) * float(np.sum(np.square(a), dtype=np.float32))

        info = {"is_success": bool(done), "success_now": bool(success_now)}
        return obs, reward, done, truncated, info

    def close(self):
        if self.handle is not None:
            try:
                self.handle.close()
            except Exception:
                pass
            self.handle = None


def main():
    parser = argparse.ArgumentParser(
        description="PPO training for Task1 (defaults match project v3; see module constants).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--stage", type=str, default="static", choices=["static", "slow_dynamic"])
    parser.add_argument("--render", action="store_true", help="MuJoCo viewer (requires --n_env 1)")
    parser.add_argument("--n_env", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--total_timesteps", type=int, default=2_000_000)
    parser.add_argument("--n_steps", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--lr_final", type=float, default=1e-5)
    parser.add_argument("--use_lr_schedule", action="store_true", default=True)

    parser.add_argument("--single_arm", type=str, default="none", choices=["none", "left", "right"])
    parser.add_argument("--fixed_target", action="store_true", help="Fixed cube pose (debug)")
    parser.add_argument("--success_bonus", type=float, default=100.0)
    parser.add_argument("--collision_penalty_scale", type=float, default=0.0)
    parser.add_argument(
        "--wrist_table_penalty_scale",
        type=float,
        default=DEFAULT_WRIST_TABLE_PENALTY_SCALE,
        help="Penalty per step if wrist_1/2/3 link geoms contact worktable_top. 0 disables.",
    )
    parser.add_argument("--action_penalty_scale", type=float, default=0.0)

    # Env (defaults = module DEFAULT_* ; 一般不必改)
    parser.add_argument("--max_delta_ctrl", type=float, default=DEFAULT_MAX_DELTA_CTRL)
    parser.add_argument("--joint_limit_epsilon", type=float, default=DEFAULT_JOINT_LIMIT_EPSILON)
    parser.add_argument("--action_smoothing_coef", type=float, default=DEFAULT_ACTION_SMOOTHING_COEF)
    parser.add_argument("--action_deadzone", type=float, default=DEFAULT_ACTION_DEADZONE)
    parser.add_argument("--action_raw_lpf_coef", type=float, default=DEFAULT_ACTION_RAW_LPF_COEF)
    parser.add_argument("--action_bound", type=float, default=DEFAULT_ACTION_BOUND)
    parser.add_argument("--curriculum_episodes", type=int, default=DEFAULT_CURRICULUM_EPISODES)
    parser.add_argument("--dist_success_start", type=float, default=DEFAULT_DIST_SUCCESS_START)
    parser.add_argument("--dist_success_end", type=float, default=DEFAULT_DIST_SUCCESS_END)
    parser.add_argument("--ori_success_start", type=float, default=DEFAULT_ORI_SUCCESS_START)
    parser.add_argument("--ori_success_end", type=float, default=DEFAULT_ORI_SUCCESS_END)
    parser.add_argument(
        "--dual_arm_mode",
        type=str,
        default=DEFAULT_DUAL_ARM_MODE,
        choices=["both", "nearest_active"],
        help="nearest_active: only arm closer to cube moves; other held at home. Ignored if --single_arm left|right.",
    )
    parser.add_argument(
        "--touch_stop",
        action="store_true",
        help="Freeze robot on first contact with cube (visual demo; default off for RL).",
    )
    parser.add_argument(
        "--no_touch_stop",
        action="store_true",
        help=argparse.SUPPRESS,
    )  # deprecated: default is already no freeze; kept so old shell scripts still parse

    parser.add_argument("--log_dir", type=str, default=DEFAULT_LOG_DIR)
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)

    # 自动课程：Stage0 左臂+固定目标 -> Stage1 左臂+随机 -> Stage2 双臂+随机（成功率达标后换挡）
    parser.add_argument(
        "--auto_curriculum",
        action="store_true",
        help="启用自动课程 Callback（初始强制左臂+固定目标，达标后依次解锁）",
    )
    parser.add_argument(
        "--auto_curriculum_threshold",
        type=float,
        default=0.85,
        help="近期成功率 >= 该值则晋升下一阶段（仅 --auto_curriculum 时有效）",
    )
    parser.add_argument(
        "--auto_curriculum_window",
        type=int,
        default=100,
        help="统计成功率的最近回合数窗口（仅 --auto_curriculum 时有效）",
    )
    args = parser.parse_args()
    if args.no_touch_stop:
        warnings.warn("--no_touch_stop is ignored: no freeze-on-contact is already the default.", UserWarning)

    if args.auto_curriculum:
        args.single_arm = "left"
        args.fixed_target = True
        print(
            "[auto_curriculum] 初始 Stage0: single_arm=left, fixed_target=True；"
            "达标后将自动进入左臂随机目标、再进入双臂。",
            flush=True,
        )

    if args.render and args.n_env != 1:
        raise ValueError("Rendering only supported with --n_env=1.")

    def make_env():
        return URDualArmTask1Env(
            stage=args.stage,
            render=args.render,
            frame_skip=5,
            episode_len=256,
            seed=args.seed,
            single_arm=args.single_arm,
            fixed_target=args.fixed_target,
            success_bonus=args.success_bonus,
            collision_penalty_scale=args.collision_penalty_scale,
            wrist_table_penalty_scale=args.wrist_table_penalty_scale,
            action_penalty_scale=args.action_penalty_scale,
            max_delta_ctrl=args.max_delta_ctrl,
            joint_limit_epsilon=args.joint_limit_epsilon,
            action_smoothing_coef=args.action_smoothing_coef,
            action_deadzone=args.action_deadzone,
            action_raw_lpf_coef=args.action_raw_lpf_coef,
            action_bound=args.action_bound,
            curriculum_episodes=args.curriculum_episodes,
            dist_success_start=args.dist_success_start,
            dist_success_end=args.dist_success_end,
            ori_success_start=args.ori_success_start,
            ori_success_end=args.ori_success_end,
            touch_stop=bool(args.touch_stop),
            dual_arm_mode=args.dual_arm_mode,
        )

    env = make_vec_env(make_env, n_envs=args.n_env)
    # Reward normalization can destabilize training when the success bonus is large
    # and the reward distribution is highly non-stationary.
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    policy_kwargs = dict(
        activation_fn=nn.ReLU,
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=0.99,
        learning_rate=(
            (lambda progress_remaining: args.lr_final + progress_remaining * (args.learning_rate - args.lr_final))
            if args.use_lr_schedule
            else args.learning_rate
        ),
        device="cuda" if __import__("torch").cuda.is_available() else "cpu",
        tensorboard_log=args.log_dir,
        seed=args.seed,
    )

    learn_callbacks = None
    if args.auto_curriculum:
        auto_stages = [
            {"single_arm": "left", "fixed_target": True},
            {"single_arm": "left", "fixed_target": False},
            {"single_arm": "none", "fixed_target": False},
        ]
        learn_callbacks = CurriculumCallback(
            stages=auto_stages,
            upgrade_threshold=args.auto_curriculum_threshold,
            window=args.auto_curriculum_window,
            verbose=1,
        )
        print(
            f"启动自动课程学习: Stage0 -> Stage1 -> Stage2 "
            f"(晋升阈值 {args.auto_curriculum_threshold * 100:.0f}%, "
            f"窗口 {args.auto_curriculum_window} 回合)",
            flush=True,
        )

    model.learn(
        total_timesteps=args.total_timesteps,
        progress_bar=True,
        callback=learn_callbacks,
    )
    model.save(args.model_path)
    model_base, _ = os.path.splitext(args.model_path)
    env.save(model_base + "_vecnormalize.pkl")
    print(f"Saved model to: {args.model_path}")


if __name__ == "__main__":
    main()

