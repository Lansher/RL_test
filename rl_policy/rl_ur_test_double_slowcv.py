import argparse
import os
import warnings
from typing import Tuple

import gym
import mujoco
import numpy as np
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

# 与 rl_ur_train_double_slowcv.py 保持同名、同数值（验证时请与训练时一致）
# slow_dynamic：全局匀速方块（每子步强制 qvel）
DEFAULT_MAX_DELTA_CTRL = 0.08
DEFAULT_ACTION_SMOOTHING_COEF = 0.5
DEFAULT_ACTION_DEADZONE = 0.01
DEFAULT_ACTION_RAW_LPF_COEF = 0.6
DEFAULT_ACTION_BOUND = 1.0
DEFAULT_ACTION_PENALTY_SCALE = 0.1
DEFAULT_JOINT_LIMIT_EPSILON = 0.02
DEFAULT_DIST_SUCCESS_START = 0.25
DEFAULT_DIST_SUCCESS_END = 0.05
DEFAULT_ORI_SUCCESS_START = 2.0
DEFAULT_ORI_SUCCESS_END = 0.5
DEFAULT_CURRICULUM_EPISODES = 250
DEFAULT_TOUCH_STOP = False
DEFAULT_DUAL_ARM_MODE = "nearest_active"
DEFAULT_WRIST_TABLE_PENALTY_SCALE = 0.0
DEFAULT_COLLISION_PENALTY_SCALE = 2.0
DEFAULT_OBJ_LIN_SPEED = 0.2
DEFAULT_OBJ_ANG_SPEED = 0.5
DEFAULT_MODEL_PATH = "./models/ppo_ur_task1_slowcv_model.zip"


def _print_config_table(title: str, rows: list[tuple[str, object]]) -> None:
    print(f"\n[{title}]")
    width = max(len(k) for k, _ in rows)
    for k, v in rows:
        print(f"  {k.ljust(width)} : {v}")
    print("")


def _quat_dot(q1_wxyz: np.ndarray, q2_wxyz: np.ndarray) -> float:
    return float(np.dot(q1_wxyz, q2_wxyz))


def _quat_angle(q1_wxyz: np.ndarray, q2_wxyz: np.ndarray) -> float:
    q1 = q1_wxyz / (np.linalg.norm(q1_wxyz) + 1e-12)
    q2 = q2_wxyz / (np.linalg.norm(q2_wxyz) + 1e-12)
    d = abs(_quat_dot(q1, q2))
    d = float(np.clip(d, -1.0, 1.0))
    return float(2.0 * np.arccos(d))


def _quat_rotate_vector(q_wxyz: np.ndarray, v_xyz: np.ndarray) -> np.ndarray:
    """Rotate a 3D vector by quaternion (w,x,y,z)."""
    q = q_wxyz.astype(np.float32, copy=False)
    v = v_xyz.astype(np.float32, copy=False)
    q = q / (np.linalg.norm(q) + 1e-12)
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    q_vec = np.array([x, y, z], dtype=np.float32)

    # v' = v + w*t + cross(q_vec, t), where t = 2*cross(q_vec, v)
    t = 2.0 * np.cross(q_vec, v)
    v_rot = v + w * t + np.cross(q_vec, t)
    return v_rot


def _tilt_yaw_ori_angle(tool_q_wxyz: np.ndarray, obj_q_wxyz: np.ndarray) -> float:
    """
    Only enforce:
    1) tool local +Z is tilted to point downward (world -Z),
    2) tool yaw matches object's yaw by aligning local +X projected onto the XY plane.

    Returns a single angle (rad) used as orientation error.
    """
    # (1) Tilt: +Z of tool vs world down (-Z)
    z_tool_world = _quat_rotate_vector(tool_q_wxyz, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    cos_tilt = float(np.dot(z_tool_world, np.array([0.0, 0.0, -1.0], dtype=np.float32)))
    cos_tilt = float(np.clip(cos_tilt, -1.0, 1.0))
    tilt_err = float(np.arccos(cos_tilt))

    # (2) Yaw: align +X projected onto XY plane
    x_tool_world = _quat_rotate_vector(tool_q_wxyz, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    x_obj_world = _quat_rotate_vector(obj_q_wxyz, np.array([1.0, 0.0, 0.0], dtype=np.float32))

    x_tool_xy = np.array([x_tool_world[0], x_tool_world[1], 0.0], dtype=np.float32)
    x_obj_xy = np.array([x_obj_world[0], x_obj_world[1], 0.0], dtype=np.float32)
    n1 = float(np.linalg.norm(x_tool_xy))
    n2 = float(np.linalg.norm(x_obj_xy))
    if n1 < 1e-8 or n2 < 1e-8:
        yaw_err = float(np.pi)
    else:
        x_tool_xy /= n1
        x_obj_xy /= n2
        cos_yaw = float(np.dot(x_tool_xy, x_obj_xy))
        cos_yaw = float(np.clip(cos_yaw, -1.0, 1.0))
        yaw_err = float(np.arccos(cos_yaw))

    return float(max(tilt_err, yaw_err))


class URDualArmTask1Env(gym.Env):
    def __init__(
        self,
        stage: str = "static",
        render: bool = False,
        frame_skip: int = 5,
        episode_len: int = 256,
        seed: int = 0,
        single_arm: str = "none",
        fixed_target: bool = False,
        success_bonus: float = 80.0,
        collision_penalty_scale: float = DEFAULT_COLLISION_PENALTY_SCALE,
        wrist_table_penalty_scale: float = DEFAULT_WRIST_TABLE_PENALTY_SCALE,
        action_penalty_scale: float = DEFAULT_ACTION_PENALTY_SCALE,
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
        obj_lin_speed: float = DEFAULT_OBJ_LIN_SPEED,
        obj_ang_speed: float = DEFAULT_OBJ_ANG_SPEED,
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

        self.left_site_name = "left_tool0_site"
        self.right_site_name = "right_tool0_site"
        self.left_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, self.left_site_name
        )
        self.right_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, self.right_site_name
        )

        self.obj_body_name = "target_object"
        self.obj_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.obj_body_name
        )
        self.obj_jnt_id = int(self.model.body_jntadr[self.obj_body_id])
        self.obj_qposadr = int(self.model.jnt_qposadr[self.obj_jnt_id])
        self.obj_qveladr = int(self.model.jnt_dofadr[self.obj_jnt_id])
        self.obj_lin_speed = float(obj_lin_speed)
        self.obj_ang_speed = float(obj_ang_speed)
        self._obj_const_lin_vel = np.zeros(3, dtype=np.float32)
        self._obj_const_ang_vel = np.zeros(3, dtype=np.float32)

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
        self.robot_geom_ids = set()
        for gid in range(int(self.model.ngeom)):
            bodyid = int(self.model.geom_bodyid[gid])
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bodyid)
            if body_name and (body_name.startswith("left_") or body_name.startswith("right_")):
                self.robot_geom_ids.add(int(gid))

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
        self.actuator_names = list(self.joint_names)
        self.actuator_ids = []
        for name in self.actuator_names:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            self.actuator_ids.append(int(aid))

        limits = []
        # Cache joint qpos/qvel addresses for proprioception in the observation.
        self.robot_qpos_adrs = []
        self.robot_qvel_adrs = []
        for jname in self.joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            lo, hi = self.model.jnt_range[jid, 0], self.model.jnt_range[jid, 1]
            if not bool(self.model.jnt_limited[jid]):
                lo, hi = -np.pi, np.pi
            limits.append((float(lo), float(hi)))
        self.joint_limits = np.asarray(limits, dtype=np.float32)
        for jname in self.joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            self.robot_qpos_adrs.append(int(self.model.jnt_qposadr[jid]))
            self.robot_qvel_adrs.append(int(self.model.jnt_dofadr[jid]))
        self.robot_qpos_adrs = np.asarray(self.robot_qpos_adrs, dtype=np.int64)  # (12,)
        self.robot_qvel_adrs = np.asarray(self.robot_qvel_adrs, dtype=np.int64)  # (12,)

        if action_bound <= 0.0:
            raise ValueError("action_bound must be positive.")
        self.action_bound = float(action_bound)
        self.action_space = spaces.Box(
            low=-self.action_bound,
            high=self.action_bound,
            shape=(12,),
            dtype=np.float32,
        )

        # Rate limit joint targets to reduce "target jumps" -> contact instability.
        self.max_delta_ctrl = float(max_delta_ctrl)  # rad per decision step
        # If a joint is already near a limit, suppress delta that would push it further.
        self.joint_limit_epsilon = float(joint_limit_epsilon)
        # Low-pass filter for delta_q to reduce high-frequency chatter.
        self.action_smoothing_coef = float(action_smoothing_coef)
        self.action_deadzone = float(action_deadzone)
        self.action_raw_lpf_coef = float(action_raw_lpf_coef)
        self.prev_ctrl = np.zeros(12, dtype=np.float32)
        self.prev_a_filtered = np.zeros(12, dtype=np.float32)
        # In single-arm debug modes we treat the other arm as "truly fixed".
        # We will hard-hold the fixed arm's qpos/qvel during rollout.
        self.fixed_hold_qpos = None
        self.fixed_hold_qvel = None

        # Touch-stop (freeze on first contact with target). Disable for RL training if needed.
        self.touch_stop_enabled = bool(touch_stop)
        self.has_touched_object = False
        self.touch_hold_qpos = None
        self.prev_delta_q = np.zeros(12, dtype=np.float32)
        self.home_left_qpos = None
        self.home_right_qpos = None

        # Observation:
        # robot_qpos(12) + robot_qvel(12) + lpos/lquat/rpos/rquat(14) +
        # rel_lpos/rel_rpos(6) + object_pos/object_quat/object_vel(13)
        obs_dim = 57
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

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

        self.home_key_id = None
        for k in range(int(self.model.nkey)):
            if self.model.key(k).name == "home":
                self.home_key_id = int(k)
                break
        if self.home_key_id is None:
            raise ValueError("Keyframe 'home' not found in scene_dual_arm.xml.")

        # x/y is read from geom_xpos[worktable_top] at reset time
        self.table_center_xy = np.array([0.0, 0.0], dtype=np.float32)
        self.table_half_extents_xy = np.array([0.15, 0.30], dtype=np.float32)
        self.obj_half_extents = np.array([0.02, 0.02, 0.02], dtype=np.float32)
        self.table_top_z = 0.73  # legacy
        self.obj_clearance = 0.016

        if dist_success_end <= 0.0 or dist_success_end > dist_success_start:
            raise ValueError("Need 0 < dist_success_end <= dist_success_start.")
        if ori_success_end <= 0.0 or ori_success_end >= ori_success_start:
            raise ValueError("Need 0 < ori_success_end < ori_success_start.")
        self.dist_success_start = float(dist_success_start)
        self.dist_success_end = float(dist_success_end)
        self.ori_success_start = float(ori_success_start)
        self.ori_success_end = float(ori_success_end)
        self.curriculum_episodes = max(1, int(curriculum_episodes))
        self.success_hold_steps = 1

        # Small x/y randomization around worktable top center
        self.xy_randomize = True
        self.xy_rand_max = np.array([0.06, 0.08], dtype=np.float32)
        self.xy_safe_margin = 0.04  # keep sphere away from edges (m)

    def _current_success_thresholds(self) -> Tuple[float, float]:
        """Return (dist_success, ori_success) for current curriculum episode."""
        t = float(min(1.0, self.episode_count / max(1, self.curriculum_episodes)))
        dist_success = float(self.dist_success_start + t * (self.dist_success_end - self.dist_success_start))
        ori_success = float(self.ori_success_start + t * (self.ori_success_end - self.ori_success_start))
        return dist_success, ori_success

    def _get_site_pos_quat(self, site_id: int):
        pos = np.array(self.data.site(site_id).xpos, dtype=np.float32)
        xmat = np.array(self.data.site(site_id).xmat, dtype=np.float64)
        quat64 = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat64, xmat)
        return pos, quat64.astype(np.float32)

    def _get_object_pos_quat_vel(self):
        body = self.data.body(self.obj_body_id)
        obj_pos = np.array(body.xpos, dtype=np.float32)
        obj_quat = np.array(body.xquat, dtype=np.float32)
        cvel = np.array(body.cvel, dtype=np.float32)
        obj_linvel = cvel[:3]
        obj_angvel = cvel[3:6]
        return obj_pos, obj_quat, obj_linvel, obj_angvel

    def _map_action_to_ctrl(self, action: np.ndarray) -> np.ndarray:
        b = float(self.action_bound)
        a = np.clip(np.asarray(action, dtype=np.float32), -b, b)
        normalized = (a + b) / (2.0 * b)
        lo = self.joint_limits[:, 0]
        hi = self.joint_limits[:, 1]
        ctrl = lo + normalized * (hi - lo)
        return ctrl.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.episode_count += 1
        self.success_streak = 0
        self.dist_success_current, self.ori_success_current = self._current_success_thresholds()

        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key_id)
        mujoco.mj_forward(self.model, self.data)
        self.step_number = 0
        self.goal_reached = False

        # 方块 x/y：单臂锚在对应臂末端投影。
        # 双臂（single_arm=none）：验证脚本不用「左右末端中点」作锚——否则方块总落在两臂中间，
        # 对称位置易导致策略不敢分工去抓；改为每回合随机以左或右臂末端在桌面上的投影为锚。
        table_top_world_xy = np.array(
            self.data.geom_xpos[self.table_geom_id][:2], dtype=np.float32
        )
        table_top_world_z = float(self.data.geom_xpos[self.table_geom_id][2])
        obj_z = table_top_world_z + float(self.obj_half_extents[2]) + float(self.obj_clearance)

        table_half = np.array(self.model.geom_size[self.table_geom_id][:2], dtype=np.float32)
        cx = float(table_top_world_xy[0])
        cy = float(table_top_world_xy[1])
        hx = float(table_half[0])
        hy = float(table_half[1])
        m = float(self.xy_safe_margin)
        lpos_h = np.array(self.data.site(self.left_site_id).xpos, dtype=np.float32)
        rpos_h = np.array(self.data.site(self.right_site_id).xpos, dtype=np.float32)
        if self.single_arm == "left":
            anchor_x = float(np.clip(lpos_h[0], cx - hx + m, cx + hx - m))
            anchor_y = float(np.clip(lpos_h[1], cy - hy + m, cy + hy - m))
        elif self.single_arm == "right":
            anchor_x = float(np.clip(rpos_h[0], cx - hx + m, cx + hx - m))
            anchor_y = float(np.clip(rpos_h[1], cy - hy + m, cy + hy - m))
        else:
            use_left = bool(self.np_random.uniform() < 0.5)
            if use_left:
                anchor_x = float(np.clip(lpos_h[0], cx - hx + m, cx + hx - m))
                anchor_y = float(np.clip(lpos_h[1], cy - hy + m, cy + hy - m))
            else:
                anchor_x = float(np.clip(rpos_h[0], cx - hx + m, cx + hx - m))
                anchor_y = float(np.clip(rpos_h[1], cy - hy + m, cy + hy - m))

        # slow_dynamic：与训练一致——木块开局在对应臂末端投影附近（可抓），再沿 +Y 移动。
        if self.stage == "slow_dynamic":
            oh = self.obj_half_extents
            m = float(self.xy_safe_margin)
            xmin = float(cx - hx + float(oh[0]) + m)
            xmax = float(cx + hx - float(oh[0]) - m)
            ymin = float(cy - hy + float(oh[1]) + m)
            ymax = float(cy + hy - float(oh[1]) - m)
            if self.single_arm == "left":
                anchor = lpos_h
            else:
                anchor = rpos_h
            bel_x = float(np.clip(anchor[0], xmin, xmax))
            bel_y = float(np.clip(anchor[1], ymin, ymax))
            if self.fixed_target:
                obj_x = bel_x
                obj_y = bel_y
                yaw = 0.0
            elif self.xy_randomize:
                obj_x = float(np.clip(bel_x + self.np_random.uniform(-0.03, 0.03), xmin, xmax))
                obj_y = float(np.clip(bel_y + self.np_random.uniform(0.0, 0.04), ymin, ymax))
                yaw = float(self.np_random.uniform(-np.pi, np.pi))
            else:
                obj_x = bel_x
                obj_y = bel_y
                yaw = float(self.np_random.uniform(-np.pi, np.pi))
        else:
            obj_x = anchor_x
            obj_y = anchor_y
            yaw = 0.0 if self.fixed_target else float(self.np_random.uniform(-np.pi, np.pi))

        # If using random x/y, avoid spawning unreachable targets.
        if self.xy_randomize and not self.fixed_target and self.stage != "slow_dynamic":
            avail = np.maximum(table_half - self.xy_safe_margin, 0.0)
            max_off = avail

            # Reachability threshold (meters).
            reach_dist = float(self.dist_success_current) * 1.6
            reach_dist = float(np.clip(reach_dist, 0.2, 0.6))

            max_attempts = 25
            accepted = False
            for _ in range(max_attempts):
                cand_x = anchor_x + float(self.np_random.uniform(-max_off[0], max_off[0]))
                cand_y = anchor_y + float(self.np_random.uniform(-max_off[1], max_off[1]))
                cand_x = float(np.clip(cand_x, cx - hx + m, cx + hx - m))
                cand_y = float(np.clip(cand_y, cy - hy + m, cy + hy - m))
                cand_yaw = float(self.np_random.uniform(-np.pi, np.pi))

                # Candidate placement.
                self.data.qpos[self.obj_qposadr : self.obj_qposadr + 3] = np.array(
                    [cand_x, cand_y, obj_z], dtype=np.float32
                )
                self.data.qpos[self.obj_qposadr + 3 : self.obj_qposadr + 7] = np.array(
                    [0.0, np.cos(cand_yaw / 2.0), np.sin(cand_yaw / 2.0), 0.0],
                    dtype=np.float32,
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
                    # 验证：允许「更近的一侧」够得着即可，不要求左右同时近（避免只剩中点可行）
                    ok = min(dL, dR) < reach_dist

                if ok:
                    obj_x = cand_x
                    obj_y = cand_y
                    yaw = cand_yaw
                    accepted = True
                    break

            # If not accepted, 回退到锚点
            if not accepted:
                obj_x = anchor_x
                obj_y = anchor_y
                yaw = float(self.np_random.uniform(-np.pi, np.pi))
                self.data.qpos[self.obj_qposadr : self.obj_qposadr + 3] = np.array(
                    [obj_x, obj_y, obj_z], dtype=np.float32
                )

        # Final placement.
        self.data.qpos[self.obj_qposadr : self.obj_qposadr + 3] = np.array(
            [obj_x, obj_y, obj_z], dtype=np.float32
        )
        self.data.qpos[self.obj_qposadr + 3 : self.obj_qposadr + 7] = np.array(
            [0.0, np.cos(yaw / 2.0), np.sin(yaw / 2.0), 0.0], dtype=np.float32
        )

        if self.fixed_target or self.stage == "static":
            lin_v = np.zeros(3, dtype=np.float32)
            ang_v = np.zeros(3, dtype=np.float32)
        elif self.stage == "slow_dynamic":
            # 世界系 Y：右臂侧 → 左臂侧为 +Y；沿桌面 Y 平移，无自转。
            speed = float(self.obj_lin_speed)
            lin_v = np.array([0.0, speed, 0.0], dtype=np.float32)
            ang_v = np.zeros(3, dtype=np.float32)
        else:
            lin_v = np.zeros(3, dtype=np.float32)
            ang_v = np.zeros(3, dtype=np.float32)

        self.data.qvel[self.obj_qveladr : self.obj_qveladr + 3] = lin_v
        self.data.qvel[self.obj_qveladr + 3 : self.obj_qveladr + 6] = ang_v
        self._obj_const_lin_vel = np.asarray(lin_v, dtype=np.float32).copy()
        self._obj_const_ang_vel = np.asarray(ang_v, dtype=np.float32).copy()

        mujoco.mj_forward(self.model, self.data)

        # Initialize previous ctrl to the current joint positions at `home`.
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

        # Proprioception: robot joint states (qpos/qvel).
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
        """Return True when target_object geom is in contact with something other than the table."""
        if self.target_geom_id < 0:
            return False
        table_id = int(self.table_geom_id) if self.table_geom_id >= 0 else -1

        for c in self.data.contact:
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 == self.target_geom_id and g2 != table_id:
                return True
            if g2 == self.target_geom_id and g1 != table_id:
                return True
        return False

    def _is_robot_geom_contact_pair(self, g1: int, g2: int) -> bool:
        """True if contact is between two robot geoms and not involving the target."""
        if g1 in self.robot_geom_ids and g2 in self.robot_geom_ids:
            if g1 == int(self.target_geom_id) or g2 == int(self.target_geom_id):
                return False
            return True
        return False

    def _robot_self_or_base_collision(self) -> bool:
        """Detect collisions that usually cause whipping: robot-robot or robot-base_col."""
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

    def _apply_object_constant_velocity(self) -> None:
        """slow_dynamic：每物理子步后强制方块 qvel 为本回合目标，实现全局匀速。"""
        if self.stage != "slow_dynamic":
            return
        a = int(self.obj_qveladr)
        self.data.qvel[a : a + 3] = self._obj_const_lin_vel
        self.data.qvel[a + 3 : a + 6] = self._obj_const_ang_vel

    def _compute_reward_done(self):
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
            # Avoid hard "min" switching between arms.
            d = 0.5 * dL + 0.5 * dR
            ori_angle = 0.5 * ori_angle_L + 0.5 * ori_angle_R

        # Positive shaping: closer -> larger reward in (0, 1], far -> ~0.
        pos_reward = float(1.0 / (1.0 + 5.0 * d))
        ori_reward = -0.4 * np.arctan(float(ori_angle))

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
                if self.base_geom_id >= 0:
                    if c.geom1 == self.base_geom_id or c.geom2 == self.base_geom_id:
                        collision_penalty -= float(self.collision_penalty_scale)
                        break

                # 3) Additional collision: robot self collision (robot <-> robot)
                if self._is_robot_geom_contact_pair(int(c.geom1), int(c.geom2)):
                    collision_penalty -= float(self.collision_penalty_scale)
                    break

        wrist_table_penalty = 0.0
        if self.wrist_table_penalty_scale != 0.0 and self.table_geom_id >= 0 and self.robot_geom_ids:
            tid = int(self.table_geom_id)
            for c in self.data.contact:
                g1, g2 = int(c.geom1), int(c.geom2)
                if g1 == tid and g2 in self.robot_geom_ids:
                    wrist_table_penalty -= float(self.wrist_table_penalty_scale)
                    break
                if g2 == tid and g1 in self.robot_geom_ids:
                    wrist_table_penalty -= float(self.wrist_table_penalty_scale)
                    break

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

        success_bonus_now = self.success_bonus if success_now else 0.0
        reward = pos_reward + ori_reward + collision_penalty + wrist_table_penalty + success_bonus_now
        return float(reward), bool(success_now)

    def step(self, action):
        # Touch-stop: if we already touched the object, keep ctrl frozen.
        if self.touch_stop_enabled and self.has_touched_object:
            ctrl_target = self.prev_ctrl
        else:
            # Incremental delta control: a in [-action_bound, action_bound] maps to delta joint radians.
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
            self._apply_object_constant_velocity()

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
            # stop integrating further in this decision step to reduce visible whipping.
            if (not self.has_touched_object) and self._robot_self_or_base_collision():
                break
            # Early stop within the same decision step when hold=1.
            if self.success_hold_steps == 1 and self._success_now():
                break
            if not np.isfinite(self.data.qacc).all():
                unstable = True
                break

        self.step_number += 1
        obs = self._get_observation()

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
        description="Evaluate policy trained with rl_ur_train_double_slowcv.py (全局匀速 slow_dynamic).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="slow_dynamic",
        choices=["static", "slow_dynamic"],
        help="须与训练一致；slowcv 训练默认 slow_dynamic",
    )
    parser.add_argument(
        "--obj_lin_speed",
        type=float,
        default=DEFAULT_OBJ_LIN_SPEED,
        help="slow_dynamic：沿 +Y 速率 (m/s)，须与训练一致",
    )
    parser.add_argument(
        "--obj_ang_speed",
        type=float,
        default=DEFAULT_OBJ_ANG_SPEED,
        help="保留参数；slow_dynamic 下恒为 0",
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--single_arm", type=str, default="none", choices=["none", "left", "right"])
    parser.add_argument("--fixed_target", action="store_true")
    parser.add_argument(
        "--wrist_table_penalty_scale",
        type=float,
        default=DEFAULT_WRIST_TABLE_PENALTY_SCALE,
        help="Must match training. 0 disables table-contact penalty.",
    )

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
        help="Must match training.",
    )
    parser.add_argument(
        "--touch_stop",
        action="store_true",
        help="Must match training. Freeze on contact (default off).",
    )
    parser.add_argument("--no_touch_stop", action="store_true", help=argparse.SUPPRESS)

    parser.add_argument(
        "--vecnormalize_path",
        type=str,
        default=None,
        help="VecNormalize pkl; default: <model_stem>_vecnormalize.pkl",
    )
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.no_touch_stop:
        warnings.warn("--no_touch_stop is ignored: no freeze-on-contact is already the default.", UserWarning)
    _print_config_table(
        "test/default_constants",
        [
            ("DEFAULT_MAX_DELTA_CTRL", DEFAULT_MAX_DELTA_CTRL),
            ("DEFAULT_ACTION_SMOOTHING_COEF", DEFAULT_ACTION_SMOOTHING_COEF),
            ("DEFAULT_ACTION_DEADZONE", DEFAULT_ACTION_DEADZONE),
            ("DEFAULT_ACTION_RAW_LPF_COEF", DEFAULT_ACTION_RAW_LPF_COEF),
            ("DEFAULT_ACTION_BOUND", DEFAULT_ACTION_BOUND),
            ("DEFAULT_JOINT_LIMIT_EPSILON", DEFAULT_JOINT_LIMIT_EPSILON),
            ("DEFAULT_DIST_SUCCESS_START", DEFAULT_DIST_SUCCESS_START),
            ("DEFAULT_DIST_SUCCESS_END", DEFAULT_DIST_SUCCESS_END),
            ("DEFAULT_ORI_SUCCESS_START", DEFAULT_ORI_SUCCESS_START),
            ("DEFAULT_ORI_SUCCESS_END", DEFAULT_ORI_SUCCESS_END),
            ("DEFAULT_CURRICULUM_EPISODES", DEFAULT_CURRICULUM_EPISODES),
            ("DEFAULT_TOUCH_STOP", DEFAULT_TOUCH_STOP),
            ("DEFAULT_DUAL_ARM_MODE", DEFAULT_DUAL_ARM_MODE),
            ("DEFAULT_WRIST_TABLE_PENALTY_SCALE", DEFAULT_WRIST_TABLE_PENALTY_SCALE),
            ("DEFAULT_OBJ_LIN_SPEED", DEFAULT_OBJ_LIN_SPEED),
            ("DEFAULT_OBJ_ANG_SPEED", DEFAULT_OBJ_ANG_SPEED),
            ("DEFAULT_MODEL_PATH", DEFAULT_MODEL_PATH),
        ],
    )
    _print_config_table(
        "test/run_args",
        [
            ("stage", args.stage),
            ("obj_lin_speed", args.obj_lin_speed),
            ("obj_ang_speed", args.obj_ang_speed),
            ("render", args.render),
            ("model_path", args.model_path),
            ("single_arm", args.single_arm),
            ("fixed_target", args.fixed_target),
            ("wrist_table_penalty_scale", args.wrist_table_penalty_scale),
            ("max_delta_ctrl", args.max_delta_ctrl),
            ("joint_limit_epsilon", args.joint_limit_epsilon),
            ("action_smoothing_coef", args.action_smoothing_coef),
            ("action_deadzone", args.action_deadzone),
            ("action_raw_lpf_coef", args.action_raw_lpf_coef),
            ("action_bound", args.action_bound),
            ("curriculum_episodes", args.curriculum_episodes),
            ("dist_success_start", args.dist_success_start),
            ("dist_success_end", args.dist_success_end),
            ("ori_success_start", args.ori_success_start),
            ("ori_success_end", args.ori_success_end),
            ("dual_arm_mode", args.dual_arm_mode),
            ("touch_stop", args.touch_stop),
            ("vecnormalize_path", args.vecnormalize_path),
            ("n_episodes", args.n_episodes),
            ("seed", args.seed),
        ],
    )

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(args.model_path)

    def make_env():
        return URDualArmTask1Env(
            stage=args.stage,
            render=args.render,
            episode_len=256,
            seed=args.seed,
            single_arm=args.single_arm,
            fixed_target=args.fixed_target,
            wrist_table_penalty_scale=args.wrist_table_penalty_scale,
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
            obj_lin_speed=args.obj_lin_speed,
            obj_ang_speed=args.obj_ang_speed,
        )

    env = make_vec_env(make_env, n_envs=1)
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    if args.vecnormalize_path is None:
        model_base, _ = os.path.splitext(args.model_path)
        vecnormalize_path = model_base + "_vecnormalize.pkl"
    else:
        vecnormalize_path = args.vecnormalize_path

    if vecnormalize_path is not None and os.path.exists(vecnormalize_path):
        env = VecNormalize.load(vecnormalize_path, env)
        # Hard-disable normalization updates during evaluation.
        # This prevents obs statistics from drifting and keeps reward normalization fixed.
        env.training = False
        env.norm_reward = False
        print(f"[INFO] Successfully loaded VecNormalize stats from: {vecnormalize_path}")
    else:
        raise FileNotFoundError(
            f"VecNormalize stats not found: {vecnormalize_path}\n"
            "Expected file alongside the .zip model, e.g. <model_stem>_vecnormalize.pkl. "
            "Please ensure you have trained with VecNormalize and copied the corresponding pkl."
        )

    # Bind env to the model to keep wrapper behavior consistent.
    model = PPO.load(args.model_path, env=env)

    successes = 0
    episode_rewards = []
    for ep in range(args.n_episodes):
        obs = env.reset()
        ep_reward = 0.0
        done = False
        last_info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            ep_reward += float(rewards[0])
            last_info = infos[0]
            done = bool(dones[0])

        episode_rewards.append(ep_reward)
        if last_info.get("is_success", False):
            successes += 1

        print(f"Episode {ep+1}/{args.n_episodes}: success={last_info.get('is_success', False)} reward={ep_reward:.2f}")

    success_rate = successes / float(args.n_episodes)
    mean_ep_reward = float(np.mean(episode_rewards))
    print(f"Task1 success_rate = {success_rate:.4f} ({successes}/{args.n_episodes})")
    print(f"Task1 mean_episode_reward = {mean_ep_reward:.4f}")
    env.close()


if __name__ == "__main__":
    main()

