import argparse
import os
from typing import Tuple

import gym
import mujoco
import numpy as np
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
import torch.nn as nn


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
        action_penalty_scale: float = 0.0,
    ):
        super().__init__()
        self.stage = stage
        self.render_mode = render
        self.frame_skip = int(frame_skip)
        self.episode_len = int(episode_len)
        self.np_random = np.random.default_rng(seed)

        if single_arm not in {"none", "left", "right"}:
            raise ValueError("single_arm must be one of: none, left, right")
        self.single_arm = single_arm
        self.fixed_target = bool(fixed_target)
        self.success_bonus = float(success_bonus)
        self.collision_penalty_scale = float(collision_penalty_scale)
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

        # Build joint limits for mapping action in [-1, 1] -> ctrl
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

        # Action: normalized delta-joint command in [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)

        # Incremental control rate limit: action is scaled to delta-radians per step.
        self.max_delta_ctrl = 0.15  # rad per decision step
        self.prev_ctrl = np.zeros(12, dtype=np.float32)
        # In single-arm debug modes we treat the other arm as "truly fixed".
        # Position actuators are compliant under contact, so "fixed ctrl target"
        # can still drift. We will hard-hold the fixed arm's qpos/qvel during rollout.
        self.fixed_hold_qpos = None
        self.fixed_hold_qvel = None

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

        # Curriculum + success gating:
        # - dist_success/ori_success start large (easy), then become tighter with episodes.
        # - success is considered "done" only after success holds for several steps.
        # 6DOF curriculum (distance + orientation both must satisfy)
        self.dist_success_start = 0.30
        # More lenient end threshold to avoid "never reaching done" early.
        self.dist_success_end = 0.18
        # 6DOF: orientation must actually match (angle between tool and object)
        # Looser early curriculum so the agent gets enough successful samples.
        self.ori_success_start = 2.0  # rad (~114deg)
        self.ori_success_end = 1.3  # rad (~75deg)
        self.curriculum_episodes = 200
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
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        normalized = (a + 1.0) / 2.0  # [0,1]
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
        for i, aid in enumerate(self.actuator_ids):
            self.data.ctrl[aid] = float(robot_qpos[i])

        mujoco.mj_forward(self.model, self.data)

        obs = self._get_observation()
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

        pos_reward = -np.arctan(d)  # [-pi/2, 0]

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
        reward = pos_reward + ori_reward + collision_penalty + success_bonus_now
        return float(reward), bool(success_now)

    def step(self, action):
        # Incremental delta control: action in [-1, 1] directly maps to delta joint radians.
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)  # (12,)
        delta_q = a * self.max_delta_ctrl  # (12,)
        ctrl_target = self.prev_ctrl + delta_q

        # Single-arm mode: keep the other arm fixed.
        if self.single_arm == "left":
            ctrl_target[6:] = self.prev_ctrl[6:]
        elif self.single_arm == "right":
            ctrl_target[:6] = self.prev_ctrl[:6]

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
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            if self.handle is not None:
                self.handle.sync()
            # Re-apply hold after physics integration to prevent drift.
            self._apply_fixed_arm_hold()
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
            return obs, -10.0, True, True, {"is_success": False, "unstable": True}

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
            a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, default="static", choices=["static", "slow_dynamic"])
    parser.add_argument("--render", action="store_true", help="Enable MuJoCo viewer (n_env must be 1)")
    parser.add_argument("--n_env", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)

    # Training length / batch settings
    parser.add_argument("--total_timesteps", type=int, default=2_000_000)
    parser.add_argument("--n_steps", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--lr_final", type=float, default=1e-5)
    parser.add_argument("--use_lr_schedule", action="store_true", default=True)

    # Reward shaping / debugging
    parser.add_argument("--single_arm", type=str, default="none", choices=["none", "left", "right"])
    parser.add_argument("--fixed_target", action="store_true", help="Disable object randomization (baseline debug).")
    parser.add_argument("--success_bonus", type=float, default=100.0)
    parser.add_argument("--collision_penalty_scale", type=float, default=0.0)
    parser.add_argument("--action_penalty_scale", type=float, default=0.0)

    parser.add_argument("--log_dir", type=str, default="./logs/ppo_ur_task1_run_v1/")
    parser.add_argument("--model_path", type=str, default="./models/ppo_ur_task1_run_v1_model.zip")
    args = parser.parse_args()

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
            action_penalty_scale=args.action_penalty_scale,
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

    model.learn(total_timesteps=args.total_timesteps, progress_bar=True)
    model.save(args.model_path)
    model_base, _ = os.path.splitext(args.model_path)
    env.save(model_base + "_vecnormalize.pkl")
    print(f"Saved model to: {args.model_path}")


if __name__ == "__main__":
    main()

