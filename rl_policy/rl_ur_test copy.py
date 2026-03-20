import argparse
import os
from typing import Tuple

import gym
import mujoco
import numpy as np
from gym import spaces
from stable_baselines3 import PPO


def _quat_dot(q1_wxyz: np.ndarray, q2_wxyz: np.ndarray) -> float:
    return float(np.dot(q1_wxyz, q2_wxyz))


def _quat_angle(q1_wxyz: np.ndarray, q2_wxyz: np.ndarray) -> float:
    q1 = q1_wxyz / (np.linalg.norm(q1_wxyz) + 1e-12)
    q2 = q2_wxyz / (np.linalg.norm(q2_wxyz) + 1e-12)
    d = abs(_quat_dot(q1, q2))
    d = float(np.clip(d, -1.0, 1.0))
    return float(2.0 * np.arccos(d))


class URDualArmTask1Env(gym.Env):
    def __init__(
        self,
        stage: str = "static",
        render: bool = False,
        frame_skip: int = 5,
        episode_len: int = 256,
        seed: int = 0,
    ):
        super().__init__()
        self.stage = stage
        self.render_mode = render
        self.frame_skip = int(frame_skip)
        self.episode_len = int(episode_len)
        self.np_random = np.random.default_rng(seed)

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

        self.target_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "target_object_geom"
        )
        self.table_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "worktable_top"
        )

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
        for jname in self.joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            lo, hi = self.model.jnt_range[jid, 0], self.model.jnt_range[jid, 1]
            if not bool(self.model.jnt_limited[jid]):
                lo, hi = -np.pi, np.pi
            limits.append((float(lo), float(hi)))
        self.joint_limits = np.asarray(limits, dtype=np.float32)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)

        # Rate limit joint targets to reduce "target jumps" -> contact instability.
        # ctrl is in joint radians (mapped from action into each joint ctrlrange).
        self.max_delta_ctrl = 0.15  # rad per decision step
        self.prev_ctrl = np.zeros(12, dtype=np.float32)

        obs_dim = 3 + 4 + 3 + 4 + 3 + 4 + 6
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
        # Box object half-extents: (x, y, z) in meters
        self.obj_half_extents = np.array([0.02, 0.02, 0.02], dtype=np.float32)
        self.table_top_z = 0.73  # legacy
        self.obj_clearance = 0.016

        # Curriculum + success gating:
        # 6DOF curriculum (distance + orientation both must satisfy)
        self.dist_success_start = 0.30
        self.dist_success_end = 0.12
        # Looser early curriculum so the agent gets enough successful samples.
        self.ori_success_start = 2.0  # rad (~114deg)
        self.ori_success_end = 1.0  # rad (~57deg)
        self.curriculum_episodes = 200
        # For debugging/troubleshooting: temporarily require only 1-step hold.
        # This helps distinguish "never reaching thresholds" vs "reaching but not maintaining".
        self.success_hold_steps = 1

        # Small x/y randomization around worktable top center (no table height change).
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
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        normalized = (a + 1.0) / 2.0
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

        # Place object at the current worktable_top center (x/y)
        table_top_world_xy = np.array(
            self.data.geom_xpos[self.table_geom_id][:2], dtype=np.float32
        )
        obj_x = float(table_top_world_xy[0])
        obj_y = float(table_top_world_xy[1])

        if self.xy_randomize:
            table_half = np.array(self.model.geom_size[self.table_geom_id][:2], dtype=np.float32)
            avail = np.maximum(table_half - self.xy_safe_margin, 0.0)
            max_off = avail
            dx = float(self.np_random.uniform(-max_off[0], max_off[0]))
            dy = float(self.np_random.uniform(-max_off[1], max_off[1]))
            obj_x += dx
            obj_y += dy

        # Place object on top of the table (use actual world z from geom_xpos)
        table_top_world_z = float(self.data.geom_xpos[self.table_geom_id][2])
        obj_z = table_top_world_z + float(self.obj_half_extents[2]) + float(self.obj_clearance)
        self.data.qpos[self.obj_qposadr : self.obj_qposadr + 3] = np.array(
            [obj_x, obj_y, obj_z], dtype=np.float32
        )
        # Requirement: local +Z should always point toward the ground (world -Z).
        # Keep +Z down by applying a fixed 180deg flip about X, then randomize yaw around world Z.
        # With MuJoCo qpos order [w, x, y, z], resulting quaternion is:
        #   q = [0, cos(yaw/2), sin(yaw/2), 0]
        yaw = float(self.np_random.uniform(-np.pi, np.pi))
        self.data.qpos[self.obj_qposadr + 3 : self.obj_qposadr + 7] = np.array(
            [0.0, np.cos(yaw / 2.0), np.sin(yaw / 2.0), 0.0], dtype=np.float32
        )

        if self.stage == "static":
            lin_v = np.zeros(3, dtype=np.float32)
            ang_v = np.zeros(3, dtype=np.float32)
        elif self.stage == "slow_dynamic":
            speed = 0.05
            theta = self.np_random.uniform(0.0, 2.0 * np.pi)
            lin_v = np.array([speed * np.cos(theta), speed * np.sin(theta), 0.0], dtype=np.float32)
            ang_speed = 0.5
            axis = self.np_random.normal(size=3).astype(np.float32)
            axis = axis / (np.linalg.norm(axis) + 1e-12)
            ang_v = ang_speed * axis
        else:
            lin_v = np.zeros(3, dtype=np.float32)
            ang_v = np.zeros(3, dtype=np.float32)

        self.data.qvel[self.obj_qveladr : self.obj_qveladr + 3] = lin_v
        self.data.qvel[self.obj_qveladr + 3 : self.obj_qveladr + 6] = ang_v

        mujoco.mj_forward(self.model, self.data)

        # Initialize previous ctrl for rate limiting.
        self.prev_ctrl = np.asarray(self.data.ctrl, dtype=np.float32).copy()

        obs = self._get_observation()
        return obs, {}

    def _get_observation(self) -> np.ndarray:
        lpos, lquat = self._get_site_pos_quat(self.left_site_id)
        rpos, rquat = self._get_site_pos_quat(self.right_site_id)
        opos, oquat, olinvel, oangvel = self._get_object_pos_quat_vel()
        obs = np.concatenate([lpos, lquat, rpos, rquat, opos, oquat, olinvel, oangvel]).astype(
            np.float32
        )
        return obs

    def _compute_reward_done(self):
        lpos, lquat = self._get_site_pos_quat(self.left_site_id)
        rpos, rquat = self._get_site_pos_quat(self.right_site_id)
        opos, oquat, _, _ = self._get_object_pos_quat_vel()

        dL = float(np.linalg.norm(lpos - opos))
        dR = float(np.linalg.norm(rpos - opos))
        d = min(dL, dR)

        pos_reward = -np.arctan(d)

        if dL <= dR:
            ori_angle = _quat_angle(lquat, oquat)
        else:
            ori_angle = _quat_angle(rquat, oquat)
        ori_reward = -0.4 * np.arctan(ori_angle)

        collision_penalty = 0.0
        if self.table_geom_id >= 0 and self.target_geom_id >= 0:
            for c in self.data.contact:
                if (c.geom1 == self.table_geom_id and c.geom2 == self.target_geom_id) or (
                    c.geom2 == self.table_geom_id and c.geom1 == self.target_geom_id
                ):
                    collision_penalty -= 0.5
                    break

        # Success criteria: "任一臂到 + 姿态角对齐"
        dist_success = float(self.dist_success_current)
        ori_success = float(self.ori_success_current)

        # Compute both orientation angles for OR success logic.
        ori_angle_L = _quat_angle(lquat, oquat)
        ori_angle_R = _quat_angle(rquat, oquat)

        success_left = (dL < dist_success) and (ori_angle_L < ori_success)
        success_right = (dR < dist_success) and (ori_angle_R < ori_success)
        success_now = bool(success_left or success_right)

        success_bonus_now = 2.0 if success_now else 0.0
        reward = pos_reward + ori_reward + collision_penalty + success_bonus_now
        return float(reward), bool(success_now)

    def step(self, action):
        ctrl_target = self._map_action_to_ctrl(action)
        # Limit how much each joint target can change per decision step.
        delta = ctrl_target - self.prev_ctrl
        delta = np.clip(delta, -self.max_delta_ctrl, self.max_delta_ctrl)
        ctrl = self.prev_ctrl + delta
        self.prev_ctrl = ctrl.copy()

        for i, aid in enumerate(self.actuator_ids):
            self.data.ctrl[aid] = float(ctrl[i])

        unstable = False
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            if self.handle is not None:
                self.handle.sync()
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
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(args.model_path)

    env = URDualArmTask1Env(stage=args.stage, render=args.render, episode_len=256, seed=args.seed)
    model = PPO.load(args.model_path)

    successes = 0
    episode_rewards = []
    for ep in range(args.n_episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        ep_reward = 0.0
        done = False
        truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += float(reward)
        episode_rewards.append(ep_reward)
        if info.get("is_success", False):
            successes += 1

        print(f"Episode {ep+1}/{args.n_episodes}: success={info.get('is_success', False)} reward={ep_reward:.2f}")

    success_rate = successes / float(args.n_episodes)
    print(f"Task1 success_rate = {success_rate:.4f} ({successes}/{args.n_episodes})")
    env.close()


if __name__ == "__main__":
    main()

