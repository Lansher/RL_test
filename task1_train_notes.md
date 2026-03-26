# Task1 训练笔记（UR 双臂 6DOF 对齐）

下面这份笔记是基于你当前仓库的实现整理的，目标是：让你这个强化学习初学者也能看懂“环境怎么定义、PPO 怎么训练、成功率怎么评估、怎么可视化看末端如何靠近/碰到方块”。

仓库里实现的关键入口文件：
- 训练：`rl_policy/rl_ur_train.py`
- 评估：`rl_policy/rl_ur_test.py`

场景文件（MuJoCo XML）：
- `mujoco_asserts/universal_robots_ur5e/scene_dual_arm.xml`

**最少参数用法（v3 默认写进代码）**：`rl_ur_train.py` / `rl_ur_test.py` 顶部 **`DEFAULT_*`** 常量已包含 `max_delta_ctrl、动作平滑/死区/LPF、课程长度、touch_stop、默认 log/model 路径`。日常只需例如：  
`python rl_policy/rl_ur_train.py --n_env 4 --total_timesteps 2000000`；验证：`python rl_policy/rl_ur_test.py`（默认读 `./models/ppo_ur_task1_run_v3_model.zip`）。需改超参时再传对应 CLI。

**双臂 `nearest_active`（默认）**：`--dual_arm_mode nearest_active` 时，方块离哪侧末端更近，**仅该侧 6 关节**按策略运动，另一侧关节目标固定为 **keyframe `home`** 并在子步硬保持；`both` 为双臂同时按策略动。`--single_arm left|right` 时仍以单臂调试为准，不使用该逻辑。

**最新合并（2026-03-21）**：第 **§10.7** 节已写入本轮关于「`max_delta_ctrl` 很小仍肩/肘大幅摆动」的**用户观测、机理、代码改动与 CLI**；与 §10.2–10.4 的 MDP/PPO 改动互补。

**最新合并（2026-03）**：第 **§10.14** 节汇总了关于 **6DOF 阈值课程 CLI**、**`--auto_curriculum` 初始 Stage 与 `auto_stages` 一致性（先右后左五阶段）**、**「碰到≠成功 / 成功后仍滑移」机理**、**`success_freeze` 与 `--no_success_freeze`**、**单臂随机方块采样锚点（避免方块总在桌面几何中心 → 偏向对侧臂）** 的对话结论；与 §10.9 / §10.9.1 互补。

**最新合并（2026-03）**：第 **§10.15** 节写入 **慢速动态 `slow_dynamic`、双机脚本 `rl_ur_*_double_slowcv.py`、全局匀速（每子步强制 `qvel`）** 及 **方块沿世界 +Y（右→左）平移、角速度为 0** 等说明。

**最新合并（2026-03-26）**：第 **§10.16** 节写入本轮关于 **姿态判定口径更新** 的结论：由“完整四元数误差”改为 **`tool +Z` 朝下（world -Z）+ 与方块坐标轴（XY 平面 heading）对齐** 的组合误差；并已在 `rl_ur_train_double_slowcv.py` 与 `rl_ur_test_double_slowcv.py` 同步。

---

## 1. 原理：强化学习在做什么？

你可以把 Task1 理解成一个循环：

1. 环境 `env` 提供当前状态 `obs`
2. 智能体（PPO policy）根据 `obs` 输出动作 `action`
3. 环境执行动作，MuJoCo 进行仿真一步（或多步）
4. 环境返回新的状态 `obs'`、奖励 `reward`、是否成功 `done`
5. PPO 用这些交互数据更新策略，使得“奖励更高、成功更频繁”

Task1 里的“成功”由代码里的判定条件决定（目前是任一臂末端到方块距离 + 姿态满足阈值）。

---

## 2. 场景与对象：为什么要加 `freejoint`？

抓取/拖拽“动态物体”最关键的一点是：物体必须能在物理仿真里移动。

在你的 `scene_dual_arm.xml` 里：
- 桌子碰撞体用 `worktable_top`（primitive box）
- 可抓取方块（动态物体）用 `target_object`（`type="box"`，`size` 为半尺寸）
- `target_object` 带 `freejoint`，这样方块才会被碰撞/拖拽时移动
- 同时方块上画了 `target_object_x_axis/y_axis/z_axis` 坐标轴，方便你观察 6DOF 对齐的效果
- 另外：底座 OBJ 也新增了启用碰撞的 geom（例如 `base_col`），用于提升物理接触的合理性

注意：MuJoCo 规定 `freejoint` 必须挂在 `worldbody` 的顶层 body，不能放在嵌套 body 里（否则会报错：`free joint can only be used on top level`）。

---

## 3. Task1 环境的“MDP 定义”（对应代码）

下面按 `URDualArmTask1Env` 的实现解释观测、动作、奖励、成功判定。

### 3.1 观测空间（Observation）

在 `rl_ur_train.py` 里，观测维度大致是：

`obs = [左末端 pos(3) + 左末端 quat(4) + 右末端 pos(3) + 右末端 quat(4) + 方块 pos(3) + 方块 quat(4) + 方块线速度(3) + 方块角速度(3)]`

总维度 = 27（代码里 `obs_dim = 3 + 4 + 3 + 4 + 3 + 4 + 6`）。

### 3.2 动作空间（Action）

动作维度是 12：
- 左臂 6 个关节
- 右臂 6 个关节

动作来自 PPO 的输出是归一化值 `[-1, 1]`，然后环境把它映射到每个关节的真实 `range`，再写入 `data.ctrl`。

动作映射核心逻辑在 `_map_action_to_ctrl()`：

```python
normalized = (a + 1.0) / 2.0  # [-1,1] -> [0,1]
ctrl = lo + normalized * (hi - lo)
```

### 3.3 reset：如何放置方块（这是你“方块放桌面上”的关键）

在 `reset()` 里：
- 先用 keyframe 把仿真重置到 `home`
- 然后用 `data.geom_xpos[worktable_top]` 读取桌面碰撞体的世界坐标
- 把 `target_object` 的 x/y 放到桌面碰撞盒范围内（可随机化），避免策略只记住固定位置
- z 放到桌面上方一点：`table_top_world_z + block_half_extents_z + clearance`
- 方块姿态要满足约束：**方块局部 +Z 始终朝向地面（world -Z）**。因此只随机绕世界 Z 轴的 yaw，禁止“翻倒”

关键代码片段（已简化理解）：

```python
mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key_id)
mujoco.mj_forward(self.model, self.data)

table_top_world_xy = self.data.geom_xpos[self.table_geom_id][:2]
table_top_world_z = self.data.geom_xpos[self.table_geom_id][2]

obj_x, obj_y = table_top_world_xy
obj_z = table_top_world_z + self.obj_half_extents[2] + self.obj_clearance

self.data.qpos[self.obj_qposadr : self.obj_qposadr + 3] = [obj_x, obj_y, obj_z]
# 约束：方块局部 +Z 始终朝向地面（world -Z）
yaw = rng.uniform(-pi, pi)
self.data.qpos[self.obj_qposadr + 3 : self.obj_qposadr + 7] = [
    0.0, cos(yaw/2), sin(yaw/2), 0.0
]
```

这确保“方块在桌面范围内随机”，并且不会因为 XML/脚本坐标不一致导致初始摆放错误。

### 3.4 奖励（Reward）与成功（Success）

奖励由两部分组成：
- 位置奖励：末端到方块距离越小越好（越接近 0 越大）
- 姿态奖励：鼓励“更近的那一臂”同时满足 **`tool +Z` 朝下** 且 **与方块轴向对齐**（slowcv 双臂脚本中已由完整四元数误差更新为组合误差）

并加了一个简单碰撞惩罚：如果方块与桌面碰撞接触，则扣一点。

成功判定在 `_compute_reward_done()`：

```python
dL = ||left_tool - block||
dR = ||right_tool - block||

ori_angle_L = orientation_error_tool_to_obj(left_tool_quat, block_quat)
ori_angle_R = orientation_error_tool_to_obj(right_tool_quat, block_quat)

success_left  = (dL < dist_success) and (ori_angle_L < ori_success)
success_right = (dR < dist_success) and (ori_angle_R < ori_success)
success_now = success_left or success_right
```

其中 `orientation_error_tool_to_obj` 在 slowcv 双臂脚本中定义为：
- `tilt_error`：工具局部 `+Z` 与世界 `-Z` 的夹角（要求“始终朝下”）
- `yaw_error`：工具局部 `+X` 与方块局部 `+X` 在 XY 平面的夹角（要求“与方块坐标轴对齐”）
- `ori_error = max(tilt_error, yaw_error)`（用 `max` 保证两项都满足才过阈值）

此外还有 `success_hold_steps`：
- `success_now=True` 只表示“某一步刚满足阈值”
- 只有连续满足 `success_hold_steps` 步，才算一次最终成功（done）

`dist_success_current/ori_success_current` 会随 episode 数逐步收紧（curriculum），让训练从“容易获得正样本”逐步变成“真正 6DOF 对齐”。

### 3.5 关键参数含义（新手最常用）
- `dist_success_current`：距离阈值（米）。用于判断“末端是否足够靠近方块”。
- `ori_success_current`：姿态角阈值（弧度）。用于判断“末端坐标系是否与方块坐标系对齐”。
- `curriculum_episodes`：在多少个 episode 内把阈值从 `*_start` 线性收紧到 `*_end`。这样能避免训练一开始就太难导致几乎没有 success 样本。**默认 5000**（可用 CLI `--curriculum_episodes` 覆盖；旧版代码曾为 200）。
- `success_hold_steps`：成功需要连续满足多少步，才算一次最终成功（`done`）。作用是抑制偶然命中阈值带来的噪声学习。
- `xy_randomize` / `xy_safe_margin`：控制方块在桌面碰撞盒内随机 `x/y`，并保持离桌沿一定安全边距，避免物体采样到不合理区域。

### 3.6 设计约束思路（建模时的“规则”）
- `freejoint`：让方块成为可被碰撞推动/接触带动的动态刚体。
- “方块 z 轴朝向地面”：任务从“同时学会翻转正确姿态 + 位姿对齐”简化为“学会 6DOF 对齐（主要是 yaw 方向）”，减少数值不稳定。
- “任一臂 OR 逻辑”：如果目标是“任一臂完成对齐”，success 不应强制使用“更近那一臂”的硬切换；OR 逻辑更贴合任务本意。

---

## 4. 训练流程（PPO）

训练入口在 `rl_ur_train.py` 的 `main()`：
- 创建 `URDualArmTask1Env`
- 用 `make_vec_env` 组成 vector env（`n_env` 控制并行数量）
- 用 SB3 的 `PPO`：
  - 策略网络 MlpPolicy
  - `n_steps` 控制每次收集多少步经验
  - `batch_size` / `n_epochs` 控制优化频率

训练过程日志里会显示 rollout 的 `success_rate`（由你上面的 success 判定触发）。

---

## 5. 评估流程（success_rate 统计）

评估入口是 `rl_ur_test.py`：
- 加载 PPO 模型
- 循环 `n_episodes` 次
- 每个 episode 从环境 `reset()` 开始
- 使用 `model.predict(..., deterministic=True)` 做确定性控制
- 统计 `info["is_success"]` 的比例，输出 `Task1 success_rate`

---

## 6. 跑例程（你可以直接复制运行）

### 6.1 训练（debug 级别，先跑通流程）

```bash
conda run -n RL_test python rl_policy/rl_ur_train.py \
  --stage static \
  --n_env 1 \
  --total_timesteps 20000 \
  --n_steps 64 \
  --batch_size 64 \
  --n_epochs 2 \
  --learning_rate 3e-4 \
  --log_dir ./logs/ppo_ur_task1_render/ \
  --model_path ./models/ppo_ur_task1_render_model.zip
```

如果你显卡/ CUDA 不可用，代码内部会自动选 `cpu`。

### 6.2 评估（看 success_rate）

```bash
conda run -n RL_test python rl_policy/rl_ur_test.py \
  --stage static \
  --model_path ./models/ppo_ur_task1_render_model.zip \
  --n_episodes 50
```

### 6.3 用可视化看末端如何靠近方块（关键：你要的 demo）

训练可视化：

```bash
conda run -n RL_test python rl_policy/rl_ur_train.py \
  --stage static \
  --render \
  --n_env 1 \
  --total_timesteps 20000 \
  --n_steps 64 \
  --batch_size 64 \
  --n_epochs 2 \
  --learning_rate 3e-4 \
  --log_dir ./ppo_ur_task1_render/ \
  --model_path ./ppo_ur_task1_render_model.zip
```

评估可视化（更适合快速看“末端怎么碰方块”）：

```bash
conda run -n RL_test python rl_policy/rl_ur_test.py \
  --stage static \
  --render \
  --model_path ./ppo_ur_task1_debug_model.zip \
  --n_episodes 1
```

如果你在无显示环境跑 viewer（SSH/服务器），可能需要用 `xvfb-run`（你项目笔记里已经给过方式）。你先告诉我你运行环境是什么（本机桌面还是远程 SSH），我再给你对应命令。

---

## 7. 初学者常见坑（结合我们排错过的点）

### 7.1 `freejoint` 放错位置

必须在 `worldbody` 顶层。

### 7.2 `keyframe` 的 `qpos` 维度不匹配

当你给 scene 引入了 `freejoint`，模型的 `nq` 会变大。
你需要确保：
- `keyframe` 里的 `qpos` 长度等于新模型的 `nq`

### 7.3 success 判定调太严格会导致 success 永远是 0

因为成功判定同时包含“距离 + 姿态（6DOF）”以及“保持步数（`success_hold_steps`）”，一旦设置过严，训练里正样本会变得非常稀少，success 就会长期为 0。

推荐的解决顺序（我们已经验证过）：先用“只看距离”的诊断确认机械臂确实够得到方块；再启用 6DOF success（并用任一臂 OR 逻辑）；最后再用 curriculum 逐步收紧 `dist_success_current/ori_success_current`，让成功真正对应末端坐标系与方块同位姿。

---

## 8. 下一步怎么做（你可以按这个节奏继续）

1. 先用“只看距离”的诊断 success 看机械臂是否能稳定够到方块（距离瓶颈通常比姿态瓶颈更容易处理）
2. 再启用 6DOF success 判定，并使用“任一臂 OR 逻辑”：左臂满足就算成功、右臂满足也算成功
3. 用 `success_hold_steps`（1 -> 3）做稳定性排查，避免偶然满足阈值就结束
4. 最后用 curriculum 逐步收紧 `dist_success_current/ori_success_current`，让成功真正对应你要的末端坐标系与方块同位姿

---
## 9. 2026-03-20 对话合并记录

### 9.1 目标与现状
本轮核心目标是提升 Task1 的 `success_rate`，同时尽量缓解 MuJoCo 数值不稳定（`WARNING: Nan, Inf or huge value in QACC ...`）。

### 9.2 成功率提升：成功口径与奖励一致化
1) 姿态对齐的“口径”需要与 `ori_reward`、`success_now` 保持一致，避免 reward 更容易高、done 却更严格导致学习信号方向相反。
2) 当前实现仍使用工具末端与方块的四元数角度（`_quat_angle(lquat, oquat)`），同时由于方块初始化约束了局部 +Z 朝下，使任务难度主要集中在绕 Z 的变化上（但代码口径仍是四元数角）。
3) curriculum 阈值的作用是“先让成功更容易出现，让策略能学到可行行为，再逐步收紧”。当前默认代码参数为 `dist_success_end=0.12`、`ori_success_end=1.0`，如果你发现 `success_rate` 长期为 0，再考虑继续放宽或延长 `curriculum_episodes`。
4) static 阶段 object 姿态范围会影响正样本的可达性；放宽/收紧策略通常和 curriculum 联动使用。

在“新训练模型”评估中，`Task1 success_rate` 达到约 `0.48 (24/50)`，并继续出现少量 QACC 警告，说明稳定性仍有偶发因素，但 success 明显提升。

### 9.3 数值稳定性处理（QACC）
1) `reset()` 中移除了额外的 `mujoco.mj_step`，避免一开始就触发接触冲击导致爆炸。
2) `step()` 中对数值不稳定做提前截断：检查 `obs` 与 `self.data.qacc`；并在 `frame_skip` 子步中尽早发现不稳定直接 break。
3) 进一步改动 reward 中距离部分的梯度：避免硬 `min(dL,dR)` 引起的“更近那只臂切换”导致梯度不连续；用更平滑的距离聚合方式替代。

### 9.4 MDP/可学习性增强（obs 与控制）
1) 观测增强：obs 维度从 27 扩展到 57，增加 robot proprioception（`robot_qpos/qvel`）与末端到目标的相对位置（`rel_lpos/rel_rpos`）。
2) reward 平滑：双臂距离不再使用硬 `min(dL,dR)`，而是用加权平均/平滑聚合以减少梯度口径突变。
3) 动作改为增量控制（Δq）：action 映射为 `delta_q = action * delta_q_max`，再叠加到 `prev_ctrl_joint` 得到关节目标角。

### 9.5 TensorBoard 训练进度可视化
在 `rl_ur_train.py` 中加入 SB3 Callback `SuccessTensorboardCallback`，记录并写入 TensorBoard：
- `rollout/mean_reward`
- `rollout/success_now_mean`
- `rollout/success_rate`
- `rollout/unstable_ratio`

训练完成后可用：
`tensorboard --logdir <log_dir> --port 6006`
查看曲线。

### 9.6 CLI 指令（无 render）
训练（默认双臂、随机目标；会自动保存 `*_vecnormalize.pkl`）：
```bash
conda run -n RL_test python rl_policy/rl_ur_train.py \
  --stage static \
  --n_env 4 \
  --total_timesteps 2000000 \
  --n_steps 4096 \
  --batch_size 1024 \
  --n_epochs 10 \
  --learning_rate 3e-4 \
  --lr_final 1e-5 \
  --log_dir ./logs/ppo_ur_task1_run_v2/ \
  --model_path ./models/ppo_ur_task1_run_v2_model.zip
```

验证（自动加载 `./models/*_vecnormalize.pkl`；确保 success_rate 口径一致）：
```bash
conda run -n RL_test python rl_policy/rl_ur_test.py \
  --stage static \
  --model_path ./models/ppo_ur_task1_run_v2_model.zip \
  --n_episodes 50
```

单臂 debug（固定目标 baseline，建议先测是否能稳定成功）：
训练（只用左臂、关闭随机目标）：
```bash
conda run -n RL_test python rl_policy/rl_ur_train.py \
  --stage static \
  --n_env 4 \
  --single_arm left \
  --fixed_target \
  --success_bonus 100 \
  --collision_penalty_scale 0 \
  --action_penalty_scale 0 \
  --total_timesteps 300000 \
  --n_steps 4096 \
  --batch_size 1024 \
  --n_epochs 10 \
  --learning_rate 3e-4 \
  --lr_final 1e-5 \
  --log_dir ./logs/ppo_ur_task1_left_fixed_v2/ \
  --model_path ./models/ppo_ur_task1_left_fixed_v2_model.zip
```

验证（需要与训练的 `--single_arm/--fixed_target` 一致）：
```bash
conda run -n RL_test python rl_policy/rl_ur_test.py \
  --stage static \
  --single_arm left \
  --fixed_target \
  --model_path ./models/ppo_ur_task1_left_fixed_v2_model.zip \
  --n_episodes 50
```

### 9.7 MuJoCo viewer 暂停相关
1) viewer 是否能“暂停”取决于代码里是否仍在调用 `mujoco.mj_step`。
2) 使用键盘回调实现暂停/继续时，修复了 `nonlocal paused` 导致的 SyntaxError，改为模块级 `global paused`，并使用 Space 键切换。

### 9.8 代码加载与 keyframe 相关排错（home ctrl size）
出现过 MuJoCo 报错：
`invalid ctrl size, expected length 12`
原因是 `scene_dual_arm.xml` 的 `home ctrl` 长度曾被设置成 19（与执行器数量 nu=12 不一致）；之后修正为 12 个数（匹配 nu）。

---
## 10. 2026-03-21 本轮对话合并记录（入门流程 + 代码诊断修复 + 提升正确率）

### 10.1 你这次给出的“致命缺陷诊断与直接修复方案”（原文整段）
```text
以下是具体的专业诊断与直接的修复方案：1. 致命缺陷：观测空间丢失“本体感受”在 _get_observation (第 225 行) 和 obs_dim (第 132 行) 的定义中，你没有将机器人自己的关节角度（qpos）和速度（qvel）放入观测中。后果： 你的神经网络目前只知道末端坐标和目标坐标。如果不输入当前 12 个关节的角度，神经网络在数学上绝对无法建立“关节转动与末端位移”的正向运动学映射，模型永远无法收敛。修复： 必须把前 12 个关节的 qpos 和 qvel 拼接到 obs 中。Python    # 替换 _get_observation 函数
    def _get_observation(self) -> np.ndarray:
        lpos, lquat = self._get_site_pos_quat(self.left_site_id)
        rpos, rquat = self._get_site_pos_quat(self.right_site_id)
        opos, oquat, olinvel, oangvel = self._get_object_pos_quat_vel()
        
        # 必须加入本体感受 (假设前12个是你的 UR 关节)
        robot_qpos = self.data.qpos[:12].astype(np.float32)
        robot_qvel = self.data.qvel[:12].astype(np.float32)

        # 强烈建议把相对位置也加回来
        rel_lpos = opos - lpos
        rel_rpos = opos - rpos

        obs = np.concatenate([
            robot_qpos, robot_qvel,        # 24
            lpos, lquat, rpos, rquat,      # 14
            rel_lpos, rel_rpos,            # 6
            opos, oquat, olinvel, oangvel  # 13
        ]).astype(np.float32)
        return obs
注：请务必将 __init__ 中的 obs_dim 同步修改为 57。2. 严重逻辑问题：动作空间退化为“带限幅的绝对控制”在 step 函数 (第 286 行) 中，你使用了 ctrl_target = self._map_action_to_ctrl(action)，随后又用 np.clip(delta, -self.max_delta_ctrl, ...) 进行了物理截断。后果： 这是强化学习中典型的策略失真 (Policy Distortion)。网络以为自己输出了一个直接到达目标位置的动作（比如转动 $2.0$ 弧度），期望获得高分；但环境强制把动作截断成了 $0.15$ 弧度，导致实际得分极低。网络的 Value 估计会完全崩溃。修复： 废弃 _map_action_to_ctrl，直接将网络输出的 $[-1, 1]$ 视为增量 $\Delta q$。Python    # 替换 step 函数的开头部分
    def step(self, action):
        # 1. 直接将 action 映射为增量 (无需经过 _map_action_to_ctrl)
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        delta_q = a * self.max_delta_ctrl  # 直接得出增量
        
        # 2. 在上一步的指令上累加
        ctrl_target = self.prev_ctrl + delta_q

        # 3. 降级测试支持
        if self.single_arm == "left":
            ctrl_target[6:] = self.prev_ctrl[6:]
        elif self.single_arm == "right":
            ctrl_target[:6] = self.prev_ctrl[:6]

        # 4. 限位保护
        lo = self.joint_limits[:, 0]
        hi = self.joint_limits[:, 1]
        ctrl = np.clip(ctrl_target, lo, hi).astype(np.float32)
        self.prev_ctrl = ctrl.copy()
        
        # ... 之后正常赋给 self.data.ctrl[aid] ...
3. 奖励梯度断裂：min(dL, dR) 被加回去了在 _compute_reward_done (第 254 行) 中，对于双臂模式，你使用了 d = min(dL, dR)。后果： 如同我们之前分析的，当“更近的手臂”发生切换时，梯度会发生不连续的突变，导致网络无所从事。修复： 使用求和或者 Soft-min 来代替。Python        # 替换对应的距离惩罚逻辑
        else:
            # 双臂协同模式：用平滑求和替代硬 min
            # 任何一只手靠近都会得到奖励
            d = 0.5 * dL + 0.5 * dR 
            # 姿态同样可以使用加权或者求和
            ori_angle = 0.5 * ori_angle_L + 0.5 * ori_angle_R
4. 值得肯定的工程架构除了上述回退的底层逻辑，你新增的架构设计非常专业：--fixed_target 与 --single_arm：这是解决复杂 RL 问题的最佳 Debug 实践（Curriculum Learning 的雏形）。学习率线性衰减 (LR Schedule)：能有效帮助 PPO 在训练后期稳定在局部最优解内，减少震荡。VecNormalize 保存：第 424 行的 env.save 极其关键，这保证了你测试时的观测分布与训练时一致。
根据这个内容修改我的训练代码@rl_policy/rl_ur_train.py
```

### 10.2 我们把上述诊断“落地”到了哪些代码（关键差异点）
1. **观测维度从 27 -> 57，并补齐本体感受**  
   - `robot_qpos(12)` + `robot_qvel(12)` + 两末端位姿 + 末端到方块相对位置 + 方块状态拼接。
2. **动作不再做“绝对 ctrl 映射 + 额外 clip 截断”**  
   - `action in [-1,1]` 直接解释为 `delta_q = action * max_delta_ctrl`，再叠加到 `prev_ctrl` 得到新的关节目标。
3. **双臂距离奖励不使用硬 `min(dL,dR)`**  
   - 改为加权平均：`d = 0.5*dL + 0.5*dR`（姿态同样加权平均），减少“更近手臂切换”带来的梯度不连续。
4. **训练/验证环境必须一致**  
   - `rl_policy/rl_ur_test.py` 也同步做了相同的 obs/action/reward 逻辑更新，否则会出现观测维度不匹配或 reward/done 口径错位。

### 10.3 新手理解：为什么这些改动会显著提升收敛与正确率？
- **观测包含关节 qpos/qvel**：  
  PPO 学到的是“在当前状态下，采取某种动作会让未来 reward 变大”。没有关节本体感受，策略就无法确定“我现在把关节转到了哪里”，也就无法学到“关节变化 -> 末端位移变化”的可学习映射。
- **动作用增量 Δq**：  
 强行把“策略输出的幅度”裁剪成“固定最大步长”会造成策略梯度和真实执行不一致（策略以为自己能做更大变化，但环境把变化吞掉了），这会导致 value/advantage 学得很差。
- **避免硬 min 的梯度跳变**：  
 `min(dL,dR)` 会在“哪只手更近”发生切换的瞬间改变优化器接收到的梯度方向。加权平均/平滑聚合能让梯度随状态连续变化，从而更稳定。

### 10.4 提升 `success_rate` 的“可执行”调参顺序
1. **先单臂可达，再双臂**  
   用 `--single_arm left/right` + `--fixed_target` 跑通，确认能稳定靠近/对齐，再把 `--single_arm none` 打开双臂任务。
2. **确认训练/验证一致性**  
   obs/action/reward 改动后，旧模型和旧 `*_vecnormalize.pkl` 可能不再匹配。务必用新版本训练得到的新模型 + 对应 vecnormalize 统计文件。
3. **把成功判定调到“能学到”**  
   如果 `success_rate` 长期为 0：优先放宽 `dist_success_end/ori_success_end` 或延长 `curriculum_episodes`，保证正样本出现；否则 PPO 只能靠噪声更新。
4. **调动作尺度（max_delta_ctrl）**  
   `max_delta_ctrl` 太大容易撞击/数值不稳定；太小会让进展过慢、正样本稀少。建议先在 `0.05~0.2` 尝试区间快速扫。
5. **用 success_hold_steps 抑制偶然成功**  
   当前默认 `success_hold_steps=1`。当训练成功率抖动很大时，可尝试提高到 `2~3`，让 done 更可靠。
6. **必要时再开碰撞/动作惩罚**  
   当前 `collision_penalty_scale` 与 `action_penalty_scale` 默认都为 0（更利于探索）。如果出现大量不合理碰撞/抖动，再逐步加惩罚约束策略。

### 10.5 训练与验证 CLI（直接可复制）
1) 训练（建议使用新目录/新模型名，避免与旧 obs_dim 的模型混用）
```bash
conda run -n RL_test python rl_policy/rl_ur_train.py \
  --stage static \
  --n_env 4 \
  --total_timesteps 2000000 \
  --n_steps 4096 \
  --batch_size 1024 \
  --n_epochs 10 \
  --learning_rate 3e-4 \
  --lr_final 1e-5 \
  --log_dir ./logs/ppo_ur_task1_run_v2/ \
  --model_path ./models/ppo_ur_task1_run_v2_model.zip
```

2) 验证/评估
```bash
conda run -n RL_test python rl_policy/rl_ur_test.py \
  --stage static \
  --model_path ./models/ppo_ur_task1_run_v2_model.zip \
  --n_episodes 50
```

3) 单臂 debug（固定目标 baseline，优先确认可达性）
```bash
conda run -n RL_test python rl_policy/rl_ur_train.py \
  --stage static \
  --n_env 4 \
  --single_arm left \
  --fixed_target \
  --success_bonus 100 \
  --collision_penalty_scale 0 \
  --action_penalty_scale 0 \
  --total_timesteps 300000 \
  --n_steps 4096 \
  --batch_size 1024 \
  --n_epochs 10 \
  --learning_rate 3e-4 \
  --lr_final 1e-5 \
  --log_dir ./logs/ppo_ur_task1_left_fixed_v2/ \
  --model_path ./models/ppo_ur_task1_left_fixed_v2_model.zip
```

### 10.6 你要求“把这一整段话也加进md文件中”（原文）
```text
@task1_train_notes.md 整理这个md文件。并将目前的对话合并到md文件中。注意，你是强化学习的专家，我是新手，你需要告诉我流程和原理，分析重要的代码以及参数，并告诉我怎么样可以提升正确率。把这一整段话也加进md文件中
```

### 10.7 本轮对话合并：`max_delta_ctrl` 几乎无效与肩/肘大幅摆动（完整记录）

本节合并 **2026-03-21** 起与用户关于「缩小 `max_delta_ctrl` 仍抖动」的对话：含**用户现象描述**、**机理分析**、**已落地的工程修改**与**可复现 CLI**。

#### 10.7.1 用户观测（原话摘要）

- 将 `--max_delta_ctrl` 设为 **`0.01`** 时，**几乎仍无改善**。
- **从初始位到靠近方块的过程中**就会出现摆动；此时末端与小方块在高度上**仍有一段距离**（尚未接触）。
- **`right_shoulder_lift_joint` 的摆动更明显**（相对肘部等）。

#### 10.7.2 为何「缩小 Δq」常常解决不了 30°–60° 级别的摆动？

核心结论：**大幅摆动往往不主要由「每步关节目标变化量」单独决定**，而是 **高刚度位置伺服 + 策略输出换向 + 重力/惯性** 共同作用。

1. **MuJoCo `position` 执行器刚度（`kp`/`kv`）**  
   - 在 `mujoco_asserts/universal_robots_ur5e/dual_ur5e.xml` 中，肩/肘类关节曾使用 **较高的 `kp`（约 600–700 量级）**。  
   - **很小的关节角跟踪误差**也会通过 PD 产生**很大的力矩**；策略若在正负方向上频繁微调，易与重力、惯性形成 **PD 极限环（limit cycle）**，视觉上像**大角度来回甩**。  
   - 因此仅把 `max_delta_ctrl` 调到 `0.01`，只能缩小「目标角」的步进，**不一定能消除**这种伺服层面的振荡倾向。

2. **策略在 `[-1, 1]` 上高频换向**  
   - 若网络输出在 **+1 / -1** 附近交替，只对 **`delta_q`** 做指数平滑时，仍可能出现明显抖动。  
   - 更直接的做法是对**归一化动作 `a` 本身**先做 **EMA（低通）** 或 **死区（deadzone）**，再计算 `delta_q = a * max_delta_ctrl`。

3. **为何 `right_shoulder_lift_joint` 更明显？**  
   - 该轴承担**抬升/压低**整段手臂与末端高度的作用，且与**重力矩**耦合强；在高 `kp` 下更容易成为**主导振荡模态**，肘部常表现为**耦合跟随**。

#### 10.7.3 已落地的工程修改（文件与参数）

| 类别 | 内容 |
|------|------|
| **XML 伺服增益** | `dual_ur5e.xml`：适度降低 `position` / `position_limited` 的 **`kp` / `kv`**，减轻肩/肘 PD 极限环（具体数值以仓库内 XML 为准）。 |
| **环境动作预处理** | `rl_ur_train.py`、`rl_ur_test.py`：新增 CLI 参数 **`--action_deadzone`**、**`--action_raw_lpf_coef`**（对 **`a` 先** EMA / 死区，再乘 `max_delta_ctrl`）；与原有 **`--action_smoothing_coef`**（对 `delta_q` 平滑）可叠加使用。 |

**参数含义（便于新手）**

- **`action_deadzone`**：若 `|a| < 死区阈值`，则置 0，抑制「噪声型」小幅指令。  
- **`action_raw_lpf_coef`**：\(a \leftarrow (1-c)\,a + c\,\text{prev\_a}\)，**`c` 越大越平滑**（典型可试 **0.85–0.95**）。  
- **`action_smoothing_coef`**：对 **`delta_q`** 的平滑；与 **`action_raw_lpf_coef`** 作用层级不同，前者在「已缩放后的增量」上，后者在「归一化动作」上。

**重要提示**：修改 XML 的 `kp/kv` 会改变**仿真动力学**；旧模型是在旧动力学下训练的，**可先肉眼看抖动是否减轻**；若成功率或行为明显变差，需用**当前 XML** **重新训练**并保留对应 `*_vecnormalize.pkl`。

#### 10.7.4 验证 CLI（渲染，建议组合）

```bash
conda run -n RL_test python rl_policy/rl_ur_test.py \
  --render --stage static \
  --model_path ./models/你的模型.zip \
  --single_arm right --fixed_target \
  --max_delta_ctrl 0.03 \
  --action_raw_lpf_coef 0.92 \
  --action_deadzone 0.08 \
  --action_smoothing_coef 0.85 \
  --n_episodes 2
```

训练脚本同样支持上述参数，例如：

```bash
conda run -n RL_test python rl_policy/rl_ur_train.py \
  --stage static --n_env 4 \
  --max_delta_ctrl 0.05 \
  --action_raw_lpf_coef 0.90 \
  --action_deadzone 0.05 \
  --action_smoothing_coef 0.80 \
  --log_dir ./logs/ppo_ur_task1_run_v2/ \
  --model_path ./models/ppo_ur_task1_run_v2_model.zip
```

（按实际模型名与日志目录调整。）

#### 10.7.5 若仍不满意时的下一步（备忘）

- 在**不**破坏任务的前提下，可对 **`right_shoulder_lift`** 单独再降一档 `kp/kv`（需改 XML）。  
- 训练侧可增加 **关节速度惩罚** 或 **靠近关节限位惩罚**（需在 `reward` 中显式加项，当前代码未默认开启）。  
- 确认 `VecNormalize` 与训练时 **obs 统计** 一致，避免把「归一化后的观测抖动」误判为机械抖动。

**本节与上文关系**：§10.2–§10.4 描述 MDP 与 PPO 侧改动；**§10.7 专门记录本轮「仿真控制刚度 + 动作预处理」对话**，与「观测/奖励/Δq 逻辑」互补。

### 10.8 动作输出边界收缩（`--action_bound`）

若任务主要在 **home 附近** 小范围运动，可将策略输出从默认 **`[-1, 1]`** 收窄为对称区间 **`[-b, b]`**（如 **`b=0.5`**），在 MDP 层面直接减少无效探索；环境内对 `a` 的 `clip` 与 `gym.spaces.Box` 一致。

- **训练 / 验证须使用相同的 `action_bound`**，否则动作分布与 PPO 训练时不一致。  
- 旧模型是在 `action_bound=1.0` 下训练的，**改用更小的 `b` 需要重新训练**（或继续微调）。  
- CLI：`--action_bound 0.5`（默认 `1.0`）。

示例：

```bash
conda run -n RL_test python rl_policy/rl_ur_train.py \
  --stage static --n_env 4 \
  --single_arm right --fixed_target \
  --action_bound 0.5 \
  --max_delta_ctrl 0.08 \
  --log_dir ./logs/ppo_ur_task1_right_bound05/ \
  --model_path ./models/ppo_ur_task1_right_bound05_model.zip
```

### 10.9 慢课程 + 单臂左基线（成功率曲线平滑上升）

**目标**：让距离/姿态成功阈值在更多回合内逐步收紧（默认 **5000** 个 episode 内从宽到严），便于曲线平滑涨到高成功率；死区宜小（**0.01** 或 **0**），保留末端微调能力。

- **`--curriculum_episodes`**：课程长度（默认 **5000**，原为代码内写死 200）。可与训练日志中的 `episode` 对齐理解：\(t=\min(1,\texttt{episode\_count}/\texttt{curriculum\_episodes})\)。

**单臂左 + 固定目标 + 慢课程 + 小死区（示例）**

```bash
conda run -n RL_test python rl_policy/rl_ur_train.py \
  --stage static \
  --n_env 4 \
  --single_arm left \
  --fixed_target \
  --curriculum_episodes 5000 \
  --action_deadzone 0.01 \
  --action_raw_lpf_coef 0.90 \
  --total_timesteps 2000000 \
  --log_dir ./logs/ppo_ur_task1_run_v3/ \
  --model_path ./models/ppo_ur_task1_run_v3_model.zip
```

验证时请与训练保持一致的 **`curriculum_episodes`、动作相关参数**：

```bash
conda run -n RL_test python rl_policy/rl_ur_test.py \
  --stage static \
  --model_path ./models/ppo_ur_task1_run_v3_model.zip \
  --single_arm left \
  --fixed_target \
  --curriculum_episodes 5000 \
  --action_deadzone 0.01 \
  --action_raw_lpf_coef 0.90 \
  --n_episodes 50
```

### 10.9.1 自动课程（外部监控 + 动态换挡，`--auto_curriculum`）

**思路**：用 `CurriculumCallback` 统计最近 N 个**已结束回合**的成功率，≥ 阈值（默认 **85%**）则通过 `VecEnv.env_method("set_curriculum_stage", ...)` 让所有并行子环境**不停训**切换难度：

- **Stage0**：右臂 + 固定目标  
- **Stage1**：右臂 + 随机目标  
- **Stage2**：左臂 + 固定目标  
- **Stage3**：左臂 + 随机目标  
- **Stage4**：双臂 + 随机目标  

**启动示例**（阶段更多时建议总步数 **5e6～1e7** 量级）：

```bash
conda run -n RL_test python rl_policy/rl_ur_train.py \
  --stage static \
  --n_env 1024 \
  --auto_curriculum \
  --auto_curriculum_threshold 0.85 \
  --auto_curriculum_window 100 \
  --total_timesteps 5000000 \
  --log_dir ./logs/ppo_ur_task1_auto_curriculum/ \
  --model_path ./models/ppo_ur_task1_auto_curriculum_model.zip
```

启用 `--auto_curriculum` 时会**强制**初始为 `single_arm=right`、`fixed_target=True`（与 Stage0 一致）。与 **6DOF 阈值课程**（`dist_success_*` / `ori_success_*`）可同时使用：前者管「单臂/双臂与目标是否随机」，后者管「成功判据多严」。

**易错点（曾导致单侧 success 率异常低）**：`make_env()` 里 **Stage0 的 `args.single_arm` / `fixed_target`** 必须与代码中的 **`auto_stages[0]`** 完全一致。若初始环境练的是 **右臂**，而 `auto_stages` 在晋升时才第一次写成 **左臂**，则左臂从未经历「固定目标」热身、直接上随机目标，**左臂成功率会崩**。当前仓库在 `main()` 里对二者做了校验；改课程表时务必**同时改**初始参数与 `auto_stages` 列表。

### 10.10 奖励与数值不稳定（2026 更新）

- **距离项**：由「越远越扣分」`-arctan(d)` 改为 **正向塑形** `pos_reward = 1/(1+5*d)`：越近越接近 1，越远趋近 0，避免仅靠距离项造成的持续负漂移（姿态项等仍按原逻辑叠加）。
- **MuJoCo 不稳定**：`qacc` 非有限等触发早终时，回报由 **-10** 改为 **-1000**，强烈抑制导致仿崩溃的策略行为。

### 10.13 腕部与桌面碰撞惩罚

- 若 **`left/right_*_wrist_1_link` / `wrist_2_link` / `wrist_3_link`** 上任一 **碰撞 geom** 与 **`worktable_top`** 发生接触，本步奖励扣除 **`wrist_table_penalty_scale`**（默认 **2.0**，CLI `--wrist_table_penalty_scale`，0 关闭）。
- 若场景里对 **wrist_3 + 桌** 做了 `<contact><exclude .../></contact>`，则 **wrist_3 与桌面不会产生接触**，该项惩罚对 wrist_3 不生效；wrist_1/2 仍生效。

### 10.12 末端与 `worktable_top` 持续碰撞（刮桌面）

- **原因**：腕部 `eef_collision` 胶囊在方块附近下探时，易与桌面碰撞体长时间接触、产生摩擦与抖动。
- **处理**：在 `scene_dual_arm.xml` 中增加 `<contact><exclude .../></contact>`：`worktable` 分别与 `left_wrist_3_link`、`right_wrist_3_link` 排除碰撞，**仅关闭腕部末端与桌面的碰撞**；前臂/上臂与桌面、方块与桌面仍正常碰撞。若需完全物理真实，可删 `exclude` 并改用奖励惩罚桌面接触。

### 10.11 `success_rate` 长期为 0：先别急着退回 `curriculum_episodes`

- **放慢课程（如 5000 episode）** 只会让「距离/姿态阈值」在更长时间内保持**更宽松**，**不会**解释「完全学不会成功」；退回 200 反而会让阈值**更早收紧**，一般**不利于**破零。
- **更常见原因：触碰即冻结（touch-stop）**：一旦末端碰到方块就锁死关节，若此时还未同时满足「距离 + 姿态」成功条件，**整局无法再调整**，`success_rate` 会长期为 0。  
  **默认已关闭 touch-stop**（无需 CLI）；需要「碰方块就冻结」时用 **`--touch_stop`**（演示用，不利 RL 学 6DOF）。

### 10.14 对话纪要：6DOF 课程、自动课程、成功冻结与现象解释（2026-03）

本节把近期实现/讨论合并进笔记，便于论文与复现实验对照。

#### 10.14.1 六自由度「阈值」课程（与自动课程正交）

在 **距离 + 姿态** 同时满足时才记成功的前提下，可用 **episode 计数** 把成功阈值从宽慢慢收到严（线性插值），避免一上来太难没有正样本。

- **CLI**：`--dist_success_start` / `--dist_success_end`、`--ori_success_start` / `--ori_success_end`、`--curriculum_episodes`（模块内 `DEFAULT_*` 与 `rl_ur_test.py` 应对齐）。  
- **语义**：`end` 比 `start` **更严** = 阈值**更小**（距离更小、姿态角更小）。  
- **训练与验证**须使用**同一套**参数（及 VecNormalize 等），否则成功率不可比。

#### 10.14.2 自动课程 `--auto_curriculum`（先右后左，共五段）

见 **§10.9.1**。当前默认路线为：

| 阶段 | 控制臂 | 方块位姿 |
|------|--------|----------|
| 0 | 右 | 固定 |
| 1 | 右 | 随机 |
| 2 | 左 | 固定 |
| 3 | 左 | 随机 |
| 4 | 双臂 | 随机 |

**生成位置（与 `reset()` 一致）**：`stage 0/1` 下单臂为 **右** 时，方块 **x/y 锚在右臂末端在桌面上的投影**（裁剪进桌面碰撞盒）；`stage 2/3` 单臂为 **左** 时锚在 **左臂末端投影**；**固定**与**随机**均使用该锚点（不再把固定目标放在桌面几何中心）。`stage 4` **双臂** 时，锚点取 **home 位形下左右末端在桌面上的中点**；随机采样时要求 **左、右臂末端到方块距离均**小于可达阈值（双臂重叠可达区），若多次拒绝则回退到该锚点。

**晋升**：`CurriculumCallback` 用最近 N 个**已结束回合**的成功率，≥ 阈值（默认 0.85）则调用 `set_curriculum_stage`；**不中断训练**。阶段变多后建议**加大总步数**（如 **5e6～1e7**）或略**降低**阈值，避免长期卡段。

#### 10.14.3 为何「碰到方块」后末端仍像沿原方向动？

几件事经常混在一起，需要分开说：

1. **碰到 ≠ 代码里的 success**：成功是 **6DOF**（距离与姿态角都在 `dist_success_current` / `ori_success_current` 内）。仅接触或距离够但姿态不对时，策略会继续输出动作。  
2. **成功判在「本步控制 + 物理子步」之后**：本步 `ctrl` 已积分，关节仍有**速度/惯性**，画面可能仍有一点滑动。  
3. **动作死区 / 对 raw action 的 LPF / 对 delta 的 smoothing** 会让增量在时间上**延续**，成功附近也像「沿原趋势」动。  
4. **`--touch_stop`** 默认关；打开后会在**首次接触**时冻结（与 6DOF 成功不是同一逻辑）。

#### 10.14.4 成功当步「零速度 + 锁控制」：`success_freeze`（训练/测试一致）

为减轻「对齐成功那一帧仍滑移」的观感，环境中增加了 **`_apply_success_freeze()`**（默认开启）：

- 机械臂关节 **qvel → 0**，**ctrl 锁到当前 qpos**（限位内），并清空 **`prev_ctrl` / `prev_delta_q` / `prev_a_filtered`**；  
- 方块 **freejoint** 线/角速度置零；  
- 调用 **`mujoco.mj_forward`**。

**CLI**：`--no_success_freeze` 关闭上述行为（默认**不**传 = 开启）。**训练与验证建议一致**；若要与**旧 checkpoint（未冻结动力学）**严格对齐，验证时可加 `--no_success_freeze`。

#### 10.14.5 单臂阶段随机方块：为何曾「总在右臂下方」、如何修

场景中 **`worktable_top` 的几何中心**在世界系里**不一定在左右臂中间**（底座与桌子相对位置导致中心更靠近某一侧）。旧逻辑在 **`fixed_target=False`** 时以桌面中心为基准做均匀扰动；若拒绝采样 25 次仍不通过可达性检验，会**回退到中心**。结果是：**左臂单臂训练**时，方块仍经常落在**更靠近右臂工作区**的区域，左臂 `success_rate` 会异常低。

**当前实现**（`reset()`）：**单臂**（`single_arm=left|right`）时锚点为 **该臂末端 site 的 (x,y)**（裁剪进桌面盒）；**固定目标**（`fixed_target=True`）也放在该锚点，不再用桌面几何中心。**双臂**（`single_arm=none`）锚点为 **home 下左右末端在桌面上的中点**；随机采样时要求 **左、右臂到方块距离均**小于可达阈值（重叠区），失败则回退锚点。

#### 10.14.6 观测 / 奖励 / 并行与惩罚渐进（与 `rl_ur_train.py` 对齐）

- **观测**：基础约 **57 维**（含双臂 **qpos/qvel** 本体感受）；`--no_enriched_obs` 可关。**富集观测**（默认开）再拼 **EE 局部相对位姿 + 方块线速度 EMA**，约 **72 维**。  
- **距离奖励**：`--reward_distance_mode`：`inverse` / `gaussian` / `hybrid`；`--gaussian_dist_alpha` 调高斯形状。  
- **平滑与能耗**：`--action_delta_penalty_scale`（\(\|a_t-a_{t-1}\|^2\)）、`--qvel_penalty_scale`（关节速度 L2）。  
- **向量化**：`--n_env>1` 时使用 **`SubprocVecEnv`** 多进程采样。  
- **惩罚渐进**：`--penalty_ramp_timesteps T` 时，碰撞/腕桌惩罚在 **T** 步内从 **0 线性拉满**（`PenaltyRampCallback` + 环境 `set_penalty_ramp_coef`）；`make_env` 将 `penalty_ramp_init=0`。

**PPO**：`--n_steps`、`--batch_size`、`--use_lr_schedule` 等见脚本 `--help`。

**双机场景脚本** `rl_ur_train_double.py` / `rl_ur_test_double.py` 若需相同能力，请与单臂训练脚本手动对齐。

### 10.15 对话合并：慢速动态「全局匀速」、独立脚本与方块初始方向（2026-03）

本节合并对话中关于 **`--stage slow_dynamic`**、**全局匀速**实现方式、以及 **方块线速度 / 姿态 / 角速度**如何采样的说明；便于与 §3.3 / §5 / README「慢速动态」对照。

#### 10.15.1 为何单独提供 `*_slowcv.py` 脚本

在保留原有 **`rl_ur_train_double.py` / `rl_ur_test_double.py` 不改**的前提下，另增一对脚本，专门实现 **slow_dynamic 下目标方块整局保持恒定速度**（避免因摩擦导致速度衰减）：

- **训练**：`rl_policy/rl_ur_train_double_slowcv.py`
- **评估**：`rl_policy/rl_ur_test_double_slowcv.py`

默认日志与模型路径与原版区分（例如 `./logs/ppo_ur_task1_slowcv/`、`./models/ppo_ur_task1_slowcv_model.zip`，以文件内常量为准）。训练与评估须**成对使用**，且 **`--obj_lin_speed` 与 `--stage`** 与训练一致（`--obj_ang_speed` 为保留参数，slow_dynamic 下角速度为 0）。

#### 10.15.2 「全局匀速」在代码里如何实现

在 **`reset()`** 中为本回合采样方块的线速度 `lin_v`、角速度 `ang_v`，写入 MuJoCo `data.qvel`，并**拷贝**到：

- `_obj_const_lin_vel`（3 维）
- `_obj_const_ang_vel`（3 维）

在 **`step()`** 的每个物理子步中，于 **`mujoco.mj_step` 之后**（以及与机械臂 hold 等逻辑之后）调用 **`_apply_object_constant_velocity()`**：若 `stage == "slow_dynamic"`，则将方块 freejoint 对应自由度上的 **`qvel` 强制重置为上述常量向量**。

效果：**速度大小与方向在本回合内保持不变**（不受桌面摩擦等使速度慢慢变小的影响）；观测中的 **`olinvel` / `oangvel`** 在强制后与常量一致，便于策略利用速度信息做追踪。

#### 10.15.3 方块平移方向与姿态（以 `*_slowcv.py` 为准）

以下均针对 **`slow_dynamic`** 且非「固定目标 / 纯 static」时。

**（1）姿态（方块平放在桌面、绕竖直轴朝向）**

- 约束：**方块局部 +Z 指向地面（世界 −Z）**，避免翻倒。
- 用 **绕世界 Z 的 yaw**（弧度）写四元数，MuJoCo 顺序 `[w,x,y,z]` 下形如：  
  `[0,\ \cos(yaw/2),\ \sin(yaw/2),\ 0]`。
- **`yaw` 来自 `reset()` 里桌面上的位置/采样流程**（与可达性、锚点等有关）。

**（2）线速度（沿桌面长边，右→左）**

- 场景世界系：**左臂 base 约 `Y=+0.232`，右臂约 `Y=-0.232`**（见 `dual_ur5e.xml`），**从右臂侧移向左臂侧**为 **世界 +Y** 方向。
- **`worktable_top`** 碰撞盒在桌面上沿 **Y** 方向较长（`size` 中第二维为半长边），平移沿 **Y** 与「从一端到另一端」一致。
- 线速度大小：`speed = obj_lin_speed`（CLI **`--obj_lin_speed`**，默认约 **0.05 m/s**）。
- 世界系恒为：  
  **`lin_v = [0,\ speed,\ 0]`**  
  即 **仅沿 Y 匀速平移**，无 X / Z 分量。

**（3）角速度**

- **`ang_v = 0`**：慢速动态下**无自转**，仅平移。

**说明**：在 **`rl_ur_*_double_slowcv.py`** 且 **`stage=slow_dynamic`** 时，方块 **不再**放在桌面几何「最右缘」（该点常超出右臂工作空间）；改为以 **home 位形下右臂末端在桌面上的投影**为锚（`single_arm=left` 时改为左臂），**X/Y** 在 `worktable_top` 范围内小幅随机，并略向 **+Y**（皮带下游）偏移；随后 **`lin_v=[0,+v,0]`** 沿 **+Y** 运动。训练与测试脚本已对齐。若要「主要由右臂抓取」，可 **`--single_arm right`**；`nearest_active` 下开局木块靠近右臂时通常也是右臂先动。

#### 10.15.4 CLI 与复现提示

- **`--obj_lin_speed`**：沿 **+Y** 的线速度大小（m/s）。
- **`--obj_ang_speed`**：保留兼容；slow_dynamic 下**不使用**（角速度为 0）。
- 评估时 **`--obj_lin_speed` / `--stage`** 须与训练一致。

### 10.16 对话合并：姿态成功判定更新（+Z 朝下 + 轴向对齐，2026-03-26）

本节合并本轮对话中对“姿态定义”的最终结论，并统一训练/测试口径，避免再出现“只要求朝下但不对齐”或“完整四元数过约束导致腕部奇异”的混淆。

#### 10.16.1 需求澄清（最终口径）

用户最终要求不是“只看 `tool +Z` 朝下”，而是**同时**满足：

1. **`tool +Z` 始终朝下**（对齐世界 `-Z`）  
2. **工具坐标轴与方块坐标轴对齐**（在桌面任务中重点是绕 `Z` 的 heading/yaw 对齐）

因此姿态成功条件应保留“垂直下压”与“轴向对齐”两项，而不是只保留单轴倾斜角。

#### 10.16.2 已落地代码（train/test 同步）

文件：
- `rl_policy/rl_ur_train_double_slowcv.py`
- `rl_policy/rl_ur_test_double_slowcv.py`

已新增并使用如下逻辑：

- `z_tilt_angle_from_quat(q)`：计算工具局部 `+Z` 与世界 `-Z` 的夹角 `tilt_error`
- `yaw_angle_between_tool_and_obj_x_axes_xy(lquat, oquat)`：比较工具与方块局部 `+X` 在 XY 平面的夹角 `yaw_error`
- `orientation_error_tool_to_obj(lquat, oquat) = max(tilt_error, yaw_error)`

并将 `_success_now()` 与 `_compute_reward_done()` 中原先的：

`_quat_angle(tool_quat, obj_quat)`

替换为：

`_orientation_error_tool_to_obj(tool_quat, obj_quat)`

这样训练与评估对“姿态达标”的定义一致。

#### 10.16.3 为什么用 `max(tilt, yaw)` 而不是加权和

- 用 `max` 时，`ori_error < ori_success` 等价于“**tilt 和 yaw 两项都小于阈值**”。
- 若用加权和，可能出现“一项很差但被另一项抵消”的误判，不符合“必须既朝下又对齐轴向”的任务约束。

#### 10.16.4 参数解释与调参建议（对应新口径）

- `ori_success_end` 现在可直观理解为“tilt 与 yaw 都要小于该角度阈值（弧度）”。
- 若策略先学会靠近但姿态不过关，可先放宽 `ori_success_start/end`，待稳定后再逐步收紧。
- 建议优先排查顺序：
  1) `dist_success_end` 是否可达  
  2) `ori_success_end` 是否过严  
  3) 是否出现腕部撞桌导致早终（已加入早终可抑制 reward hacking）

#### 10.16.5 文档口径说明

本笔记包含多轮迭代记录（章节编号按历史保留，个别小节顺序不严格按时间）。后续以本节与 §10.15 的组合为准：

- §10.15：慢速动态目标（全局匀速 +Y）
- §10.16：姿态成功定义（+Z 朝下 + 与方块轴向对齐）

---
