# Task1 训练笔记（UR 双臂 6DOF 对齐）

下面这份笔记是基于你当前仓库的实现整理的，目标是：让你这个强化学习初学者也能看懂“环境怎么定义、PPO 怎么训练、成功率怎么评估、怎么可视化看末端如何靠近/碰到方块”。

仓库里实现的关键入口文件：
- 训练：`rl_policy/rl_ur_train.py`
- 评估：`rl_policy/rl_ur_test.py`

场景文件（MuJoCo XML）：
- `mujoco_asserts/universal_robots_ur5e/scene_dual_arm.xml`

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
- 姿态奖励：鼓励“更近的那一臂”与方块的四元数接近（当前姿态项是 reward，用作训练信号）

并加了一个简单碰撞惩罚：如果方块与桌面碰撞接触，则扣一点。

成功判定在 `_compute_reward_done()`：

```python
dL = ||left_tool - block||
dR = ||right_tool - block||

ori_angle_L = quat_angle(left_tool_quat, block_quat)
ori_angle_R = quat_angle(right_tool_quat, block_quat)

success_left  = (dL < dist_success) and (ori_angle_L < ori_success)
success_right = (dR < dist_success) and (ori_angle_R < ori_success)
success_now = success_left or success_right  # “任一臂 6DOF 对齐”
```

此外还有 `success_hold_steps`：
- `success_now=True` 只表示“某一步刚满足阈值”
- 只有连续满足 `success_hold_steps` 步，才算一次最终成功（done）

`dist_success_current/ori_success_current` 会随 episode 数逐步收紧（curriculum），让训练从“容易获得正样本”逐步变成“真正 6DOF 对齐”。

### 3.5 关键参数含义（新手最常用）
- `dist_success_current`：距离阈值（米）。用于判断“末端是否足够靠近方块”。
- `ori_success_current`：姿态角阈值（弧度）。用于判断“末端坐标系是否与方块坐标系对齐”。
- `curriculum_episodes`：在多少个 episode 内把阈值从 `*_start` 线性收紧到 `*_end`。这样能避免训练一开始就太难导致几乎没有 success 样本。
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
1) 将姿态对齐的判定从“全四元数旋转角”改为“绕世界 Z 轴 yaw 差值”（wrap 到 [-pi, pi] 后取绝对值）。
2) `ori_reward` 与 success 的 yaw-only 判定保持一致，避免 reward 与 done 不匹配。
3) curriculum 阈值放宽：`dist_success_end` 从 `0.12 -> 0.18`，`ori_success_end` 从 `1.0 -> 1.3`。
4) static 阶段 object yaw 随 curriculum 从较小范围逐步扩大（pi/4 -> pi），更易获得正样本。

在“新训练模型”评估中，`Task1 success_rate` 达到约 `0.48 (24/50)`，并继续出现少量 QACC 警告，说明稳定性仍有偶发因素，但 success 明显提升。

### 9.3 数值稳定性处理（QACC）
1) `reset()` 中移除了额外的 `mujoco.mj_step`，避免一开始就触发接触冲击导致爆炸。
2) `step()` 中对数值不稳定做提前截断：检查 `obs` 与 `self.data.qacc`；并在 `frame_skip` 子步中尽早发现不稳定直接 break。
3) 进一步改动 reward 中距离部分的梯度：将硬 `min(dL,dR)` 改为 smooth-min（softmin），缓解 reward 梯度断裂。

### 9.4 MDP/可学习性增强（obs 与控制）
1) 观测增强：obs 维度从 27 扩展到 57，增加 robot proprioception（`robot_qpos/qvel`）与末端到目标的相对位置（`rel_lpos/rel_rpos`）。
2) reward 平滑：距离项使用 smooth-min，替代硬 min。
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
  --log_dir ./logs/ppo_ur_task1_run_v1/ \
  --model_path ./models/ppo_ur_task1_run_v1_model.zip
```

验证（自动加载 `./models/*_vecnormalize.pkl`；确保 success_rate 口径一致）：
```bash
conda run -n RL_test python rl_policy/rl_ur_test.py \
  --stage static \
  --model_path ./models/ppo_ur_task1_run_v1_model.zip \
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
  --log_dir ./logs/ppo_ur_task1_left_fixed_v1/ \
  --model_path ./models/ppo_ur_task1_left_fixed_v1_model.zip
```

验证（需要与训练的 `--single_arm/--fixed_target` 一致）：
```bash
conda run -n RL_test python rl_policy/rl_ur_test.py \
  --stage static \
  --single_arm left \
  --fixed_target \
  --model_path ./models/ppo_ur_task1_left_fixed_v1_model.zip \
  --n_episodes 50
```

### 9.7 MuJoCo viewer 暂停相关
1) viewer 是否能“暂停”取决于代码里是否仍在调用 `mujoco.mj_step`。
2) 使用键盘回调实现暂停/继续时，修复了 `nonlocal paused` 导致的 SyntaxError，改为模块级 `global paused`，并使用 Space 键切换。

### 9.8 代码加载与 keyframe 相关排错（home ctrl size）
出现过 MuJoCo 报错：
`invalid ctrl size, expected length 12`
原因是 `scene_dual_arm.xml` 的 `home ctrl` 长度曾被设置成 19（与执行器数量 nu=12 不一致）；之后修正为 12 个数（匹配 nu）。

