# `rl_policy` 代码说明

本目录存放强化学习训练/测试脚本，按机器人与任务大致分为两类：

- **UR5e 双臂 Task1**（当前主线）
- **AgileX Piper**（IK 与抓取的早期/独立实验）

---

## 一、UR5e 双臂 Task1（主线）

### 1) `rl_ur_train_double_slowcv.py`（推荐训练入口）
- 作用：训练双臂 UR5e 在 `slow_dynamic` 场景下的策略。
- 特点：
  - 目标方块做“全局匀速”运动（每个子步强制 `qvel`）。
  - 沿世界系 `+Y` 方向移动（右臂侧 -> 左臂侧）。
  - 支持 `VecNormalize`、课程阈值、动作平滑/死区/低通、`waist_action_scale`、`ent_coef` 等参数。
- 对应评估脚本：`rl_ur_test_double_slowcv.py`

### 2) `rl_ur_test_double_slowcv.py`（推荐评估入口）
- 作用：评估 `rl_ur_train_double_slowcv.py` 训练出的模型。
- 特点：
  - 加载并固定 `VecNormalize` 统计（评估不更新）。
  - 输出每个 episode 的成功与回报。
  - 末端相对目标的距离与姿态信息（含四元数/欧拉角输出）。

### 3) `rl_ur_train_double.py`
- 作用：双臂 UR5e 的常规训练脚本（非 slowcv 专用）。
- 说明：是 `*_slowcv.py` 的基础版本之一，适合对照实验或回归检查。

### 4) `rl_ur_test_double.py`
- 作用：`rl_ur_train_double.py` 对应评估脚本。

### 5) `rl_ur_train.py`
- 作用：Task1 训练的通用/历史主干脚本。
- 特点：包含较完整的课程逻辑（如自动课程回调）与大量训练开关。

### 6) `rl_ur_test.py`
- 作用：`rl_ur_train.py` 的对应评估脚本。

### 7) `rl_ur_train_single.py` / `rl_ur_test_single.py`
- 作用：单臂模式训练与评估（调试可达性、动作稳定性常用）。

### 8) `rl_ur_train_ori.py` / `rl_ur_test_ori.py`
- 作用：较早版本/备份入口（`ori`），用于回溯对比，不建议作为当前主线训练入口。

---

## 二、AgileX Piper（独立实验线）

### 1) `rl_piper_ik_train.py` / `rl_piper_ik_test.py`
- 作用：Piper 机械臂 IK（逆运动学）相关训练与测试。
- 场景：`mujoco_asserts/agilex_piper/scene.xml`

### 2) `rl_piper_grasp_train.py` / `rl_piper_grasp_test.py`
- 作用：Piper 抓取任务训练与测试。
- 场景：`mujoco_asserts/agilex_piper_grasp/scene.xml`

---

## 三、建议使用方式（当前项目）

如果你现在主要做 UR5e 双臂慢速动态目标追踪/对齐，建议固定使用下面这对脚本：

- 训练：`rl_ur_train_double_slowcv.py`
- 评估：`rl_ur_test_double_slowcv.py`

并保持训练与评估参数对齐（如 `stage`、`obj_lin_speed`、动作相关参数、课程阈值、`VecNormalize` 文件）。

