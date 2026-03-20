## Task1 环境部署与场景集成记录（对话整理）

### 1. 任务概览（来自 `RL_test/README.md`）

本项目主题为：**基于强化学习的机械臂抓取**，其中课程必做为 **Task 1：UR 双臂机器人强化学习**。

Task 1 的核心交付要求包括：

- 在 `mujoco_asserts/universal_robots_ur5e/scene_dual_arm.xml` 基础上完成**双臂动态协作抓取场景**搭建
  - 目标物体高度：与机械臂基座的相对高度约 **800–900mm**
  - 目标物体需要具有“动态方向/运动方式”（例如静态、慢速动态、可选双臂协同互斥惩罚）
- 在 `rl_policy/` 目录编写训练与评估代码：
  - `rl_ur_train.py`（训练）
  - `rl_ur_test.py`（评估，输出 RL 成功率）
- MDP 建模（观测空间、动作空间、奖励函数）以及课程学习（静态抓取 → 慢速动态 → 双臂协同可选）

项目提交材料要求包含：**训练过程曲线（TensorBoard 截图）**、**评估结果（RL 成功率）**、**推理分析**等。

#### 快速验收标准（用于指导你怎么改）
1. XML 场景：`scene_dual_arm.xml` 能稳定 `mujoco.MjModel.from_xml_path(...)` 加载，并且桌子/物体在预期高度范围内。
2. 训练可运行：`rl_policy/` 的 Task1 训练入口能跑通（不因 obs/action 维度或 reward 发散崩溃）。
3. 可评估：`rl_ur_test.py` 能输出明确 `success_rate`（成功率）。
4. 报告可复现：报告里给出关键超参/训练时长、`--n_env`、随机种子/评估 episode 数。

---

### 2. 主流程（从环境到跑 demo 的工程化步骤）

#### 2.1 环境依赖与 MuJoCo 场景

1. 进入项目目录并创建 Conda 环境：
   - `cd /home/bns/sevenT/ly/RL_test`
   - `conda create -n RL_test python=3.10.9 -y`
   - `conda activate RL_test`
2. 安装依赖（按 `requirements.txt` 对应 GPU 方案或 CPU 方案均可，本文按 GPU/cuda 方向配置）。
3. 验证 MuJoCo XML 能否加载：
   - 通过 `mujoco.MjModel.from_xml_path(".../scene_dual_arm.xml")` 做“加载成功”确认。

#### 2.2 “跑 demo”与可视化/渲染问题

在无图形环境（或 SSH 场景）下运行 MuJoCo viewer 可能触发 GLFW 报错：

- `GLFWError: X11: The DISPLAY environment variable is missing`
- `ERROR: could not initialize GLFW`

解决方式之一：使用 `xvfb-run` 提供虚拟显示：

```bash
sudo apt install -y xvfb
xvfb-run -s "-screen 0 1920x1080x24" python rl_policy/rl_piper_ik_test.py
```

另一个与“viewer 导入方式”相关的问题：

- 在某次尝试中，代码报错：`AttributeError: module 'mujoco' has no attribute 'viewer'`
- 修复：显式导入 `mujoco.viewer`

```python
import mujoco
import mujoco.viewer  # 关键
```

---

### 3. 场景扩展：在 XML 中加入桌子与桌上动态可抓取物体

#### 3.1 当前 `scene_dual_arm.xml` 的现状

`scene_dual_arm.xml` 主要结构为：

- `<include file="dual_ur5e.xml"/>`：引入双臂机器人主体（MJCF）
- `<worldbody>` 里包含：
  - `floor` 平面
  - `left_target/right_target`：`mocap="true"` 的可视化坐标轴（用于标定目标，不是可碰撞抓取物）

因此加入“桌子 + 桌上移动物体”应当主要改 `scene_dual_arm.xml` 的 `<worldbody>`，而不需要动 `dual_ur5e.xml` 的机器人定义。

#### 3.2 高度要求的坐标换算

从 `dual_ur5e.xml` 可见两臂基座 `left_robot_base/right_robot_base` 的 `pos` 为 `... z=1.0`。

README 给出：目标物体相对基座高度约 `0.8–0.9m`，因此目标物体世界坐标高度应约为：

- `z_obj = z_base + (0.8~0.9) = 1.8 ~ 1.9 m`

同时桌子高度也需要对应抬升，否则机械臂可能无法到达或发生不合理穿模/浮空。

#### 3.3 XML 改法（推荐的最小可用版本）

1. 在 `<worldbody>` 中添加桌子：
   - 先用 `box` 形状做桌面与桌腿（稳、简单、碰撞易调）
2. 在桌面附近添加一个可碰撞的物体：
   - 用 `freejoint` 使其成为真正动力学刚体（可被碰撞扰动/被抓取后移动）
3. 让“物体移动”的专业做法：
   - **推荐先用 mocap 驱动**（在 Python 每步更新 `data.mocap_pos/mocap_quat`）
   - 或升级到更物理的约束/执行器控制（复杂度更高）

#### 建议的验证检查点（改完立刻确认）
1. 第一次加桌子/物体：先只做几何（`geom type="box"/"mesh"`），确保 XML 能加载。
2. 确认高度：把桌面与目标物体放到预期高度（例如 `z≈1.8~1.9m`），用 viewer 目视确认机械臂末端“能到”。
3. 确认接触：打开碰撞后观察是否“不穿模/不爆炸”（不要求精确物理，但要基本稳定）。

---

### 4. SolidWorks 模型接入 MuJoCo 的转换与 XML 引用

#### 4.1 OBJ/STL 转换链路

推荐链路：

- SolidWorks → 导出 **OBJ**（优先）或导出 **STL**
- 若导出 STL，可先用 Blender/MeshLab/其他工具转 OBJ

#### 4.2 单位与尺度（非常关键）

MuJoCo 默认单位按 **米**。

SolidWorks 常用建模单位是 **毫米**，因此 OBJ 需要在 MJCF 引用时做缩放：

- 毫米 → 米：`scale="0.001 0.001 0.001"`

#### 4.3 在 MJCF 中注册 mesh 与使用 mesh geom

在 `scene_dual_arm.xml` 的 `<asset>` 中注册 mesh：

```xml
<mesh name="custom_base_mesh" file="custom/your_base.obj" scale="0.001 0.001 0.001"/>
<material name="base_mat" rgba="0.2 0.2 0.2 1"/>
```

再在 `<worldbody>` 或对应 body 下使用：

```xml
<geom name="base_vis" type="mesh" mesh="custom_base_mesh" material="base_mat"
      contype="0" conaffinity="0"/>
```

> 专业实践：视觉 mesh 用复杂网格，碰撞建议用 primitive/简化几何（箱体/凸包），避免 RL 训练中接触不稳定与计算开销过大。

#### 导入后快速验证（最省时间的顺序）
1. 先保证 `scene_dual_arm.xml` 不报错加载。
2. 再确认 mesh/geom 名称确实存在（例如 `geom type=mjGEOM_MESH`）。
3. 如果“加载成功但看不到”：通常是尺度（`scale` 或导出单位）或旋转（`pos/quat/euler`）不对。

---

### 5. OBJ 原点与相对 base_link 位姿的定义方式

#### 5.1 两层概念

- **OBJ 原点/坐标系**：由你导出 OBJ 时网格的局部坐标决定（最推荐在建模软件中把参考点放到原点）
- **相对 base_link 位姿**：由 MJCF 的 `body pos` / `quat`（或 `geom pos/quat`）决定

#### 5.2 推荐策略

- 在 SolidWorks/Blender 中把 OBJ 原点放到你希望对齐的参考点（如桌面中心、底座安装点中心）
- 在 MJCF 里用 `body pos="X Y Z" quat="w x y z"` 把物体放到相对基座/世界的位置
- 如果需要二次修正（例如模型朝向不一致），可在 `geom pos/quat` 里做偏置补偿

#### 标定与验证流程（把“对不齐”变成可操作步骤）
1. 先在导出阶段把 OBJ 原点放到你希望的参考点（桌面中心/安装面中心等）。
2. 在 MJCF 里先用 `body pos + quat` 做整体放置。
3. 如仍不对：只改一个旋转自由度（优先改 `quat`），每次改完重加载 XML 并目视检查。

---

### 6. 验证 OBJ 是否成功导入：加载成功判定 + 可视化检查

#### 6.1 “无 GUI”的成功判定

直接加载 XML：

```python
import mujoco
m = mujoco.MjModel.from_xml_path(xml_path)
```

并进一步确认：

- `geom` 是否存在且类型为 `mjGEOM_MESH`
- `mesh` 资源是否存在

示例（用于确认 `base_vis` / `custom_base_mesh`）：

```python
gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "base_vis")
mid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MESH, "custom_base_mesh")
```

#### 6.2 遇到的 OBJ 解析失败与修复

你遇到的错误：

- `ValueError: could not parse OBJ file '.../dual_ur5e_base.obj'`
- 原因定位：OBJ 文件里面片行格式不被 MuJoCo OBJ 解析器接受，例如出现类似 `f 19// 17// 173//` 的不规范面片语法（`//` 后缺少完整索引）

修复手段（工程上最省事的路径）：

- 将 OBJ 的 `f` 面片转换为 MuJoCo 友好的格式：`f v1 v2 v3`
- 对多边形面做三角化并输出到新文件，例如：
  - `dual_ur5e_base_fixed.obj`
- 修改 `scene_dual_arm.xml` 里 mesh 引用指向修复后的 OBJ

随后 `mujoco.MjModel.from_xml_path(scene_dual_arm.xml)` 成功通过：

- `loaded OK`
- `ngeom=75`

#### 排错顺序（建议你每次都按这个来）
1. 优先确认是解析错误还是运行错误：`from_xml_path` 阶段报错通常是路径/语法/OBJ 格式问题。
2. 若明确是 OBJ 解析失败：定位到报错行号附近的 `f` 面片语法，重写/重导出为兼容的三角面片（`f v1 v2 v3`）。
3. 若加载成功但看不到/方向不对：再检查 `scale` 与 `pos/quat/euler`。

---

### 7. 遇到的问题汇总（按影响程度）

1. GLFW / 可视化失败（DISPLAY 缺失）
   - 定位：viewer 相关脚本会在 GLFW 初始化阶段直接失败。
   - 处理：无 GUI 用 `xvfb-run`；有桌面但 DISPLAY 异常则修复 `DISPLAY`。

2. `mujoco.viewer` 导入方式错误
   - 定位：代码中调用 `mujoco.viewer...`，但只执行了 `import mujoco`。
   - 处理：补上 `import mujoco.viewer`（让 viewer 子模块可用）。

3. OBJ 解析失败（OBJ face 语法不兼容）
   - 定位：`from_xml_path` 报错 `could not parse OBJ file ...`，错误里会给到 OBJ 行号；通常是 `f` 面片语法不兼容。
   - 处理：重写/重导出 OBJ，使面片变为 MuJoCo 兼容三角面（`f v1 v2 v3`），并更新 XML 指向修复后的文件。

4. 依赖/认知澄清：MuJoCo 仅读取 XML（MJCF）
   - 定位：你修改了 `.urdf` 但仿真现象不变。
   - 处理：仿真以 `.xml` 为准，把变更落在 `scene_*.xml / dual_ur5e.xml` 等 MJCF 上。

---

### 8. 后续建议（你可以继续推进的方向）

1. 在 `scene_dual_arm.xml` 的 `<worldbody>` 中接入桌子（视觉 mesh + 碰撞 primitive）。
2. 在桌面附近接入可抓取物体：
   - 先用 `freejoint` 刚体保证它“确实参与动力学/接触”
   - 先用 mocap 在 Python 中驱动物体运动，做静态/慢速动态 curriculum，避免一开始就引入复杂控制器。
3. 训练 Task1 前先做一致性检查：
   - 目标高度与基座布局匹配（例如 `z≈1.8~1.9m`）
   - obs/action/reward 的维度与含义严格一致，避免 reward 设计与成功判定“逻辑不对齐”。

---

### 9. 后续对场景/机械臂参数的改动记录

这一节偏“操作流程/修改点”，重点记录：改哪里、怎么改、怎么验证，而不是只记最终参数值。

#### 9.1 `scene_dual_arm.xml`：增大地板覆盖范围（加大工作空间）
1. 在 `scene_dual_arm.xml` 的 `<worldbody>` 找到 `geom name="floor"`。
2. 调整它的 `size="sx sy sz"` 的前两项（`sx sy` 控制地板跨度），保持厚度项 `sz` 不变或按需微调。
3. 验证方式：重新加载 XML（`mujoco.MjModel.from_xml_path(...)`），并用 viewer 观察是否覆盖你后续桌子/物体的运动范围。

#### 9.2 `scene_dual_arm.xml`：对导入 mesh 做复合旋转
1. 找到你导入并用于视觉的 `geom name="base_vis"`（`type="mesh"`，引用 `mesh="custom_base_mesh"`）。
2. 若需要组合旋转（例如：先绕 X +90°，再绕 Z -90°），优先用“最终等价四元数”写到 `quat="w x y z"`，避免欧拉角顺序/参考系差异导致的反向或偏差。
3. 如确需用欧拉角：需严格结合 `dual_ur5e.xml` 中的 `eulerseq="zyx"`，否则同一组数对应的旋转意义会变。

> 以本次为例：`base_vis` 使用了等价的最终 `quat`（MuJoCo `quat` 顺序为 `w x y z`）。

#### 9.3 `scene_dual_arm.xml`：加入碰撞模型（mesh 视觉 + 碰撞 geom）
1. 先明确视觉 geom 的碰撞状态：当前 `base_vis` 通常设置为 `contype="0" conaffinity="0"`，因此它只负责显示，不参与碰撞。
2. 添加碰撞时，推荐流程是“视觉与碰撞分离”：
   - 保留 `base_vis` 做视觉：`contype=0 conaffinity=0`
   - 新增一个碰撞 geom：`contype=1 conaffinity=1`
3. 碰撞 geom 的选型顺序：
   - 快速验证：先用同 mesh 做碰撞（`type="mesh"`）确认不会再出现 XML 解析/基础接触问题
   - 提升稳定性：将碰撞 geom 替换为简单 primitive（如 `type="box"`/`type="capsule"`），减少不稳定接触与计算开销
4. 验证方式：加载 XML 成功后，再用 viewer/仿真检查是否出现接触（可通过碰撞反应、接触力/位姿变化等间接判断）。

#### 9.4 `dual_ur5e.xml`：沿 z 轴整体上移机械臂基座
1. 在 `dual_ur5e.xml` 找到 `left_robot_base` 与 `right_robot_base` 的 `<body ... pos="x y z" ...>`。
2. 将两者的 `pos` 中 `z` 增加 1m（其他维度保持不变），实现整体抬升。
3. 验证方式：重新加载 XML；同时检查 Task1 的目标高度/桌面高度设置是否仍满足“机械臂末端可达”与“物体在桌面上表面附近运动”。

#### 9.5 `dual_ur5e.xml`：将某一侧基座旋转 30°
1. 在 `dual_ur5e.xml` 找到对应 body（例如 `left_robot_base`）。
2. 修改其 `quat="w x y z"` 为所需旋转角对应的四元数。
3. 注意：如果你希望另一侧也同步同样旋转，应同时修改 `right_robot_base`；否则左右布局会发生相对偏置。
4. 验证方式：重新加载 XML；并在 viewer 里确认旋转方向是否符合预期（必要时再取反角）。

