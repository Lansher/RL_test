import os
import mujoco
import mujoco.viewer

xml_path = "/home/bns/sevenT/ly/RL_test/mujoco_asserts/universal_robots_ur5e/scene_dual_arm.xml"
keyframe_name = "home"  # 用 XML 里 keyframe 的 qpos 作为初始角度

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# 找到 keyframe id
key_id = None
for k in range(int(model.nkey)):
    if model.key(k).name == keyframe_name:
        key_id = int(k)
        break
if key_id is None:
    raise ValueError(f"Keyframe '{keyframe_name}' not found in scene_dual_arm.xml")

# 用 keyframe 的 qpos 初始化
mujoco.mj_resetDataKeyframe(model, data, key_id)
mujoco.mj_forward(model, data)

# 速度清零，避免“静止时也在动”
data.qvel[:] = 0.0

# 让 position actuator 继续“保持当前 pose”（后续一旦你恢复 mj_step，不会突然拉飞）
for aid in range(int(model.nu)):
    act_name = model.actuator(aid).name
    if not act_name:
        continue
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, act_name)
    if jid >= 0:
        qpos_adr = int(model.jnt_qposadr[jid])
        data.ctrl[aid] = float(data.qpos[qpos_adr])

viewer = mujoco.viewer.launch_passive(model, data)
viewer.cam.distance = 3
viewer.cam.azimuth = 0
viewer.cam.elevation = -30

paused = True

# 如果 glfw 可用，就用 Space 键切换暂停/继续
try:
    import glfw

    def on_key(window, key, scancode, action, mods):
        global paused
        if action == glfw.PRESS and key == glfw.KEY_SPACE:
            paused = not paused

    if hasattr(viewer, "window"):
        glfw.set_key_callback(viewer.window, on_key)
except Exception:
    # 没有 glfw/窗口句柄时，仍然保持 paused=True 不会抖动
    pass

while viewer.is_running():
    if not paused:
        mujoco.mj_step(model, data)
    viewer.sync()