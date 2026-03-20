import mujoco
import mujoco.viewer

# 最小场景：地面 + 光照 + 空世界（不包含任何机械臂/碰撞体）
xml = """
<mujoco model="minimal_window">
  <option timestep="0.002" integrator="implicitfast"/>
  <worldbody>
    <geom name="floor" type="plane" pos="0 0 0" size="1 1 0.01" rgba="0.2 0.2 0.2 1"/>
    <light pos="0 0 3" directional="true" dir="0 0 -1"/>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

viewer = mujoco.viewer.launch_passive(model, data)
viewer.cam.distance = 2.0
viewer.cam.azimuth = 0.0
viewer.cam.elevation = -20.0

while viewer.is_running():
    mujoco.mj_step(model, data)  # 也可以删掉这一行让物体完全不动
    viewer.sync()
