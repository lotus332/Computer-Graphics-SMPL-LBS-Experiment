import os
import numpy as np
import torch
import smplx

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ============================================================
# 1. 路径设置
# ============================================================

MODEL_PATH = "/home/lzyuan/Documents/lotusBNUComVision_SMPL/SMPL_NEUTRAL.pkl"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. 加载 SMPL
# ============================================================

model = smplx.create(
    MODEL_PATH,
    model_type="smpl",
    gender="neutral",
    ext="pkl",
    num_betas=10,
    batch_size=1
)

v_template = model.v_template.detach().cpu().numpy()
faces = model.faces
lbs_weights = model.lbs_weights.detach().cpu().numpy()

print("v_template:", v_template.shape)
print("faces:", faces.shape)
print("lbs_weights:", lbs_weights.shape)


# ============================================================
# 3. SMPL 24关节名称
# ============================================================

joint_names = [
    "Pelvis",          # 0
    "Left Hip",        # 1
    "Right Hip",       # 2
    "Spine 1",         # 3
    "Left Knee",       # 4
    "Right Knee",      # 5
    "Spine 2",         # 6
    "Left Ankle",      # 7
    "Right Ankle",     # 8
    "Spine 3",         # 9
    "Left Foot",       # 10
    "Right Foot",      # 11
    "Neck",            # 12
    "Left Collar",     # 13
    "Right Collar",    # 14
    "Head",            # 15
    "Left Shoulder",   # 16
    "Right Shoulder",  # 17
    "Left Elbow",      # 18
    "Right Elbow",     # 19
    "Left Wrist",      # 20
    "Right Wrist",     # 21
    "Left Hand",       # 22
    "Right Hand"       # 23
]


# ============================================================
# 4. 坐标转换
# ============================================================

def convert_for_visualization(vertices):
    """
    SMPL 原始坐标中人体竖直方向主要是 Y 轴。
    Matplotlib 3D 默认使用 Z 轴作为屏幕竖直方向。

    因此仅在绘图时做坐标转换：

        原始: (X, Y, Z)
        显示: (X, Z, Y)

    注意：
    这不会修改真正用于 LBS 计算的 v_template。
    """

    v_vis = vertices[:, [0, 2, 1]].copy()

    return v_vis


# ============================================================
# 5. 辅助函数：保持三维坐标等比例
# ============================================================

def set_axes_equal(ax, vertices):

    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]

    max_range = max(
        x.max() - x.min(),
        y.max() - y.min(),
        z.max() - z.min()
    ) / 2.0

    mid_x = (x.max() + x.min()) / 2.0
    mid_y = (y.max() + y.min()) / 2.0
    mid_z = (z.max() + z.min()) / 2.0

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # 防止 Matplotlib 把人体比例压扁
    ax.set_box_aspect((1, 1, 1))


# ============================================================
# 6. 创建用于绘图的模板顶点
# ============================================================

v_vis = convert_for_visualization(v_template)

triangles = v_vis[faces]


# ============================================================
# 7. Stage A：单关节权重热力图
# ============================================================

# 选择左肘关节
joint_id = 18

vertex_weights = lbs_weights[:, joint_id]

print()
print("Selected joint:", joint_id, joint_names[joint_id])
print("Weight min:", vertex_weights.min())
print("Weight max:", vertex_weights.max())
print("Weight mean:", vertex_weights.mean())

# 一个三角形由三个顶点构成
# 这里使用三个顶点权重的平均值作为面片颜色
face_weights = vertex_weights[faces].mean(axis=1)

norm = Normalize(vmin=0.0, vmax=1.0)
cmap = plt.get_cmap("turbo")

face_colors = cmap(norm(face_weights))


# ============================================================
# 8. 绘制单关节权重图
# ============================================================

fig = plt.figure(figsize=(9, 10))

ax = fig.add_subplot(
    111,
    projection="3d"
)

mesh = Poly3DCollection(
    triangles,
    facecolors=face_colors,
    linewidths=0.01
)

mesh.set_edgecolor("none")

ax.add_collection3d(mesh)

set_axes_equal(ax, v_vis)

# ------------------------------------------------------------
# 关键：正面视角
# ------------------------------------------------------------
ax.view_init(
    elev=0,
    azim=-90
)

ax.set_title(
    f"(a) Template Mesh + Skinning Weight\n"
    f"Joint {joint_id}: {joint_names[joint_id]}",
    fontsize=15
)

ax.set_xlabel("X")
ax.set_ylabel("Depth")
ax.set_zlabel("Height")

# 去掉一些多余空白
ax.grid(False)

# Colorbar
sm = cm.ScalarMappable(
    norm=norm,
    cmap=cmap
)

sm.set_array([])

cbar = fig.colorbar(
    sm,
    ax=ax,
    fraction=0.03,
    pad=0.05
)

cbar.set_label(
    "Skinning Weight"
)

plt.tight_layout()

stage_a_path = os.path.join(
    OUTPUT_DIR,
    "stage_a_template_weights.png"
)

plt.savefig(
    stage_a_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:", stage_a_path)


# ============================================================
# 9. 全关节主导权重分布
# ============================================================

# 每个顶点影响最大的关节编号
dominant_joint = np.argmax(
    lbs_weights,
    axis=1
)

# 每个顶点最大的蒙皮权重
dominant_weight = np.max(
    lbs_weights,
    axis=1
)

# ------------------------------------------------------------
# 对每个三角面片：
# 选择三个顶点中出现次数最多的主导关节
# ------------------------------------------------------------

face_joint = []

for f in faces:

    joints = dominant_joint[f]

    counts = np.bincount(
        joints,
        minlength=24
    )

    face_joint.append(
        np.argmax(counts)
    )

face_joint = np.array(face_joint)


# 三个顶点最大权重的平均值
face_strength = dominant_weight[faces].mean(axis=1)


# ============================================================
# 10. 创建 24 个关节的离散颜色
# ============================================================

joint_cmap = plt.get_cmap(
    "tab20",
    24
)

face_colors_all = joint_cmap(
    face_joint
)

# 用透明度表现主导权重强度
face_colors_all[:, 3] = (
    0.45
    +
    0.55 * face_strength
)


# ============================================================
# 11. 绘制全关节主导权重图
# ============================================================

fig = plt.figure(
    figsize=(9, 10)
)

ax = fig.add_subplot(
    111,
    projection="3d"
)

mesh_all = Poly3DCollection(
    triangles,
    facecolors=face_colors_all,
    linewidths=0.01
)

mesh_all.set_edgecolor(
    "none"
)

ax.add_collection3d(
    mesh_all
)

set_axes_equal(
    ax,
    v_vis
)

# 正视图
ax.view_init(
    elev=0,
    azim=-90
)

ax.set_title(
    "Dominant Joint Weight Distribution",
    fontsize=15
)

ax.set_xlabel("X")
ax.set_ylabel("Depth")
ax.set_zlabel("Height")

ax.grid(False)

plt.tight_layout()

all_weights_path = os.path.join(
    OUTPUT_DIR,
    "all_joint_weights.png"
)

plt.savefig(
    all_weights_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved:",
    all_weights_path
)


print()
print("Task 2 finished successfully.")
