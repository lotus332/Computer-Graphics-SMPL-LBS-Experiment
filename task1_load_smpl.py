import os
import torch
import smplx

# 当前目录下的 SMPL 模型文件
model_path = "/home/lzyuan/Documents/lotusBNUComVision_SMPL/SMPL_NEUTRAL.pkl"

print("========== SMPL LBS Experiment: Task 1 ==========")
print("Model path:", model_path)
print("Model exists:", os.path.exists(model_path))

# 加载 SMPL 模型
model = smplx.create(
    model_path,
    model_type="smpl",
    gender="neutral",
    ext="pkl",
    num_betas=10,
    batch_size=1
)

print("\n========== Basic Model Information ==========")

# 1. 模板顶点
v_template = model.v_template
print("v_template shape:", tuple(v_template.shape))
print("Number of vertices:", v_template.shape[0])

# 2. 三角面片
faces = model.faces
print("faces shape:", faces.shape)
print("Number of faces:", faces.shape[0])

# 3. 关节回归器
J_regressor = model.J_regressor
print("J_regressor shape:", tuple(J_regressor.shape))
print("Number of joints:", J_regressor.shape[0])

# 4. shape 参数
betas = model.betas
print("betas shape:", tuple(betas.shape))
print("Number of betas:", betas.shape[1])

# 5. 蒙皮权重
lbs_weights = model.lbs_weights
print("lbs_weights shape:", tuple(lbs_weights.shape))

# 额外检查
print("\n========== Additional Information ==========")
print("shapedirs shape:", tuple(model.shapedirs.shape))
print("posedirs shape:", tuple(model.posedirs.shape))
print("parents shape:", tuple(model.parents.shape))

print("\nTask 1 finished successfully.")
