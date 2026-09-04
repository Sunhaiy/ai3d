# Tiny Image-to-3D

在 RTX 4060 8GB 上从随机权重训练一个小型“单张图片生成 3D”模型。网页默认使用 `128 x 128` 输入图片和 `32 x 32 x 32` 体素，并通过 marching cubes 导出 Blender 可以打开的平滑 `.obj`。

完整流程：

`输入图片 + 对应 3D 网格 -> 配对训练集 -> 图片编码器 -> 体素解码器 -> OBJ`

这是用于跑通原理的第一版，不是高质量商品级 Image-to-3D。训练和推理完全在本地进行，不使用预训练权重。

## 网页工作台

安装完依赖后启动：

```powershell
cd E:\ai3d
.\start_web.ps1
```

脚本会把服务作为独立后台进程启动，关闭当前 PowerShell 后仍可使用。浏览器打开 `http://127.0.0.1:8000`。网页支持上传图片、选择 `runs` 目录中的检查点、调整体素阈值、查看各阶段日志、旋转预览结果并下载 OBJ/NPY。

顶栏切换到“训练”即可在网页中完成整个训练流程：

1. 数据根目录填写包含 `meshes` 和 `images` 的目录，例如 `E:\modeldata`。
2. 点击“制作训练集”，等待模型体素化完成。
3. 设置训练轮数和批大小，点击“开始训练”。

处理后的训练集保存在 `E:\ai3d\data\image3d_pairs.npz`，体素化预览保存在 `E:\ai3d\data\image3d_previews`。批大小表示一次参数更新同时处理多少张图片；越大通常吞吐越高、显存占用也越高，显存不足时应调小。

每次新训练都会在 `runs` 目录保存成独立 `.pt` 文件。可以填写“训练成果名称”；留空时使用 `image3d_年月日_时分秒.pt`，名称重复时自动追加 `_02`，不会覆盖旧成果。“初始模型”选择“从零开始”会使用随机权重，选择任意已有检查点则会把兼容权重迁移到新任务，训练结果仍另存为新文件。相同网络结构会完整加载，不同图片或体素分辨率会只加载尺寸兼容的层。

训练页会实时显示总进度、epoch、batch、训练/验证 loss、IoU、GPU 利用率、显存、温度和子进程日志。训练时图片生成会自动停用，防止两项任务同时占用显存。

点击“暂停训练”后，程序会先完成当前 batch，再把模型、AdamW 优化器、混合精度缩放器、当前 epoch 和 batch 保存到独立续训检查点。状态变成“已暂停”后会释放 GPU，可以关闭网页或重启服务；下次点击“继续训练”会从该 batch 后继续。“结束本次训练”会删除续训检查点，但不会删除已经保存的最佳模型。

体素档位：

- `16³`：最快，仅用于验证流程。
- `32³`：RTX 4060 8GB 推荐档，默认批大小 32。
- `64³`：轮廓更细，默认批大小 16，制作数据和训练明显更慢。
- `128³`：实验档，默认批大小 4；214 个模型完整训练可能需要数小时，建议先用 `32³` 验证数据方向和训练曲线。

训练数据按“多张图片索引到一个唯一体素目标”保存，不会再为同一个模型的 56 张图片重复存储 56 份高分辨率体素。

项目自带 `demo_untrained.pt` 用于验证网页流程，它没有经过训练。完成正式训练后刷新网页状态，`image_to_3d.pt` 会自动出现在检查点列表中。

## 最重要的数据要求

每条训练数据必须同时包含：

1. 一张物体图片，作为模型输入。
2. 图片中同一个物体的 3D 网格，作为正确答案。

只有图片、没有对应 3D 模型，无法用这套监督训练方法从零学习 Image-to-3D。

支持的图片格式：`.png`、`.jpg`、`.jpeg`、`.webp`、`.bmp`。

支持的 3D 格式：`.obj`、`.stl`、`.ply`、`.glb`、`.gltf`、`.3mf`。

## 安装

```powershell
cd E:\ai3d
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

确认 CUDA：

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 路线 A：真实图片和 3D 配对

图片文件名需要和 3D 文件名一致：

```text
D:\image3d_data\meshes\
  chair_0001.obj
  chair_0002.glb

D:\image3d_data\images\
  chair_0001.png
  chair_0002.png
```

同一个 3D 可以配多张图片，额外图片使用双下划线后缀：

```text
chair_0001.png
chair_0001__side.jpg
chair_0001__back.png
```

这些图片都会匹配 `chair_0001.obj`，目标保持为同一个标准朝向的 3D。所有 3D 应保持一致的顶部和正面方向，图片也应尽量采用一致的拍摄距离、背景和构图。

制作配对数据：

```powershell
python prepare_image3d.py `
  --meshes "D:\image3d_data\meshes" `
  --images "D:\image3d_data\images" `
  --resolution 32 `
  --image-size 128 `
  --output data/image3d_pairs.npz `
  --preview-dir data/image3d_previews
```

转换完成后，检查预览目录中的：

- `preview_0000_input.png`：模型实际看到的输入。
- `preview_0000_target.obj`：这张图片对应的正确 3D 答案。

## 路线 B：从 3D 自动造训练图片

没有配套图片时，可以先从 3D 网格自动生成简化的正交投影视图：

```powershell
python prepare_image3d.py `
  --meshes "D:\my_3d_models" `
  --views 4 `
  --output data/image3d_pairs.npz `
  --preview-dir data/image3d_previews
```

200 个 3D 使用 4 个方向会得到 800 对训练数据。这条路线最容易验证模型确实学会了图片到体素的映射，但它生成的是黑底深度轮廓图，因此训练出的模型只适合输入相似风格的图片。

## 训练

```powershell
python train_image_to_3d.py `
  --data data/image3d_pairs.npz `
  --epochs 100 `
  --batch-size 32 `
  --output runs/image_to_3d.pt
```

日志会分别显示训练集和验证集的 `loss` 与 `iou`。同一个 3D 的不同图片会被放在同一侧，避免验证数据泄漏。验证损失下降、IoU 上升说明模型正在学习；脚本会自动保存验证损失最低的检查点。模型大小会随体素分辨率和潜空间维度增加。

## 用一张图片生成 3D

```powershell
python reconstruct.py `
  --checkpoint runs/image_to_3d.pt `
  --image "D:\test_images\chair.png" `
  --output result.obj
```

同时得到：

- `result.obj`：导入 Blender 查看。
- `result.npy`：原始体素数组。

结果为空时尝试 `--threshold 0.40`，糊成一大块时尝试 `--threshold 0.50`。

## 一键验证

从 3D 自动造视图、训练并拿第一张预览重建：

```powershell
.\run_image_to_3d.ps1 -MeshDir "D:\my_3d_models" -Views 4 -Epochs 100
```

已经有真实配对图片时：

```powershell
.\run_image_to_3d.ps1 `
  -MeshDir "D:\image3d_data\meshes" `
  -ImageDir "D:\image3d_data\images" `
  -Epochs 100
```

## 200 个模型如何准备

- 先只选一个类别，例如全部是椅子。
- 所有 3D 统一朝向，顶部统一为 `+Z`。
- 使用封闭、无严重破面的网格。
- 真实图片尽量背景干净、主体居中、大小接近。
- 每个 3D 最好准备 2-4 个角度的图片。
- 先用预览目录中的输入图测试重建，再尝试训练分布之外的照片。

## 测试

```powershell
pytest -q
```

旧的 `generate_data.py`、`train.py` 和 `sample.py` 是无图片条件的 3D VAE 实验。Image-to-3D 主流程使用 `prepare_image3d.py`、`train_image_to_3d.py` 和 `reconstruct.py`。
