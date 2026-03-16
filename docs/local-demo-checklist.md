# VidGen 本地可演示部署检查清单

本文档用于定义 VidGen 达到“本地可演示部署”所需满足的最低标准。

这里的目标不是生产部署，而是：

- 在单机环境中稳定运行真实模型
- 能提交真实任务并产出视频结果
- 能查看输出、控制链路信息和失败原因
- 能让另一位同事按文档完成复现

## 1. 环境准备

- [ ] 使用 `VidGen` conda 环境，Python 版本固定为 `3.11.15`
- [ ] 安装运行依赖，包括 ML 依赖
- [ ] 确认以下模块都可导入：
  - `PIL`
  - `torch`
  - `diffusers`
  - `transformers`
  - `controlnet_aux`
  - `imageio`
- [ ] 在 README 中写清楚环境激活与启动命令
- [ ] 明确演示运行模式：
  - CPU 演示模式
  - GPU 演示模式

## 2. 本地模型准备

- [ ] `stable-diffusion-xl-base-1.0` 已放在 `storage/models/`
- [ ] `stable-video-diffusion-img2vid-xt` 已放在 `storage/models/`
- [ ] 如果需要演示 FLUX，`FLUX.1-dev` 已放在 `storage/models/`
- [ ] 已确认模型目录结构可被 diffusers 正确加载
- [ ] `app/core/model_registry.py` 已配置为本地模型目录优先
- [ ] 以下最小模型检查脚本可以通过：
  - `scripts/test_local_sdxl_load.py`
  - `scripts/test_local_svd_load.py`

## 3. 真实任务执行

- [ ] `reference_image_path -> SVD` 能在本地完成真实 job
- [ ] job 状态能正确落到 `succeeded` 或 `failed`
- [ ] 能生成以下输出产物：
  - preview 图片
  - mp4 视频
  - metadata json
- [ ] 至少有 1 个基于 reference image 的 case 可稳定用于演示

## 4. 控制链路

- [ ] `provider_execution` 能正确写出 provider artifacts
- [ ] `generation_context` 能正确消费 provider artifacts
- [ ] controlled run 的输出 metadata 中包含 routing 信息
- [ ] 控制开启时能看到以下 artifact 类型：
  - `skeleton_sequence`
  - `depth_manifest`
  - `camera_manifest`
  - `transition_manifest`
- [ ] baseline 与 controlled 的 routing summary 不同

## 5. 演示输入资源

- [ ] 准备 2 到 3 条固定 prompt 用于演示
- [ ] 准备 2 到 3 张固定 reference image 用于演示
- [ ] 演示 reference image 统一放在 `storage/regression_inputs/` 或其他有文档说明的位置
- [ ] 重复性 case 统一写入 `tests/fixtures/regression_cases.json`

## 6. 运行配置

- [ ] README 中写清楚真实本地推理所需环境变量：
  - `VIDGEN_USE_MOCK_PIPELINE=false`
  - `VIDGEN_TASK_MODE=eager`
  - `VIDGEN_DEFAULT_MODEL=stable-video-diffusion-img2vid`
  - `VIDGEN_DEVICE=cpu` 或 `cuda`
- [ ] 写清楚输出目录位置
- [ ] 写清楚模型目录位置

## 7. API 演示路径

- [ ] `uvicorn app.main:app` 能成功启动
- [ ] `GET /health` 返回正常
- [ ] `POST /v1/videos:generate` 能创建真实 job
- [ ] `GET /v1/jobs/{job_id}` 能返回结果 metadata
- [ ] `scripts/submit_demo.py` 可作为命令行演示入口使用

## 8. 输出检查

- [ ] 输出统一写入 `storage/outputs/`
- [ ] 每个完成的 job 都有 preview 图和视频
- [ ] 输出 metadata 中包含：
  - `provider_execution`
  - `generation_context`
  - `selected_candidate`
  - `routing_summary`
  - `score`

## 9. 回归基线

- [ ] 最小控制回归脚本可运行
- [ ] 至少 1 个 regression case 能完整跑完
- [ ] 会生成 summary 文件：
  - `storage/regression/control-regression-summary.json`
- [ ] 会生成 markdown 报告：
  - `storage/regression/control-regression-summary.md`
- [ ] 报告中能体现：
  - `baseline`
  - `pose_depth`

## 10. 故障定位

- [ ] 文档中写清楚失败后应该先查看：
  - `storage/jobs/<job_id>.json`
  - `storage/outputs/<job_id>.json`
  - 终端 traceback
- [ ] 文档中写清楚如何区分：
  - 模型加载失败
  - provider / preprocessor 失败
  - inference 时失败

## 11. 文档完整性

- [ ] README 已说明本地环境准备
- [ ] README 已说明本地模型放置方式
- [ ] README 已说明如何提交真实 job
- [ ] README 已说明结果输出位置
- [ ] README 已说明如何运行控制回归脚本
- [ ] 文档与当前实现状态保持一致

## 12. 本地可演示部署完成标准

当以下条件全部满足时，可以认为项目达到了“本地可演示部署”：

- [ ] 其他同事能激活环境并启动服务
- [ ] 能提交真实 job
- [ ] 能在本地产生 preview 和视频
- [ ] 输出 json 中能看到控制链路信息
- [ ] 至少有 1 个稳定 demo case 和 1 个稳定 regression case
- [ ] README 足够支撑第二个人完成复现
