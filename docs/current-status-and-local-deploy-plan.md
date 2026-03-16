# VidGen 当前阶段总结与本地部署执行计划

## 一、文档目的

本文档基于 [architecture.md](/b:/agent/MyCode/VidGen/docs/architecture.md) 对 VidGen 当前进展做阶段性总结，并给出一套可执行的本地部署计划。

这里的“本地部署”目标不是生产上线，而是：

- 在单机环境中完成真实模型运行
- 支持输入提示词并生成 AI 视频
- 支持查看生成结果、控制链路信息和回归结果
- 支持他人按文档复现基本流程

## 二、当前项目进行到哪一步了

结合架构文档，项目整体分为三个阶段：

1. 阶段一：MVP 打通  
   重点是 prompt 到视频生成的最小可用链路
2. 阶段二：一致性与控制增强  
   重点是镜头规划、pose/depth/camera 控制、候选排序和质量验证
3. 阶段三：产品化与部署  
   重点是训练、服务化、监控、调度、成本与稳定性

### 当前判断

VidGen 目前已经明显超过“纯 MVP 骨架阶段”，进入了**阶段二的前中段**。  
更准确地说，当前状态属于：

**MVP 主链路已完成，真实控制链路已接入，正在向“可验证、可稳定演示”的阶段推进。**

## 三、已经完成的能力

### 1. 基础生成链路

以下主链路已经打通：

- `prompt -> generation_profile`
- `generation_profile -> prompt_bundle`
- `prompt_bundle -> shot_plan`
- `shot_plan -> control_plan`
- `control_plan -> adapter_plan`
- `adapter_plan -> provider_execution`
- `provider_execution -> generation_context`
- `generation_context -> pipeline injection`
- `candidates -> scoring/filtering`
- `job metadata / API result`

### 2. 基础工程能力

已经具备：

- FastAPI 接口
- job 创建与查询
- eager 本地任务执行
- Celery 集成骨架
- 本地 job store
- metadata 输出与持久化

### 3. 本地真实模型能力

已经完成：

- 本地 `SDXL` 模型目录加载
- 本地 `SVD` 模型目录加载
- 本地 `FLUX.1-dev` 路径支持
- 最小 `SDXL` smoke test
- 最小 `SVD` smoke test
- `reference_image_path -> SVD` 真实路径

### 4. 控制链路能力

已经完成：

- `control_plan` 生成
- `adapter_plan` 映射
- provider artifact 生成
- `generation_context` 消费
- `condition_router` 注入推理参数
- `routing_summary` 写入输出 metadata

### 5. 真实预处理器接入

已接入或已具备接入路径：

- `OpenPose`
- `Depth Anything`
- camera / transition 相关中间件

当前是“真实预处理优先，失败则回退”的模式。

### 6. 最小回归基线

已经完成：

- 固定 regression cases
- 回归脚本
- JSON summary 输出
- Markdown 报告输出
- 中途持续写 summary，避免长时间运行后无结果

## 四、部分完成但还需要继续打磨的部分

### 1. 控制条件已接入，但质量收益还没有系统验证

现在已经能证明：

- pose/depth 条件会进入控制链路
- metadata 中能看见 provider artifacts 和 routing summary

但还没有完全证明：

- 这些控制条件是否稳定提升了视频质量
- 对不同 prompt 是否都能带来可控收益
- 不同模式下是否会引入明显退化

### 2. Camera / transition 还处于轻量实现阶段

虽然 camera 和 transition 已经接入：

- `camera_manifest`
- `transition_manifest`

但目前还偏“基础可运行”：

- camera 更像参数注入和后处理引导
- transition 更像轻量 blend/stitching

离架构里更成熟的“镜头级生成控制”和“分段拼接”还有距离。

### 3. 评测体系还比较初级

目前有：

- scoring
- 最小回归
- routing 差异验证

但还没有：

- 更完整的 benchmark
- 稳定的多 case 自动评测体系
- 长期趋势对比机制

## 五、还没有完成的内容

以下内容在架构上仍属于未完成状态：

- 更强的时序一致性模块
- 更成熟的 transition stitching
- 更成熟的 camera-aware generation
- 更完整的候选重生成与重排序策略
- 完整训练与微调链路
- 产品化部署体系
- 监控、告警、成本控制
- 多 GPU / 调度 / 容器化
- 完整的线上服务治理

## 六、当前阶段的实际结论

目前的 VidGen 可以被描述为：

**一个已经具备本地真实视频生成能力，并且真实控制链路已经进入可验证状态的文本到视频系统。**

它还不能被描述为：

- 生产可部署系统
- 稳定高吞吐服务
- 已完成系统化质量验证的可控视频平台

所以当前最合理的阶段结论是：

> 项目已经从“架构与 mock 联调阶段”进入“真实控制链路工程验证阶段”，正处于本地可演示部署前的收口期。

## 七、距离本地可部署还差什么

如果目标只是“本地可演示部署”，距离并不算远。  
目前最大的缺口已经不是模型或架构设计，而是以下四类事情：

1. 稳定一个可重复成功的 demo case
2. 把本地输入资源、模型路径、环境配置彻底固定
3. 把 API 提交与结果查看路径跑顺
4. 把错误定位和回归结果整理清楚

也就是说，当前最需要做的是“收口”和“固化”，不是继续扩 schema 或重构架构。

## 八、本地部署的完成标准

当满足以下条件时，可以认为 VidGen 达到了“本地可演示部署”：

- 能启动本地 API
- 能提交一条真实 prompt 任务
- 能输出 preview 和 mp4
- 输出 metadata 中能看到控制链路信息
- 至少有一个稳定 demo case
- 至少有一个稳定 regression case
- README 足够让第二个人照着跑通

## 九、可执行计划

下面这份计划按执行顺序排列，优先做能最快提升演示可用性的部分。

### Phase A：锁定稳定的本地 demo baseline

目标：先确认一条最稳定、最容易展示的本地真实链路。

任务：

- 固定标准运行环境：
  - `VidGen` conda 环境
  - Python `3.11.15`
- 固定标准模型：
  - `stable-diffusion-xl-base-1.0`
  - `stable-video-diffusion-img2vid-xt`
- 固定标准运行模式：
  - `VIDGEN_USE_MOCK_PIPELINE=false`
  - `VIDGEN_TASK_MODE=eager`
  - `VIDGEN_DEFAULT_MODEL=stable-video-diffusion-img2vid`
  - `VIDGEN_DEVICE=cpu` 或 `cuda`
- 锁定 1 个稳定 demo case：
  - 固定 prompt
  - 固定 reference image
  - 固定 generation profile
  - 固定 seed 和尺寸
- 确认这条路径可以重复成功

完成标准：

- 至少 1 条 case 能稳定跑到 `succeeded`
- 能生成 preview 和 mp4
- metadata 完整

### Phase B：收口本地输入资源

目标：把 reference image 和 regression case 管理规范化。

任务：

- 创建输入目录：
  - `storage/regression_inputs/`
- 把用于 demo 和 regression 的图片统一放入该目录
- 更新 `tests/fixtures/regression_cases.json`
- 确保每个 case 都包含：
  - `case_id`
  - `prompt`
  - `reference_image_path`
  - `generation_profile`
  - `parameters`
  - `seed`

完成标准：

- 输入资源位置固定
- 不再依赖临时手工图片路径

### Phase C：完成最小控制回归闭环

目标：让控制链路有可重复、可比对、可查看的验证结果。

任务：

- 运行 `scripts/run_control_regression.py`
- 至少完成 1 个 case 的：
  - `baseline`
  - `pose_depth`
- 确认输出文件存在：
  - `storage/regression/control-regression-summary.json`
  - `storage/regression/control-regression-summary.md`
- 检查 markdown 报告里的关键字段：
  - `routing_differs_from_baseline`
  - `controlled_has_pose`
  - `controlled_has_depth`
  - `controlled_video_conditioning_used`
  - `score_delta`

完成标准：

- 回归脚本可用
- 可以基于报告判断控制链路是否生效

### Phase D：验证控制链路的真实作用

目标：确认 `pose/depth` 不只是有 artifact，而是真的进入推理。

任务：

- 比较 baseline 与 controlled 的：
  - provider artifacts
  - generation context
  - routing summary
  - 输出结果
- 对至少 1 个 case 做人工检查
- 记录当前结论：
  - 条件是否进入模型输入
  - 条件是否导致 routing 差异
  - 条件是否影响结果或评分

完成标准：

- 至少 1 个 case 可以说明控制链路“确实参与推理”

### Phase E：跑通本地 API 演示流程

目标：让项目不只是能用脚本跑，还能通过 API 演示。

任务：

- 启动本地 API
- 检查 `/health`
- 用 API 创建真实 job
- 查询 job detail
- 验证结果文件与接口返回一致
- 确认 `scripts/submit_demo.py` 可作为演示入口

完成标准：

- 通过 API 可提交 prompt 并查看结果

### Phase F：补齐错误定位与运行说明

目标：把“出错后怎么查”也变成文档的一部分。

任务：

- 在 README 中明确：
  - 环境准备
  - 模型目录
  - 真实 job 运行方式
  - 输出位置
  - regression 运行方式
- 文档中明确失败时先查：
  - `storage/jobs/<job_id>.json`
  - `storage/outputs/<job_id>.json`
  - 终端 traceback
- 记录常见错误类型：
  - 模型路径问题
  - 预处理器问题
  - 推理失败

完成标准：

- 别人拿到项目后，能独立定位基本问题

## 十、建议的优先顺序

如果目标是尽快达到本地可部署演示，建议按这个顺序推进：

1. 锁定 1 个稳定 demo case
2. 统一本地输入资源目录
3. 跑出第一份完整 regression 报告
4. 跑通 API 提交到结果查看的闭环
5. 补齐 README 和错误定位说明

## 十一、不建议现在优先做的事情

以下事项重要，但不建议在本地演示部署前优先投入：

- 大规模训练与微调
- 多 GPU 调度
- Docker / K8s 生产部署
- 成本监控与复杂服务治理
- 大规模自动 benchmark

这些属于后续产品化阶段，不应该阻塞当前“本地可演示”目标。

## 十二、阶段性总结

当前项目已经具备：

- 本地真实模型运行能力
- 文本到视频主链路
- 真实控制链路接入
- 控制回归基线能力
- 本地 demo 的工程基础

当前最合理的工作重心是：

- 固定稳定 demo case
- 收口输入、模型和配置
- 强化回归与结果可解释性
- 打通 API 演示闭环

一句话总结：

> VidGen 已经完成了“从架构骨架到真实本地运行”的关键跨越，当前距离本地可演示部署不远，接下来的重点不是重新设计系统，而是把现有真实链路稳定下来、标准化下来、文档化下来。
