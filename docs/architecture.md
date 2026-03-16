# VidGen 架构方案

## 1. 项目目标

构建一个基于现有开源模型进行组合、微调和工程封装的文本到视频 AI 系统。系统接收自然语言提示词，输出一段数秒长、内容连贯、主体稳定、具备基础运动逻辑的视频。

约束前提：

- 不从零训练基础大模型。
- 以开源图像/视频生成模型为核心能力基座。
- 优先追求“可落地、可迭代、可评估”的工程路线，而不是一次性追求 Sora 级别能力。

---

## 2. 总体建设策略

推荐采用“三层能力 + 三阶段实施”的策略。

三层能力：

1. 视觉生成层：负责主体、场景、风格、细节质量。
2. 时序一致性层：负责跨帧角色一致、背景稳定、闪烁控制。
3. 运动控制层：负责镜头运动、物体位移、基础物理逻辑。

三阶段实施：

1. MVP 阶段：先生成“短、稳、可看”的视频。
2. 增强阶段：提高时序一致性和运动可控性。
3. 产品化阶段：支持垂类微调、评测、服务化与优化部署。

---

## 3. 分阶段实施路线图

## 阶段一：MVP 打通

### 目标

实现最小可用文本到视频链路，支持 3 到 5 秒、低到中等分辨率的视频生成。

### 推荐方案

- 文本理解：`Qwen2.5-7B-Instruct` 或 `Llama 3.1 8B Instruct`
- 关键帧/首帧图像生成：`FLUX.1-dev` 或 `SDXL`
- 视频扩展：`Stable Video Diffusion XT`
- 工程框架：`Diffusers + PyTorch + FastAPI`

### 工作流

1. 用户输入提示词。
2. LLM 将提示词改写为结构化视频提示：
   - 主体
   - 场景
   - 风格
   - 动作
   - 镜头语言
   - 负面提示词
3. 图像模型生成高质量首帧或少量关键帧。
4. 视频模型基于首帧做视频扩展，生成 16 到 32 帧短视频。
5. 后处理模块做插帧、超分、去闪烁、编码输出。

### 阶段产出

- 文本到视频基础 API
- Prompt 重写模块
- 图像到视频推理管线
- 基础任务队列
- 基础视频评测脚本

### 关键挑战

- 首帧质量高，但后续帧容易漂移。
- 动作逻辑弱，容易出现“有动感但无明确动作”。
- 视频时长短，镜头表达有限。

### 解决方向

- 使用高质量图像模型先提升首帧。
- 把动作描述结构化，例如“人物向前走 3 步，镜头缓慢推进”。
- 先限制场景复杂度和动作幅度，避免一开始就做高动态视频。

---

## 阶段二：一致性与运动增强

### 目标

让视频更稳定，主体跨帧更一致，动作更可控，时长延长到 5 到 8 秒。

### 推荐方案

- 视频基座：`CogVideoX-5B` 或 `ModelScope T2V` 作为直接文本到视频基座
- 时序增强：`AnimateDiff`
- 结构控制：`ControlNet`（深度图、边缘图、姿态图）
- 光流与一致性评估：`RAFT`
- 视频插帧：`RIFE`

### 工作流增强

1. LLM 将提示词拆成镜头计划：
   - 场景设定
   - 主体属性
   - 动作序列
   - 镜头运动
   - 时长与节奏
2. 规划器生成中间控制信号：
   - pose 序列
   - depth / canny 条件
   - camera motion 标签
3. 视频模型按条件生成候选视频。
4. 一致性模块进行重打分，筛掉抖动、主体崩坏、动作错乱的结果。
5. 对保留结果做插帧、超分、颜色稳定化。

### 阶段产出

- 镜头规划模块
- 可控视频生成模块
- 视频重排序与质量筛选模块
- 视频后处理流水线

### 关键挑战

- 主体 ID 一致性仍然容易失控，尤其是人物脸部和服装。
- 复杂动作容易导致肢体错误。
- 控制条件越多，生成自由度越低，结果可能僵硬。

### 解决方向

- 对角色、服装、背景做 LoRA 微调或参考图约束。
- 把复杂动作分解为短片段生成，再拼接或续写。
- 控制信号只保留高价值条件，避免过拟合控制。

---

## 阶段三：垂类微调与生产化

### 目标

面向特定场景提升稳定性、可控性和吞吐，例如营销短视频、二次元角色视频、商品展示视频。

### 推荐方案

- 微调方式：`LoRA`、`DreamBooth`、`Temporal LoRA`
- 训练框架：`Diffusers + PEFT + Accelerate + DeepSpeed`
- 存储：`PostgreSQL + MinIO`
- 调度：`Celery + Redis`
- 监控：`Prometheus + Grafana`
- 部署：`Docker + NVIDIA Container Toolkit`

### 可落地能力

- 风格化视频生成
- 指定角色一致性
- 商品/品牌视觉一致性
- 模板化镜头生成
- 批量任务提交与回放

### 阶段产出

- 垂类训练数据集
- 自动评测仪表盘
- 模型版本管理
- 在线推理服务
- A/B 测试与成本监控

### 关键挑战

- 训练数据质量直接决定上限。
- GPU 成本高，推理延迟难控。
- 多模型串联后，排障复杂。

### 解决方向

- 优先建设数据清洗和标注标准。
- 针对推理做量化、分辨率分级、异步队列。
- 引入完整观测链路，记录 prompt、seed、模型版本、耗时、评分。

---

## 4. 推荐的具体模型与工具

## 4.1 文本理解与提示词增强

推荐模型：

- `Qwen2.5-7B-Instruct`
- `Llama 3.1 8B Instruct`

职责：

- 把自然语言转换为适合视频模型的结构化 prompt
- 自动补齐动作、镜头、风格、时长、主体属性
- 生成负面提示词
- 生成分镜或 shot plan

推荐输出结构：

```json
{
  "subject": "a young woman in a red coat",
  "scene": "snowy street at night",
  "style": "cinematic, realistic",
  "action": "walking slowly toward the camera",
  "camera": "slow dolly in",
  "duration_sec": 4,
  "negative_prompt": "blurry, distorted hands, flicker, duplicate person"
}
```

## 4.2 图像生成

推荐优先级：

1. `FLUX.1-dev`
2. `SDXL`

作用：

- 生成首帧
- 生成关键帧
- 生成角色参考图

选择理由：

- `FLUX.1-dev` 语义理解强，首帧质量通常更高。
- `SDXL` 生态成熟，LoRA、ControlNet、社区工具更完整。

## 4.3 视频生成

推荐组合：

- MVP：`Stable Video Diffusion XT`
- 增强：`CogVideoX-5B`
- 可控运动补充：`AnimateDiff`

选型建议：

- 如果优先要快速打通工程链路，先用 `SVD`。
- 如果优先做直接文本到视频能力和更长视频，尝试 `CogVideoX`。
- 如果已有 SD 系列图像生态，希望复用 ControlNet/LoRA，可引入 `AnimateDiff`。

## 4.4 条件控制与时序辅助

推荐工具：

- `ControlNet`
- `OpenPose`
- `MiDaS` / `Depth Anything`
- `RAFT`
- `RIFE`

作用：

- `ControlNet`：约束姿态、边缘、深度结构
- `OpenPose`：人物动作骨架控制
- `Depth Anything`：场景深度先验
- `RAFT`：光流估计与时序一致性评估
- `RIFE`：插帧，提升流畅度

## 4.5 训练与部署框架

推荐工具链：

- 训练：`PyTorch`、`Diffusers`、`Accelerate`、`PEFT`
- 服务：`FastAPI`
- 队列：`Celery` 或 `RQ`
- 缓存/消息：`Redis`
- 元数据：`PostgreSQL`
- 对象存储：`MinIO`
- 监控：`Prometheus`、`Grafana`
- 打包：`Docker`

---

## 5. 系统模块设计

建议拆成以下模块。

## 5.1 Prompt Orchestrator

输入自然语言，输出结构化生成任务。

职责：

- prompt 清洗
- 敏感词/风险词过滤
- 分镜拆解
- 生成负面提示词
- 生成参数模板，例如时长、帧数、宽高比、seed

输出：

- `shot_plan`
- `prompt_bundle`
- `control_plan`

## 5.2 Visual Generator

负责生成首帧、关键帧和参考图。

职责：

- 首帧生成
- 多候选采样
- 风格一致性约束
- 角色参考图缓存

## 5.3 Video Generator

负责从文本、图像或控制条件生成视频。

职责：

- text-to-video
- image-to-video
- keyframe-to-video
- 分段续写生成

建议支持三种模式：

1. 纯文本直出
2. 首帧驱动视频
3. 关键帧 + 控制条件驱动视频

## 5.4 Temporal Consistency Module

负责跨帧稳定性。

职责：

- 光流一致性检查
- 去闪烁
- 角色一致性重打分
- 背景稳定度检查

常用策略：

- 候选视频重排序
- 对低质量片段局部重生成
- 分段拼接后统一做稳定化

## 5.5 Motion Planner

负责基础运动逻辑。

职责：

- 将动作描述映射为 pose / camera motion / timing
- 支持简单运动模板：
  - 行走
  - 转头
  - 挥手
  - 镜头推进
  - 平移跟随

建议先模板化，而不是一开始做通用物理推理。

## 5.6 Post Processing

职责：

- 插帧
- 超分辨率
- 去噪
- 色彩稳定化
- 视频编码
- 音频占位或后续配乐接口

## 5.7 Evaluation & Feedback

职责：

- 文本视频对齐评分
- 美学评分
- 时序一致性评分
- 用户反馈采集
- 生成参数回放

推荐指标：

- `CLIPScore`：文本和视频帧语义对齐
- `FVD`：视频分布质量
- 光流误差：时序稳定
- 人工双盲评分：最终体验

---

## 6. 参考系统架构

```text
[Web / API]
    |
    v
[Prompt Orchestrator]
    |
    +--> [Shot Planner]
    |
    +--> [Control Signal Generator]
    |
    v
[Visual Generator] ---> [Reference Frame Store]
    |
    v
[Video Generator]
    |
    v
[Temporal Consistency Module]
    |
    v
[Post Processing]
    |
    +--> [Object Storage: MinIO]
    +--> [Metadata DB: PostgreSQL]
    |
    v
[Evaluation + Ranking + Feedback]
```

---

## 7. 推荐的工程落地方案

## 7.1 MVP 版本建议

适合两到四周打通：

- API：`FastAPI`
- 模型调用：`Diffusers`
- 图像模型：`FLUX.1-dev` 或 `SDXL`
- 视频模型：`Stable Video Diffusion XT`
- 队列：`Redis + Celery`
- 存储：本地磁盘 + `MinIO`

MVP 功能范围：

- 单 prompt 生成 3 到 4 秒视频
- 支持 2 到 4 个候选结果
- 支持基础负面提示词
- 支持插帧和导出 MP4

## 7.2 中期版本建议

适合一到两个月：

- 增加 `CogVideoX`
- 引入 `ControlNet + OpenPose`
- 增加自动评测
- 支持角色 LoRA
- 支持镜头模板

## 7.3 生产版本建议

适合持续演进：

- 多 GPU 推理调度
- 模型版本路由
- 异步任务与失败重试
- 成本与性能监控
- 数据闭环与自动微调

---

## 8. 各阶段关键挑战与解决方向汇总

## 挑战一：帧间闪烁和主体漂移

表现：

- 人脸变化
- 衣服颜色漂移
- 背景纹理跳动

解决方向：

- 首帧/关键帧约束
- Temporal LoRA
- 光流约束和候选重排序
- 分段生成后做稳定化

## 挑战二：运动缺乏逻辑

表现：

- 角色似乎在动，但动作不可解释
- 镜头语言随机

解决方向：

- 先做动作模板库
- prompt 结构化成“主体 + 动作 + 镜头 + 时序”
- 使用 pose/depth 控制而不是纯文本硬推

## 挑战三：时长扩展困难

表现：

- 越长越容易崩
- 前后片段风格不一致

解决方向：

- 分段生成 + bridge frame 续写
- 保留角色参考图和场景 embedding
- 对片段做统一调色和稳定化

## 挑战四：训练数据建设难

表现：

- 视频数据清洗成本高
- 标签不标准导致动作学习差

解决方向：

- 优先做垂类数据集
- 建立统一标签规范：主体、动作、镜头、场景、风格
- 把高质量生成结果也纳入闭环训练数据

## 挑战五：推理成本高

表现：

- 生成慢
- GPU 占用高

解决方向：

- 分辨率分级
- 先低清生成，再选优超分
- LoRA 热加载
- 批处理与异步队列

---

## 9. 推荐的起步组合

如果目标是“最快做出能演示的系统”，推荐：

- LLM：`Qwen2.5-7B-Instruct`
- 首帧生成：`FLUX.1-dev`
- 视频生成：`Stable Video Diffusion XT`
- 后处理：`RIFE + FFmpeg`
- 服务：`FastAPI + Redis + Celery`

如果目标是“更强的文本直出视频能力”，推荐：

- LLM：`Qwen2.5-7B-Instruct`
- 视频生成：`CogVideoX-5B`
- 条件控制：`ControlNet + OpenPose`
- 评测：`CLIPScore + FVD + RAFT`

---

## 10. 最终建议

这个项目不应从“训练一个全能视频模型”开始，而应从“搭一个稳定的视频生成系统”开始。最合理的路线是：

1. 先用高质量图像模型解决首帧质量。
2. 再用开源视频模型解决短视频生成。
3. 再通过控制条件、时序模块和 LoRA 微调提升一致性。
4. 最后建设评测、数据闭环和服务化体系。

如果要立刻开工，优先开发顺序建议是：

1. Prompt Orchestrator
2. 首帧生成
3. Image-to-Video
4. 后处理
5. 评测与筛选
6. 可控运动和垂类微调
