# 更新记录 (Changelog)

本项目遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/) 语义化版本规范，所有显著的更新与变更均记录于此。

---

## [v1.4.0] - 2026-11-23

### 🔄 虚拟模型与多轮会话自适应降级策略 (Virtual Model & Fallback Degradation)
- **新增虚拟模型（Virtual Model）**：
  - 引入虚拟模型（默认标识 `auto`，支持通过 `VIRTUAL_MODEL_NAME` 自定义或扩展别名），并在 `GET /v1/models` 和 `GET /v1/models/{model}` 中向客户端完整暴露。
  - 可在 `.env` 中设置 `DEFAULT_MODEL=auto`，使所有未显式指定模型的请求默认获得高可用降级保护。
- **可配置备选模型链与失败阈值**：
  - 支持配置 `FALLBACK_MODELS` 备选模型列表（如 `gemini-4-hydrogen,gemini-3.8-flash`），按顺序依次尝试。
  - 支持配置 `FALLBACK_THRESHOLD`（连续失败次数阈值，默认 `1` 次）。
- **多轮会话状态感知与平滑降级**：
  - 自动识别客户端会话 ID（支持头部识别与根提示词语义指纹识别）。
  - 若当前模型失败达到阈值，自动降级至备选列表中的下一个模型并立即执行重试，客户端无需报错重试。
  - 同一会话的多轮后续对话将自动保持在降级后的稳定模型上，避免重复在故障模型上等待。
- **新会话自动重置**：
  - 当检测到新对话开始（上下文无历史助理回复）时，自动重置会话模型指针，重新回到第一个优选模型尝试。
- **全链路流式与非流式切流保护**：
  - 流式 SSE 会话在前置数据阶段遇到上游断流或错误时，由适配器在内部捕获并无缝切流至降级模型，保证客户端接收完整有效流。

---

## [v1.3.0] - 2026-11-23

### 📁 日志按日分卷与存储目录重构 (Logging Storage & Rotation)
- **指定日志存储目录**：
  - 日志配置从单一文件路径模式重构为指定存储目录模式（默认 `LOG_DIR=logs`），支持通过 `.env` 或命令行 `--log-dir` 指定。
  - 启动时自动检测并安全创建目录，向后平滑兼容 `--log-file` 参数。
- **每日独立文件与分卷命名（`yyyy-MM-dd.##.log`）**：
  - 日志每天自动生成独立的日期文件，格式遵循 `yyyy-MM-dd.##.log`（例如：`2026-11-23.01.log`）。
  - 日期跨天（午夜 00:00）时，日志流自动无缝平滑轮换到新日期的 `.01.log`。
- **单文件 30MB 上限与序号自增**：
  - 单个日志文件大小最大限制为 30MB（可通过 `MAX_LOG_SIZE_MB` 配置）。
  - 当文件达到 30MB 上限时，序号自动递增分卷（如 `2026-11-23.02.log`、`2026-11-23.03.log` 等）。
  - 服务重启时智能探查当前日期已存在的日志分卷，若最新分卷未满 30MB 则继续追加写入，避免重启造成文件碎片化。

---

## [v1.2.1] - 2026-11-20

### 📝 文档与规范 (Documentation)
- **文档全面完善**：在 `README.md` 中补充企业级网络高可用与双协议降级设计的实现细节。
- **配置与架构指引**：增加多模型适配器（`adapters/`）扩展指南与 Visual Studio Copilot Agent 模式对接说明。
- **更新记录归档**：新增 `CHANGELOG.md`，规范化记录项目从诞生到高可用优化的完整演变历程。

---

## [v1.2.0] - 2026-11-20

### 🚀 优化与高可用 (Resilience & Performance)
- **双协议自适应降级轮换（HTTP/2 ↔ HTTP/1.1）**：
  - 针对 Visual Studio Copilot Agent 超长会话（如 140+ 轮消息、60+ 本地工具）思考时间长达 30~50 秒的场景，首发请求默认采用 HTTP/2 进行多路复用与心跳保活。
  - 当本地代理软件（如 Clash、V2Ray 等）或中间 NAT 网关因长时间静默产生流去同步（Stream Desync）导致断开时，重试请求自动降级为 HTTP/1.1 纯净长连接，大幅提升代理穿透与连接稳定性。
- **全量传输层异常捕获与重试**：
  - 将重试异常捕获基类扩展为 `httpx.TransportError`，全量覆盖 `ReadError`、`RemoteProtocolError`、`ConnectError`、`ReadTimeout`、`WriteError`、`NetworkError`、`ProxyError` 等 15 种网络错误，以及 `httpcore` 和系统级网络超时异常。
- **工具调用中断断连优雅收尾**：
  - 若模型在长时间推理并已完整产生工具调用（`tool_calls`）后发生底层网络断连，适配器自动补充 `finish_reason: "tool_calls"` 并优雅收尾，杜绝下发错误提示文本，保护 Visual Studio Copilot 客户端免受 JSON 解析报错影响。
- **可配置超时与重试参数**：
  - 配置文件 `.env` 和 `.env.example` 新增 `GEMINI_TIMEOUT`（默认 180 秒）和 `GEMINI_MAX_RETRIES`（默认 3 次）。

---

## [v1.1.0] - 2026-11-20

### 🐛 缺陷修复 (Bug Fixes)
- **解决 SSE 过程中的 `RemoteProtocolError: Server disconnected`**：
  - 针对 Gemini 复杂推理（`gemini-4-hydrogen`）时因等待时间过长而遭遇的上游静默断开问题，引入指数退避自动重试机制（默认退避 1s、2s、4s）。
- **延迟首包机制（Deferred SSE Initial Chunk）**：
  - 修复此前在连接建立前过早向下游客户端输出 `role: assistant` 首包导致网络异常时无法重试的问题。
  - 调整为仅在上游确认 HTTP 200 并返回首个实际数据块后，才向下游输出初始包。确保在真正接收到模型内容前发生的一切断线与重试对 Copilot 客户端完全透明、静默自愈。
- **HTTP/2 支持引入**：
  - `httpx.AsyncClient` 开启 `http2=True`，减少长连接握手开销。

---

## [v1.0.1] - 2026-11-16

### ✨ 新增功能 (Features)
- **精细化日志等级配置**：
  - 新增 `LOG_LEVEL` 环境变量与命令行 `--log-level` 参数（支持 `INFO`、`DEBUG`、`WARNING`、`ERROR`，默认为 `INFO`）。
- **请求/应答内容脱敏与降噪**：
  - 在 `INFO` 等级下，控制台仅输出单行结构化摘要：包含请求 ID、方法、路径、模型名称、状态码、精确耗时（ms）、Token 统计或 SSE Chunk 数与客户端 IP。
  - `INFO` 等级下坚决不打印请求体的 Message、Prompt 以及响应正文，既杜绝隐私敏感信息外泄，又避免多轮长对话刷屏。
  - 完整的 Headers、参数体与模型返回明细仅在 `DEBUG` 等级下输出，方便问题排查。

---

## [v1.0.0] - 2026-11-16

### 🎉 初始版本发布 (Initial Release)
- **模块化插件式适配器架构**：
  - 设计抽象基类 `BaseAIAdapter`，解耦路由层与上游模型提供商实现。
  - 采用注册表中心机制（`ADAPTER_REGISTRY`），提供 `aistudio_gemini`（Google AI Studio Gemini）专属适配器，便于未来平滑扩展 Claude、DeepSeek、Ollama 等其他提供商。
- **OpenAI 兼容协议标准实现**：
  - 支持 `GET /models` 与 `GET /v1/models` 模型列表动态查询。
  - 支持 `GET /models/{model}` 与 `GET /v1/models/{model}` 单模型详情与别名映射查询（如 `gpt-4o` 自动映射）。
  - 支持 `POST /v1/chat/completions` 流式 SSE（`stream: true`）与非流式对话补全。
  - 支持 `POST /v1/completions` 代码行内补全（Visual Studio Ghost Text / FIM 填充）。
  - 支持 `GET /health` 健康探针接口。
- **Visual Studio Copilot Agent 工具全兼容**：
  - 完整适配 Agent 模式中的复杂工具与函数调用（Function Calling）。
  - 自动清洗和转换 Visual Studio 发送的 OpenAPI Schema（移除 `x-mcp-header` 等 Google 不支持的属性）。
  - 针对 Gemini 2.5/3.x/4.x 系列模型的多轮工具上下文，实现 `thoughtSignature` 签名缓存与丢失自愈。
- **安全与防泄漏体系**：
  - 敏感密钥统一交由本地 `.env` 管理，并提供 `.env.example` 模板。
  - 配置完善的 `.gitignore`，严防 `.env`、日志文件和 Python 缓存误入代码仓库。
  - 日志中间件对 API Key 和 Auth 头部自动做脱敏遮蔽处理。
- **自动化测试与运行脚本**：
  - 编写涵盖 8 大核心功能的端到端测试套件 `test_adapter.py`。
  - 提供 Windows 便捷启动脚本 `start.bat`。
