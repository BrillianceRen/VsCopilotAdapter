# Copilot OpenAI Adapter (多提供商适配器架构 · 当前专用于 Google AI Studio Gemini)

本项目基于 Python (FastAPI + httpx) 构建，将各种 AI 大模型接口转换为标准 **OpenAI 兼容接口**，专为 **Visual Studio 内置 Copilot** 以及各类 OpenAI 兼容客户端（VS Code、Continue、CodeGPT 等）设计。

当前已完整实现 **Google AI Studio Gemini** 适配器，并采用了清晰、解耦的可扩展插件式多适配器架构（`adapters/`），以便未来轻松扩充其他 AI 提供商（如 Claude, DeepSeek, Ollama 等）。

> [!IMPORTANT]
> **安全规范与零泄露保障**：
> 本项目的敏感密钥（API Key）一律通过本地配置文件 `.env` 或环境变量读取，代码及文档中无任何硬编码密钥，且 `.env` 与日志文件均已被 `.gitignore` 保护，严禁提交到代码仓库。

---

## 模块化架构设计

```text
copilot-openai-adapter/
├── adapters/
│   ├── __init__.py           # 适配器注册表与动态工厂函数 (get_adapter)
│   ├── base.py               # 抽象基类 BaseAIAdapter（规范标准接口）
│   └── aistudio_gemini.py    # Google AI Studio Gemini 专用适配器实现
├── .env                      # 本地私密配置文件（包含 API Key，被 gitignore）
├── .env.example              # 安全配置示例模板
├── .gitignore                # Git 忽略规则，防止私密及日志泄漏
├── adapter.log               # 运行及全量调试日志（自动轮转备份）
├── requirements.txt          # Python 依赖清单
├── server.py                 # FastAPI 主服务端（通用路由与全链路日志中间件）
├── start.bat                 # Windows 一键启动脚本
└── test_adapter.py           # 8项端到端自动化测试套件
```

### 未来扩充其他 AI 适配器指南
如果未来需要接入其他模型（例如 Claude、DeepSeek 等）：
1. 在 `adapters/` 目录下新建实现类（例如 `adapters/deepseek.py`），继承 `BaseAIAdapter`。
2. 实现 `provider_name`、`default_model`、`list_models`、`get_model`、`chat_completions`、`chat_completions_stream` 等方法。
3. 在 `adapters/__init__.py` 的 `ADAPTER_REGISTRY` 字典中注册新类名。
4. 修改 `.env` 中的 `AI_PROVIDER=deepseek` 即可平滑切换，主服务无需改动任何路由代码。

---

## 核心功能与特性

1. **安全配置与隐私保护**：
   - 密钥采用 `.env` 配置文件与环境变量隔离，严禁代码硬编码。
   - 提供安全模板 [.env.example](file:///d:/Program/AI/copilot-openai-adapter/.env.example)。
   - 配备 [.gitignore](file:///d:/Program/AI/copilot-openai-adapter/.gitignore)，自动忽略 `.env`、日志文件和缓存文件，杜绝密钥泄露风险。
   - 所有 Gemini 上游通信均通过安全请求头 `x-goog-api-key` 传输，避免在 URL 参数中暴露密钥。
   - 调试日志自动对 `authorization` 和 `x-goog-api-key` 请求头进行脱敏掩码处理（如 `AQ.Ab8...****`）。

2. **全链路请求与响应详细日志记录（调试利器）**：
   - 自动拦截并记录每一个进入的 HTTP 请求（包含客户端 IP、请求时间、脱敏请求头、JSON 请求体格式化输出）。
   - 自动记录每一个响应（包含 HTTP 状态码、请求耗时毫秒数、响应 JSON 内容）。
   - **流式 SSE 深度支持**：针对 Copilot 流式对话，既保证每个数据块实时送达客户端，又在后台完整聚合流式结果、事件总数与耗时并打印。
   - **双路输出**：同时打印到控制台屏幕并持久化保存到 [adapter.log](file:///d:/Program/AI/copilot-openai-adapter/adapter.log)（支持 UTF-8 编码、自动按 10MB 轮转备份）。

3. **完全实现 OpenAI 模型接口（满足题目要求）**：
   - `GET /models` 与 `GET /v1/models`：动态拉取 Gemini 官方支持的模型列表，并补充标准别名（如 `gpt-4o`、`gpt-4-turbo` 等）。
   - `GET /models/{model}` 与 `GET /v1/models/{model}`：支持按模型 ID（如 `gemini-3.8-flash`、`gpt-4o`、`models/gemini-3.8-flash`）精确查询模型元数据。

4. **完整支持 Copilot 核心推理与 Agent 函数调用 (Tools/Function Calling)**：
   - **Agent Tools 深度支持**：全兼容 Visual Studio Copilot 的 Agent 模式工具调用（如 `view`、`grep_search`、`edit` 等 170+ 项目操作工具），自动清洗 MCP Schema（移除 `x-mcp-header` 等非法属性）并转换为 Gemini 兼容的 OpenAPI 3.0 格式。
   - **多轮 thoughtSignature 校验保持**：完美解决 Gemini 2.5/3.x 在多轮工具调用上下文时的签名缺失校验问题。
   - `POST /v1/chat/completions`：支持流式 SSE（`stream: true`）与非流式问答，多轮对话上下文合并，支持系统提示词（System Prompt）以及工具调用状态机。
   - `POST /v1/completions`：支持 Visual Studio 代码内联补全（Ghost text / FIM 代码填充）。
   - `POST /v1/embeddings`：支持向量嵌入模型调用。

5. **低延迟与性能优化**：
   - 默认采用最新的高性能模型 `gemini-3.8-flash`。
   - 针对 Copilot 场景优化了思考预算（`thinkingBudget: 0`），确保代码生成极速返回，防止隐藏思考过程耗尽 Token 或导致超时。

---

## 环境配置

在项目根目录下配置 `.env` 文件：
```bash
# 复制模板文件
cp .env.example .env
```
在 `.env` 中填入你的配置：
```env
# AI 提供商（当前为 aistudio_gemini）
AI_PROVIDER=aistudio_gemini

# Google Gemini API Key
GEMINI_API_KEY=your_gemini_api_key_here

# 服务监听地址与端口
HOST=0.0.0.0
PORT=8000

# 默认模型
DEFAULT_MODEL=gemini-3.8-flash

# 关闭深度思考过程以确保代码快速补全
GEMINI_THINKING_BUDGET=0

# 日志文件路径
LOG_FILE=adapter.log
```

---

## 启动服务

### 方式 1：双击批处理文件启动（Windows 推荐）
直接双击运行目录下的：
```bat
start.bat
```

### 方式 2：使用 Python 命令行启动
```bash
python server.py --host 0.0.0.0 --port 8000
```
*(如果环境中有多个 Python 版本，可指定实际路径执行，如：`C:\Users\Brilliance\AppData\Local\Python\bin\python.exe server.py`)*

启动成功后，控制台输出如下：
```text
============================================================
 Copilot OpenAI Adapter Server (Multi-Provider)
============================================================
 AI Provider        : aistudio_gemini
 API Key            : AQ.Ab8...PxNQ
 Default Model      : gemini-3.8-flash
 Listening Address  : http://0.0.0.0:8000
 OpenAI Base URL    : http://127.0.0.1:8000/v1
 Models Endpoint    : http://127.0.0.1:8000/v1/models
 Log File           : D:\Program\AI\copilot-openai-adapter\adapter.log
============================================================
```

---

## Visual Studio / Copilot 配置方法

在 Visual Studio 2022 中配置自定义 OpenAI 兼容接口（或通过 Copilot 自定义端点配置）：

1. **Base URL / Endpoint**：
   ```text
   http://127.0.0.1:8000/v1
   ```
   *(服务已同时兼容带 `/v1` 和不带 `/v1` 的路径)*

2. **API Key**：
   填入你的 Gemini Key 或任意占位符（如 `sk-custom`），服务端将自动使用 `.env` 中配置的有效密钥。

3. **Model Name**：
   ```text
   gemini-3.8-flash
   ```
   *(如果客户端只认 OpenAI 标准名，也可直接填写 `gpt-4o` 或 `gpt-3.5-turbo`，适配器会自动映射到 `gemini-3.8-flash`)*

---

## 自动化测试验证

项目中提供了完整的端到端自动化测试脚本 [test_adapter.py](file:///d:/Program/AI/copilot-openai-adapter/test_adapter.py)：

```bash
python test_adapter.py
```

测试内容包括：
- [x] 健康检查接口 (`GET /health`)
- [x] 模型列表查询 (`GET /v1/models`)
- [x] 指定模型查询 (`GET /v1/models/gemini-3.8-flash`)
- [x] 映射模型查询 (`GET /v1/models/gpt-4o`)
- [x] 对话补全非流式请求 (`POST /v1/chat/completions`)
- [x] 对话补全流式 SSE 请求 (`POST /v1/chat/completions?stream=true`)
- [x] 代码内联补全请求 (`POST /v1/completions`)
- [x] Agent 工具/函数调用流式识别 (`tools / functionCalling`)
