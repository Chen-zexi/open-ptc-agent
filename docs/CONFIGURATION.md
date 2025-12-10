# Configuration Guide

[English](CONFIGURATION.md) | [中文](zh/CONFIGURATION.md)

This guide covers all configuration options for Open PTC Agent.

## Overview

The project uses two main configuration files:

| File | Purpose |
|------|---------|
| `config.yaml` | Main configuration - LLM selection, MCP servers, Daytona sandbox, security, storage, logging |
| `llms.json` | LLM provider definitions - model IDs, SDKs, API keys |

Credentials are stored separately in `.env` (see `.env.example`).

### Config File Search Paths

Configuration files are searched in order (first found wins):

**SDK/Library usage:**
1. Current working directory
2. Git repository root
3. `~/.ptc-agent/`

**CLI (`ptc-agent` command):**
1. `~/.ptc-agent/` (user config takes precedence)
2. Current working directory

**Environment Variable Overrides:**
```bash
PTC_CONFIG_FILE=/path/to/config.yaml  # Override config.yaml path
PTC_LLMS_FILE=/path/to/llms.json      # Override llms.json path
```

---

## config.yaml

### LLM Selection

```yaml
llm:
  # Reference to LLM definition in llms.json
  name: "claude-sonnet-4-5"
```

**Inline Definition** (alternative to llms.json):

```yaml
llm:
  name: "claude-haiku-4-5"
  model_id: "claude-haiku-4-5-20251107"
  provider: "anthropic"
  sdk: "langchain_anthropic.ChatAnthropic"
  api_key_env: "ANTHROPIC_API_KEY"
```

Available options depend on what's defined in `llms.json`. Pre-configured models:

- `claude-sonnet-4-5`, `claude-opus-4-5` - Anthropic Claude
- `gpt-5.1-codex`, `gpt-5.1-codex-mini` - OpenAI GPT
- `gemini-3-pro`, `gemini-3-pro-image` - Google Gemini
- `glm-4.6` - Z.AI/Zhipu GLM
- `minimax-m2-stable` - Minimax
- `deepseek-v3.2` - DeepSeek
- `doubao-seed-code` - Volcengine/ByteDance Doubao
- `qwen3-max` - Dashscope/Alibaba Qwen
- `kimi-k2-thinking` - Moonshot Kimi

---

### Daytona Sandbox

```yaml
daytona:
  base_url: "https://app.daytona.io/api"
  auto_stop_interval: 3600      # 1 hour - sandbox auto-stops after inactivity
  auto_archive_interval: 86400  # 24 hours - sandbox archived
  auto_delete_interval: 604800  # 7 days - sandbox deleted
  python_version: "3.12"

  # Snapshot configuration (recommended for faster startup)
  snapshot_enabled: true        # Use snapshots for 7x faster initialization
  snapshot_name: "open-ptc-v1"  # Base name (hash appended automatically)
  snapshot_auto_create: true    # Create snapshot if missing
```

**Snapshots**: First sandbox creation takes ~10-15 minutes to install dependencies. With snapshots enabled, subsequent sandboxes initialize in ~8 seconds.

---

### Security

```yaml
security:
  # Execution limits
  max_execution_time: 300   # 5 minutes max per code execution
  max_code_length: 10000    # 10KB max code size
  max_file_size: 10485760   # 10MB max file size

  # Code validation
  enable_code_validation: true

  # Allowed Python imports (whitelist)
  allowed_imports:
    - os
    - sys
    - json
    - yaml
    - requests
    - datetime
    - pathlib
    - typing
    - re
    - math
    - random
    - time
    - collections
    - itertools
    - functools
    - subprocess
    - shutil

  # Blocked code patterns (security)
  blocked_patterns:
    - "eval("
    - "exec("
    - "__import__"
    - "compile("
    - "globals("
    - "locals("
```

---

### MCP Servers

```yaml
mcp:
  servers:
    - name: "tavily"
      enabled: true                    # Toggle server on/off
      description: "Web search engine"
      instruction: "Use for web searches..."
      tool_exposure_mode: "summary"    # "summary" or "full"
      transport: "stdio"               # "stdio", "http", or "sse"
      command: "npx"
      args: ["-y", "tavily-mcp@latest"]
      env:
        TAVILY_API_KEY: "${TAVILY_API_KEY}"

    - name: "alphavantage"
      enabled: false
      transport: "http"
      url: "https://mcp.alphavantage.co/mcp?apikey=${ALPHA_VANTAGE_API_KEY}"

  # Tool discovery
  tool_discovery_enabled: true
  lazy_load: true       # Load tools on-demand (recommended)
  cache_duration: 300   # Cache tool metadata for 5 minutes
```

**Transport Types**:
- `stdio` - Standard I/O (most common, runs as subprocess)
- `http` - HTTP endpoint
- `sse` - Server-Sent Events

**Tool Exposure Modes**:
- `summary` - Brief tool descriptions (recommended)
- `full` - Complete signatures with all parameters (use for frequently called tools)

**Adding a Custom MCP Server**:

```yaml
- name: "my-custom-server"
  enabled: true
  description: "My custom MCP server"
  instruction: "When to use this server..."
  tool_exposure_mode: "summary"
  transport: "stdio"
  command: "uv"
  args: ["run", "python", "mcp_servers/my_server.py"]
  env:
    MY_API_KEY: "${MY_API_KEY}"
```

**Path Resolution for Local MCP Servers**:

Relative paths in MCP server configurations are resolved in the following order:
1. Relative to the config file directory (e.g., `~/.ptc-agent/`)
2. Relative to the current working directory (fallback)

This allows you to place MCP server files alongside your config:
```
~/.ptc-agent/
├── config.yaml
└── mcp_servers/
    └── my_server.py
```

With config:
```yaml
args: ["run", "python", "mcp_servers/my_server.py"]  # Resolved relative to ~/.ptc-agent/
```

Absolute paths are also supported and used as-is:
```yaml
args: ["run", "python", "/path/to/my_server.py"]
```

---

### Filesystem

```yaml
filesystem:
  working_directory: "/home/daytona"  # Sandbox root directory
  allowed_directories:
    - "/home/daytona"
    - "/tmp"
  enable_path_validation: true        # Validate paths against allowed list
```

---

### Cloud Storage

```yaml
storage:
  # Provider for auto-uploading charts/images
  # Options: s3, r2, oss, none
  provider: "s3"
```

All credentials loaded from `.env`. Configure one of:

**Cloudflare R2** (recommended - zero egress fees):
```bash
R2_ACCOUNT_ID=your-account-id
R2_ACCESS_KEY_ID=your-access-key
R2_SECRET_ACCESS_KEY=your-secret-key
R2_BUCKET_NAME=your-bucket
R2_PUBLIC_URL_BASE=https://your-bucket.r2.dev  # Optional
```

**AWS S3**:
```bash
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
S3_BUCKET_NAME=your-bucket
S3_REGION=us-east-1
S3_PUBLIC_URL_BASE=https://your-bucket.s3.amazonaws.com  # Optional
```

**Alibaba Cloud OSS**:
```bash
OSS_ACCESS_KEY_ID=your-access-key
OSS_ACCESS_KEY_SECRET=your-secret-key
OSS_BUCKET_NAME=your-bucket
OSS_REGION=oss-cn-hangzhou
OSS_ENDPOINT=oss-cn-hangzhou.aliyuncs.com
```

Set `provider: "none"` to disable image uploads.

---

### Logging

```yaml
logging:
  level: "INFO"        # DEBUG, INFO, WARNING, ERROR, CRITICAL
  format: "json"       # "json" or "text"
  file: "logs/ptc.log"
```

---

### Agent

```yaml
agent:
  # Use custom filesystem tools with advanced features
  # true: Custom tools (Read, Write, Edit, Glob, Grep) with more options
  # false: DeepAgent's native middleware tools
  use_custom_filesystem_tools: true

  # Enable ViewImageMiddleware for multimodal LLM image analysis
  enable_view_image: true
```

Custom tools provide:
- Grep with output_mode, multiline, context lines, file type filtering
- Advanced glob patterns
- Line-numbered file reading

---

### Subagents

```yaml
subagents:
  enabled:
    - "general-purpose"  # Full execute_code, filesystem, vision tools
    - "research"         # Web search (Tavily) + think tool
```

Available subagents:
- `general-purpose` - Complex multi-step tasks with code execution and file access
- `research` - Web research with strategic reflection

Subagents run in the background by default, allowing the main agent to continue working.

---

### CLI (ptc-agent only)

```yaml
cli:
  theme: "auto"       # auto, dark, light
  palette: "nord"     # Color scheme
```

**Theme Options:**
- `auto` - Detect from terminal (default)
- `dark` - Force dark mode
- `light` - Force light mode

**Palette Options:**
- `emerald`, `cyan`, `amber`, `teal` - Accent colors
- `nord`, `gruvbox`, `catppuccin`, `tokyo_night` - Full themes

**Environment Variable Overrides:**
```bash
PTC_THEME=dark        # Override theme
PTC_PALETTE=gruvbox   # Override palette
NO_COLOR=1            # Disable all colors
```

---

## llms.json

Defines available LLM providers.

### Structure

```json
{
  "llms": {
    "model-name": {
      "model_id": "actual-model-id",
      "provider": "anthropic|openai|google",
      "sdk": "langchain_module.ClassName",
      "api_key_env": "ENV_VAR_NAME",
      "base_url": "https://custom-endpoint",  // Optional
      "output_version": "responses/v1",       // Optional (OpenAI)
      "use_previous_response_id": true,       // Optional (OpenAI)
      "parameters": {                         // Optional - model-specific
        "key": "value"
      }
    }
  }
}
```

### Required Fields

| Field | Description |
|-------|-------------|
| `model_id` | The actual model identifier sent to the API |
| `provider` | Provider name: `anthropic`, `openai`, `google` |
| `sdk` | LangChain SDK class: `langchain_anthropic.ChatAnthropic`, `langchain_openai.ChatOpenAI`, etc. |
| `api_key_env` | Environment variable containing the API key |

### Optional Fields

| Field | Description |
|-------|-------------|
| `base_url` | Custom API endpoint (for proxies or alternative providers) |
| `output_version` | OpenAI-specific output format |
| `use_previous_response_id` | OpenAI-specific response chaining |
| `parameters` | Model-specific parameters (e.g., `reasoning.effort` for OpenAI, `enable_thinking` for Qwen) |

### Adding a New LLM Provider

#### Step 1: Choose the Right SDK

| Model Type | Recommended SDK | Notes |
|------------|-----------------|-------|
| Anthropic native | `langchain_anthropic.ChatAnthropic` | Claude models |
| OpenAI native | `langchain_openai.ChatOpenAI` | GPT models, supports responses API |
| Google | `langchain_google_genai.ChatGoogleGenerativeAI` | Gemini models |
| OpenAI-compatible with reasoning | `langchain_deepseek.ChatDeepSeek` or `langchain_qwq.ChatQwen` | Use to capture reasoning/thinking tokens |
| Interleaved thinking models | `langchain_anthropic.ChatAnthropic` | For Minimax, Kimi K2 - use provider's Anthropic endpoint |

#### SDK Selection Guidelines

1. **Standard OpenAI-compatible models**: Use `langchain_openai.ChatOpenAI`

2. **Models with reasoning/thinking output**: If the model outputs reasoning tokens via OpenAI completion API, use `langchain_deepseek.ChatDeepSeek` or `langchain_qwq.ChatQwen` to properly capture the reasoning content.
   ```json
   {
     "qwen3-max": {
       "sdk": "langchain_qwq.ChatQwen",
       "parameters": { "enable_thinking": true }
     }
   }
   ```

3. **OpenAI Responses API support**: Some models (like Doubao) support OpenAI's responses API.
   ```json
   {
     "doubao-seed-code": {
       "sdk": "langchain_openai.ChatOpenAI",
       "output_version": "responses/v1",
       "use_previous_response_id": true
     }
   }
   ```

4. **Interleaved thinking models (Minimax, Kimi K2)**: These have native interleaved thinking similar to Claude. Use `langchain_anthropic.ChatAnthropic` with the provider's Anthropic-compatible endpoint.
   ```json
   {
     "kimi-k2-thinking": {
       "sdk": "langchain_anthropic.ChatAnthropic",
       "base_url": "https://api.moonshot.ai/anthropic"
     }
   }
   ```
   > **Note**: Third-party providers (e.g., OpenRouter) may have poor support for interleaved thinking - prefer direct provider endpoints.

#### Step 2: Verify Base URL and Regional Settings

- Check provider documentation for the correct endpoint
- Be aware of regional differences (e.g., `api.z.ai` vs regional variants)
- Chinese providers often have different endpoints for different regions

#### Step 3: Add Configuration

1. Add definition to `llms.json`:
```json
{
  "llms": {
    "my-custom-model": {
      "model_id": "custom-model-v1",
      "provider": "my-provider",
      "sdk": "langchain_anthropic.ChatAnthropic",
      "api_key_env": "MY_CUSTOM_API_KEY",
      "base_url": "https://api.my-provider.com/anthropic"
    }
  }
}
```

2. Add API key to `.env`:
```bash
MY_CUSTOM_API_KEY=your-api-key
```

3. Update `config.yaml`:
```yaml
llm:
  name: "my-custom-model"
```

---

## Example Configurations

### Minimal Setup

Only requires one LLM provider and Daytona. Uses the bundled yfinance MCP server (no additional API keys needed).

**config.yaml**:
```yaml
llm:
  name: "claude-sonnet-4-5"  # Or any configured model

mcp:
  servers:
    - name: "tavily"
      enabled: false  # Requires TAVILY_API_KEY

    - name: "yfinance"
      enabled: true
      transport: "stdio"
      command: "uv"
      args: ["run", "python", "mcp_servers/yfinance_mcp_server.py"]

storage:
  provider: "none"
```

**.env**:
```bash
# One LLM provider (choose one)
ANTHROPIC_API_KEY=your-key
# or OPENAI_API_KEY, GEMINI_API_KEY, ZAI_API_KEY, etc.

# Daytona (required)
DAYTONA_API_KEY=your-key
```

### Full Setup (Multiple LLMs + Storage)

**config.yaml**:
```yaml
llm:
  name: "claude-opus-4-5"

daytona:
  snapshot_enabled: true
  snapshot_name: "ptc-full-v1"

mcp:
  servers:
    - name: "tavily"
      enabled: true
      transport: "stdio"
      command: "npx"
      args: ["-y", "tavily-mcp@latest"]
      env:
        TAVILY_API_KEY: "${TAVILY_API_KEY}"

    - name: "yfinance"
      enabled: true
      transport: "stdio"
      command: "uv"
      args: ["run", "python", "mcp_servers/yfinance_mcp_server.py"]

storage:
  provider: "r2"

logging:
  level: "DEBUG"
```

**.env**:
```bash
ANTHROPIC_API_KEY=your-key
DAYTONA_API_KEY=your-key
TAVILY_API_KEY=your-key
R2_ACCOUNT_ID=your-account
R2_ACCESS_KEY_ID=your-key
R2_SECRET_ACCESS_KEY=your-secret
R2_BUCKET_NAME=your-bucket
LANGSMITH_API_KEY=your-key
```
