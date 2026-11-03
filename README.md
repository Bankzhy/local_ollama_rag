# Fully Local Standard RAG (Ollama + Chroma)

面向 Windows / Python 的可运行基础版。**模型和向量检索全部运行在本机**，不调用在线 LLM 或搜索引擎；首次安装 Python 包和下载 Ollama 权重需要联网。

## 0. 系统要求

- 已安装 Ollama（本机服务监听 `127.0.0.1:11434`）
- Python 3.10–3.12（推荐 3.11）
- 可运行 `qwen3:4b` 的内存/显存；资源较少时可以换成较小模型
- 支持带文字层的 `.pdf`，以及 `.docx`、`.txt`、`.md`；扫描 PDF 暂时不支持 OCR

## 1. Ollama 模型下载（联网完成一次即可）

在 PowerShell 中：

```powershell
ollama pull qwen3:4b
ollama pull qwen3-embedding:0.6b
ollama list
```

如果之前用的是其他 LLM 名称，例如 `qwen3:4b-instruct`，可通过 `$env:RAG_LLM = "qwen3:4b-instruct"` 指定。

`qwen3-embedding:0.6b` 只用于将文档和问题转换为数值向量，**不是回答问题的 LLM**。

## 2. 安装依赖（在项目目录执行）

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

如果没有 Python 3.11，可先执行 `py -0p` 列出已安装版本，再将 `-3.11` 改为你的版本。

## 3. 放入自己的专业文档

复制文件到 `docs/`，支持嵌套文件夹。例如：

```text
local_ollama_rag/
  docs/
    example_software.txt
    paper1.pdf
    manuals/
      domain_guide.docx
```

为了测试是否安装成功，已经附带 `example_software.txt`；之后可删除并改为真实文件。

## 4. 构建本地向量索引

```powershell
.\.venv\Scripts\python.exe local_rag.py index
```

步骤：PDF/Word 提取文字 → 按字符切分（默认900字符，重叠150字符） → Ollama `/api/embed` → Chroma 持久化到 `chroma_db/`。

**注意：再次运行 index 会重建整个 collection**，使其与当前 `docs/` 中的文档一致。文档增加、修改或删除之后需要重新运行。

## 5. 检索或问答

只检查检索：

```powershell
.\.venv\Scripts\python.exe local_rag.py search "What is Feature Envy?"
```

生成回答：

```powershell
.\.venv\Scripts\python.exe local_rag.py ask "What is Feature Envy?" --show-context
```

中文问题也可以：

```powershell
.\.venv\Scripts\python.exe local_rag.py ask "什么是 Long Method，什么条件下可考虑 Extract Method？" --top-k 4
```

持续问答：

```powershell
.\.venv\Scripts\python.exe local_rag.py chat
```

注意：`chat` 是多轮输入界面，但每轮问题独立运行 RAG，不自动关联前文；这符合无对话记忆的基本基线。

## 6. 可选配置

```powershell
$env:RAG_LLM = "qwen3:4b"              # 生成模型
$env:RAG_EMBED = "qwen3-embedding:0.6b"  # 向量模型
$env:RAG_CHUNK_SIZE = "900"             # 字符数，非 token 数
$env:RAG_CHUNK_OVERLAP = "150"
$env:RAG_EMBED_BATCH = "8"
```

改变 Embedding 模型时，**必须重新运行 `index`**。两个阶段一定使用相同的 Embedding 模型。

## 7. 常见错误

- **无法连接 Ollama**：检查 Ollama 是否运行；在 PowerShell 中执行 `ollama list` 和 `ollama run qwen3:4b`。
- **model not found**：执行 `ollama pull qwen3:4b` 和 `ollama pull qwen3-embedding:0.6b`，核对 `ollama list` 中的名称。
- **PDF 索引 0 个片段**：很可能是扫描版图片 PDF，需要先 OCR；脚本没有 OCR 功能。
- **检索结果不准确**：尝试减小 chunk 大小、改变 Top-K，或换更好的嵌入模型；每次改嵌入模型后重建索引。
- **答案不准确**：先执行 `search` 看检索结果，判断是检索问题还是 LLM 生成问题。
- **资料中找不到答案**：模型应明确表示证据不足，但无法从机制上保证绝不幻觉，需人工核验。
- **不是完全零网络流量**：初次安装依赖和 `ollama pull` 需要联网。安装完毕后业务调用仅使用本地 Ollama 地址；Chroma 匿名遥测已禁用。

## 8. 实验含义

这是 **Standard Dense RAG Baseline**，不是你计划的 Ontology + Knowledge Graph 方法。

- 向量库只存文本片段、Embedding 和来源元数据，并没有自动把文档转换为实体关系规则。
- 可用于后续与结构化知识表示方法对照。
- 记录 `top_k`、chunk 大小、embedding 模型、生成模型和 Prompt，保证论文实验可复现。
- 来源 `[S1]` 等来自真实本地文档片段，引用正确性仍需检查；cosine distance 不是可信度概率。

参考官方文档：
- https://docs.ollama.com/api/embed
- https://docs.ollama.com/api/chat
- https://docs.trychroma.com/docs/collections/configure
- https://docs.trychroma.com/docs/querying-collections/query-and-get
