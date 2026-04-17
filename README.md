# Knowledge Agent

一个最小 Python 项目：输入一段文本，调用 LLM，把内容整理成结构化知识卡片，并打印为 JSON。

## 项目结构

```text
knowledge-agent/
├── knowledge_card.py   # 主脚本：输入文本，调用 LLM，输出 JSON
├── notes_store.py      # 预留：后续可把知识卡片保存到 notes.json
├── requirements.txt    # Python 依赖
└── README.md           # 使用说明
```

## 安装

建议使用 Python 3.10 或更高版本。

```bash
pip install -r requirements.txt
```

## 配置 API Key

脚本默认使用 OpenAI SDK。请先设置环境变量：

PowerShell:

```powershell
$env:OPENAI_API_KEY="你的 API Key"
```

macOS / Linux:

```bash
export OPENAI_API_KEY="你的 API Key"
```

也可以通过 `OPENAI_MODEL` 指定模型；不设置时默认使用 `gpt-4.1-mini`。

## 使用

直接传入文本：

```bash
python knowledge_card.py "Prompt engineering is the practice of designing inputs for language models."
```

或者从标准输入读取：

```bash
echo "机器学习是一种让计算机从数据中学习规律的方法。" | python knowledge_card.py
```

输出示例：

```json
{
  "topic": "Prompt Engineering",
  "summary": "Prompt engineering focuses on designing effective inputs for language models.",
  "keywords": ["prompt", "language model", "AI"],
  "suggested_category": "Artificial Intelligence",
  "deepen": ["How do prompts affect model behavior?", "What are common prompt patterns?"]
}
```

## JSON 字段说明

- `topic`: 主题名称
- `summary`: 简短总结
- `keywords`: 关键词列表
- `suggested_category`: 建议分类
- `deepen`: 后续可以深入思考或学习的问题列表

## 后续接入 notes.json

项目已经预留了 `notes_store.py`。之后如果想把每次生成的知识卡片保存起来，可以在 `knowledge_card.py` 中调用：

```python
from notes_store import save_note

save_note(card)
```

这会把卡片追加保存到 `notes.json`。
