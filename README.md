Kevin's AI learning project uses a local Chroma vector database and Ollama to help learners prepare for the Canadian drone pilot exams.

## 本地多智能体备考

应用、向量检索和对话模型均在本机运行，不需要 Azure/Foundry 或 Gemini API。启动后由三个专职教员协作：

- **导航规划教员**：输入“我完全不懂，第一天该学什么”或 `outline`，获取入门说明和模块化学习计划。
- **知识拆解与翻译教员**：提出概念问题，获得中文讲解、英文术语对照及 PDF 文件/页码出处。
- **实战场景与考官**：输入“出一道关于大风天气飞行的模拟题”，作答后获得批改；答错时转交知识教员补课，并将主题记入本次运行的薄弱点，后续可输入“继续”强化练习。

三个角色共享现有 RAG 检索器和本地 Chroma 资料库。学习进度目前保存在当前进程内，退出后不持久化。有关真实地点空域的题目只会在检索资料能支持时给出确定结论。

## 首次安装（Windows）

1. 安装 [Ollama for Windows](https://ollama.com/download)。
2. 在 PowerShell 下载并测试低内存模型：

   ```powershell
   ollama run qwen2.5:0.5b
   ```

   首次运行会下载模型。测试完输入 `/bye` 退出。
3. 在项目虚拟环境中安装 Python 依赖并启动：

   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
   .\.venv\Scripts\python.exe .\app.py
   ```

应用使用 2048-token 上下文，并在空闲 30 秒后让 Ollama 卸载模型以释放内存。小模型回答质量有限，考试法规内容请对照手册核实。
