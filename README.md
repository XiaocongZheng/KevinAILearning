Kevin's AI learning project uses a local Chroma vector database and Gemini to help learners prepare for the Canadian drone pilot exams.

## 本地多智能体备考

应用仍在本机运行，复用 `drone_pdfs/`、Chroma 向量库和 Gemini API，不需要 Azure/Foundry 部署。启动后由三个专职教员协作：

- **导航规划教员**：输入“我完全不懂，第一天该学什么”或 `outline`，获取入门说明和模块化学习计划。
- **知识拆解与翻译教员**：提出概念问题，获得中文讲解、英文术语对照及 PDF 文件/页码出处。
- **实战场景与考官**：输入“出一道关于大风天气飞行的模拟题”，作答后获得批改；答错时转交知识教员补课，并将主题记入本次运行的薄弱点，后续可输入“继续”强化练习。

三个角色共享现有 RAG 检索器和本地 Chroma 资料库。学习进度目前保存在当前进程内，退出后不持久化。有关真实地点空域的题目只会在检索资料能支持时给出确定结论。
