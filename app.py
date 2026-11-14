import sys
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdf_indexing import (
    backup_legacy_chroma,
    clear_index_manifest,
    synchronize_pdf_index,
)
from langchain_community.embeddings import HuggingFaceEmbeddings  # 引入本地向量模型
from study_agents import StudyAssistant, StudyProgress

BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# 2. 基础路径配置
# ---------------------------------------------------------------------------
PDF_DIR = BASE_DIR / "drone_pdfs"
CHROMA_DIR = BASE_DIR / "chroma_drone_db"
INDEX_MANIFEST = BASE_DIR / "pdf_index_manifest.sqlite3"

print("正在初始化无人机大容量资料 RAG 学习系统...")

# 本地 CPU 向量模型与低内存本地对话模型
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatOllama(
    model="qwen2.5:0.5b",
    temperature=0.2,
    num_ctx=2048,
    keep_alive="30s",
)

# ---------------------------------------------------------------------------
# 3. 增量同步 PDF 与持久化向量数据库
# ---------------------------------------------------------------------------
PDF_DIR.mkdir(parents=True, exist_ok=True)
legacy_backup = backup_legacy_chroma(CHROMA_DIR, INDEX_MANIFEST)
if legacy_backup is not None:
    print(
        "旧向量库没有索引清单，已保留为备份并准备重新建立索引: "
        f"{legacy_backup}"
    )

chroma_has_data = CHROMA_DIR.is_dir() and any(CHROMA_DIR.iterdir())
if INDEX_MANIFEST.exists() and not chroma_has_data:
    print("未找到向量数据库，将清空旧索引清单并重新索引 PDF。")
    clear_index_manifest(INDEX_MANIFEST)

vectorstore = Chroma(
    persist_directory=str(CHROMA_DIR), embedding_function=embeddings
)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
)
synchronize_pdf_index(
    PDF_DIR,
    vectorstore,
    INDEX_MANIFEST,
    text_splitter,
)

if not any(
    path.is_file() and path.suffix.lower() == ".pdf"
    for path in PDF_DIR.rglob("*")
):
    print(f"警告: '{PDF_DIR}' 文件夹中没有找到任何 PDF 文件！")
    sys.exit(0)

# ---------------------------------------------------------------------------
# 4. 初始化多 Agent 学习系统，共享 RAG 检索器与资料库
# ---------------------------------------------------------------------------
retriever = vectorstore.as_retriever(
    search_type="similarity", search_kwargs={"k": 5}
)
study_assistant = StudyAssistant(llm, retriever)
study_progress = StudyProgress()

# ---------------------------------------------------------------------------
# 5. 交互控制台
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  🚁 无人机多智能体协同备考系统已就绪！")
print("  - 导航规划教员：输入“我完全不懂，第一天该学什么”")
print("  - 知识拆解与翻译教员：输入术语或问题，如“什么是 VLOS？”")
print("  - 实战场景与考官：输入“出一道关于大风天气飞行的模拟题”")
print("  - 输入 'outline' 查看学习大纲，输入 'exit' 退出")
print("=" * 60 + "\n")

while True:
    try:
        user_input = input("请输入你的学习主题或指令: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit"]:
            print("学习结束，祝你飞行安全！")
            break

        if user_input.lower() == "outline":
            user_input = "请根据资料库为零基础学员制定学习大纲和今天的学习任务"

        response = study_assistant.respond(user_input, study_progress)
        print(f"\n[{response.agent_name}]")
        print(response.text)
        if response.sources:
            print("\n📖 资料出处：")
            for source in response.sources:
                print(f"  • {source}")

        print("\n" + "=" * 60 + "\n")

    except KeyboardInterrupt:
        print("\n学习已退出。")
        break
    except Exception as e:
        print(f"\n发生错误: {e}\n")