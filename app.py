from dotenv import load_dotenv
import os
import sys
import time
from pathlib import Path
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdf_indexing import (
    DailyEmbeddingQuotaExceeded,
    DailyQuotaCooldownActive,
    RateLimitedGoogleGenerativeAIEmbeddings,
    backup_legacy_chroma,
    clear_index_manifest,
    synchronize_pdf_index,
    wait_until_retry_at,
)

# ---------------------------------------------------------------------------
# 1. 加载项目本地的 .env 文件（实现 API Key 隔离，不污染全局环境）
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

if not os.environ.get("GOOGLE_API_KEY"):
    print(
        "错误: 未在当前项目目录下检测到 .env 文件或 GOOGLE_API_KEY，请检查配置！"
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# 2. 基础路径配置
# ---------------------------------------------------------------------------
PDF_DIR = BASE_DIR / "drone_pdfs"
CHROMA_DIR = BASE_DIR / "chroma_drone_db"
INDEX_MANIFEST = BASE_DIR / "pdf_index_manifest.sqlite3"

print("正在初始化无人机大容量资料 RAG 学习系统...")

# 初始化 Gemini 向量模型与大模型
embeddings = RateLimitedGoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

# Gemini 对话模型
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

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
while True:
    try:
        synchronize_pdf_index(
            PDF_DIR,
            vectorstore,
            INDEX_MANIFEST,
            text_splitter,
        )
        break
    except DailyEmbeddingQuotaExceeded as error:
        print(f"\n{error}")
        retry_at = error.retry_after_timestamp or (
            time.time() + 24 * 60 * 60
        )
    except DailyQuotaCooldownActive as error:
        print(f"\n{error}")
        retry_at = error.retry_after_timestamp

    if not wait_until_retry_at(retry_at):
        sys.exit(0)

if not any(
    path.is_file() and path.suffix.lower() == ".pdf"
    for path in PDF_DIR.rglob("*")
):
    print(f"警告: '{PDF_DIR}' 文件夹中没有找到任何 PDF 文件！")
    sys.exit(0)

# ---------------------------------------------------------------------------
# 4. 构建 RAG 检索链与导师 Prompt
# ---------------------------------------------------------------------------
# 每次检索最相关的 5 个片段
retriever = vectorstore.as_retriever(
    search_type="similarity", search_kwargs={"k": 5}
)

study_system_prompt = (
    "你是一位资深的无人机飞行教员和专业技术翻译。\n"
    "用户正在通过庞大的无人机技术资料库学习。请根据下面检索到的参考资料（Context），"
    "为用户提供结构化的深度学习指导。\n\n"
    "你的回答必须严格包含以下几个部分：\n"
    "1. 📌 **核心知识点总结**：提炼出该主题最关键的技术、操作规范或安全警告。\n"
    "2. 🇨🇳 **重点术语与难点翻译**：如果资料包含英文，请给出专业、地好的中文翻译和原词对照。\n"
    "3. 🗺️ **原始文件阅读导航**：明确指出这些知识点分布在哪个文件的具体哪一页。\n"
    "4. 💡 **教员提示**：补充一条实操建议或易错避坑指南。\n\n"
    "如果参考资料中没有包含相关内容，请如实回答：“在现有的无人机学习资料中未找到该内容。”\n\n"
    "参考资料片段：\n{context}"
)

study_prompt = ChatPromptTemplate.from_messages([
    ("system", study_system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, study_prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# ---------------------------------------------------------------------------
# 5. 交互控制台
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  🚁 60MB 无人机资料库 AI 学习导师已就绪！")
print("  - 输入具体问题（如：‘电池存放与充电规范’、‘失联返航设置’）")
print("  - 输入 'outline' 可让 AI 结合资料梳理整体学习大纲")
print("  - 输入 'exit' 退出")
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
            query_text = "请根据资料库的内容，为我生成一份系统化的无人机学习大纲，划分出基础理论、飞行前准备、应急处置和法规政策等模块。"
        else:
            query_text = user_input

        print("\n[AI 导师正在 60MB 资料库中精准检索并提炼中...]")
        response = rag_chain.invoke({"input": query_text})

        print("\n" + response["answer"])

        # 溯源信息打印
        print("\n" + "─" * 40)
        print("📖 原始文件精确溯源：")
        sources = set()
        for doc in response["context"]:
            file_name = os.path.basename(doc.metadata.get("source", "未知文件"))
            page_num = doc.metadata.get("page", 0) + 1
            sources.add(f"  • 手册: 《{file_name}》 ➔ 建议翻阅第 {page_num} 页")

        for source in sources:
            print(source)

        print("\n" + "=" * 60 + "\n")

    except KeyboardInterrupt:
        print("\n学习已退出。")
        break
    except Exception as e:
        print(f"\n发生错误: {e}\n")