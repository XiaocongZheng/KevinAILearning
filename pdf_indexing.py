import hashlib
import re
import sqlite3
import threading
import time
from contextlib import closing
from pathlib import Path
from typing import List, Optional

from google.api_core.exceptions import ResourceExhausted
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import TextSplitter
from pydantic import PrivateAttr


MIN_REQUEST_INTERVAL_SECONDS = 0.8
MAX_QUOTA_RETRIES = 5
MAX_RETRY_DELAY_SECONDS = 60
DAILY_QUOTA_FALLBACK_COOLDOWN_SECONDS = 24 * 60 * 60
DOCUMENTS_PER_INDEX_BATCH = 100


class DailyEmbeddingQuotaExceeded(RuntimeError):
    def __init__(self, message: str, retry_after_seconds: Optional[float]) -> None:
        self.retry_after_seconds = retry_after_seconds
        retry_timestamp = None
        if retry_after_seconds is not None:
            retry_timestamp = time.time() + retry_after_seconds
            retry_at = time.strftime(
                "%Y-%m-%d %H:%M:%S",
                time.localtime(retry_timestamp),
            )
            message += f" Google 建议于本地时间 {retry_at} 后重试。"
        self.retry_after_timestamp = retry_timestamp
        super().__init__(message)


class DailyQuotaCooldownActive(RuntimeError):
    def __init__(self, retry_after_timestamp: float) -> None:
        self.retry_after_timestamp = retry_after_timestamp
        retry_at = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(retry_after_timestamp)
        )
        super().__init__(
            "Google 每日嵌入配额仍在冷却期，程序暂不发送请求。"
            f"预计可于本地时间 {retry_at} 后重试。"
        )


def wait_until_retry_at(retry_at: float) -> bool:
    retry_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(retry_at))
    print(
        "程序将保持运行，等待 Gemini 配额恢复，预计本地时间 "
        f"{retry_time} 自动继续。按 Ctrl+C 可停止等待；索引进度已保存。"
    )
    try:
        while True:
            remaining = retry_at - time.time()
            if remaining <= 0:
                return True
            time.sleep(min(60, remaining))
    except KeyboardInterrupt:
        print("\n已停止等待。索引进度已保存，可稍后重新运行程序。")
        return False


class RateLimitedGoogleGenerativeAIEmbeddings(GoogleGenerativeAIEmbeddings):
    _request_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    _last_request_at: float = PrivateAttr(default=0.0)

    def embed_documents(
        self,
        texts: List[str],
        *,
        batch_size: int = 100,
        task_type: Optional[str] = None,
        titles: Optional[List[str]] = None,
        output_dimensionality: Optional[int] = None,
    ) -> List[List[float]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if titles is not None and len(titles) != len(texts):
            raise ValueError("titles must have the same length as texts")

        embeddings: List[List[float]] = []
        title_offset = 0
        batches = GoogleGenerativeAIEmbeddings._prepare_batches(texts, batch_size)
        for batch in batches:
            batch_titles = None
            if titles is not None:
                batch_titles = titles[title_offset : title_offset + len(batch)]
            title_offset += len(batch)
            embeddings.extend(
                self._embed_batch_with_retry(
                    batch,
                    batch_size=batch_size,
                    task_type=task_type,
                    titles=batch_titles,
                    output_dimensionality=output_dimensionality,
                )
            )
        return embeddings

    def _embed_batch_with_retry(
        self,
        batch: List[str],
        *,
        batch_size: int,
        task_type: Optional[str],
        titles: Optional[List[str]],
        output_dimensionality: Optional[int],
    ) -> List[List[float]]:
        for attempt in range(MAX_QUOTA_RETRIES + 1):
            with self._request_lock:
                wait_seconds = MIN_REQUEST_INTERVAL_SECONDS - (
                    time.monotonic() - self._last_request_at
                )
                if wait_seconds > 0:
                    time.sleep(wait_seconds)
                self._last_request_at = time.monotonic()

                try:
                    return super().embed_documents(
                        batch,
                        batch_size=batch_size,
                        task_type=task_type,
                        titles=titles,
                        output_dimensionality=output_dimensionality,
                    )
                except Exception as error:
                    if not self._caused_by_quota_error(error):
                        raise
                    error_text = self._exception_chain_text(error)
                    if self._is_daily_quota_error(error_text):
                        retry_after = self._retry_after_seconds(error_text)
                        if retry_after is None:
                            retry_after = DAILY_QUOTA_FALLBACK_COOLDOWN_SECONDS
                        raise DailyEmbeddingQuotaExceeded(
                            "Google 免费层每日嵌入请求配额已用尽。"
                            "本批次尚未写入向量库；进度已保存，"
                            "请等配额恢复后再运行。",
                            retry_after,
                        ) from error
                    if attempt == MAX_QUOTA_RETRIES:
                        raise

            delay = min(2 ** (attempt + 1), MAX_RETRY_DELAY_SECONDS)
            print(f"Gemini 嵌入请求触发配额限制，{delay} 秒后重试...")
            time.sleep(delay)

        raise RuntimeError("Embedding retry loop exited unexpectedly")

    @staticmethod
    def _caused_by_quota_error(error: BaseException) -> bool:
        current: Optional[BaseException] = error
        while current is not None:
            if isinstance(current, ResourceExhausted):
                return True
            current = current.__cause__
        return False

    @staticmethod
    def _exception_chain_text(error: BaseException) -> str:
        messages = []
        current: Optional[BaseException] = error
        while current is not None:
            messages.append(str(current))
            current = current.__cause__
        return " ".join(messages).lower()

    @staticmethod
    def _is_daily_quota_error(error_text: str) -> bool:
        return "per day" in error_text or "perday" in error_text

    @staticmethod
    def _retry_after_seconds(error_text: str) -> Optional[float]:
        match = re.search(r"retry in ([0-9hms.]+)", error_text)
        if match is None:
            return None

        multipliers = {"h": 3600, "m": 60, "s": 1}
        parts = re.findall(r"([0-9]+(?:\.[0-9]+)?)([hms])", match.group(1))
        if not parts:
            return None
        return sum(float(amount) * multipliers[unit] for amount, unit in parts)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            block = file.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def clear_index_manifest(manifest_path: Path) -> None:
    if not manifest_path.exists():
        return
    with closing(sqlite3.connect(manifest_path)) as connection:
        with connection:
            table = connection.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type = 'table' AND name = 'pdf_index'
                """
            ).fetchone()
            if table is None:
                return
            connection.execute("DELETE FROM pdf_index")


def synchronize_pdf_index(
    pdf_directory: Path,
    vectorstore: VectorStore,
    manifest_path: Path,
    text_splitter: TextSplitter,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_directory.mkdir(parents=True, exist_ok=True)

    with closing(sqlite3.connect(manifest_path)) as connection:
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS pdf_index (
                    source TEXT PRIMARY KEY,
                    sha256 TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('pending', 'indexed')),
                    next_chunk INTEGER NOT NULL DEFAULT 0,
                    retry_after REAL
                )
                """
            )
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(pdf_index)")
            }
            if "next_chunk" not in columns:
                connection.execute(
                    "ALTER TABLE pdf_index ADD COLUMN next_chunk INTEGER NOT NULL DEFAULT 0"
                )
            if "retry_after" not in columns:
                connection.execute("ALTER TABLE pdf_index ADD COLUMN retry_after REAL")
            connection.commit()

            pdf_paths = {
                str(path.resolve()): path
                for path in pdf_directory.rglob("*")
                if path.is_file() and path.suffix.lower() == ".pdf"
            }
            indexed_rows = connection.execute(
                "SELECT source, sha256, status FROM pdf_index"
            ).fetchall()
            indexed_sources = {row[0] for row in indexed_rows}

            for stale_source in sorted(indexed_sources - pdf_paths.keys()):
                vectorstore.delete(where={"source": stale_source})
                connection.execute(
                    "DELETE FROM pdf_index WHERE source = ?", (stale_source,)
                )
                connection.commit()
                print(f"已从索引移除不存在的 PDF: {stale_source}")

            cooldown = connection.execute(
                """
                SELECT MAX(retry_after) FROM pdf_index
                WHERE status = 'pending' AND retry_after > ?
                """,
                (time.time(),),
            ).fetchone()[0]
            if cooldown is not None:
                raise DailyQuotaCooldownActive(cooldown)

            for source, pdf_path in sorted(pdf_paths.items()):
                file_hash = sha256_file(pdf_path)
                row = connection.execute(
                    """
                    SELECT sha256, status, next_chunk
                    FROM pdf_index WHERE source = ?
                    """,
                    (source,),
                ).fetchone()
                if row is not None and row[0] == file_hash and row[1] == "indexed":
                    print(f"PDF 未变化，跳过: {pdf_path.name}")
                    continue

                is_resuming = (
                    row is not None
                    and row[0] == file_hash
                    and row[1] == "pending"
                )
                next_chunk = row[2] if is_resuming else 0
                if not is_resuming or next_chunk == 0:
                    vectorstore.delete(where={"source": source})
                    next_chunk = 0

                connection.execute(
                    """
                    INSERT INTO pdf_index (
                        source, sha256, status, next_chunk, retry_after
                    )
                    VALUES (?, ?, 'pending', ?, NULL)
                    ON CONFLICT(source) DO UPDATE SET
                        sha256 = excluded.sha256,
                        status = 'pending',
                        next_chunk = excluded.next_chunk,
                        retry_after = NULL
                    """,
                    (source, file_hash, next_chunk),
                )
                connection.commit()

                print(f"正在索引 PDF: {pdf_path.name}")
                documents = PyPDFLoader(source).load()
                for document in documents:
                    document.metadata["source"] = source

                chunks = text_splitter.split_documents(documents)
                if not chunks:
                    raise ValueError(f"PDF 中没有可索引的文本: {pdf_path}")

                if next_chunk > len(chunks):
                    vectorstore.delete(where={"source": source})
                    next_chunk = 0
                    connection.execute(
                        """
                        UPDATE pdf_index
                        SET next_chunk = 0
                        WHERE source = ?
                        """,
                        (source,),
                    )
                    connection.commit()

                while next_chunk < len(chunks):
                    batch_end = min(
                        next_chunk + DOCUMENTS_PER_INDEX_BATCH, len(chunks)
                    )
                    batch = chunks[next_chunk:batch_end]
                    ids = [
                        _chunk_id(source, file_hash, chunk_index, document)
                        for chunk_index, document in enumerate(batch, start=next_chunk)
                    ]
                    try:
                        vectorstore.add_documents(batch, ids=ids)
                    except DailyEmbeddingQuotaExceeded as error:
                        retry_after = (
                            time.time() + error.retry_after_seconds
                            if error.retry_after_seconds is not None
                            else None
                        )
                        connection.execute(
                            """
                            UPDATE pdf_index
                            SET retry_after = ?
                            WHERE source = ?
                            """,
                            (retry_after, source),
                        )
                        connection.commit()
                        raise

                    next_chunk = batch_end
                    connection.execute(
                        """
                        UPDATE pdf_index
                        SET next_chunk = ?, retry_after = NULL
                        WHERE source = ?
                        """,
                        (next_chunk, source),
                    )
                    connection.commit()

                connection.execute(
                    """
                    UPDATE pdf_index
                    SET status = 'indexed', retry_after = NULL
                    WHERE source = ?
                    """,
                    (source,),
                )
                connection.commit()
                print(f"索引完成: {pdf_path.name}（{len(chunks)} 个文本块）")


def _chunk_id(source: str, file_hash: str, index: int, document: Document) -> str:
    identity = f"{source}\0{file_hash}\0{index}\0{document.page_content}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def backup_legacy_chroma(
    chroma_directory: Path, manifest_path: Path
) -> Optional[Path]:
    if manifest_path.exists() or not chroma_directory.is_dir():
        return None
    if not any(chroma_directory.iterdir()):
        return None

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    backup_path = chroma_directory.with_name(
        f"{chroma_directory.name}.legacy-{timestamp}"
    )
    suffix = 1
    while backup_path.exists():
        backup_path = chroma_directory.with_name(
            f"{chroma_directory.name}.legacy-{timestamp}-{suffix}"
        )
        suffix += 1

    chroma_directory.rename(backup_path)
    return backup_path
