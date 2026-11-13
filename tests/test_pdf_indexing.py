import io
import sqlite3
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from google.api_core.exceptions import ResourceExhausted
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import pdf_indexing


class FakeLoader:
    def __init__(self, source):
        self.source = Path(source)

    def load(self):
        return [
            Document(
                page_content=self.source.read_text(encoding="utf-8"),
                metadata={"source": str(self.source), "page": 0},
            )
        ]


class FakeSplitter:
    def split_documents(self, documents):
        chunks = []
        for document in documents:
            for content in document.page_content.split("||"):
                chunks.append(
                    Document(
                        page_content=content,
                        metadata=document.metadata.copy(),
                    )
                )
        return chunks


class FakeVectorStore:
    def __init__(self):
        self.documents = {}
        self.add_calls = 0
        self.fail_next_add = False
        self.daily_quota_on_call = None

    def delete(self, *, where):
        source = where["source"]
        self.documents = {
            document_id: document
            for document_id, document in self.documents.items()
            if document.metadata.get("source") != source
        }

    def add_documents(self, documents, *, ids):
        self.add_calls += 1
        self.documents.update(zip(ids, documents))
        if self.add_calls == self.daily_quota_on_call:
            raise pdf_indexing.DailyEmbeddingQuotaExceeded(
                "Daily quota reached", 3600
            )
        if self.fail_next_add:
            self.fail_next_add = False
            raise RuntimeError("simulated partial write")


class PdfIndexingTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.pdf_directory = self.root / "pdfs"
        self.pdf_directory.mkdir()
        self.manifest_path = self.root / "manifest.sqlite3"
        self.vectorstore = FakeVectorStore()
        self.splitter = FakeSplitter()
        self.loader_patch = patch("pdf_indexing.PyPDFLoader", FakeLoader)
        self.loader_patch.start()
        self.addCleanup(self.loader_patch.stop)

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_pdf(self, name, content):
        path = self.pdf_directory / name
        path.write_text(content, encoding="utf-8")
        return path

    def sync(self):
        with redirect_stdout(io.StringIO()):
            pdf_indexing.synchronize_pdf_index(
                self.pdf_directory,
                self.vectorstore,
                self.manifest_path,
                self.splitter,
            )

    def test_skips_unchanged_pdf(self):
        self.write_pdf("manual.pdf", "Initial manual")

        self.sync()
        self.sync()

        self.assertEqual(self.vectorstore.add_calls, 1)
        self.assertEqual(len(self.vectorstore.documents), 1)

    def test_reindexes_changed_file_and_removes_deleted_file(self):
        manual = self.write_pdf("manual.pdf", "Version one")
        obsolete = self.write_pdf("obsolete.pdf", "Remove me")
        self.sync()

        manual.write_text("Version two", encoding="utf-8")
        obsolete.unlink()
        self.sync()

        self.assertEqual(self.vectorstore.add_calls, 3)
        self.assertEqual(
            [
                document.page_content
                for document in self.vectorstore.documents.values()
            ],
            ["Version two"],
        )

    def test_failed_write_is_retried_without_duplicate_chunks(self):
        self.write_pdf("manual.pdf", "Retry this manual")
        self.vectorstore.fail_next_add = True

        with self.assertRaisesRegex(RuntimeError, "simulated partial write"):
            self.sync()

        self.sync()

        self.assertEqual(len(self.vectorstore.documents), 1)
        connection = sqlite3.connect(self.manifest_path)
        try:
            status = connection.execute(
                "SELECT status FROM pdf_index"
            ).fetchone()[0]
        finally:
            connection.close()
        self.assertEqual(status, "indexed")

    def test_embedding_retries_only_quota_errors(self):
        calls = []

        def fake_embed(self, texts, **kwargs):
            calls.append(texts)
            if len(calls) == 1:
                raise RuntimeError("wrapped quota") from ResourceExhausted("quota")
            return [[0.1] for _ in texts]

        with (
            patch.object(GoogleGenerativeAIEmbeddings, "embed_documents", fake_embed),
            patch.object(pdf_indexing, "MIN_REQUEST_INTERVAL_SECONDS", 0),
            patch.object(pdf_indexing.time, "sleep"),
        ):
            embeddings = pdf_indexing.RateLimitedGoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key="test-key",
            )
            with redirect_stdout(io.StringIO()):
                result = embeddings.embed_documents(["manual text"])

        self.assertEqual(len(calls), 2)
        self.assertEqual(result, [[0.1]])

    def test_daily_quota_stops_retries_and_reads_google_retry_delay(self):
        calls = []

        def fake_embed(self, texts, **kwargs):
            calls.append(texts)
            error = RuntimeError(
                "Quota exceeded for requests per day limit; "
                "Please retry in 12h9m40.806469051s."
            )
            raise error from ResourceExhausted("daily quota")

        with (
            patch.object(GoogleGenerativeAIEmbeddings, "embed_documents", fake_embed),
            patch.object(pdf_indexing, "MIN_REQUEST_INTERVAL_SECONDS", 0),
            patch.object(pdf_indexing.time, "sleep"),
        ):
            embeddings = pdf_indexing.RateLimitedGoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key="test-key",
            )
            with self.assertRaises(pdf_indexing.DailyEmbeddingQuotaExceeded) as error:
                embeddings.embed_documents(["manual text"])

        self.assertEqual(len(calls), 1)
        self.assertAlmostEqual(error.exception.retry_after_seconds, 43780.806, places=3)

    def test_waits_until_quota_retry_time(self):
        sleeps = []
        with (
            patch.object(pdf_indexing.time, "time", side_effect=[100, 160, 200]),
            patch.object(
                pdf_indexing.time, "sleep", side_effect=lambda delay: sleeps.append(delay)
            ),
            redirect_stdout(io.StringIO()),
        ):
            should_resume = pdf_indexing.wait_until_retry_at(200)

        self.assertTrue(should_resume)
        self.assertEqual(sleeps, [60, 40])

    def test_ctrl_c_stops_waiting_without_losing_saved_progress(self):
        with (
            patch.object(pdf_indexing.time, "time", return_value=100),
            patch.object(pdf_indexing.time, "sleep", side_effect=KeyboardInterrupt),
            redirect_stdout(io.StringIO()),
        ):
            should_resume = pdf_indexing.wait_until_retry_at(200)

        self.assertFalse(should_resume)

    def test_daily_quota_resume_waits_then_continues_from_saved_batch(self):
        self.write_pdf("manual.pdf", "one||two||three||four")
        self.vectorstore.daily_quota_on_call = 2
        with patch.object(pdf_indexing, "DOCUMENTS_PER_INDEX_BATCH", 2):
            with self.assertRaises(pdf_indexing.DailyEmbeddingQuotaExceeded):
                self.sync()

            self.assertEqual(len(self.vectorstore.documents), 4)
            self.assertEqual(self.vectorstore.add_calls, 2)
            connection = sqlite3.connect(self.manifest_path)
            try:
                progress = connection.execute(
                    "SELECT next_chunk, retry_after FROM pdf_index"
                ).fetchone()
            finally:
                connection.close()
            self.assertEqual(progress[0], 2)
            self.assertGreater(progress[1], time.time())

            with self.assertRaises(pdf_indexing.DailyQuotaCooldownActive):
                self.sync()
            self.assertEqual(self.vectorstore.add_calls, 2)

            self.vectorstore.daily_quota_on_call = None
            with patch.object(
                pdf_indexing.time, "time", return_value=progress[1] + 1
            ):
                self.sync()

        self.assertEqual(len(self.vectorstore.documents), 4)
        self.assertEqual(self.vectorstore.add_calls, 3)
        self.assertEqual(
            {document.page_content for document in self.vectorstore.documents.values()},
            {"one", "two", "three", "four"},
        )

    def test_existing_manifest_gets_resume_columns(self):
        self.write_pdf("manual.pdf", "Migrate manifest")
        connection = sqlite3.connect(self.manifest_path)
        try:
            connection.execute(
                """
                CREATE TABLE pdf_index (
                    source TEXT PRIMARY KEY,
                    sha256 TEXT NOT NULL,
                    status TEXT NOT NULL
                )
                """
            )
            connection.commit()
        finally:
            connection.close()

        self.sync()

        connection = sqlite3.connect(self.manifest_path)
        try:
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(pdf_index)")
            }
        finally:
            connection.close()
        self.assertIn("next_chunk", columns)
        self.assertIn("retry_after", columns)

    def test_legacy_chroma_is_moved_to_a_preserved_backup(self):
        chroma_directory = self.root / "chroma_drone_db"
        chroma_directory.mkdir()
        database_file = chroma_directory / "chroma.sqlite3"
        database_file.write_text("legacy", encoding="utf-8")

        backup = pdf_indexing.backup_legacy_chroma(
            chroma_directory, self.manifest_path
        )

        if backup is None:
            self.fail("expected the legacy database to be backed up")
        self.assertFalse(chroma_directory.exists())
        self.assertEqual((backup / database_file.name).read_text(), "legacy")


if __name__ == "__main__":
    unittest.main()
