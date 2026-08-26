"""读取演示知识文件，并使用 ChromaDB 返回带来源引用的检索结果。"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from chromadb.config import Settings

from app.schemas.models import KnowledgeReference


# 该集合只存储第一阶段景区服务场景的演示数据。
_COLLECTION_NAME: Final = "scenic_service_demo"
_SOURCE_ID_PATTERN: Final = re.compile(r"^- Source ID: `(?P<value>[^`]+)`$", re.MULTILINE)
_SOURCE_TITLE_PATTERN: Final = re.compile(r"^# (?P<value>.+)$", re.MULTILINE)
_SOURCE_PATH_PATTERN: Final = re.compile(
    r"^- Citation path: `(?P<value>[^`]+)`$", re.MULTILINE
)
_RETRIEVAL_KEYWORDS_PATTERN: Final = re.compile(
    r"^- Retrieval keywords: (?P<value>.+)$", re.MULTILINE
)


@dataclass(frozen=True)
class _KnowledgeDocument:
    """Markdown 文件写入 ChromaDB 前使用的内部结构。"""

    source_id: str
    source_title: str
    source_path: str
    retrieval_keywords: list[str]
    content: str


class DemoHashEmbeddingFunction(EmbeddingFunction[Documents]):
    """仅用于演示 RAG 流程的本地确定性文本向量器。"""

    def __init__(self, dimensions: int = 64) -> None:
        if dimensions < 8:
            raise ValueError("dimensions must be at least 8")
        self._dimensions = dimensions

    def __call__(self, input: Documents) -> Embeddings:
        """将每段文本转换为稳定向量，不调用网络模型。"""

        return [self._embed_text(text) for text in input]

    @staticmethod
    def name() -> str:
        """向 ChromaDB 声明该演示向量器的稳定名称。"""

        return "demo_hash_embedding_v1"

    def get_config(self) -> dict[str, int]:
        """公开向量维度，供 ChromaDB 持久化配置时识别该实现。"""

        return {"dimensions": self._dimensions}

    @staticmethod
    def build_from_config(config: dict[str, int]) -> "DemoHashEmbeddingFunction":
        """根据已保存配置重建同一维度的演示向量器。"""

        dimensions = config.get("dimensions", 64)
        if not isinstance(dimensions, int):
            raise ValueError("dimensions must be an integer")
        return DemoHashEmbeddingFunction(dimensions=dimensions)

    def _embed_text(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions

        # 英文单词和中文双字词都参与向量计算，使“卫生间没水”这类演示查询
        # 能与包含相同词语的演示资料共享特征，同时减少单个汉字造成的误命中。
        for token in self._tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, byteorder="big", signed=False)
            index = value % self._dimensions
            vector[index] += 1.0 if value % 2 else -1.0

        norm = math.sqrt(sum(component * component for component in vector))
        if norm == 0.0:
            return vector
        return [component / norm for component in vector]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        normalized = text.lower()
        latin_tokens = re.findall(r"[a-z0-9_]+", normalized)
        chinese_sequences = re.findall(r"[\u4e00-\u9fff]+", normalized)
        chinese_tokens: list[str] = []
        for sequence in chinese_sequences:
            if len(sequence) == 1:
                chinese_tokens.append(sequence)
            else:
                chinese_tokens.extend(
                    sequence[index : index + 2]
                    for index in range(len(sequence) - 1)
                )
        return [*latin_tokens, *chinese_tokens]


class ChromaKnowledgeStore:
    """只保存明确标注为演示资料的本地 ChromaDB 知识库。"""

    def __init__(
        self,
        persist_directory: Path,
        collection_name: str = _COLLECTION_NAME,
    ) -> None:
        if not collection_name.strip():
            raise ValueError("collection_name must not be empty")

        self._persist_directory = Path(persist_directory)
        self._persist_directory.mkdir(parents=True, exist_ok=True)
        self._embedding_function = DemoHashEmbeddingFunction()

        # 关闭匿名遥测：第一阶段应完全在本地运行，不能把项目文本发送给遥测服务。
        self._client = chromadb.PersistentClient(
            path=str(self._persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embedding_function,
            metadata={
                "project": "service-request-agent",
                "data_scope": "demo_only",
            },
        )

    def index_directory(self, knowledge_directory: Path) -> list[str]:
        """将知识目录中的每个 Markdown 文件写入或更新到 ChromaDB。"""

        directory = Path(knowledge_directory)
        if not directory.is_dir():
            raise ValueError(f"knowledge directory does not exist: {directory}")

        documents = [
            self._load_markdown(path) for path in sorted(directory.glob("*.md"))
        ]
        source_ids = [document.source_id for document in documents]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("knowledge source IDs must be unique")
        if not documents:
            return []

        self._collection.upsert(
            ids=source_ids,
            documents=[document.content for document in documents],
            metadatas=[
                {
                    "source_id": document.source_id,
                    "source_title": document.source_title,
                    "source_path": document.source_path,
                    "is_demo_source": True,
                }
                for document in documents
            ],
        )
        return source_ids

    def search(self, query: str, limit: int = 3) -> list[KnowledgeReference]:
        """为非空文本查询返回按相关度排序且带引用的知识来源。"""

        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be empty")
        if limit < 1:
            raise ValueError("limit must be at least 1")

        document_count = self._collection.count()
        if document_count == 0:
            return []

        result = self._collection.query(
            query_texts=[normalized_query],
            # 先取回全部演示资料，再应用保守的命中门槛；这样不会因 limit 太小而
            # 漏掉实际包含相同关键词的资料。
            n_results=document_count,
            include=["documents", "metadatas", "distances"],
        )
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]

        references: list[KnowledgeReference] = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            if document is None or metadata is None:
                continue
            # 向量库总会给出最相近的结果。演示阶段只接受资料明确声明的完整关键词，
            # 避免“咨询”等常见词让无关诉求被误判为知识命中。
            if not self._has_demo_keyword_overlap(normalized_query, document):
                continue
            references.append(
                KnowledgeReference(
                    source_id=str(metadata["source_id"]),
                    source_title=str(metadata["source_title"]),
                    source_path=str(metadata["source_path"]),
                    excerpt=self._make_excerpt(document),
                    relevance_score=self._distance_to_demo_score(float(distance)),
                )
            )
            if len(references) == limit:
                break
        return references

    def count(self) -> int:
        """返回已入库演示资料数量，供诊断和测试使用。"""

        return self._collection.count()

    @staticmethod
    def _load_markdown(path: Path) -> _KnowledgeDocument:
        content = path.read_text(encoding="utf-8")
        return _KnowledgeDocument(
            source_id=ChromaKnowledgeStore._required_metadata(
                _SOURCE_ID_PATTERN, content, "Source ID"
            ),
            source_title=ChromaKnowledgeStore._required_metadata(
                _SOURCE_TITLE_PATTERN, content, "document title"
            ),
            source_path=ChromaKnowledgeStore._required_metadata(
                _SOURCE_PATH_PATTERN, content, "Citation path"
            ),
            retrieval_keywords=ChromaKnowledgeStore._read_retrieval_keywords(
                content
            ),
            content=content,
        )

    @staticmethod
    def _required_metadata(
        pattern: re.Pattern[str], content: str, label: str
    ) -> str:
        match = pattern.search(content)
        if match is None:
            raise ValueError(f"knowledge document is missing {label}")
        return match.group("value").strip()

    @staticmethod
    def _read_retrieval_keywords(content: str) -> list[str]:
        """读取资料显式声明的演示检索词，拒绝空列表或空词。"""

        raw_keywords = ChromaKnowledgeStore._required_metadata(
            _RETRIEVAL_KEYWORDS_PATTERN,
            content,
            "Retrieval keywords",
        )
        keywords = [
            keyword.strip().strip("`").strip()
            for keyword in raw_keywords.split(",")
        ]
        if not keywords or any(not keyword for keyword in keywords):
            raise ValueError("Retrieval keywords must contain non-empty values")
        return keywords

    @staticmethod
    def _make_excerpt(document: str) -> str:
        normalized = " ".join(document.split())
        if len(normalized) <= 1_000:
            return normalized
        return f"{normalized[:997]}..."

    @staticmethod
    def _has_demo_keyword_overlap(query: str, document: str) -> bool:
        """判断查询是否包含资料显式声明的完整演示检索词。"""

        return any(
            keyword in query
            for keyword in ChromaKnowledgeStore._read_retrieval_keywords(document)
        )

    @staticmethod
    def _distance_to_demo_score(distance: float) -> float:
        """将向量距离转为展示分数；该分数不表示模型准确率。"""

        return max(0.0, min(1.0, 1.0 / (1.0 + distance)))
