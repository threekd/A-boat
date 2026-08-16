#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hello_agents RAG 补丁（持久化修复）

修复 hello_agents==0.2.9 中 RAG 索引的两个 bug：

1. `index_chunks` 默认 batch_size=64 超过 DashScope 兼容模式 embedding 接口的
   单次上限(20)，整批请求被 400 拒绝：
   `Value error, batch size is invalid, it should not be larger than 20.`

2. 失败重试时向量归一化逻辑有误：`embedder.encode()` 返回 `list[np.ndarray]`，
   被错误地包成单条向量后调用 `float(numpy数组)` 抛
   `only 0-dimensional arrays can be converted to Python scalars`，
   最终所有向量静默变成零向量。

本模块通过 monkey-patch 在运行时覆盖 `hello_agents.memory.rag.pipeline` 中的
相关函数，因此即使 `uv sync` 重装依赖包后补丁依然生效。

用法：在应用入口（main.py / agents/QA_Assistant.py 等）的最顶部引入：

    import patches.hello_agents_rag_patch  # noqa: F401
"""

from typing import List, Dict, Optional, Any
import os
import time

import numpy as np

import hello_agents.memory.rag.pipeline as _pipeline

# DashScope 兼容模式 embedding 接口单次请求最多 20 条；默认给 10 留出余量。
MAX_EMBED_BATCH_SIZE = 20
DEFAULT_EMBED_BATCH_SIZE = 10


def _normalize_embedding_result(raw, dimension: int) -> List[List[float]]:
    """将 embedder.encode 的各种返回形态统一成 List[List[float]]。

    兼容：
    - 单条向量（numpy 1-D 数组 / 普通 list / numpy 标量）
    - 批量向量（list[np.ndarray] / list[list] / 2-D numpy 数组）
    - 被多余包了一层的 [[...]] 形态
    维度不符时自动填充/截断到 dimension，单条转换失败时用零向量兜底。
    """
    if raw is None:
        return []

    def _to_flat(v):
        if hasattr(v, "tolist"):
            v = v.tolist()
        if isinstance(v, (list, tuple)):
            # 单条向量被包成 [[...]]，只解一层（递归处理多重包裹）
            if v and isinstance(v[0], (list, tuple)):
                return _to_flat(v[0])
            return [float(x) for x in v]
        # numpy 标量 / 其它可转 float 的对象
        return [float(v)]

    # 2-D numpy 数组（多行向量）直接转成 list of lists
    if isinstance(raw, np.ndarray) and raw.ndim > 1:
        raw = raw.tolist()
    elif not isinstance(raw, (list, tuple)):
        raw = [raw]

    out: List[List[float]] = []
    for item in raw:
        try:
            vec = _to_flat(item)
        except Exception:
            vec = [0.0] * dimension
        if len(vec) != dimension:
            vec = (vec + [0.0] * dimension)[:dimension]
        out.append(vec)
    return out


def index_chunks(
    store=None,
    chunks: List[Dict] = None,
    cache_db: Optional[str] = None,
    batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
    rag_namespace: str = "default",
) -> None:
    """修复版 index_chunks：安全的 batch 大小 + 健壮的向量归一化。"""
    from hello_agents.memory.embedding import get_text_embedder, get_dimension

    if not chunks:
        print("[RAG] No chunks to index")
        return

    embedder = get_text_embedder()
    dimension = get_dimension(384)

    if store is None:
        store = _pipeline._create_default_vector_store(dimension)
        print(f"[RAG] Created default Qdrant store with dimension {dimension}")

    processed_texts = [
        _pipeline._preprocess_markdown_for_embedding(c["content"]) for c in chunks
    ]

    # 显式传入优先，否则读环境变量，最后用安全默认值；并夹紧到 API 上限内
    batch_size = batch_size or int(os.getenv("EMBED_BATCH_SIZE", DEFAULT_EMBED_BATCH_SIZE))
    batch_size = max(1, min(batch_size, MAX_EMBED_BATCH_SIZE))

    print(f"[RAG] Embedding start: total_texts={len(processed_texts)} batch_size={batch_size}")

    vecs: List[List[float]] = []
    for i in range(0, len(processed_texts), batch_size):
        part = processed_texts[i:i + batch_size]
        try:
            vecs.extend(_normalize_embedding_result(embedder.encode(part), dimension))
        except Exception as e:
            print(f"[WARNING] Batch {i} encoding failed: {e}")
            print(f"[RAG] Retrying batch {i} with smaller chunks...")

            # 降级重试：先拆成 8 条的更小批次，仍失败再逐条编码
            small_size = min(8, batch_size)
            any_ok = False
            for j in range(0, len(part), small_size):
                small_part = part[j:j + small_size]
                try:
                    time.sleep(1)  # 轻微限速，避免触发频率限制
                    vecs.extend(_normalize_embedding_result(embedder.encode(small_part), dimension))
                    any_ok = True
                except Exception as e2:
                    print(f"[WARNING] 小批次 {j // small_size} 仍然失败: {e2}")
                    # 逐条重试，仍失败则用零向量兜底
                    for text in small_part:
                        try:
                            vecs.extend(_normalize_embedding_result(embedder.encode(text), dimension))
                            any_ok = True
                        except Exception as e3:
                            print(f"[WARNING] 单条向量编码失败: {e3}, 使用零向量")
                            vecs.append([0.0] * dimension)

            if not any_ok:
                print(f"[ERROR] 批次 {i} 完全失败，使用零向量")

        print(f"[RAG] Embedding progress: {min(i + batch_size, len(processed_texts))}/{len(processed_texts)}")

    # Prepare metadata with RAG tags
    metas: List[Dict] = []
    ids: List[str] = []
    for ch in chunks:
        meta = {
            "memory_id": ch["id"],
            "user_id": "rag_user",
            "memory_type": "rag_chunk",
            "content": ch["content"],  # Keep original markdown content
            "data_source": "rag_pipeline",  # RAG identification tag
            "rag_namespace": rag_namespace,
            "is_rag_data": True,  # Clear RAG data marker
        }
        # Merge chunk metadata
        meta.update(ch.get("metadata", {}))
        metas.append(meta)
        ids.append(ch["id"])

    print(f"[RAG] Qdrant upsert start: n={len(vecs)}")
    success = store.add_vectors(vectors=vecs, metadata=metas, ids=ids)
    if success:
        print(f"[RAG] Qdrant upsert done: {len(vecs)} vectors indexed")
    else:
        print(f"[RAG] Qdrant upsert failed")
        raise RuntimeError("Failed to index vectors to Qdrant")


def apply() -> None:
    """应用补丁：覆盖 pipeline 模块中的相关函数。"""
    _pipeline._normalize_embedding_result = _normalize_embedding_result
    _pipeline.index_chunks = index_chunks
    print("[patch] hello_agents RAG 补丁已应用（batch_size + 向量归一化）")


# 模块被 import 时自动应用补丁
apply()
