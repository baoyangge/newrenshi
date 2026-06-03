"""Databricks Volumes 上の原典フルパスを構築するヘルパー。

`RetrievalResult.source_document`（拡張子なしファイル名）と
`RetrievalResult.index_kind`（"faq" / "doc"）から、Volumes 上の
原典ファイルへのフルパスを組み立てる純粋関数を提供する。
"""

from __future__ import annotations

from src.config import VOLUME_CSV_PATH, VOLUME_PDF_PATH, IndexKind


def build_volume_path(source_document: str, index_kind: IndexKind) -> str:
    """RetrievalResult から Databricks Volumes 上の原典フルパスを構築する。

    Args:
        source_document: 拡張子なしのファイル名（例: "qa.20260330_法人事務"）。
            `RetrievalResult.source_document` の値をそのまま渡す。
        index_kind: ヒット元インデックス種別。"faq" なら CSV を、
            "doc" なら PDF をそれぞれ対応する Volumes ディレクトリ配下と
            みなす。

    Returns:
        Volumes フルパス（POSIX 形式の文字列）。

    Raises:
        ValueError: ``index_kind`` が "faq" / "doc" 以外の場合。
    """
    if index_kind == "faq":
        return f"{VOLUME_CSV_PATH}/{source_document}.csv"
    if index_kind == "doc":
        return f"{VOLUME_PDF_PATH}/{source_document}.pdf"
    raise ValueError(f"Unknown index_kind: {index_kind}")
