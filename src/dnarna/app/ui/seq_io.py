from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from dnarna.app.pipeline import parse_sequences


def summarize_sequences(seqs: dict[str, str], *, none_text: str, template: str) -> str:
    if not seqs:
        return none_text
    lengths = [len(s) for s in seqs.values()]
    return template.format(n=len(seqs), min_len=min(lengths), max_len=max(lengths))


@dataclass
class LoadedSeqs:
    dna: dict[str, str]
    rna: dict[str, str]


def load_path_file(path_str: str, kind: str, *, lang: str) -> dict[str, str]:
    path_str = (path_str or "").strip()
    if not path_str:
        return {}
    path = Path(path_str).expanduser()
    if not path.exists():
        st.error(f"{kind}: {path} (not found)" if lang == "en" else f"{kind} 文件不存在：{path}")
        return {}
    if not path.is_file():
        st.error(f"{kind}: {path} (not a file)" if lang == "en" else f"{kind} 路径不是文件：{path}")
        return {}
    try:
        content = path.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        st.error(
            f"{kind}: failed to read {path} ({exc})"
            if lang == "en"
            else f"{kind} 文件读取失败：{path} ({exc})"
        )
        return {}
    try:
        return parse_sequences(content)
    except Exception as exc:  # noqa: BLE001
        st.error(
            f"{kind}: failed to parse {path} ({exc})"
            if lang == "en"
            else f"{kind} 文件解析失败：{path} ({exc})"
        )
        return {}


def parse_pasted(text: str, kind: str, *, lang: str) -> dict[str, str]:
    raw = (text or "").strip()
    if not raw:
        return {}
    try:
        return parse_sequences(raw)
    except Exception as exc:  # noqa: BLE001
        st.error(
            f"{kind}: paste parse failed: {exc}"
            if lang == "en"
            else f"{kind} 粘贴内容解析失败：{exc}"
        )
        return {}

