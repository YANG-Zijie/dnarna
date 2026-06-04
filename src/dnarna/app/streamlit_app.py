import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from dnarna.app.pipeline import AppJobConfig, run_app_job
from dnarna.app.ui.i18n import build_i18n, make_translator, select_language
from dnarna.app.ui.seq_io import load_path_file, parse_pasted, summarize_sequences
from dnarna.app.ui.styles import apply_global_styles
from dnarna.data.seq.window import DEFAULT_STRIDE, DEFAULT_WINDOW_SIZE, MAX_WINDOW_SIZE


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _device_ok(device: str) -> bool:
    dev = (device or "").strip()
    if not dev:
        return True
    if dev == "cpu":
        return True
    if dev.startswith("cuda"):
        try:
            import torch
        except Exception:
            return False
        if not torch.cuda.is_available():
            return False
        if dev == "cuda":
            return True
        if ":" in dev:
            _, idx = dev.split(":", 1)
            if idx.isdigit():
                return int(idx) < torch.cuda.device_count()
    return True


st.set_page_config(
    page_title="AI-Powered DNA/RNA Pair Prediction",
    page_icon="🧬",
    layout="wide",
)

apply_global_styles()

header_left, header_right = st.columns([12, 4], vertical_alignment="center")
with header_right:
    _, lang_col = st.columns([1, 3])
    with lang_col:
        lang = select_language(default="en")

i18n = build_i18n()
t = make_translator(lang, i18n)

with header_left:
    st.title(t("title"))
    st.caption(t("subtitle"))

with st.expander(t("overview_title"), expanded=False):
    st.markdown(t("overview_body"))

models_dir = Path("models")
default_dnabert_ckpt = str(models_dir / "dnabert2_classifier.pt")
default_rnafm_ckpt = str(models_dir / "rnafm_classifier.pt")

# ----------------------------- Step 1: Sequences -----------------------------

st.subheader(t("step_title", n=1, title=t("step1")))
st.caption(t("step1_caption"))

dna_col, rna_col = st.columns(2)
with dna_col:
    dna_mode = st.radio(
        t("mode_label_dna"),
        options=["path", "paste"],
        format_func=lambda v: t("mode_path") if v == "path" else t("mode_paste"),
        horizontal=True,
        key="dna_mode",
        help=t("mode_help"),
    )
    dna_path = st.text_input(
        t("dna_path"),
        value="",
        placeholder=t("dna_path_ph"),
        disabled=dna_mode != "path",
        key="dna_path",
        help=t("path_help"),
    )
    dna_text = st.text_area(
        t("dna_paste"),
        height=180,
        placeholder=t("dna_paste_ph"),
        disabled=dna_mode != "paste",
        key="dna_text",
        help=t("dna_paste_help"),
    )

with rna_col:
    rna_mode = st.radio(
        t("mode_label_rna"),
        options=["path", "paste"],
        format_func=lambda v: t("mode_path") if v == "path" else t("mode_paste"),
        horizontal=True,
        key="rna_mode",
        help=t("mode_help"),
    )
    rna_path = st.text_input(
        t("rna_path"),
        value="",
        placeholder=t("rna_path_ph"),
        disabled=rna_mode != "path",
        key="rna_path",
        help=t("path_help"),
    )
    rna_text = st.text_area(
        t("rna_paste"),
        height=180,
        placeholder=t("rna_paste_ph"),
        disabled=rna_mode != "paste",
        key="rna_text",
        help=t("rna_paste_help"),
    )

dna_seqs = (
    parse_pasted(dna_text, "DNA", lang=lang)
    if dna_mode == "paste"
    else load_path_file(dna_path, "DNA", lang=lang)
)
rna_seqs = (
    parse_pasted(rna_text, "RNA", lang=lang)
    if rna_mode == "paste"
    else load_path_file(rna_path, "RNA", lang=lang)
)

window_col1, window_col2 = st.columns(2)
window_size = window_col1.number_input(
    t("window_size"),
    min_value=1,
    max_value=MAX_WINDOW_SIZE,
    value=DEFAULT_WINDOW_SIZE,
    step=1,
    help=t("window_size_help", max_len=MAX_WINDOW_SIZE),
)
window_stride = window_col2.number_input(
    t("window_stride"),
    min_value=1,
    max_value=int(window_size),
    value=min(DEFAULT_STRIDE, int(window_size)),
    step=1,
    help=t("window_stride_help"),
)

stats_col1, stats_col2 = st.columns(2)
stats_col1.metric(
    t("dna_metric"),
    summarize_sequences(dna_seqs, none_text=t("seq_none"), template=t("seq_summary")),
)
stats_col2.metric(
    t("rna_metric"),
    summarize_sequences(rna_seqs, none_text=t("seq_none"), template=t("seq_summary")),
)

st.divider()

# ----------------------------- Step 2: Embeddings + optional top-K -----------------------------

st.subheader(t("step_title", n=2, title=t("step2")))
st.caption(t("step2_caption"))

cuda_available = _cuda_available()
if not cuda_available:
    st.warning(t("warning_no_cuda"))

k_col1, k_col2, k_col3 = st.columns(3)
top_k_dna = k_col1.number_input(
    t("topk_dna"),
    min_value=0,
    value=0,
    step=1,
    help=t("topk_dna_help"),
)
top_k_rna = k_col2.number_input(
    t("topk_rna"),
    min_value=0,
    value=0,
    step=1,
    help=t("topk_rna_help"),
)
score_batch_size = k_col3.number_input(
    t("score_batch_size"),
    min_value=1,
    value=1024,
    step=1,
    help=t("score_batch_help"),
    disabled=int(top_k_dna) <= 0 and int(top_k_rna) <= 0,
)

embed_col1, embed_col2 = st.columns(2)
dna_embed_batch_size = embed_col1.number_input(
    t("dna_embed_batch_size"),
    min_value=1,
    value=1024,
    step=1,
    help=t("dna_embed_batch_help"),
)
rna_embed_batch_size = embed_col2.number_input(
    t("rna_embed_batch_size"),
    min_value=1,
    value=256,
    step=1,
    help=t("rna_embed_batch_help"),
)

cp_col1, cp_col2 = st.columns(2)
dnabert_ckpt = cp_col1.text_input(
    t("dnabert_ckpt"),
    value=default_dnabert_ckpt,
    placeholder=t("dnabert_ckpt_ph"),
    help=t("dnabert_ckpt_help"),
    disabled=int(top_k_dna) <= 0,
)
rnafm_ckpt = cp_col2.text_input(
    t("rnafm_ckpt"),
    value=default_rnafm_ckpt,
    placeholder=t("rnafm_ckpt_ph"),
    help=t("rnafm_ckpt_help"),
    disabled=int(top_k_rna) <= 0,
)

dev_col1, dev_col2 = st.columns(2)
dna_device = dev_col1.text_input(
    t("dna_device"),
    value="",
    help=t("dna_device_help"),
)
rna_device = dev_col2.text_input(
    t("rna_device"),
    value="",
    help=t("rna_device_help"),
)
if (dna_device or "").strip() and not _device_ok(dna_device):
    st.error(t("device_invalid", label=t("dna_device"), device=dna_device))
if (rna_device or "").strip() and not _device_ok(rna_device):
    st.error(t("device_invalid", label=t("rna_device"), device=rna_device))

st.divider()

# ----------------------------- Step 3: Pair prediction -----------------------------

st.subheader(t("step_title", n=3, title=t("step3")))
st.caption(t("step3_caption"))

pair_ckpt = st.text_input(
    t("pair_ckpt"),
    value="",
    placeholder=t("pair_ckpt_ph"),
    help=t("pair_ckpt_help"),
)

pair_row1, pair_row2, pair_row3 = st.columns(3)
pair_feature_mode = pair_row1.selectbox(
    t("pair_feature_mode"),
    options=["concat", "absdiff", "mul", "all"],
    index=0,
    help=t("pair_feature_mode_help"),
)
pair_threshold = pair_row2.number_input(
    t("pair_threshold"),
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01,
    help=t("pair_threshold_help"),
)
pair_device = pair_row3.text_input(
    t("pair_device"),
    value="",
    help=t("pair_device_help"),
)
if (pair_device or "").strip() and not _device_ok(pair_device):
    st.error(t("device_invalid", label=t("pair_device"), device=pair_device))

pair_row4, pair_row5, pair_row6 = st.columns(3)
pair_dna_block_size = pair_row4.number_input(
    t("pair_dna_block_size"),
    min_value=1,
    value=64,
    step=1,
    help=t("pair_block_help"),
)
pair_rna_block_size = pair_row5.number_input(
    t("pair_rna_block_size"),
    min_value=1,
    value=64,
    step=1,
    help=t("pair_block_help"),
)
pair_batch_size = pair_row6.number_input(
    t("pair_batch_size"),
    min_value=1,
    value=4096,
    step=1,
    help=t("pair_batch_help"),
)

with st.expander(t("pair_advanced"), expanded=False):
    adv_col1, adv_col2 = st.columns(2)
    pair_chunk_size = adv_col1.number_input(
        t("pair_chunk_size"),
        min_value=1,
        value=4096,
        step=1,
        help=t("pair_chunk_help"),
    )
    pair_num_workers = adv_col2.number_input(
        t("pair_num_workers"),
        min_value=1,
        value=1,
        step=1,
        help=t("pair_num_workers_help"),
    )
    adv_col3, adv_col4 = st.columns(2)
    pair_max_dna = adv_col3.number_input(
        t("pair_max_dna"),
        min_value=0,
        value=0,
        step=1,
        help=t("pair_max_help"),
    )
    pair_max_rna = adv_col4.number_input(
        t("pair_max_rna"),
        min_value=0,
        value=0,
        step=1,
        help=t("pair_max_help"),
    )

st.divider()

# ----------------------------- Step 4: Output dir -----------------------------

st.subheader(t("step_title", n=4, title=t("step4")))
st.caption(t("step4_caption"))
default_out = Path("local/app_runs") / datetime.datetime.now().strftime(
    "run_%Y%m%d_%H%M%S"
)
output_dir = st.text_input(
    t("output_dir"),
    value=str(default_out),
    help=t("output_help"),
)

st.divider()

# ----------------------------- Summary + run -----------------------------

prefilter_on = int(top_k_dna) > 0 or int(top_k_rna) > 0
pair_label = Path(pair_ckpt).name if pair_ckpt else t("status_not_set")
pair_device_label = pair_device.strip() or t("status_auto")

summary_left, summary_right = st.columns([9, 3], vertical_alignment="top")
with summary_left:
    st.markdown(
        f"""
            <div class="card">
            <div class="card-title">{t("status_title")}</div>
            <div class="muted">{t("status_inputs")}: DNA={len(dna_seqs)} | RNA={len(rna_seqs)}</div>
            <div class="muted">{t("status_prefilter")}: {(t("status_enabled") if prefilter_on else t("status_disabled"))}</div>
            <div class="muted">{t("status_pair")}: {pair_label} · {pair_feature_mode} · {pair_device_label}</div>
            <div class="muted">{t("status_output")}: <span class="mono">{output_dir or t("status_not_set")}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with summary_right:
    if not cuda_available:
        st.warning(t("warning_no_cuda"))
    st.caption(t("hint_run"))
    submitted = st.button(t("run"), type="primary")

if submitted:
    if not dna_seqs or not rna_seqs:
        st.error(t("need_seqs"))
        st.stop()
    if not output_dir.strip():
        st.error(t("need_out"))
        st.stop()
    if (dna_device or "").strip() and not _device_ok(dna_device):
        st.error(t("device_invalid", label=t("dna_device"), device=dna_device))
        st.stop()
    if (rna_device or "").strip() and not _device_ok(rna_device):
        st.error(t("device_invalid", label=t("rna_device"), device=rna_device))
        st.stop()
    if not pair_ckpt.strip():
        st.error(t("need_pair_ckpt"))
        st.stop()
    pair_ckpt_path = Path(pair_ckpt).expanduser()
    if not pair_ckpt_path.exists():
        st.error(t("pair_ckpt_missing", path=pair_ckpt))
        st.stop()
    if (pair_device or "").strip() and not _device_ok(pair_device):
        st.error(t("device_invalid", label=t("pair_device"), device=pair_device))
        st.stop()

    try:
        cfg = AppJobConfig(
            output_dir=output_dir.strip() or None,
            window_size=int(window_size),
            window_stride=int(window_stride),
            dnabert_checkpoint=dnabert_ckpt.strip() or None,
            rnafm_checkpoint=rnafm_ckpt.strip() or None,
            pair_checkpoint=pair_ckpt.strip() or None,
            top_k_dna=int(top_k_dna),
            top_k_rna=int(top_k_rna),
            dna_device=dna_device.strip() or None,
            rna_device=rna_device.strip() or None,
            dna_embed_batch_size=int(dna_embed_batch_size),
            rna_embed_batch_size=int(rna_embed_batch_size),
            score_batch_size=int(score_batch_size),
            save_artifacts=True,
            pair_feature_mode=str(pair_feature_mode),
            pair_dna_block_size=int(pair_dna_block_size),
            pair_rna_block_size=int(pair_rna_block_size),
            pair_batch_size=int(pair_batch_size),
            pair_chunk_size=int(pair_chunk_size),
            pair_num_workers=int(pair_num_workers),
            pair_threshold=float(pair_threshold),
            pair_device=pair_device.strip() or None,
            pair_max_dna=int(pair_max_dna),
            pair_max_rna=int(pair_max_rna),
        )
        with st.spinner(t("running")):
            result = run_app_job(dna_seqs=dna_seqs, rna_seqs=rna_seqs, cfg=cfg)
    except Exception as exc:  # noqa: BLE001
        st.error(t("failed", msg=str(exc)))
        st.exception(exc)
        st.stop()

    st.success(t("done"))
    st.write(
        t(
            "mode_line",
            mode=t("mode_all") if result.mode == "all_pairs" else t("mode_topk"),
            dna=result.used_dna_count,
            rna=result.used_rna_count,
        )
    )

    if result.dna_scores_path is not None and result.dna_scores_path.exists():
        st.markdown(t("dna_scores"))
        st.dataframe(
            pd.read_csv(result.dna_scores_path, nrows=50), use_container_width=True
        )
    if result.rna_scores_path is not None and result.rna_scores_path.exists():
        st.markdown(t("rna_scores"))
        st.dataframe(
            pd.read_csv(result.rna_scores_path, nrows=50), use_container_width=True
        )

    if result.pair_predictions_path is not None and result.pair_predictions_path.exists():
        st.markdown(t("pair_predictions"))
        st.dataframe(
            pd.read_csv(result.pair_predictions_path, nrows=200),
            use_container_width=True,
        )
        st.caption(
            t("pair_predictions_path", path=str(result.pair_predictions_path))
        )
    if result.pair_summary_path is not None and result.pair_summary_path.exists():
        st.markdown(t("pair_summary"))
        st.dataframe(
            pd.read_csv(result.pair_summary_path, nrows=200),
            use_container_width=True,
        )
        st.caption(t("pair_summary_path", path=str(result.pair_summary_path)))

    st.info(
        t(
            "used_paths",
            dna=str(result.used_dna_path),
            rna=str(result.used_rna_path),
            pairs=str(result.pair_summary_path or result.pair_predictions_path)
            if result.pair_summary_path or result.pair_predictions_path
            else t("status_not_set"),
        )
    )
    if cfg.output_dir:
        st.caption(t("artifacts", out=cfg.output_dir))

    st.download_button(
        t("download_cfg"),
        data=pd.Series(cfg.__dict__).to_json(indent=2),
        file_name="dnarna_app_config.json",
        mime="application/json",
    )
