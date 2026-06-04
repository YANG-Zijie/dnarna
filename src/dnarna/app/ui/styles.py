from __future__ import annotations

import streamlit as st


def apply_global_styles(*, primary_accent: str = "#0A7F6B") -> None:
    st.markdown(
        f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&display=swap');
            html, body, [class*="css"]  {{
                font-family: 'Space Grotesk', 'Helvetica Neue', sans-serif;
                background: radial-gradient(circle at 20% 20%, rgba(10,127,107,0.08), transparent 35%),
                            radial-gradient(circle at 80% 10%, rgba(16,52,166,0.06), transparent 32%),
                            linear-gradient(120deg, #f9fbfd 0%, #f2f6ff 100%);
            }}
            section.main > div.block-container {{
                padding-top: 2rem;
            }}
            h1, h2, h3, h4 {{
                font-weight: 600;
            }}
            .stButton>button {{
                background:{primary_accent};
                color:white;
                border:none;
                border-radius:12px;
                padding:0.75rem 1.25rem;
                font-weight:600;
                box-shadow:0 12px 30px rgba(10,127,107,0.25);
            }}
            .stButton>button:hover {{
                background:#0e8f79;
            }}
            .card {{
                background: rgba(255,255,255,0.86);
                border: 1px solid rgba(16, 52, 166, 0.09);
                border-radius: 16px;
                padding: 14px 16px;
                box-shadow: 0 10px 30px rgba(20, 40, 80, 0.06);
            }}
            .card-title {{
                font-weight: 600;
                margin-bottom: 8px;
                color: rgba(16, 52, 166, 0.86);
            }}
            .muted {{
                color: rgba(0,0,0,0.58);
                font-size: 0.95rem;
            }}
            .mono {{
                font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )
