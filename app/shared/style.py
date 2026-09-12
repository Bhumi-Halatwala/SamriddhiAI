"""app/shared/style.py
Panel-specific CSS layered on top of Streamlit's base theme.

Two visual identities:
    bank mode:   cool grey bg, navy sidebar, serif headings, dense
    user mode:   cream bg, cream sidebar, sans headings, spacious

Base colors/fonts come from .streamlit/config.toml. This module adds
banners, cards, pills, typography scale, chart defaults, and sidebar
overrides.
"""

import streamlit as st


BRAND = {
    "navy":       "#152A4E",
    "navy_soft":  "#243B61",
    "green":      "#2A7F6F",
    "green_dark": "#1E6355",
    "amber":      "#B07A2E",
    "red":        "#A84434",
    "cream":      "#FBF9F4",
    "paper":      "#FFFFFF",
    "cool":       "#F4F6F9",
    "ink":        "#16202E",
    "muted":      "#5B6675",
    "line":       "#E1E4EA",
    "line_warm":  "#E6E0D2",
}

# Chart palette (used by Altair in the panels)
CHART_IN = "#2A7F6F"
CHART_OUT = "#A84434"
CHART_SAVINGS = "#2A7F6F"
CHART_BAR = "#2A7F6F"


def apply_brand_style(mode: str = "bank") -> None:
    if mode == "bank":
        _apply_bank()
    else:
        _apply_user()


# ------------------------------------------------------------------
# Shared base — both panels get this in addition to their mode CSS
# ------------------------------------------------------------------
def _base_css() -> str:
    B = BRAND
    return f"""
    html, body, .stApp, [class*="css"] {{
        font-size: 18px !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                     Roboto, "Helvetica Neue", Arial, sans-serif;
        color: {B["ink"]};
    }}
    .stApp p, .stApp li, .stApp .stMarkdown p {{
        font-size: 18px !important;
        line-height: 1.6 !important;
        color: {B["ink"]} !important;
    }}
    .stApp .stCaption, .stApp [data-testid="stCaptionContainer"] {{
        font-size: 15px !important;
        color: {B["muted"]} !important;
    }}
    /* Force heading colors in main area — fixes "not visible" headings */
    .stApp h1, .stApp h2, .stApp h3,
    .stApp h4, .stApp h5, .stApp h6 {{
        color: {B["ink"]} !important;
        opacity: 1 !important;
    }}
    /* Altair chart wrapper — force light background */
    .stAltairChart, [data-testid="stArrowVegaLiteChart"] {{
        background: {B["paper"]} !important;
        border: 1px solid {B["line"]};
        border-radius: 6px;
        padding: 8px !important;
    }}
    /* Kill Streamlit blue links and focus outlines globally */
    a, a:visited {{ color: inherit !important; text-decoration: none !important; }}
    *:focus {{ outline: 1px solid {B["green"]} !important; box-shadow: none !important; }}
    /* Kill input underline decoration */
    input, textarea, select {{
        border: 1px solid {B["line"]} !important;
        border-radius: 4px !important;
        box-shadow: none !important;
        outline: none !important;
    }}
    """


def _apply_bank() -> None:
    B = BRAND
    st.markdown(f"""
    <style>
    {_base_css()}

    .stApp {{ background: {B["cool"]} !important; }}
    header[data-testid="stHeader"] {{ background: transparent; }}

    /* Serif section headings */
    .stApp h1, .stApp h2, .stApp h3 {{
        font-family: Georgia, Cambria, "Times New Roman", serif !important;
        font-weight: 600;
    }}
    .stApp h1 {{ font-size: 34px !important; }}
    .stApp h2 {{ font-size: 26px !important; }}
    .stApp h3 {{ font-size: 22px !important; }}
    .stApp h4, .stApp h5, .stApp h6 {{
        font-family: Georgia, Cambria, serif !important;
        font-weight: 600;
        font-size: 19px !important;
        color: {B["navy"]} !important;
        margin-top: 16px; margin-bottom: 10px;
    }}

    .sa-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 18px 24px;
        background: {B["paper"]};
        border: 1px solid {B["line"]};
        border-left: 5px solid {B["navy"]};
        border-radius: 4px;
        margin-bottom: 20px;
    }}
    .sa-header .brand {{
        font-family: Georgia, Cambria, serif;
        font-size: 22px;
        font-weight: 700;
        color: {B["navy"]};
    }}
    .sa-header .tagline {{
        font-size: 15px;
        color: {B["muted"]};
        margin-top: 4px;
    }}
    .sa-header .panel-tag {{
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: {B["navy"]};
        background: {B["cool"]};
        border: 1px solid {B["line"]};
        padding: 6px 12px;
        border-radius: 3px;
    }}

    /* Viewing: CUST label */
    .sa-viewing {{
        font-size: 16px;
        color: {B["muted"]};
        padding: 10px 14px;
        background: {B["paper"]};
        border: 1px solid {B["line"]};
        border-radius: 4px;
        margin-bottom: 14px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }}
    .sa-viewing strong {{
        color: {B["navy"]};
        font-family: "SF Mono", Consolas, Menlo, monospace;
        font-weight: 600;
        letter-spacing: 0.02em;
    }}

    /* Equal-height cards */
    .sa-card {{
        background: {B["paper"]};
        border: 1px solid {B["line"]};
        border-radius: 4px;
        padding: 16px 18px;
        margin-bottom: 10px;
        min-height: 118px;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        box-sizing: border-box;
    }}
    .sa-card .label {{
        font-size: 13px;
        color: {B["muted"]};
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
        font-weight: 500;
    }}
    .sa-card .value {{
        font-size: 24px;
        font-weight: 600;
        color: {B["navy"]};
        line-height: 1.2;
    }}
    .sa-card .sub {{
        font-size: 13px;
        color: {B["muted"]};
        margin-top: 6px;
    }}

    .sa-pill {{
        display: inline-block;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        padding: 5px 11px;
        border-radius: 3px;
        border: 1px solid transparent;
        margin-top: 6px;
    }}
    .sa-pill.ok      {{ color: {B["green_dark"]}; background: #EAF2EF; border-color: #C8DBD5; }}
    .sa-pill.watch   {{ color: {B["amber"]};      background: #F6EEDF; border-color: #E4D3B4; }}
    .sa-pill.concern {{ color: {B["red"]};        background: #F4E5E1; border-color: #DFC5BF; }}

    .sa-reco {{
        background: {B["paper"]};
        border: 1px solid {B["line"]};
        border-top: 3px solid {B["navy"]};
        border-radius: 4px;
        padding: 18px 20px;
        margin-bottom: 12px;
    }}
    .sa-reco .title {{
        font-size: 18px;
        font-weight: 600;
        color: {B["navy"]};
        margin-bottom: 8px;
    }}
    .sa-reco .body {{
        font-size: 16px;
        color: {B["ink"]};
        line-height: 1.6;
    }}
    .sa-reco .meta {{
        font-size: 14px;
        color: {B["muted"]};
        margin-top: 8px;
    }}

    .sa-info {{
        background: #EEF4F2;
        border-left: 4px solid {B["green"]};
        padding: 16px 20px;
        border-radius: 3px;
        font-size: 17px;
        color: {B["ink"]};
        line-height: 1.6;
        margin: 10px 0 16px 0;
    }}
    .sa-info .head {{
        font-weight: 600;
        color: {B["green_dark"]};
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 6px;
        font-size: 17px;
    }}
    .sa-alert {{
        background: #F7EBE7;
        border-left: 4px solid {B["red"]};
        padding: 16px 20px;
        border-radius: 3px;
        font-size: 17px;
        color: {B["ink"]};
        line-height: 1.6;
        margin: 10px 0 16px 0;
    }}
    .sa-alert .head {{
        font-weight: 600;
        color: {B["red"]};
        margin-bottom: 6px;
        font-size: 17px;
    }}

    .sa-hr {{ border: none; border-top: 1px solid {B["line"]}; margin: 24px 0; }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        border-bottom: 1px solid {B["line"]};
    }}
    .stTabs [data-baseweb="tab"] {{
        padding: 10px 16px;
        font-size: 16px;
        font-weight: 500;
    }}
    .stTabs [aria-selected="true"] {{
        color: {B["navy"]} !important;
        border-bottom: 2px solid {B["navy"]} !important;
    }}

    /* Navy sidebar */
    section[data-testid="stSidebar"] {{
        background: {B["navy"]} !important;
        border-right: 1px solid {B["navy_soft"]};
    }}
    section[data-testid="stSidebar"] * {{
        color: #E8ECF3 !important;
        font-size: 15px;
    }}
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] strong {{
        color: #FFFFFF !important;
        font-family: Georgia, Cambria, serif;
    }}
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] select {{
        background: {B["navy_soft"]} !important;
        color: #FFFFFF !important;
        border: 1px solid {B["navy_soft"]} !important;
        border-radius: 4px !important;
        box-shadow: none !important;
        font-size: 15px;
    }}
    section[data-testid="stSidebar"] hr {{
        border-top: 1px solid {B["navy_soft"]};
    }}
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] *,
    section[data-testid="stSidebar"] small {{
        color: #E8ECF3 !important;
    }}
    /* Remove the blue underline Streamlit adds to selectbox/text_input */
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
    section[data-testid="stSidebar"] div[data-baseweb="input"] > div {{
        border-bottom: none !important;
        box-shadow: none !important;
        background: {B["navy_soft"]} !important;
        border-radius: 4px !important;
    }}
    </style>
    """, unsafe_allow_html=True)


def _apply_user() -> None:
    B = BRAND
    st.markdown(f"""
    <style>
    {_base_css()}

    .stApp {{ background: {B["cream"]} !important; }}
    header[data-testid="stHeader"] {{ background: transparent; }}

    .stApp h1, .stApp h2, .stApp h3,
    .stApp h4, .stApp h5, .stApp h6 {{
        color: {B["green_dark"]} !important;
        font-weight: 600;
    }}
    .stApp h1 {{ font-size: 34px !important; }}
    .stApp h2 {{ font-size: 28px !important; }}
    .stApp h3 {{ font-size: 24px !important; }}
    .stApp h4, .stApp h5, .stApp h6 {{
        font-size: 20px !important;
        margin-top: 18px; margin-bottom: 12px;
    }}

    .sa-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 22px 28px;
        background: {B["paper"]};
        border: 1px solid {B["line_warm"]};
        border-left: 5px solid {B["green"]};
        border-radius: 8px;
        margin-bottom: 24px;
    }}
    .sa-header .brand {{
        font-size: 24px;
        font-weight: 700;
        color: {B["green_dark"]};
    }}
    .sa-header .tagline {{
        font-size: 16px;
        color: {B["muted"]};
        margin-top: 4px;
    }}
    .sa-header .panel-tag {{
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: {B["green_dark"]};
        background: {B["cream"]};
        border: 1px solid {B["line_warm"]};
        padding: 6px 14px;
        border-radius: 14px;
    }}

    .sa-viewing {{
        font-size: 17px;
        color: {B["muted"]};
        padding: 10px 16px;
        background: {B["paper"]};
        border: 1px solid {B["line_warm"]};
        border-radius: 6px;
        margin-bottom: 16px;
    }}
    .sa-viewing strong {{
        color: {B["green_dark"]};
        font-family: "SF Mono", Consolas, Menlo, monospace;
        font-weight: 600;
    }}

    /* Spacious, equal-height cards */
    .sa-card {{
        background: {B["paper"]};
        border: 1px solid {B["line_warm"]};
        border-radius: 8px;
        padding: 22px 24px;
        margin-bottom: 16px;
        min-height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        box-sizing: border-box;
    }}
    .sa-card .label {{
        font-size: 15px;
        color: {B["muted"]};
        margin-bottom: 10px;
        font-weight: 500;
    }}
    .sa-card .value {{
        font-size: 30px;
        font-weight: 600;
        color: {B["green_dark"]};
        line-height: 1.15;
    }}
    .sa-card .sub {{
        font-size: 15px;
        color: {B["muted"]};
        margin-top: 10px;
    }}

    .sa-pill {{
        display: inline-block;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.04em;
        padding: 5px 14px;
        border-radius: 14px;
        border: 1px solid transparent;
        margin-top: 8px;
    }}
    .sa-pill.ok      {{ color: {B["green_dark"]}; background: #EEF6F3; border-color: #CBE2DA; }}
    .sa-pill.watch   {{ color: {B["amber"]};      background: #FAF3E6; border-color: #E7D6B8; }}
    .sa-pill.concern {{ color: {B["red"]};        background: #F9EDE9; border-color: #E4C9C1; }}

    .sa-reco {{
        background: {B["paper"]};
        border: 1px solid {B["line_warm"]};
        border-radius: 8px;
        padding: 22px 24px;
        margin-bottom: 16px;
    }}
    .sa-reco .title {{
        font-size: 20px;
        font-weight: 600;
        color: {B["green_dark"]};
        margin-bottom: 10px;
    }}
    .sa-reco .body {{
        font-size: 17px;
        color: {B["ink"]};
        line-height: 1.65;
    }}
    .sa-reco .meta {{
        font-size: 15px;
        color: {B["muted"]};
        margin-top: 10px;
    }}

    .sa-info {{
        background: #EEF6F3;
        border-left: 4px solid {B["green"]};
        border-radius: 6px;
        padding: 20px 24px;
        font-size: 17px;
        color: {B["ink"]};
        line-height: 1.65;
        margin: 14px 0 22px 0;
    }}
    .sa-info .head {{
        font-weight: 600;
        color: {B["green_dark"]};
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 8px;
        font-size: 17px;
    }}
    .sa-alert {{
        background: #FBF1EE;
        border-left: 4px solid {B["red"]};
        border-radius: 6px;
        padding: 20px 24px;
        font-size: 17px;
        color: {B["ink"]};
        line-height: 1.65;
        margin: 14px 0 22px 0;
    }}
    .sa-alert .head {{
        font-weight: 600;
        color: {B["red"]};
        margin-bottom: 8px;
        font-size: 17px;
    }}

    .sa-hr {{ border: none; border-top: 1px solid {B["line_warm"]}; margin: 30px 0; }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        border-bottom: 1px solid {B["line_warm"]};
    }}
    .stTabs [data-baseweb="tab"] {{
        padding: 12px 20px;
        font-size: 17px;
        font-weight: 500;
    }}
    .stTabs [aria-selected="true"] {{
        color: {B["green_dark"]} !important;
        border-bottom: 2px solid {B["green"]} !important;
    }}

    /* Cream sidebar */
    section[data-testid="stSidebar"] {{
        background: {B["cream"]} !important;
        border-right: 1px solid {B["line_warm"]};
    }}
    section[data-testid="stSidebar"] * {{
        color: {B["ink"]} !important;
        font-size: 15px;
    }}
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {{
        color: {B["green_dark"]} !important;
    }}
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] select {{
        background: {B["paper"]} !important;
        color: {B["ink"]} !important;
        border: 1px solid {B["line_warm"]} !important;
        border-radius: 6px !important;
        box-shadow: none !important;
        font-size: 15px;
    }}
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
    section[data-testid="stSidebar"] div[data-baseweb="input"] > div {{
        border-bottom: none !important;
        box-shadow: none !important;
        background: {B["paper"]} !important;
        border-radius: 6px !important;
    }}
    </style>
    """, unsafe_allow_html=True)


# ------------------------------------------------------------------
# HTML helpers
# ------------------------------------------------------------------
def header(title: str, tagline: str, panel_tag: str) -> str:
    return (
        f'<div class="sa-header">'
        f'  <div>'
        f'    <div class="brand">{title}</div>'
        f'    <div class="tagline">{tagline}</div>'
        f'  </div>'
        f'  <div class="panel-tag">{panel_tag}</div>'
        f'</div>'
    )


def viewing_label(customer_id: str) -> str:
    return f'<div class="sa-viewing">Viewing: <strong>{customer_id}</strong></div>'


def metric_card(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    return (
        f'<div class="sa-card">'
        f'  <div class="label">{label}</div>'
        f'  <div class="value">{value}</div>'
        f'  {sub_html}'
        f'</div>'
    )


def status_pill(level: str) -> str:
    labels = {"ok": "Stable", "watch": "Monitor", "concern": "Needs attention"}
    css = level if level in labels else "watch"
    return f'<span class="sa-pill {css}">{labels.get(level, level.title())}</span>'


def reco_card(title: str, body: str, meta: str = "") -> str:
    meta_html = f'<div class="meta">{meta}</div>' if meta else ""
    return (
        f'<div class="sa-reco">'
        f'  <div class="title">{title}</div>'
        f'  <div class="body">{body}</div>'
        f'  {meta_html}'
        f'</div>'
    )


def info_banner(head: str, body: str, icon: str = "") -> str:
    return (
        f'<div class="sa-info">'
        f'  <div class="head">{icon} {head}</div>'
        f'  <div>{body}</div>'
        f'</div>'
    )


def alert_banner(head: str, body: str) -> str:
    return (
        f'<div class="sa-alert">'
        f'  <div class="head">{head}</div>'
        f'  <div>{body}</div>'
        f'</div>'
    )


def hr() -> str:
    return '<hr class="sa-hr"/>'


SHIELD_SVG = (
    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" '
    'stroke="#1E6355" stroke-width="2" stroke-linecap="round" '
    'stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>'
)
