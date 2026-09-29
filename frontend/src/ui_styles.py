import base64
from pathlib import Path


def apply_hospital_theme(icon_path: str | Path, title: str, subtitle: str = "") -> None:
    icon_path = Path(icon_path)
    with open(icon_path, "rb") as f:
        icon_b64 = base64.b64encode(f.read()).decode()

    import streamlit as st
    st.markdown(f"""
    <style>
    :root {{ --navy:#172033; --blue:#1565C0; --soft:#F7F9FC; --border:#D9E1EC; }}
    .stApp {{ background: #F7F9FC; color: #172033; }}
    .block-container {{ padding-top: 1.4rem !important; padding-bottom: 3rem !important; max-width: 1400px; }}
    [data-testid="stSidebar"] {{ background: #172033; }}
    [data-testid="stSidebar"] * {{ color: #F7FAFF !important; }}
    [data-testid="stSidebarNav"]::before {{
        content: "Intelligent Hospital"; display:block; color:#FFFFFF; font-size:18px; font-weight:700;
        text-align:center; padding:20px 8px 16px 42px; margin:0 8px 16px;
        border-bottom:1px solid rgba(255,255,255,.18);
        background-image:url('data:image/png;base64,{icon_b64}'); background-size:28px 28px;
        background-repeat:no-repeat; background-position:calc(50% - 86px) 15px;
    }}
    .hero {{ background:linear-gradient(135deg,#EAF3FF,#FFFFFF); border:1px solid #D5E4F7; border-radius:20px; padding:28px; margin-bottom:20px; }}
    .hero h1 {{ color:#172033; margin:0; font-size:32px; }}
    .hero p {{ color:#536174; margin:.45rem 0 0; font-size:15px; }}
    .section-title {{ color:#172033; font-size:23px; font-weight:750; margin:10px 0 12px; }}
    div[data-testid="stChatMessage"] {{ border:1px solid #D9E1EC !important; border-radius:16px !important; margin:10px 0 !important; }}
    div[data-testid="stChatMessage"] [data-testid="stChatMessageContent"] {{ color:#172033 !important; }}
    div[data-testid="stChatMessage"] p, div[data-testid="stChatMessage"] li, div[data-testid="stChatMessage"] span {{ color:#172033 !important; }}
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {{ background:#EAF3FF !important; margin-left:12% !important; }}
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {{ background:#FFFFFF !important; margin-right:12% !important; }}
    [data-testid="stChatInput"] textarea {{ color:#172033 !important; background:#FFFFFF !important; }}
    [data-testid="stChatInput"] textarea::placeholder {{ color:#6B778A !important; }}
    .metric-card {{ background:#FFFFFF; border:1px solid #D9E1EC; border-radius:16px; padding:18px; box-shadow:0 4px 18px rgba(23,32,51,.05); }}
    .metric-label {{ color:#637083; font-size:13px; }} .metric-value {{ color:#172033; font-size:28px; font-weight:800; }}
    .small-muted {{ color:#66758A; font-size:13px; }}
    .success-box {{ background:#EAF8F0; border:1px solid #BCE4CC; color:#14532D; padding:14px 16px; border-radius:12px; }}
    .info-box {{ background:#EFF6FF; border:1px solid #C7DCF8; color:#174A7E; padding:14px 16px; border-radius:12px; }}
    </style>
    <div class="hero">
      <h1><img src="data:image/png;base64,{icon_b64}" style="height:38px;vertical-align:middle;margin-right:10px;">{title}</h1>
      {f'<p>{subtitle}</p>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)
