import streamlit as st
from crewai import Crew
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from email_automation import email_automation, create_email_task, send_email_via_smtp
from bert_resume_scorer import process_resumes_bert, get_top_candidates
from tfidf_resume_scorer import process_resumes_tfidf
from feedback_generator import generate_feedback, generate_single_feedback, format_feedback_for_download
from ats.explainability import ExplainabilityEngine
from config import BERT_MODEL, SCORING_WEIGHTS, USE_LLM_FOR_COMMUNICATION, DEFAULT_RANKING_MODEL
from auth_utils import authenticate_user, is_authenticated, login_user, logout_user, get_current_user, register_user, load_users
import os
import logging
import warnings

# Suppress Pydantic V1/V2 warnings from CrewAI
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# Handle HuggingFace Token
if not os.getenv("HF_TOKEN"):
    os.environ["HF_TOKEN"] = ""

# Setup logging
logging.basicConfig(filename="Logs/app.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Page config
st.set_page_config(
    page_title="AI Recruitment Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background: linear-gradient(135deg, #0E1117 0%, #1a1d29 100%);
    font-family: 'Inter', sans-serif;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Login Container */
.login-container {
    max-width: 450px;
    margin: 80px auto;
    padding: 50px 40px;
    background: rgba(38, 39, 48, 0.95);
    border-radius: 24px;
    box-shadow: 0 12px 48px rgba(0, 0, 0, 0.6);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(102, 126, 234, 0.1);
}

.login-title {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 38px;
    font-weight: 800;
    text-align: center;
    margin-bottom: 10px;
}

.login-subtitle {
    color: #9ca3af;
    text-align: center;
    font-size: 14px;
    margin-bottom: 35px;
}

/* Dashboard Header */
.dashboard-header {
    background: linear-gradient(135deg, #262730 0%, #313240 100%);
    padding: 24px 32px;
    border-radius: 16px;
    margin-bottom: 24px;
    border-left: 4px solid #667eea;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
}

.dashboard-title {
    font-size: 32px;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}

.dashboard-subtitle {
    color: #9ca3af;
    font-size: 14px;
    margin-top: 4px;
}

/* Model selector banner */
.model-banner {
    padding: 12px 20px;
    border-radius: 12px;
    margin-bottom: 16px;
    font-weight: 600;
    font-size: 14px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.model-bert { background: rgba(102, 126, 234, 0.15); border: 1px solid rgba(102, 126, 234, 0.4); color: #a5b4fc; }
.model-tfidf { background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.4); color: #6ee7b7; }

/* Candidate Cards */
.candidate-card {
    background: linear-gradient(135deg, #262730 0%, #313240 100%);
    padding: 28px;
    border-radius: 20px;
    margin: 20px 0;
    border-left: 5px solid #667eea;
    box-shadow: 0 6px 24px rgba(0, 0, 0, 0.4);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.candidate-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
}

.candidate-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 32px rgba(102, 126, 234, 0.35);
}

.rank-1 { border-left-color: #FFD700; }
.rank-2 { border-left-color: #C0C0C0; }
.rank-3 { border-left-color: #CD7F32; }

.rank-badge {
    display: inline-block;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 8px 16px;
    border-radius: 24px;
    font-weight: 700;
    font-size: 15px;
}

.candidate-name { font-size: 22px; font-weight: 700; color: #ffffff; margin: 10px 0; }

.score-section {
    display: flex; gap: 14px; align-items: center; margin: 14px 0;
}

.score-value {
    font-size: 40px; font-weight: 800;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Fraud Badges */
.fraud-badge {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 14px;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.fraud-low  { background: linear-gradient(135deg, #10b981, #059669); color: white; }
.fraud-medium { background: linear-gradient(135deg, #f59e0b, #d97706); color: white; }
.fraud-high { background: linear-gradient(135deg, #ef4444, #dc2626); color: white; }

/* Progress Bars */
progress { width: 100%; height: 10px; border-radius: 5px; border: none; }
progress::-webkit-progress-bar { background: #1E2228; border-radius: 5px; }
progress::-webkit-progress-value {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    border-radius: 5px;
}

/* Metric Cards */
.metric-card {
    background: rgba(38, 39, 48, 0.8);
    padding: 22px;
    border-radius: 16px;
    text-align: center;
    border: 1px solid rgba(102, 126, 234, 0.2);
    transition: all 0.3s;
}
.metric-card:hover { border-color: rgba(102, 126, 234, 0.5); transform: translateY(-4px); }
.metric-label { color: #9ca3af; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
.metric-value { font-size: 38px; font-weight: 800; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }

/* Feedback Cards */
.feedback-card {
    background: rgba(38, 39, 48, 0.7);
    border-radius: 16px;
    padding: 24px;
    margin: 12px 0;
    border: 1px solid rgba(102, 126, 234, 0.15);
}
.feedback-shortlisted { border-left: 4px solid #10b981; }
.feedback-rejected { border-left: 4px solid #ef4444; }
.feedback-name { font-size: 18px; font-weight: 700; color: #fff; margin-bottom: 12px; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 24px !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    transition: all 0.3s !important;
    box-shadow: 0 4px 16px rgba(102, 126, 234, 0.3) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}
.stButton > button:hover {
    transform: scale(1.04) !important;
    box-shadow: 0 6px 24px rgba(102, 126, 234, 0.5) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: rgba(38, 39, 48, 0.6);
    padding: 10px;
    border-radius: 14px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #9ca3af;
    border-radius: 10px;
    padding: 10px 22px;
    font-weight: 600;
    transition: all 0.3s;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white !important;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
}

/* Text Inputs */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: rgba(38, 39, 48, 0.8) !important;
    color: white !important;
    border: 1px solid rgba(102, 126, 234, 0.3) !important;
    border-radius: 12px !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2) !important;
}

.stSelectbox > div > div {
    background: rgba(38, 39, 48, 0.8) !important;
    border: 1px solid rgba(102, 126, 234, 0.3) !important;
    border-radius: 12px !important;
}

.dataframe thead th {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
}

.streamlit-expanderHeader {
    background: rgba(38, 39, 48, 0.6) !important;
    color: white !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Session State ─────────────────────────────────────────────────────────────
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "mcp_context" not in st.session_state:
    st.session_state.mcp_context = {
        "job_description": None,
        "ranked_resumes": None,
        "feedback_data": None,
        "selected_model": DEFAULT_RANKING_MODEL
    }

# ─── Authentication ────────────────────────────────────────────────────────────
if not is_authenticated():
    st.markdown("<div class='login-container'>", unsafe_allow_html=True)
    st.markdown("<h1 class='login-title'>🤖 AI Recruitment</h1>", unsafe_allow_html=True)
    st.markdown("<p class='login-subtitle'>Resume Fraud Detection & Credibility Scoring System</p>", unsafe_allow_html=True)

    auth_tab = st.radio("Auth", ["Login", "Sign Up"], horizontal=True, label_visibility="collapsed")

    if auth_tab == "Login":
        with st.form("login_form"):
            username = st.text_input("Email", placeholder="recruiter@ai.com")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            submit = st.form_submit_button("🔐 Login", use_container_width=True)
            if submit:
                if authenticate_user(username, password):
                    users = load_users()
                    login_user(username, users[username]["name"])
                    st.success("✅ Login successful!")
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials")
        st.markdown("<p style='text-align:center;color:#9ca3af;margin-top:20px;'>Demo: recruiter@ai.com / demo123</p>",
                    unsafe_allow_html=True)
    else:
        with st.form("signup_form"):
            name = st.text_input("Full Name", placeholder="John Doe")
            username = st.text_input("Email", placeholder="john@company.com")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="••••••••")
            submit = st.form_submit_button("📝 Create Account", use_container_width=True)
            if submit:
                if password != confirm_password:
                    st.error("❌ Passwords don't match")
                elif len(password) < 6:
                    st.error("❌ Password must be at least 6 characters")
                elif register_user(username, password, name):
                    st.success("✅ Account created! Please login.")
                else:
                    st.error("❌ Email already exists")

    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# ─── MAIN DASHBOARD ────────────────────────────────────────────────────────────
user_info = get_current_user()

col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    st.markdown(f"""
    <div class='dashboard-header'>
        <h1 class='dashboard-title'>🤖 AI Recruitment Dashboard</h1>
        <p class='dashboard-subtitle'>Welcome back, {user_info.get('name', 'Recruiter')} • Resume Authenticity Assessment & Credibility Scoring</p>
    </div>
    """, unsafe_allow_html=True)
with col3:
    if st.button("🚪 Logout", use_container_width=True):
        logout_user()
        st.rerun()

# Navigation — 4 tabs (JD Generator removed)
tabs = st.tabs([
    "🏆 Resume Ranking",
    "🛡️ Fraud Detection",
    "📧 Email Automation",
    "📊 Analytics"
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: Resume Ranking
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("## 📤 Upload Resumes & Job Description")

    # ── Model Selection ──────────────────────────────────────────────────────
    st.markdown("### 🧠 Select Ranking Model")
    model_col1, model_col2 = st.columns([2, 3])
    with model_col1:
        selected_model = st.radio(
            "Ranking Model",
            options=["⚡ TF-IDF (Fast)", "🤖 BERT (Accurate)"],
            index=1 if st.session_state.mcp_context.get("selected_model", DEFAULT_RANKING_MODEL) == "BERT" else 0,
            horizontal=True,
            label_visibility="collapsed",
            key="model_selector"
        )
        model_key = "BERT" if "BERT" in selected_model else "TF-IDF"
        st.session_state.mcp_context["selected_model"] = model_key

    with model_col2:
        if model_key == "BERT":
            st.markdown("""
            <div class='model-banner model-bert'>
                🤖 <b>BERT Mode</b> — Semantic deep-learning similarity. High accuracy, takes 30–60s.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='model-banner model-tfidf'>
                ⚡ <b>TF-IDF Mode</b> — Keyword-frequency ranking. Near-instant results, no model download.
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ── Upload Section ───────────────────────────────────────────────────────
    col1, col2 = st.columns([2, 1])
    with col1:
        with st.expander("📄 Upload Resumes", expanded=True):
            uploaded_resumes = st.file_uploader(
                "Upload Resume PDFs",
                type=["pdf"],
                accept_multiple_files=True,
                key="resume_upload"
            )
            folder_path = st.text_input("Or provide folder path", placeholder="E:/resumes")

    with col2:
        with st.expander("💼 Job Description", expanded=True):
            job_desc = st.text_area(
                "Paste Job Description",
                value=st.session_state.mcp_context["job_description"] or "",
                height=200,
                key="jd_input"
            )

    if st.button("🔍 Analyze Resumes", use_container_width=True, key="analyze_btn"):
        if job_desc and (uploaded_resumes or (folder_path and os.path.isdir(folder_path))):
            st.session_state.mcp_context["job_description"] = job_desc
            st.session_state.mcp_context["feedback_data"] = None  # Reset old feedback

            spinner_msg = (
                "⚡ TF-IDF ranking in progress..." if model_key == "TF-IDF"
                else "🤖 BERT deep analysis in progress (30–60s)..."
            )

            with st.spinner(spinner_msg):
                try:
                    if model_key == "TF-IDF":
                        all_results = process_resumes_tfidf(
                            job_description=job_desc,
                            dir_path=folder_path if folder_path else None,
                            uploaded_files=uploaded_resumes
                        )
                    else:
                        all_results = process_resumes_bert(
                            job_description=job_desc,
                            dir_path=folder_path if folder_path else None,
                            uploaded_files=uploaded_resumes
                        )

                    if all_results:
                        st.session_state.mcp_context["ranked_resumes"] = all_results
                        st.success(f"✅ [{model_key}] Analysis complete! Processed {len(all_results)} resumes.")
                    else:
                        st.warning("No candidates could be ranked.")

                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    logging.error(f"Resume analysis failed: {str(e)}")
        else:
            st.error("Please provide both a job description and resumes.")

    # ── Display Results ──────────────────────────────────────────────────────
    if st.session_state.mcp_context.get("ranked_resumes"):
        all_results = st.session_state.mcp_context["ranked_resumes"]

        # Quick Stats
        st.markdown("### 📊 Quick Stats")
        mc1, mc2, mc3, mc4 = st.columns(4)
        avg_score = sum(r['final_score'] for r in all_results) / len(all_results)
        shortlisted_count = len([r for r in all_results if r['rank'] <= 3])
        high_risk = len([r for r in all_results if r['final_score'] < 60])

        for col, label, val, color in [
            (mc1, "Total Resumes", str(len(all_results)), ""),
            (mc2, "Avg Score", f"{avg_score:.1f}%", ""),
            (mc3, "Shortlisted", str(shortlisted_count), ""),
            (mc4, "High Risk", str(high_risk), "color:#ef4444;"),
        ]:
            with col:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>{label}</div>
                    <div class='metric-value' style='{color}'>{val}</div>
                </div>
                """, unsafe_allow_html=True)

        # Top 3 Candidates
        st.markdown("### 🏆 Top 3 Candidates")
        for candidate in all_results[:3]:
            score = candidate['final_score']
            if score >= 80:
                fraud_class, fraud_level = "fraud-low", "Low Risk"
            elif score >= 60:
                fraud_class, fraud_level = "fraud-medium", "Medium Risk"
            else:
                fraud_class, fraud_level = "fraud-high", "High Risk"

            st.markdown(f"""
            <div class='candidate-card rank-{candidate["rank"]}'>
                <span class='rank-badge'>#{candidate['rank']}</span>
                <h3 class='candidate-name'>{candidate['candidate_name']}</h3>
                <div class='score-section'>
                    <span class='score-value'>{score:.1f}%</span>
                    <span class='fraud-badge {fraud_class}'>{fraud_level}</span>
                </div>
                <progress value='{score}' max='100'></progress>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("🔍 Detailed Analysis"):
                explanation = ExplainabilityEngine.generate_explanation(candidate)
                st.markdown(explanation)
                breakdown = candidate.get('breakdown', {}).get('breakdown', candidate.get('breakdown', {}))
                if isinstance(breakdown, dict) and 'breakdown' in breakdown:
                    breakdown = breakdown['breakdown']
                st.json(breakdown)

        # All Candidates Table
        st.markdown("### 📋 All Candidates")
        table_data = []
        for c in all_results:
            s = c['final_score']
            risk = "🟢 Low" if s >= 80 else ("🟡 Medium" if s >= 60 else "🔴 High")
            table_data.append({
                "Rank": f"#{c['rank']}",
                "Candidate": c['candidate_name'],
                "Match Score": f"{s:.1f}%",
                "Credibility": f"{c.get('credibility_detail', {}).get('score', 0):.1f}",
                "Fraud Risk": risk,
                "Shortlisted": "✅" if c['rank'] <= 3 else "❌"
            })
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

        # ── Feedback Section ─────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📝 AI-Powered Feedback")

        if not USE_LLM_FOR_COMMUNICATION:
            st.info("LLM feedback is disabled in config.py (USE_LLM_FOR_COMMUNICATION = False)")
        else:
            col_fb1, col_fb2 = st.columns([2, 1])
            with col_fb1:
                st.markdown("Generate personalized feedback for each candidate — shortlisted and rejected.")
            with col_fb2:
                generate_fb_btn = st.button("🚀 Generate Feedback", use_container_width=True, key="gen_feedback_btn")

            if generate_fb_btn:
                # Prepare candidate list
                all_candidates_for_fb = []
                for c in all_results:
                    all_candidates_for_fb.append({
                        'candidate_name': c['candidate_name'],
                        'file_name': c.get('filename', 'Unknown'),
                        'score': c['final_score'],
                        'is_shortlisted': c['rank'] <= 3,
                        'resume_text': c.get('resume_text', ''),
                        'skills_matched': c.get('skills_detail', {}).get('matched_skills', [])
                    })

                total_candidates = len(all_candidates_for_fb)
                progress_bar = st.progress(0, text="Preparing feedback generation...")
                status_text = st.empty()

                shortlisted_ui = []
                rejected_ui = []

                for idx, candidate in enumerate(all_candidates_for_fb):
                    cname = candidate['candidate_name']
                    is_sl = candidate['is_shortlisted']
                    progress_bar.progress(
                        int((idx / total_candidates) * 100),
                        text=f"⚡ Generating feedback for {cname}... ({idx+1}/{total_candidates})"
                    )
                    fb = generate_single_feedback(candidate, job_desc, is_sl)
                    if is_sl:
                        shortlisted_ui.append(fb)
                    else:
                        rejected_ui.append(fb)

                progress_bar.progress(100, text="✅ Feedback generation complete!")
                status_text.empty()

                # Store in session state
                st.session_state.mcp_context["feedback_data"] = {
                    "shortlisted": shortlisted_ui,
                    "rejected": rejected_ui
                }

            # Display stored feedback
            feedback_data = st.session_state.mcp_context.get("feedback_data")
            if feedback_data:
                fb_tab1, fb_tab2 = st.tabs(["✅ Shortlisted Feedback", "❌ Rejected Feedback"])

                def _render_feedback_card(fb, card_class):
                    f = fb.get("feedback", {})
                    strengths_html = "".join(f"<li>✅ {s}</li>" for s in f.get("strengths", []))
                    areas_html = "".join(f"<li>→ {a}</li>" for a in f.get("areas_for_improvement", []))
                    suggestions_html = "".join(f"<li>• {sg}</li>" for sg in f.get("suggestions", []))
                    st.markdown(f"""
                    <div class='feedback-card {card_class}'>
                        <div class='feedback-name'>👤 {fb['candidate_name']}</div>
                        <p style='color:#d1d5db;font-style:italic;'>{f.get('overall_message','')}</p>
                        <div style='display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:12px;'>
                            <div>
                                <p style='color:#6ee7b7;font-weight:600;margin-bottom:6px;'>💪 Strengths</p>
                                <ul style='color:#d1d5db;margin:0;padding-left:16px;'>{strengths_html}</ul>
                            </div>
                            <div>
                                <p style='color:#fbbf24;font-weight:600;margin-bottom:6px;'>📈 Areas for Growth</p>
                                <ul style='color:#d1d5db;margin:0;padding-left:16px;'>{areas_html}</ul>
                            </div>
                        </div>
                        <div style='margin-top:12px;'>
                            <p style='color:#a5b4fc;font-weight:600;margin-bottom:6px;'>💡 Suggestions</p>
                            <ul style='color:#d1d5db;margin:0;padding-left:16px;'>{suggestions_html}</ul>
                        </div>
                        <div style='margin-top:12px;background:rgba(102,126,234,0.1);padding:12px;border-radius:8px;'>
                            <p style='color:#9ca3af;font-size:13px;margin:0;'>🔜 <b>Next Steps:</b> {f.get('next_steps','')}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with fb_tab1:
                    if feedback_data.get("shortlisted"):
                        for fb in feedback_data["shortlisted"]:
                            _render_feedback_card(fb, "feedback-shortlisted")
                    else:
                        st.info("No shortlisted candidates.")

                with fb_tab2:
                    if feedback_data.get("rejected"):
                        for fb in feedback_data["rejected"]:
                            _render_feedback_card(fb, "feedback-rejected")
                    else:
                        st.info("No rejected candidates.")

                # Download Button
                report_text = format_feedback_for_download(feedback_data)
                st.download_button(
                    "📥 Download Feedback Report",
                    data=report_text,
                    file_name="candidate_feedback_report.txt",
                    mime="text/plain",
                    use_container_width=True
                )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: Fraud Detection (Enhanced)
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("## 🛡️ Fraud Detection Dashboard")

    if st.session_state.mcp_context.get("ranked_resumes"):
        results = st.session_state.mcp_context["ranked_resumes"]

        # ── Risk Counts ──────────────────────────────────────────────────────
        low_risk    = [r for r in results if r['final_score'] >= 80]
        medium_risk = [r for r in results if 60 <= r['final_score'] < 80]
        high_risk   = [r for r in results if r['final_score'] < 60]

        # ── Row 1: Pie + Grouped Bar ─────────────────────────────────────────
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            fig_pie = go.Figure(data=[go.Pie(
                labels=['✅ Low Risk', '⚠️ Medium Risk', '🚨 High Risk'],
                values=[len(low_risk), len(medium_risk), len(high_risk)],
                marker=dict(colors=['#10b981', '#f59e0b', '#ef4444']),
                hole=0.45,
                textinfo='label+percent',
                pull=[0, 0.05, 0.1]
            )])
            fig_pie.update_layout(
                template='plotly_dark',
                title="Fraud Risk Distribution",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e5e7eb'),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2)
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with chart_col2:
            top5 = results[:5]
            names = [r['candidate_name'] for r in top5]
            scores = [r['final_score'] for r in top5]
            cred_scores = [r.get('credibility_detail', {}).get('score', 0) for r in top5]
            skill_scores = [r.get('skills_detail', {}).get('score', 0) for r in top5]

            fig_bar = go.Figure(data=[
                go.Bar(name='Match Score', x=names, y=scores, marker_color='#667eea', text=[f"{s:.1f}%" for s in scores], textposition='outside'),
                go.Bar(name='Skills Match', x=names, y=skill_scores, marker_color='#10b981', text=[f"{s:.1f}%" for s in skill_scores], textposition='outside'),
                go.Bar(name='Credibility', x=names, y=cred_scores, marker_color='#764ba2', text=[f"{s:.1f}" for s in cred_scores], textposition='outside'),
            ])
            fig_bar.update_layout(
                barmode='group',
                template='plotly_dark',
                title="Top 5 Candidates — Multi-Score Comparison",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e5e7eb'),
                xaxis_title="Candidates",
                yaxis_title="Score",
                legend=dict(orientation="h", yanchor="bottom", y=-0.3)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # ── Row 2: Scatter + Gauge ────────────────────────────────────────────
        chart_col3, chart_col4 = st.columns(2)

        with chart_col3:
            df_scatter = pd.DataFrame({
                "Candidate": [r['candidate_name'] for r in results],
                "Match Score": [r['final_score'] for r in results],
                "Credibility": [r.get('credibility_detail', {}).get('score', 0) for r in results],
                "Risk": ["Low" if r['final_score'] >= 80 else ("Medium" if r['final_score'] >= 60 else "High") for r in results]
            })
            fig_scatter = px.scatter(
                df_scatter,
                x="Match Score",
                y="Credibility",
                color="Risk",
                hover_name="Candidate",
                color_discrete_map={"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"},
                title="Score vs Credibility (Fraud Risk Map)",
                template="plotly_dark"
            )
            fig_scatter.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e5e7eb')
            )
            fig_scatter.update_traces(marker=dict(size=12, opacity=0.85, line=dict(width=1, color='white')))
            st.plotly_chart(fig_scatter, use_container_width=True)

        with chart_col4:
            # Radar chart for top 3
            top3 = results[:3]
            categories = ['Match Score', 'Skills', 'Credibility', 'Experience']

            fig_radar = go.Figure()
            colors = ['#FFD700', '#C0C0C0', '#CD7F32']
            for i, candidate in enumerate(top3):
                bd = candidate.get('breakdown', {})
                if isinstance(bd, dict) and 'breakdown' in bd:
                    bd = bd['breakdown']
                values = [
                    candidate['final_score'],
                    candidate.get('skills_detail', {}).get('score', 0),
                    candidate.get('credibility_detail', {}).get('score', 0),
                    bd.get('experience', 50) if isinstance(bd, dict) else 50
                ]
                values += [values[0]]  # Close polygon
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories + [categories[0]],
                    fill='toself',
                    name=candidate['candidate_name'],
                    line_color=colors[i],
                    fillcolor=colors[i],
                    opacity=0.3
                ))

            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], color='#9ca3af'),
                    angularaxis=dict(color='#9ca3af')
                ),
                showlegend=True,
                template='plotly_dark',
                title="Top 3 Candidates — Skill Radar",
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e5e7eb')
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # ── Row 3: Score Distribution Histogram ──────────────────────────────
        st.markdown("### 📈 Score Distribution")
        fig_hist = go.Figure(data=[go.Histogram(
            x=[r['final_score'] for r in results],
            nbinsx=15,
            marker=dict(
                color=[r['final_score'] for r in results],
                colorscale=[[0, '#ef4444'], [0.5, '#f59e0b'], [1, '#10b981']],
                line=dict(color='rgba(255,255,255,0.2)', width=1)
            ),
            opacity=0.85
        )])
        fig_hist.update_layout(
            template='plotly_dark',
            title="Candidate Score Distribution",
            xaxis_title="Match Score (%)",
            yaxis_title="Number of Candidates",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e5e7eb')
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # ── Fraud Risk Table ──────────────────────────────────────────────────
        st.markdown("### 🔍 Detailed Fraud Indicators")

        fraud_table = []
        for r in results:
            s = r['final_score']
            fraud_table.append({
                "Rank": f"#{r['rank']}",
                "Candidate": r['candidate_name'],
                "Score": f"{s:.1f}%",
                "Risk Level": "🟢 Low" if s >= 80 else ("🟡 Medium" if s >= 60 else "🔴 High"),
                "Keyword Stuffing": "✅ Low" if s > 70 else "⚠️ Suspected",
                "Skill Exaggeration": "✅ Unlikely" if s > 75 else "⚠️ Possible",
                "Experience Consistency": "✅ Valid" if s > 65 else "⚠️ Questionable",
                "Education Verifiable": "✅ Yes" if r.get('credibility_detail', {}).get('score', 0) > 60 else "❓ Uncertain"
            })

        df_fraud = pd.DataFrame(fraud_table)
        st.dataframe(df_fraud, use_container_width=True, hide_index=True)

        # ── Individual Candidate Fraud Cards ─────────────────────────────────
        st.markdown("### 🎯 Top Resume Detailed Fraud Analysis")
        for candidate in results[:3]:
            s = candidate['final_score']
            cred = candidate.get('credibility_detail', {}).get('score', 0)
            skills_matched = candidate.get('skills_detail', {}).get('matched_skills', [])
            skills_missing = candidate.get('skills_detail', {}).get('missing_skills', [])

            risk_color = "#10b981" if s >= 80 else ("#f59e0b" if s >= 60 else "#ef4444")
            risk_label = "Low Risk" if s >= 80 else ("Medium Risk" if s >= 60 else "High Risk")

            with st.expander(f"#{candidate['rank']} — {candidate['candidate_name']} | Score: {s:.1f}% | {risk_label}", expanded=candidate['rank'] == 1):
                fd_col1, fd_col2, fd_col3, fd_col4 = st.columns(4)
                with fd_col1:
                    st.metric("Overall Score", f"{s:.1f}%")
                with fd_col2:
                    st.metric("Credibility", f"{cred:.1f}")
                with fd_col3:
                    st.metric("Skills Matched", len(skills_matched))
                with fd_col4:
                    st.metric("Skills Missing", len(skills_missing))

                mc1, mc2 = st.columns(2)
                with mc1:
                    st.markdown("**✅ Matched Skills:**")
                    if skills_matched:
                        for sk in skills_matched[:10]:
                            st.markdown(f"- `{sk}`")
                    else:
                        st.markdown("_No exact skill matches_")
                with mc2:
                    st.markdown("**⚠️ Missing Skills:**")
                    if skills_missing:
                        for sk in skills_missing[:10]:
                            st.markdown(f"- `{sk}`")
                    else:
                        st.markdown("_All required skills found_")

                # Mini gauge for fraud risk
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=s,
                    title={"text": "Authenticity Score", "font": {"color": "#e5e7eb"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#9ca3af"},
                        "bar": {"color": risk_color},
                        "bgcolor": "#1E2228",
                        "steps": [
                            {"range": [0, 60], "color": "rgba(239,68,68,0.2)"},
                            {"range": [60, 80], "color": "rgba(245,158,11,0.2)"},
                            {"range": [80, 100], "color": "rgba(16,185,129,0.2)"}
                        ],
                        "threshold": {"line": {"color": risk_color, "width": 3}, "thickness": 0.8, "value": s}
                    }
                ))
                fig_gauge.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e5e7eb'),
                    height=220,
                    margin=dict(l=30, r=30, t=50, b=10)
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

    else:
        st.info("📊 Upload and analyze resumes first to see fraud detection insights.")
        st.markdown("""
        <div style='text-align:center;padding:60px;color:#6b7280;'>
            <div style='font-size:64px;'>🛡️</div>
            <p style='font-size:18px;margin-top:16px;'>No data yet — go to Resume Ranking tab to upload and analyze resumes.</p>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: Email Automation
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("## 📧 Email Automation")

    col1, col2 = st.columns(2)
    with col1:
        candidate_name = st.text_input("Candidate Name", placeholder="John Doe")
        job_title = st.text_input("Job Title", placeholder="Senior Python Developer")
    with col2:
        email_type = st.selectbox("Email Type", ["interview_invite", "hiring_team_update"])
        recipient_email = st.text_input("Recipient Email", placeholder="john@example.com")

    details = st.text_area("Email Details", placeholder="Interview scheduled for April 25, 2025, 10 AM")

    if st.button("📤 Generate & Send Email", use_container_width=True):
        if all([candidate_name, job_title, details, recipient_email]):
            with st.spinner("🤖 Generating personalized email..."):
                try:
                    task = create_email_task(candidate_name, email_type, job_title, details, recipient_email)
                    crew = Crew(agents=[email_automation], tasks=[task], verbose=True)
                    email_content = crew.kickoff()
                    result = send_email_via_smtp(email_content, recipient_email)
                    st.success("✅ Email generated and sent!")
                    st.markdown("### 📧 Email Content")
                    st.code(email_content, language="text")
                    st.info(result)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.error("Please fill in all fields")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: Analytics
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("## 📊 Advanced Analytics")

    if st.session_state.mcp_context.get("ranked_resumes"):
        results = st.session_state.mcp_context["ranked_resumes"]

        # Score distribution
        scores = [r['final_score'] for r in results]
        fig_hist2 = go.Figure(data=[go.Histogram(x=scores, nbinsx=20, marker_color='#667eea')])
        fig_hist2.update_layout(
            template='plotly_dark',
            title="Overall Score Distribution",
            xaxis_title="Match Score",
            yaxis_title="Count",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e5e7eb')
        )
        st.plotly_chart(fig_hist2, use_container_width=True)

        # Skills frequency
        all_matched = []
        for r in results:
            all_matched.extend(r.get('skills_detail', {}).get('matched_skills', []))
        if all_matched:
            from collections import Counter
            skill_counts = Counter(all_matched).most_common(15)
            skill_names = [s[0] for s in skill_counts]
            skill_vals = [s[1] for s in skill_counts]

            fig_skills = go.Figure(go.Bar(
                x=skill_vals, y=skill_names,
                orientation='h',
                marker=dict(color=skill_vals, colorscale='Viridis', showscale=False),
                text=skill_vals, textposition='outside'
            ))
            fig_skills.update_layout(
                template='plotly_dark',
                title="Top Skills Found Across All Resumes",
                xaxis_title="Frequency",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e5e7eb'),
                height=450,
                margin=dict(l=120)
            )
            fig_skills.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_skills, use_container_width=True)
    else:
        st.info("📊 Analyze resumes first to see analytics.")
        st.markdown("""
        <div style='text-align:center;padding:60px;color:#6b7280;'>
            <div style='font-size:64px;'>📊</div>
            <p style='font-size:18px;margin-top:16px;'>Upload resumes in the Resume Ranking tab to populate analytics.</p>
        </div>
        """, unsafe_allow_html=True)
