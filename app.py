
import json
import math
import re
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import joblib
from urllib.parse import urlparse
import tldextract
from xgboost import XGBClassifier
from transformers import AutoTokenizer, BertForSequenceClassification
from groq import Groq
from huggingface_hub import hf_hub_download, snapshot_download


st.set_page_config(page_title="Decoy", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; }

    .stApp {
        background: radial-gradient(circle at 15% 0%, #16213e 0%, #0d0d15 45%, #0a0a10 100%);
    }

    /* Hero header */
    .hero {
        text-align: center;
        padding: 28px 20px 8px 20px;
    }
    .hero-title {
        font-size: 44px;
        font-weight: 700;
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
        letter-spacing: -1px;
    }
    .hero-sub {
        color: #8b92a8;
        font-size: 15px;
        font-weight: 500;
    }
    .pipeline-pill {
        display: inline-block;
        background: rgba(96,165,250,0.08);
        border: 1px solid rgba(96,165,250,0.25);
        color: #93c5fd;
        padding: 4px 14px;
        border-radius: 999px;
        font-size: 12px;
        margin: 3px;
        font-weight: 500;
    }

    /* Verdict banners */
    .verdict-safe {
        background: linear-gradient(135deg, rgba(34,197,94,0.15), rgba(34,197,94,0.03));
        border: 1px solid rgba(34,197,94,0.4);
        box-shadow: 0 0 30px rgba(34,197,94,0.08);
        padding: 22px 26px;
        border-radius: 16px;
        color: #d1fae5;
        margin-bottom: 16px;
    }
    .verdict-phishing {
        background: linear-gradient(135deg, rgba(239,68,68,0.18), rgba(239,68,68,0.03));
        border: 1px solid rgba(239,68,68,0.45);
        box-shadow: 0 0 30px rgba(239,68,68,0.1);
        padding: 22px 26px;
        border-radius: 16px;
        color: #fee2e2;
        margin-bottom: 16px;
    }
    .verdict-title { font-size: 24px; font-weight: 700; margin: 0; font-family: 'Space Grotesk', sans-serif; }
    .verdict-sub { font-size: 13px; opacity: 0.75; margin-top: 4px; }

    /* Score cards */
    .score-card {
        background: rgba(255,255,255,0.03);
        backdrop-filter: blur(10px);
        padding: 20px 16px;
        border-radius: 14px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.08);
        transition: all 0.2s ease;
    }
    .score-card:hover {
        border-color: rgba(96,165,250,0.4);
        transform: translateY(-2px);
    }
    .score-value { font-size: 30px !important; font-weight: 700; margin: 4px 0 0 0; font-family: 'Space Grotesk', sans-serif; }
    .score-label { color: #8b92a8; font-size: 12px; margin: 0; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; }
    .score-icon { font-size: 20px; }

    /* Explanation box */
    .explain-box {
        background: rgba(96,165,250,0.06);
        border-left: 3px solid #60a5fa;
        border-radius: 0 12px 12px 0;
        padding: 18px 20px;
        color: #cbd5e1;
        font-size: 14.5px;
        line-height: 1.65;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #12121f, #0a0a10);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    .sidebar-stat {
        background: rgba(255,255,255,0.03);
        border-radius: 10px;
        padding: 10px 12px;
        margin-bottom: 8px;
        font-size: 13px;
        color: #94a3b8;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .sidebar-stat b { color: #e2e8f0; }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 10px 22px;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        box-shadow: 0 4px 20px rgba(139,92,246,0.4);
        transform: translateY(-1px);
    }

    /* Input fields */
    .stTextInput>div>div>input, .stTextArea textarea {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.03);
        border-radius: 10px 10px 0 0;
        padding: 10px 18px;
        font-weight: 600;
    }

    footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def download_artifacts():
    xgb_path = hf_hub_download(repo_id="Dhanushram26/xgboost_url_phishing.json", filename="xgboost_url_phishing.json")
    transformer_path = hf_hub_download(repo_id="Dhanushram26/char_transformer_url.pth", filename="char_transformer_url.pth")
    meta_path = hf_hub_download(repo_id="Dhanushram26/url_ensemble_meta.pkl", filename="url_ensemble_meta.pkl")
    vocab_path = hf_hub_download(repo_id="Dhanushram26/char_to_idx.json", filename="char_to_idx.json")
    tranco_path = hf_hub_download(repo_id="Dhanushram26/tranco_lookup.csv", filename="tranco_lookup.csv")
    email_dir = snapshot_download(repo_id="Dhanushram26/decoy-models", allow_patterns="email_phishing_v6/*")
    email_dir = f"{email_dir}/email_phishing_v6"
    return xgb_path, transformer_path, meta_path, vocab_path, tranco_path, email_dir

XGB_PATH, TRANSFORMER_PATH, META_PATH, CHAR_VOCAB_PATH, TRANCO_LOOKUP_PATH, EMAIL_MODEL_PATH = download_artifacts()

MAX_LEN = 160
TOP_N_TRUSTED = 100000
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")

FEATURE_COLS = [
    'Url_length', 'domain_length', 'num_dots', 'num_hyphen', 'num_underscores',
    'num_digits', 'has_ip', 'num_subdomains', 'has_at_symbols',
    'num_special_characters', 'entrophy', 'path_length', 'num_params',
    'is_suspicious_tld', 'brand_impersonation', 'is_known_domain', 'domain_trust_score'
]

SUSPICIOUS_TLDS = {'tk', 'ml', 'ga', 'cf', 'gq', 'xyz', 'top', 'work', 'click', 'link', 'live', 'buzz'}
KNOWN_BRANDS = ['paypal', 'google', 'amazon', 'microsoft', 'apple', 'facebook',
                'netflix', 'bankofamerica', 'wellsfargo', 'chase', 'instagram',
                'linkedin', 'ebay', 'dropbox', 'adobe']

# writable cache dir for tldextract's public suffix list (HF filesystem can be read-only elsewhere)
_tld_extractor = tldextract.TLDExtract(cache_dir="/tmp/tldextract_cache")

class CharTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_heads=4, num_layers=2, max_len=MAX_LEN, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_embedding(positions)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)

@st.cache_resource
def load_artifacts():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    xgb_model = XGBClassifier()
    xgb_model.load_model(XGB_PATH)

    with open(CHAR_VOCAB_PATH, "r") as f:
        char_to_idx = json.load(f)
    vocab_size = len(char_to_idx)

    transformer_model = CharTransformer(vocab_size=vocab_size).to(device)
    transformer_model.load_state_dict(torch.load(TRANSFORMER_PATH, map_location=device))
    transformer_model.eval()

    meta_model = joblib.load(META_PATH)

    tranco_df = pd.read_csv(TRANCO_LOOKUP_PATH)
    tranco_rank_lookup = dict(zip(tranco_df['domain'], tranco_df['rank']))

    email_tokenizer = AutoTokenizer.from_pretrained(EMAIL_MODEL_PATH)
    email_model = BertForSequenceClassification.from_pretrained(EMAIL_MODEL_PATH).to(device)
    email_model.eval()

    groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

    return {
        "device": device, "xgb_model": xgb_model, "char_to_idx": char_to_idx,
        "transformer_model": transformer_model, "meta_model": meta_model,
        "tranco_rank_lookup": tranco_rank_lookup,
        "email_tokenizer": email_tokenizer, "email_model": email_model,
        "groq_client": groq_client,
    }

ART = load_artifacts()

def shannon_entrophy(s):
    if not s:
        return 0
    prob = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in prob)

def get_domain_trust(url):
    ext = _tld_extractor(url)
    registered_domain = f"{ext.domain}.{ext.suffix}"
    rank = ART["tranco_rank_lookup"].get(registered_domain, None)
    if rank is None:
        return 0, 0
    return 1, max(0, 1 - (rank / TOP_N_TRUSTED))

def extract_features(url):
    parsed = urlparse(url)
    domain = parsed.netloc
    ext = _tld_extractor(url)
    domain_clean = ext.domain.lower()

    contains_brand = 1 if any(b in domain_clean for b in KNOWN_BRANDS) else 0
    is_actual_brand_domain = 1 if domain_clean in KNOWN_BRANDS else 0
    brand_impersonation = 1 if (contains_brand and not is_actual_brand_domain) else 0
    is_known_domain, domain_trust_score = get_domain_trust(url)

    return {
        'Url_length': len(url), 'domain_length': len(domain),
        'num_dots': url.count('.'), 'num_hyphen': url.count('-'),
        'num_underscores': url.count('_'), 'num_digits': sum(c.isdigit() for c in url),
        'has_ip': 1 if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', domain) else 0,
        'num_subdomains': domain.count('.') - 1 if domain.count('.') > 0 else 0,
        'has_at_symbols': 1 if '@' in url else 0,
        'num_special_characters': len(re.findall(r'[^\w\s]', url)),
        'entrophy': shannon_entrophy(url), 'path_length': len(parsed.path),
        'num_params': url.count('=') + url.count('&'),
        'is_suspicious_tld': 1 if ext.suffix in SUSPICIOUS_TLDS else 0,
        'brand_impersonation': brand_impersonation,
        'is_known_domain': is_known_domain,
        'domain_trust_score': domain_trust_score,
    }

def encode_url(url, char_to_idx, max_len=MAX_LEN):
    url = url.lower()
    unk = char_to_idx.get('<UNK>', 0)
    ids = [char_to_idx.get(c, unk) for c in url[:max_len]]
    ids += [char_to_idx.get('<PAD>', 0)] * (max_len - len(ids))
    return ids

def predict_url(url):
    feats = extract_features(url)
    is_known, trust_score = get_domain_trust(url)
    is_impersonation_signal = feats['brand_impersonation'] or feats['is_suspicious_tld']

    # Allowlist short-circuit: well-established domain, no impersonation/suspicious-TLD red flags
    if is_known and trust_score > 0.5 and not is_impersonation_signal:
        return {"url": url, "xgb_prob": 0.02, "transformer_prob": 0.02,
                "final_prob": 0.02, "verdict": "Legitimate", "matched": "allowlist"}

    features_df = pd.DataFrame([feats])[FEATURE_COLS]
    xgb_prob = float(ART["xgb_model"].predict_proba(features_df)[:, 1][0])

    encoded = encode_url(url, ART["char_to_idx"])
    x_tensor = torch.tensor([encoded], dtype=torch.long).to(ART["device"])
    with torch.no_grad():
        output = ART["transformer_model"](x_tensor)
        transformer_prob = float(torch.softmax(output, dim=1)[:, 1].item())

    meta_input = np.array([[xgb_prob, transformer_prob]])
    final_prob = float(ART["meta_model"].predict_proba(meta_input)[:, 1][0])
    verdict = "Phishing" if final_prob > 0.5 else "Legitimate"

    return {"url": url, "xgb_prob": xgb_prob, "transformer_prob": transformer_prob,
            "final_prob": final_prob, "verdict": verdict, "matched": "model"}

def detect_bec_signals(email_text):
    """Rule-assist layer for BEC/subtle phishing that BERT alone under-detects
    (secrecy requests, sender-unreachable pretext, financial asks, account-scare language)."""
    text = email_text.lower()
    signals = {}

    signals['secrecy_request'] = bool(re.search(
        r"keep (this|it) (confidential|between us|quiet|private)|don'?t (tell|loop in|mention|discuss)|"
        r"confidential(ly)? for now|can'?t discuss.*over email", text))

    signals['claims_unreachable'] = bool(re.search(
        r"can'?t talk|in a meeting|unreachable|mid-?flight|back.?to.?back (meetings|interviews|calls)|"
        r"tied up|won'?t be reachable|stuck in|no signal|spotty (signal|reception)|heading into", text))

    signals['financial_ask'] = bool(re.search(
        r"wire transfer|wire \$|gift cards?|banking details|send (money|funds|payment)|"
        r"account (number|routing)|process(ing)? the payment|update.*payment method|"
        r"push through|authorize the (payment|transfer)|action(ing)? a payment|handle the payment|"
        r"get to my banking|process(ing)? .*payment|get.*over to the account|send.*to the account|"
        r"\$[\d,]+", text))

    has_urgency = bool(re.search(r"urgent(ly)?|time sensitive|immediately|asap|today|before (end of day|close of business)|within \d+ hours?", text))
    has_specifics = bool(re.search(r"#\d+|order\s*#|invoice\s*#|ticket\s*#|reference\s*#|ending in \d{4}", text))
    signals['urgency_no_specifics'] = has_urgency and not has_specifics

    signals['account_scare_language'] = bool(re.search(
        r"account.*(suspended|under review|locked|policy violation)|storage.*(almost full|will be deleted)|"
        r"two-factor authentication.*disabled|fraud alert|unusual activity|unauthorized (charge|access)", text))

    bec_triad = sum([signals['secrecy_request'], signals['claims_unreachable'], signals['financial_ask']])

    if bec_triad >= 2:
        rule_score = 0.92
    elif signals['account_scare_language']:
        rule_score = 0.85
    elif signals['secrecy_request'] or signals['claims_unreachable']:
        rule_score = 0.55
    elif signals['urgency_no_specifics']:
        rule_score = 0.35
    else:
        rule_score = 0.0

    signals['rule_score'] = rule_score
    return signals


def predict_email(email_text):
    inputs = ART["email_tokenizer"](email_text, truncation=True, padding=True,
                                     max_length=256, return_tensors='pt').to(ART["device"])
    with torch.no_grad():
        outputs = ART["email_model"](**inputs)
        bert_prob = float(torch.softmax(outputs.logits, dim=1)[:, 1].item())

    signals = detect_bec_signals(email_text)
    rule_score = signals['rule_score']

    # weighted blend when both signals moderately agree — boosts instead of capping at max,
    # since two moderate signals together are stronger evidence than either alone
    if bert_prob > 0.25 and rule_score > 0.25:
        combined_prob = bert_prob + rule_score * (1 - bert_prob)
    else:
        combined_prob = max(bert_prob, rule_score)

    verdict = "Phishing" if combined_prob > 0.5 else "Safe"
    return {"email_prob": combined_prob, "bert_prob": bert_prob, "rule_score": rule_score, "verdict": verdict}

def explain_verdict(url_result=None, email_result=None):
    if not ART["groq_client"]:
        return "_(Set GROQ_API_KEY in secrets to enable AI explanations.)_"

    context = ""
    if url_result:
        context += f"""URL analyzed: {url_result['url']}
Final verdict: {url_result['verdict']}
Detection path: {"trusted domain allowlist" if url_result.get('matched')=='allowlist' else "full ML model pipeline"}
Note: for scores below, higher = more likely phishing.
- XGBoost model (structural/lexical features): {url_result['xgb_prob']:.2f}
- Character-Transformer model (raw URL text patterns): {url_result['transformer_prob']:.2f}
- Combined ensemble score: {url_result['final_prob']:.2f}
"""
    if email_result:
        context += f"""Email content analyzed.
Final verdict: {email_result['verdict']}
- BERT model: {email_result['email_prob']:.2f} probability of phishing
"""

    prompt = f"""You are a cybersecurity assistant explaining phishing detection results to a user.

{context}

Important: the URL and email were evaluated independently — no combined verdict between them.
If a URL matched the trusted domain allowlist, briefly mention it's a well-established, popular domain rather than describing model scores as the primary reason.

Explain in 2-3 sentences per input why it was flagged this way, referencing the actual scores accurately. Be concise and clear for a non-technical user."""

    response = ART["groq_client"].chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content

def render_verdict_banner(verdict, matched=None):
    is_safe = verdict in ("Legitimate", "Safe")
    css_class = "verdict-safe" if is_safe else "verdict-phishing"
    icon = "✅" if is_safe else "🚨"
    sub = ""
    if matched == "allowlist":
        sub = '<div class="verdict-sub">Matched trusted domain allowlist — high-confidence known-safe site</div>'
    st.markdown(f"<div class='{css_class}'><p class='verdict-title'>{icon} {verdict}</p>{sub}</div>", unsafe_allow_html=True)

def render_score_card(col, label, value, icon):
    with col:
        st.markdown(
            f"<div class='score-card'><span class='score-icon'>{icon}</span>"
            f"<p class='score-label'>{label}</p>"
            f"<p class='score-value'>{value:.0%}</p></div>",
            unsafe_allow_html=True
        )

with st.sidebar:
    st.markdown("## 🛡️ Decoy")
    st.caption("Multi-layer phishing detection engine")
    st.markdown("---")
    st.markdown("**Detection Pipeline**")
    st.markdown('<div class="sidebar-stat">🔗 <b>URL</b> — XGBoost + Char-Transformer + Domain-Trust Allowlist</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-stat">📧 <b>Email</b> — Fine-tuned BERT classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-stat">🤖 <b>Explanation</b> — Groq (Llama 3.3 70B)</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Built by Dhanush")

    
st.markdown("""
<div class="hero">
    <div class="hero-title">🛡️ Decoy</div>
    <div class="hero-sub">Multi-layer phishing detection — URL structure, character patterns, domain trust & AI reasoning</div>
    <div style="margin-top: 14px;">
        <span class="pipeline-pill">⚡ XGBoost</span>
        <span class="pipeline-pill">🔤 Char-Transformer</span>
        <span class="pipeline-pill">🌐 Domain Trust</span>
        <span class="pipeline-pill">🧠 BERT</span>
        <span class="pipeline-pill">💬 LLM Reasoning</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.write("")
tab1, tab2 = st.tabs(["🔗  Check URL", "📧  Check Email"])

with tab1:
    url_input = st.text_input("Paste a URL to check:", placeholder="https://example.com/login")
    if st.button("Analyze URL", type="primary") and url_input:
        with st.spinner("Running detection models..."):
            url_result = predict_url(url_input)

        render_verdict_banner(url_result["verdict"], url_result.get("matched"))

        if url_result.get("matched") != "allowlist":
            c1, c2, c3 = st.columns(3)
            render_score_card(c1, "XGBoost", url_result["xgb_prob"], "⚡")
            render_score_card(c2, "Char-Transformer", url_result["transformer_prob"], "🔤")
            render_score_card(c3, "Ensemble", url_result["final_prob"], "🎯")
            st.progress(url_result["final_prob"])

        with st.spinner("🤖 AI analyzing threat patterns..."):
            explanation = explain_verdict(url_result=url_result)
        st.markdown(f"<div class='explain-box'>{explanation}</div>", unsafe_allow_html=True)

with tab2:
    email_input = st.text_area("Paste email content to check:", height=200,
                                placeholder="Paste the email body here...")
    if st.button("Analyze Email", type="primary") and email_input:
        with st.spinner("Running BERT classifier..."):
            email_result = predict_email(email_input)

        render_verdict_banner(email_result["verdict"])

        c1, _, _ = st.columns(3)
        render_score_card(c1, "BERT Confidence", email_result["email_prob"], "🧠")

        with st.spinner("🤖 AI analyzing threat patterns..."):
            explanation = explain_verdict(email_result=email_result)
        st.markdown(f"<div class='explain-box'>{explanation}</div>", unsafe_allow_html=True)
