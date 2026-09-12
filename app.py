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
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from huggingface_hub import snapshot_download

st.set_page_config(page_title="Decoy.ai", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

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

    /* Footer bar (replaces old sidebar dev-card) */
    .app-footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background: rgba(10, 10, 15, 0.85);
        backdrop-filter: blur(8px);
        text-align: center;
        padding: 8px 0;
        font-size: 12.5px;
        color: #8b92a8;
        border-top: 1px solid rgba(255,255,255,0.06);
        z-index: 999;
    }
    .app-footer a {
        color: #a78bfa;
        text-decoration: none;
        margin: 0 6px;
        font-weight: 500;
    }
    .app-footer a:hover {
        color: #ffffff;
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

# ==== Hugging Face Hub model download ====
HF_REPO_ID = "DhanushramS/Decoy_model"

@st.cache_resource
def get_model_dir():
    return snapshot_download(repo_id=HF_REPO_ID, repo_type="model")

MODEL_DIR = get_model_dir()

ARTIFACT_DIR = f"{MODEL_DIR}/url_pipeline"
XGB_PATH = f"{ARTIFACT_DIR}/xgboost_url_phishing.json"
TRANSFORMER_PATH = f"{ARTIFACT_DIR}/char_transformer_url.pth"
META_PATH = f"{ARTIFACT_DIR}/url_ensemble_meta.pkl"
CHAR_VOCAB_PATH = f"{ARTIFACT_DIR}/char_to_idx.json"
TRANCO_LOOKUP_PATH = f"{ARTIFACT_DIR}/tranco_lookup.csv"

EMAIL_MODEL_DIR = f"{MODEL_DIR}/email_phishing"
EMAIL_BERT_PATH = f"{EMAIL_MODEL_DIR}/bert"
EMAIL_XGB_PATH = f"{EMAIL_MODEL_DIR}/xgb_structural.json"
EMAIL_META_PATH = f"{EMAIL_MODEL_DIR}/meta_model.pkl"
EMAIL_ENGINEERED_COLS_PATH = f"{EMAIL_MODEL_DIR}/engineered_cols.pkl"
EMAIL_MAX_LEN = 256

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

# ---- v8 email pipeline: brand-domain matching, tactic lexicon, features ----
BRAND_REAL_DOMAINS = {
    'paypal': ['paypal.com'], 'google': ['google.com', 'accounts.google.com'],
    'amazon': ['amazon.com', 'amazon.in', 'amazon.co.uk'],
    'microsoft': ['microsoft.com', 'live.com', 'outlook.com', 'office.com'],
    'apple': ['apple.com', 'icloud.com'], 'facebook': ['facebook.com', 'fb.com'],
    'netflix': ['netflix.com'], 'bankofamerica': ['bankofamerica.com'],
    'wellsfargo': ['wellsfargo.com'], 'chase': ['chase.com'],
    'instagram': ['instagram.com'], 'linkedin': ['linkedin.com'],
    'ebay': ['ebay.com'], 'dropbox': ['dropbox.com'], 'adobe': ['adobe.com'],
}
TACTIC_LEXICON = {
    'urgency': ['act now', 'immediately', 'urgent', 'right away', 'within 24 hours',
                'within 48 hours', 'expires today', 'time sensitive', 'final notice',
                'last chance', 'today only'],
    'fear_threat': ['suspended', 'unauthorized', 'unusual activity', 'legal action',
                    'account locked', 'will be terminated', 'permanently deleted',
                    'security alert', 'compromised', 'suspicious login'],
    'reward_greed': ['congratulations', "you've been selected", 'claim your', 'winner',
                      'free gift', 'gift card', 'reward', 'bonus', 'exclusive offer'],
    'authority': ['irs', 'government', 'legal department', 'law enforcement',
                  'court order', 'compliance', 'audit'],
    'action_request': ['click here', 'verify your', 'confirm your', 'update your payment',
                        'log in now', 'reset your password', 'provide your',
                        'wire transfer', 'send gift card'],
}
EMAIL_ENGINEERED_COLS = [
    'num_urls_in_email', 'brand_mentioned', 'brand_domain_match', 'brand_mismatch',
    'brand_named_no_link', 'tactic_urgency', 'tactic_fear_threat', 'tactic_reward_greed',
    'tactic_authority', 'tactic_action_request', 'tactic_total', 'generic_greeting',
    'sentiment_compound', 'sentiment_negative', 'high_risk_combo',
]
EMAIL_URL_RE = re.compile(r'https?://[^\s<>"\')\]]+')
EMAIL_ADDR_RE = re.compile(r'\S+@\S+\.\S+')

def normalize_email_text(text):
    """Matches the normalization used during v8 BERT training — apply this
    BEFORE feeding to BERT, but AFTER extracting engineered features (which
    need the real URLs intact)."""
    text = EMAIL_URL_RE.sub(' <URL> ', text)
    text = EMAIL_ADDR_RE.sub(' <EMAIL> ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()

def extract_email_engineered_features(raw_email_text):
    """Run on RAW text (real URLs intact) — do NOT normalize first."""
    text_lower = raw_email_text.lower()
    urls = EMAIL_URL_RE.findall(raw_email_text)
    brands = [b for b in KNOWN_BRANDS if re.search(rf'(?<![a-z]){re.escape(b)}(?![a-z])', text_lower)]

    feats = {'num_urls_in_email': len(urls)}
    if not brands:
        feats.update({'brand_mentioned': 0, 'brand_domain_match': 0,
                       'brand_mismatch': 0, 'brand_named_no_link': 0})
    elif not urls:
        feats.update({'brand_mentioned': 1, 'brand_domain_match': 0,
                       'brand_mismatch': 0, 'brand_named_no_link': 1})
    else:
        linked_domains = {_tld_extractor(u).domain + '.' + _tld_extractor(u).suffix for u in urls}
        match = any(linked_domains & set(BRAND_REAL_DOMAINS.get(b, [])) for b in brands)
        feats.update({'brand_mentioned': 1, 'brand_domain_match': int(match),
                       'brand_mismatch': int(not match), 'brand_named_no_link': 0})

    for category, phrases in TACTIC_LEXICON.items():
        feats[f'tactic_{category}'] = sum(1 for p in phrases if p in text_lower)
    feats['tactic_total'] = sum(feats[f'tactic_{c}'] for c in TACTIC_LEXICON)

    first_line = text_lower.strip().split('\n')[0]
    feats['generic_greeting'] = int(bool(
        re.search(r'dear (customer|user|member|valued customer|candidate|applicant)', first_line)
    ))

    polarity = _vader.polarity_scores(raw_email_text[:1000])  # truncated, matches training
    feats['sentiment_compound'] = polarity['compound']
    feats['sentiment_negative'] = polarity['neg']

    feats['high_risk_combo'] = int(
        (feats['tactic_urgency'] > 0 or feats['tactic_fear_threat'] > 0) and
        (feats['brand_mismatch'] == 1 or feats['brand_named_no_link'] == 1 or
         (feats['generic_greeting'] == 1 and feats['tactic_action_request'] > 0))
    )
    return feats

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

    email_tokenizer = AutoTokenizer.from_pretrained(EMAIL_BERT_PATH)
    email_bert_model = BertForSequenceClassification.from_pretrained(EMAIL_BERT_PATH).to(device)
    email_bert_model.eval()

    email_xgb_model = XGBClassifier()
    email_xgb_model.load_model(EMAIL_XGB_PATH)

    email_meta_model = joblib.load(EMAIL_META_PATH)
    email_engineered_cols = joblib.load(EMAIL_ENGINEERED_COLS_PATH)

    vader_analyzer = SentimentIntensityAnalyzer()

    groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

    return {
        "device": device, "xgb_model": xgb_model, "char_to_idx": char_to_idx,
        "transformer_model": transformer_model, "meta_model": meta_model,
        "tranco_rank_lookup": tranco_rank_lookup,
        "email_tokenizer": email_tokenizer, "email_bert_model": email_bert_model,
        "email_xgb_model": email_xgb_model, "email_meta_model": email_meta_model,
        "email_engineered_cols": email_engineered_cols, "vader_analyzer": vader_analyzer,
        "groq_client": groq_client,
    }

ART = load_artifacts()
_vader = ART["vader_analyzer"]


# FEATURE ENGINEERING

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


# Inference
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

def predict_email(raw_email_text):
    # Step 1: engineered features from RAW text (real URLs, real brand names)
    feats = extract_email_engineered_features(raw_email_text)

    # Step 2: normalize THEN run BERT — matches v8 training distribution
    normalized_text = normalize_email_text(raw_email_text)
    inputs = ART["email_tokenizer"](normalized_text, truncation=True, padding=True,
                                     max_length=EMAIL_MAX_LEN, return_tensors='pt').to(ART["device"])
    with torch.no_grad():
        bert_prob = float(torch.softmax(ART["email_bert_model"](**inputs).logits, dim=1)[0, 1].item())

    # Step 3: XGBoost on engineered features
    feats_df = pd.DataFrame([feats])[ART["email_engineered_cols"]]
    xgb_prob = float(ART["email_xgb_model"].predict_proba(feats_df)[:, 1][0])

    # Step 4: meta-learner
    meta_input = np.array([[bert_prob, xgb_prob]])
    final_prob = float(ART["email_meta_model"].predict_proba(meta_input)[:, 1][0])

    # Step 5: deterministic override — mandatory brand-impersonation rule +
    # zero-red-flag ceiling. The meta-learner alone learns to ignore these
    # signals on in-distribution training data, so they're enforced as rules
    # here, same pattern as the URL pipeline's allowlist short-circuit.
    ml_prob_before_override = final_prob
    override_applied = False
    if feats['brand_mismatch'] == 1 and final_prob < 0.75:
        final_prob = 0.75
        override_applied = True
    elif feats['high_risk_combo'] == 1 and final_prob < 0.65:
        final_prob = 0.65
        override_applied = True
    elif (feats['brand_mismatch'] == 0 and feats['brand_named_no_link'] == 0
          and feats['tactic_total'] == 0 and feats['num_urls_in_email'] == 0
          and feats['generic_greeting'] == 0 and final_prob > 0.4):
        final_prob = 0.4
        override_applied = True

    verdict = "Phishing" if final_prob >= 0.5 else "Safe"
    return {
        "email_prob": final_prob, "bert_prob": bert_prob, "xgb_prob": xgb_prob,
        "verdict": verdict, "override_applied": override_applied,
        "ml_prob_before_override": ml_prob_before_override,
        "brand_mismatch": feats['brand_mismatch'], "high_risk_combo": feats['high_risk_combo'],
    }

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
Detection path: {"rule-based adjustment (brand impersonation or high-risk pattern detected)" if email_result.get('override_applied') else "full ML ensemble"}
- BERT model (semantic/contextual): {email_result['bert_prob']:.2f} probability of phishing
- XGBoost model (engineered features — brand impersonation, urgency tactics, sentiment): {email_result['xgb_prob']:.2f}
- Combined ensemble score: {email_result['email_prob']:.2f}
"""

    prompt = f"""You are a cybersecurity assistant explaining phishing detection results to a user.

{context}

Important: the URL and email were evaluated independently — no combined verdict between them.
If a URL matched the trusted domain allowlist, briefly mention it's a well-established, popular domain rather than describing model scores as the primary reason.

Explain in 2-3 sentences per input why it was flagged this way, referencing the actual scores accurately. Be concise and clear for a non-technical user."""

    response = ART["groq_client"].chat.completions.create(
        model="openai/gpt-oss-120b",
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
    st.markdown("## Decoy.ai")
    st.caption("Multi-layer phishing detection engine")
    st.markdown("---")
    st.markdown("**Detection Pipeline**")
    st.markdown('<div class="sidebar-stat">🔗 <b>URL</b> — XGBoost + Char-Transformer + Domain-Trust Allowlist</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-stat">📧 <b>Email</b> — BERT + XGBoost + Brand-Impersonation Rules</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-stat">🤖 <b>Explanation</b> — Groq (gpt-oss-120b)</div>', unsafe_allow_html=True)


st.markdown("""
<div class="hero">
    <div class="hero-title">Decoy.ai</div>
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
        with st.spinner("Running BERT + XGBoost ensemble..."):
            email_result = predict_email(email_input)

        render_verdict_banner(email_result["verdict"])

        if email_result.get("override_applied"):
            st.caption(
                f"⚠️ Rule-based adjustment applied — raw ML score was "
                f"{email_result['ml_prob_before_override']:.0%}, adjusted to "
                f"{email_result['email_prob']:.0%} due to "
                f"{'brand impersonation' if email_result['brand_mismatch'] else 'risk pattern'} detection."
            )

        c1, c2, c3 = st.columns(3)
        render_score_card(c1, "BERT", email_result["bert_prob"], "🧠")
        render_score_card(c2, "XGBoost", email_result["xgb_prob"], "⚡")
        render_score_card(c3, "Ensemble", email_result["email_prob"], "🎯")

        with st.spinner("🤖 AI analyzing threat patterns..."):
            explanation = explain_verdict(email_result=email_result)
        st.markdown(f"<div class='explain-box'>{explanation}</div>", unsafe_allow_html=True)

st.markdown("""
<div class="app-footer">
    Decoy.ai &nbsp;·&nbsp; Built by Dhanush &nbsp;·&nbsp;
    <a href="https://github.com/Dhanushram2612" target="_blank">GitHub</a> &nbsp;·&nbsp;
    <a href="https://www.linkedin.com/in/dhanushram-s-967b81309/" target="_blank">LinkedIn</a>
</div>
""", unsafe_allow_html=True)
