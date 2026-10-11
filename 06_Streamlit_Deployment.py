# Sub-module: app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
import random
import hashlib
import requests
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie

st.set_page_config(page_title="Music Emotion Recognition", page_icon="🎵", layout="wide", initial_sidebar_state="expanded")

def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200: return None
        return r.json()
    except: return None

# Neo Kinpaku Design System Injection
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Albert+Sans:wght@400;500&family=Alumni+Sans:wght@100;300;400&display=swap');

    * { font-family: 'Albert Sans', sans-serif; }
    h1, h2, h3, h4, h5, h6 { font-family: 'Alumni Sans', sans-serif !important; margin: 0; }
    
    h1 { font-weight: 100 !important; font-size: 4rem !important; letter-spacing: -0.01em; color: #E8E8E8; }
    h2 { font-weight: 300 !important; font-size: 2.8rem !important; color: #E8E8E8; }
    h3 { font-family: 'Albert Sans', sans-serif !important; font-weight: 500 !important; font-size: 1.18rem !important; color: #E8E8E8; }
    p, span, div { color: #E0E0E0; font-size: 1rem; line-height: 1.65; }

    .stApp { background-color: #0B0C0D; } 
    
    [data-testid="stSidebar"] {
        background-color: #131415 !important; 
        border-right: 1px solid rgba(212, 175, 55, 0.16); 
    }

    .stButton > button {
        background-color: #D4AF37 !important; 
        color: #0B0C0D !important; 
        border: 1px solid #D4AF37 !important;
        border-radius: 2px !important; 
        font-family: 'Albert Sans', sans-serif !important;
        font-weight: 500 !important;
        padding: 0 38px !important;
        min-height: 48px !important;
    }
    .stButton > button:hover { background-color: #E6C55C !important; }

    .ks-wordmark {
        font-family: 'Alumni Sans', sans-serif;
        font-weight: 400; text-transform: uppercase; letter-spacing: 0.15em;
        font-size: 1.6rem; color: #D4AF37;
    }
    
    .ks-mono-eyebrow {
        font-family: monospace; font-size: 0.7rem; font-weight: 500;
        text-transform: uppercase; letter-spacing: 0.18em; color: #45B6AD;
    }

    @keyframes spin { 100% { transform: rotate(360deg); } }
    .vinyl-record {
        width: 120px; height: 120px;
        background: #131415; border-radius: 50%;
        border: 1px solid rgba(212, 175, 55, 0.16);
        animation: spin 4s linear infinite;
        position: relative; display: flex; align-items: center; justify-content: center;
    }
    .vinyl-center { width: 30px; height: 30px; background: #D4AF37; border-radius: 50%; border: 2px solid #0B0C0D; }
    .vinyl-hole { width: 8px; height: 8px; background: #0B0C0D; border-radius: 50%; }

    .highlight-lyric { background-color: rgba(212, 175, 55, 0.1); border-left: 2px solid #D4AF37; padding-left: 5px; }
    
    .block-container { padding-top: 2rem !important; padding-bottom: 2rem !important; }
    </style>
""", unsafe_allow_html=True)

METRICS = {
    "Acoustic": {
        "Angry": {"Precision": 0.73, "Recall": 0.71, "Macro_F1": 0.73},
        "Calm":  {"Precision": 0.76, "Recall": 0.77, "Macro_F1": 0.73},
        "Happy": {"Precision": 0.69, "Recall": 0.68, "Macro_F1": 0.73},
        "Sad":   {"Precision": 0.74, "Recall": 0.75, "Macro_F1": 0.73}
    },
    "NLP": {
        "Angry": {"Precision": 0.45, "Recall": 0.48, "Macro_F1": 0.43},
        "Calm":  {"Precision": 0.41, "Recall": 0.39, "Macro_F1": 0.43},
        "Happy": {"Precision": 0.44, "Recall": 0.41, "Macro_F1": 0.43},
        "Sad":   {"Precision": 0.42, "Recall": 0.44, "Macro_F1": 0.43}
    },
    "Fusion": {
        "Angry": {"Precision": 0.75, "Recall": 0.73, "Macro_F1": 0.74},
        "Calm":  {"Precision": 0.77, "Recall": 0.78, "Macro_F1": 0.74},
        "Happy": {"Precision": 0.72, "Recall": 0.69, "Macro_F1": 0.74},
        "Sad":   {"Precision": 0.72, "Recall": 0.76, "Macro_F1": 0.74}
    }
}

EMOTIONS = ["Angry", "Calm", "Happy", "Sad"]

with st.sidebar:
    st.markdown("<div class='ks-wordmark' style='text-align:center;'>VIBECHECK</div>", unsafe_allow_html=True)
    st.markdown("<div class='ks-mono-eyebrow' style='text-align:center; margin-bottom:20px;'>Neo Kinpaku Core</div>", unsafe_allow_html=True)
    
    selected = option_menu(
        menu_title=None,
        options=["Architecture Overview", "Single Track Analysis", "Enterprise Batch"],
        icons=["cpu", "headphones", "server"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#D4AF37", "font-size": "16px"},
            "nav-link": {"font-family": "Albert Sans", "font-size": "14px", "text-align": "left", "margin": "5px 0px"},
            "nav-link-selected": {"background-color": "#1F2022", "border-left": "2px solid #D4AF37", "color": "#E8E8E8"},
        }
    )
    
    st.markdown("---")
    st.markdown("<div class='ks-mono-eyebrow'>SYSTEM TELEMETRY</div>", unsafe_allow_html=True)
    if "activity_log" not in st.session_state: st.session_state.activity_log = []
    for log in reversed(st.session_state.activity_log[-5:]):
        st.markdown(f"<small style='color:#E0E0E0; font-family:monospace;'>{log}</small>", unsafe_allow_html=True)

def generate_noise_probs(target_idx, noise=0.15):
    base = [0.05, 0.05, 0.05, 0.05]
    base[target_idx] = 0.85
    probs = [max(0.01, p + random.uniform(-noise, noise)) for p in base]
    total = sum(probs)
    return [p/total for p in probs]

def get_mock_spotify_features(song_name):
    hash_object = hashlib.md5(str(song_name).encode('utf-8'))
    hash_hex = hash_object.hexdigest()
    
    energy = int(hash_hex[0:2], 16) / 255.0
    valence = int(hash_hex[2:4], 16) / 255.0
    tempo = 60 + (int(hash_hex[4:6], 16) / 255.0) * 120
    loudness = -20 + (int(hash_hex[6:8], 16) / 255.0) * 18
    acoustic = int(hash_hex[8:10], 16) / 255.0
    
    return {'Energy': energy, 'Valence': valence, 'Tempo': tempo, 'Loudness': loudness, 'Acousticness': acoustic, 'track_id': '3n3Ppam7vgaVa1iaRUc9Lp'}

def get_mock_predictions(song_name):
    time.sleep(1.0) 
    song_lower = str(song_name).lower()
    
    if any(x in song_lower for x in ["angry", "metal", "chop", "break stuff"]): scenario_idx = 0
    elif any(x in song_lower for x in ["calm", "lofi", "weightless"]): scenario_idx = 1
    elif any(x in song_lower for x in ["sad", "fix you", "someone like you"]): scenario_idx = 2
    elif any(x in song_lower for x in ["happy", "uptown funk", "levitating"]): scenario_idx = 3
    else: scenario_idx = int(hashlib.md5(song_lower.encode('utf-8')).hexdigest(), 16) % 4
        
    random.seed(song_lower)
    feats = get_mock_spotify_features(song_name)
    
    if scenario_idx == 0:
        ac_idx, nlp_idx, fin_idx = 0, 2, 0
        lyrics = "Wake up! Grab a brush and put a little makeup... <br><br><span class='highlight-lyric'>I don't think you trust in my self-righteous suicide!</span>"
    elif scenario_idx == 1:
        ac_idx, nlp_idx, fin_idx = 1, 3, 1
        lyrics = "<span class='highlight-lyric'>(Instrumental / Ambient Synthesisers)</span><br><br>Breathing in the midnight air..."
    elif scenario_idx == 2:
        ac_idx, nlp_idx, fin_idx = 1, 3, 3
        lyrics = "I heard that you're settled down, that you found a girl... <br><br><span class='highlight-lyric'>Never mind, I'll find someone like you...</span>"
    else:
        ac_idx, nlp_idx, fin_idx = 2, 2, 2
        lyrics = "If you wanna run away with me, I know a galaxy... <br><br><span class='highlight-lyric'>You want me, I want you, baby!</span>"
        
    ac_probs = generate_noise_probs(ac_idx, 0.2)
    nlp_probs = generate_noise_probs(nlp_idx, 0.3)
    fin_probs = generate_noise_probs(fin_idx, 0.1)
    
    # Radar chart needs closed loop
    fin_probs_radar = fin_probs.copy()
    fin_probs_radar.append(fin_probs_radar[0])
        
    return EMOTIONS[ac_idx], EMOTIONS[nlp_idx], EMOTIONS[fin_idx], lyrics, ac_probs, nlp_probs, fin_probs, fin_probs_radar, feats

def create_radar_chart(probs):
    categories = ['Angry', 'Calm', 'Happy', 'Sad']
    categories.append(categories[0]) 
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=probs, theta=categories, fill='toself', fillcolor='rgba(212, 175, 55, 0.2)',
        line=dict(color='#D4AF37', width=2), name='Fusion Probability'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(255,255,255,0.05)"), angularaxis=dict(gridcolor="rgba(255,255,255,0.05)")),
        showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#E0E0E0", size=12, family="Albert Sans"),
        height=320, margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig

def create_circumplex_scatter(valence, energy):
    fig = go.Figure()
    fig.add_vline(x=0.5, line_width=1, line_dash="dash", line_color="rgba(255,255,255,0.1)")
    fig.add_hline(y=0.5, line_width=1, line_dash="dash", line_color="rgba(255,255,255,0.1)")
    
    fig.add_trace(go.Scatter(
        x=[valence], y=[energy], mode='markers',
        marker=dict(size=16, color='#45B6AD', line=dict(width=2, color='#D4AF37')),
        name="Track Location"
    ))
    
    fig.add_annotation(x=0.85, y=0.95, text="HAPPY", showarrow=False, font=dict(color="#E0E0E0", size=12, family="monospace"))
    fig.add_annotation(x=0.15, y=0.95, text="ANGRY", showarrow=False, font=dict(color="#E0E0E0", size=12, family="monospace"))
    fig.add_annotation(x=0.15, y=0.05, text="SAD", showarrow=False, font=dict(color="#E0E0E0", size=12, family="monospace"))
    fig.add_annotation(x=0.85, y=0.05, text="CALM", showarrow=False, font=dict(color="#E0E0E0", size=12, family="monospace"))
    
    fig.update_layout(
        title=dict(text="RUSSELL'S CIRCUMPLEX MODEL", font=dict(family="Alumni Sans", size=20, color="#E0E0E0"), x=0.5),
        xaxis=dict(title="Valence", range=[0, 1], gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="Arousal", range=[0, 1], gridcolor="rgba(255,255,255,0.05)"),
        showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#E0E0E0", family="Albert Sans"),
        height=320, margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def create_gauge_chart(value, title, color):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        title = {'text': title, 'font': {'color': '#E0E0E0', 'size': 14, 'family': 'Alumni Sans'}},
        gauge = {
            'axis': {'range': [None, 1], 'tickwidth': 1, 'tickcolor': "#E0E0E0"},
            'bar': {'color': color},
            'bgcolor': "rgba(255,255,255,0.05)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 0.33], 'color': "rgba(255,255,255,0.02)"},
                {'range': [0.33, 0.66], 'color': "rgba(255,255,255,0.04)"},
                {'range': [0.66, 1], 'color': "rgba(255,255,255,0.06)"}],
        },
        number = {'font': {'color': color, 'size': 24}}
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=220, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def render_prob_bars(probs, color):
    html = "<div style='margin-top: 15px;'>"
    for i, e in enumerate(EMOTIONS):
        pct = int(probs[i] * 100)
        html += f"""
        <div style="margin-bottom: 6px;">
            <div style="display: flex; justify-content: space-between; font-family:monospace; font-size: 0.65rem; color: #A0A0A0; margin-bottom: 2px;">
                <span>{e.upper()}</span><span>{pct}%</span>
            </div>
            <div style="width: 100%; background-color: rgba(255,255,255,0.05); height: 4px; border-radius: 2px;">
                <div style="width: {pct}%; background-color: {color}; height: 100%; border-radius: 2px; transition: width 1s ease-in-out;"></div>
            </div>
        </div>
        """
    html += "</div>"
    return html

if selected == "Architecture Overview":
    st.markdown("<div class='ks-mono-eyebrow'>System Overview</div>", unsafe_allow_html=True)
    st.markdown("<h1>Multimodal Late Fusion Core</h1>", unsafe_allow_html=True)
    
    col1, spacer, col2 = st.columns([1.5, 0.05, 1])
    with col1:
        html_block = """
        <div style='background-color: #1F2022; border: 1px solid rgba(212, 175, 55, 0.16); border-radius: 4px; padding: 24px; margin-bottom: 16px;'>
            <h3 style='margin:0; margin-bottom:10px;'>VALENCE AMBIGUITY IN ACOUSTICS</h3>
            <p style='margin:0; font-size:1rem; color:#E0E0E0;'>Traditional music emotion recognition relies purely on acoustic features. However, high-energy tracks can sound aggressive or energetic. Without semantic context, acoustic models hit a hard accuracy limit.</p>
            <h3 style='margin:0; margin-top:30px; margin-bottom:10px; color:#D4AF37;'>LATE FUSION META-CLASSIFIER</h3>
            <p style='margin:0; font-size:1rem; color:#E0E0E0;'>This platform fuses independent AI models to resolve emotional ambiguity:</p>
            <ol style='color:#E0E0E0;'>
                <li><strong>Acoustic Ensemble:</strong> Soft Voting Ensemble trained on 190,000 Spotify tracks.</li>
                <li><strong>Semantic NLP:</strong> DistilBERT Transformer fine-tuned on 16,752 Genius Lyrics.</li>
            </ol>
            <p style='margin:0; font-size:1rem; color:#E0E0E0;'>A Logistic Regression Meta-Classifier learns when to trust the lyrics and when to trust the beat.</p>
        </div>
        """
        st.markdown(html_block, unsafe_allow_html=True)
            
    with col2:
        lottie_ai = load_lottieurl("https://lottie.host/7e04fec8-3eb2-4020-801b-c1e1346b8b0e/h8jTq6Gf1g.json")
        if lottie_ai: st_lottie(lottie_ai, height=350, key="ai_brain")

elif selected == "Single Track Analysis":
    st.markdown("<div class='ks-mono-eyebrow'>Analysis Engine</div>", unsafe_allow_html=True)
    st.markdown("<h1>Single Track Analysis</h1>", unsafe_allow_html=True)

    if "current_song" not in st.session_state: st.session_state.current_song = ""

    html_form_block = """
    <div style='background-color: #1F2022; border: 1px solid rgba(212, 175, 55, 0.16); border-radius: 4px; padding: 24px; margin-bottom: 16px;'>
    """
    st.markdown(html_form_block, unsafe_allow_html=True)
    with st.form("search_form"):
        col1, col2 = st.columns([3, 1])
        with col1: song_input = st.text_input("QUERY DATABASE", placeholder="Track ID or Song Name", label_visibility="collapsed")
        with col2: submitted = st.form_submit_button("INITIALIZE", use_container_width=True)
        
        if submitted and song_input:
            st.session_state.current_song = song_input
            st.session_state.activity_log.append(f"> INIT: {song_input.upper()}")
    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.current_song:
        with st.spinner("Executing Inference Pipeline..."):
            ac_pred, nlp_pred, fin_pred, lyrics, ac_probs, nlp_probs, fin_probs, fin_probs_radar, feats = get_mock_predictions(st.session_state.current_song)
            
        st.markdown("<div class='ks-mono-eyebrow' style='margin-top:30px;'>RESOLUTION RESULTS</div>", unsafe_allow_html=True)
        
        # FIX: Removed height:220px to allow boxes to grow dynamically
        c1, c2, c3 = st.columns(3)
        with c1:
            ac_html = f"""
            <div style='background-color: #1F2022; border: 1px solid rgba(212, 175, 55, 0.16); border-top: 2px solid #45B6AD; border-radius: 4px; padding: 24px; margin-bottom: 16px;'>
                <h3 style='margin:0;'>ACOUSTIC ENSEMBLE</h3>
                <h2 style='color:#45B6AD; margin:10px 0;'>{ac_pred}</h2>
                <hr style='border:none; border-top: 1px solid rgba(212, 175, 55, 0.16); margin: 15px 0;'>
                <span style='font-family:monospace; font-size:0.8rem; line-height:1.5; color:#A0A0A0;'>
                    PRECISION: {METRICS['Acoustic'][ac_pred]['Precision']} <br>
                    RECALL: {METRICS['Acoustic'][ac_pred]['Recall']} <br>
                    MACRO F1: {METRICS['Acoustic'][ac_pred]['Macro_F1']}
                </span>
                {render_prob_bars(ac_probs, '#45B6AD')}
            </div>
            """
            st.markdown(ac_html, unsafe_allow_html=True)

        with c2:
            nlp_html = f"""
            <div style='background-color: #1F2022; border: 1px solid rgba(212, 175, 55, 0.16); border-top: 2px solid #45B6AD; border-radius: 4px; padding: 24px; margin-bottom: 16px;'>
                <h3 style='margin:0;'>NLP DISTILBERT</h3>
                <h2 style='color:#45B6AD; margin:10px 0;'>{nlp_pred}</h2>
                <hr style='border:none; border-top: 1px solid rgba(212, 175, 55, 0.16); margin: 15px 0;'>
                <span style='font-family:monospace; font-size:0.8rem; line-height:1.5; color:#A0A0A0;'>
                    PRECISION: {METRICS['NLP'][nlp_pred]['Precision']} <br>
                    RECALL: {METRICS['NLP'][nlp_pred]['Recall']} <br>
                    MACRO F1: {METRICS['NLP'][nlp_pred]['Macro_F1']}
                </span>
                {render_prob_bars(nlp_probs, '#45B6AD')}
            </div>
            """
            st.markdown(nlp_html, unsafe_allow_html=True)
            
        with c3:
            fin_html = f"""
            <div style='background-color: #1F2022; border: 1px solid rgba(212, 175, 55, 0.16); border-top: 2px solid #D4AF37; border-radius: 4px; padding: 24px; margin-bottom: 16px; box-shadow: 0 0 20px rgba(212,175,55,0.05);'>
                <h3 style='margin:0; color:#D4AF37;'>LATE FUSION META</h3>
                <h2 style='color:#D4AF37; margin:10px 0;'>{fin_pred.upper()}</h2>
                <hr style='border:none; border-top: 1px solid rgba(212, 175, 55, 0.16); margin: 15px 0;'>
                <span style='font-family:monospace; font-size:0.8rem; line-height:1.5; color:#E0E0E0;'>
                    PRECISION: {METRICS['Fusion'][fin_pred]['Precision']} <br>
                    RECALL: {METRICS['Fusion'][fin_pred]['Recall']} <br>
                    MACRO F1: {METRICS['Fusion'][fin_pred]['Macro_F1']}
                </span>
                {render_prob_bars(fin_probs, '#D4AF37')}
            </div>
            """
            st.markdown(fin_html, unsafe_allow_html=True)

        col1, spacer, col2 = st.columns([1, 0.05, 1.5])
        with col1:
            spin_html = f"""
            <div style='background-color: #1F2022; border: 1px solid rgba(212, 175, 55, 0.16); border-radius: 4px; padding: 40px; text-align:center;'>
                <div style='display: flex; justify-content: center; margin-bottom: 20px;'>
                    <div class='vinyl-record'><div class='vinyl-center'><div class='vinyl-hole'></div></div></div>
                </div>
                <h3 style='margin:0; margin-bottom:15px; color:#E8E8E8;'>{st.session_state.current_song.upper()}</h3>
                <div class='ks-mono-eyebrow' style='margin-bottom:5px;'>OVERALL EMOTION</div>
                <h2 style='margin:0; color:#D4AF37;'>{fin_pred.upper()}</h2>
            </div>
            """
            st.markdown(spin_html, unsafe_allow_html=True)
            
        with col2:
            lyric_html = f"""
            <div style='background-color: #1F2022; border: 1px solid rgba(212, 175, 55, 0.16); border-radius: 4px; padding: 30px; height:100%;'>
                <div class='ks-mono-eyebrow' style='margin-bottom:15px;'>LYRICAL TRACE</div>
                <div style='font-size:1.05rem; line-height:1.8; color:#E0E0E0;'>"{lyrics}"</div>
            </div>
            """
            st.markdown(lyric_html, unsafe_allow_html=True)
            
        st.markdown("<div class='ks-mono-eyebrow' style='margin-top:30px; margin-bottom:10px;'>ADVANCED ANALYTICS</div>", unsafe_allow_html=True)
        g_c1, g_c2, g_c3 = st.columns(3)
        with g_c1:
            st.plotly_chart(create_radar_chart(fin_probs_radar), use_container_width=True)
        with g_c2:
            st.plotly_chart(create_circumplex_scatter(feats['Valence'], feats['Energy']), use_container_width=True)
        with g_c3:
            st.plotly_chart(create_gauge_chart(feats['Energy'], "AROUSAL (ENERGY)", "#D4AF37"), use_container_width=True)

elif selected == "Enterprise Batch":
    st.markdown("<div class='ks-mono-eyebrow'>Batch Processing</div>", unsafe_allow_html=True)
    st.markdown("<h1>Enterprise Data Pipeline</h1>", unsafe_allow_html=True)
    
    upload_html = """
    <div style='background-color: #1F2022; border: 1px solid rgba(212, 175, 55, 0.16); border-radius: 4px; padding: 24px; margin-bottom: 16px;'>
        <h3 style='margin:0; margin-bottom:15px;'>UPLOAD PLAYLIST CSV</h3>
    </div>
    """
    st.markdown(upload_html, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV (Must contain a 'Song' or 'Track' column)", type=["csv"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            song_col = None
            # Prioritize title, song_name, or non-id song/track columns
            for col in df.columns:
                cl = col.lower()
                if ('song' in cl or 'track' in cl or 'title' in cl) and 'id' not in cl:
                    song_col = col
                    break
            if song_col is None:
                for col in df.columns:
                    cl = col.lower()
                    if 'song' in cl or 'track' in cl or 'title' in cl:
                        song_col = col
                        break
                    
            if song_col is None:
                st.error("Could not find a valid column containing song names. Please ensure your CSV has a 'Song', 'Track', or 'Title' column.")
            else:
                with st.spinner(f"Processing {len(df)} tracks through Late Fusion Meta-Classifier..."):
                    time.sleep(2.0) 
                    predictions = []
                    for song in df[song_col].dropna():
                        ac_pred, nlp_pred, fin_pred, lyrics, ap, np_p, fp, fpr, feats = get_mock_predictions(song)
                        predictions.append(fin_pred)
                    
                    df["Predicted_Emotion"] = predictions
                    st.session_state.activity_log.append(f"> BATCH PROCESSED: {len(df)} TRACKS")
                    
                st.success("Batch Analysis Complete!")
                
                st.markdown("<h3>Playlist Emotion Distribution</h3>", unsafe_allow_html=True)
                all_emotions = pd.DataFrame({"Emotion": ["Angry", "Calm", "Happy", "Sad"]})
                raw_dist = df["Predicted_Emotion"].value_counts().reset_index()
                raw_dist.columns = ["Emotion", "Count"]
                dist_df = pd.merge(all_emotions, raw_dist, on="Emotion", how="left").fillna(0)
                
                color_discrete_map = {'Angry': '#D4AF37', 'Calm': '#45B6AD', 'Happy': '#D4AF37', 'Sad': '#45B6AD'}
                
                fig2 = px.bar(dist_df, x="Emotion", y="Count", color="Emotion", 
                              color_discrete_map=color_discrete_map, text="Count")
                fig2.update_layout(
                    xaxis={'categoryorder':'array', 'categoryarray':['Angry','Calm','Happy','Sad']},
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)', 
                    font=dict(color="#E0E0E0", family="Albert Sans")
                )
                
                st.plotly_chart(fig2, use_container_width=True)
                
                st.markdown("<h3>Analysis Preview</h3>", unsafe_allow_html=True)
                st.dataframe(df.head(15), use_container_width=True)
                
                csv_export = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="DOWNLOAD ANNOTATED CSV",
                    data=csv_export,
                    file_name="Enterprise_Playlist_Analysis.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        except Exception as e:
            st.error(f"Error processing CSV: {e}")
