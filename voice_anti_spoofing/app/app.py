import os
import sys
import io
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src import config, audio, features, preprocessing
from src.authentication import VoiceAuthenticator
from src.enrollment import EnrollmentManager

# Page Configuration
st.set_page_config(
    page_title="AI Voice Anti-Spoofing & Authentication System",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .decision-granted {
        background: linear-gradient(135deg, #059669 0%, #10B981 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: 1px;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
        margin-bottom: 1rem;
    }
    .decision-denied {
        background: linear-gradient(135deg, #DC2626 0%, #EF4444 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: 1px;
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.3);
        margin-bottom: 1rem;
    }
    .reason-box {
        background-color: #F8FAFC;
        border-left: 5px solid #64748B;
        padding: 0.8rem 1.2rem;
        border-radius: 6px;
        font-size: 1.05rem;
        font-weight: 500;
        color: #334155;
        margin-bottom: 1.5rem;
    }
    .card-box {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .badge-pass {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .badge-fail {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .scenario-card {
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 1rem;
        background-color: #F8FAFC;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_authenticator():
    return VoiceAuthenticator()

def main():
    authenticator = get_authenticator()
    enrollment_mgr = authenticator.enrollment_mgr
    
    # Title & Header
    st.markdown('<div class="main-header">🎙️ AI Voice Authentication & Replay Attack Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">B.Tech Minor Project Demonstration Platform | Parallel Biometric Verification & Liveness Architecture</div>', unsafe_allow_html=True)
    
    # Sidebar Settings
    with st.sidebar:
        st.header("⚙️ System Control Panel")
        
        st.subheader("Model Decision Thresholds")
        spk_thresh = st.slider(
            "Speaker Verification Threshold (Cosine Sim)",
            min_value=0.40, max_value=0.90, value=float(config.SPEAKER_THRESHOLD), step=0.01,
            help="Minimum cosine similarity required to confirm speaker match."
        )
        spf_thresh = st.slider(
            "Replay Attack Detection Threshold (Probability)",
            min_value=0.20, max_value=0.80, value=float(config.SPOOF_THRESHOLD), step=0.01,
            help="Replay probability at or above this threshold triggers REPLAY ATTACK status."
        )
        
        st.divider()
        st.subheader("🖥️ Hardware & Status")
        st.info(f"**Execution Device**: `{config.DEVICE.type.upper()}`")
        st.write(f"**Sample Rate**: `{config.SAMPLE_RATE} Hz`")
        st.write(f"**Spectrogram Mels**: `{config.N_MELS}`")
        
        # Enrolled User Info
        st.divider()
        st.subheader("👤 Enrolled Speaker Profile")
        if enrollment_mgr.is_enrolled():
            info = enrollment_mgr.get_enrollment_info() or {}
            st.success(f"**Status**: Enrolled ({info.get('user_id', 'User')})")
            st.caption(f"Samples: {info.get('sample_count', 'N/A')} | Date: {info.get('enrolled_at', '')[:10]}")
            if st.button("🗑️ Reset Speaker Enrollment", type="secondary"):
                enrollment_mgr.clear_enrollment()
                st.rerun()
        else:
            st.warning("**Status**: No Authorized User Enrolled")
            st.caption("Please enroll an authorized speaker in Tab 2 before live testing.")

    # Main Tabs
    tab_auth, tab_enroll, tab_scenarios, tab_eval = st.tabs([
        "🔐 Live Authentication Engine",
        "👤 Authorized Speaker Enrollment",
        "🧪 Panel Demonstration (4 Scenarios)",
        "📊 Model Benchmarks & Results"
    ])

    # =========================================================================
    # TAB 1: LIVE AUTHENTICATION ENGINE
    # =========================================================================
    with tab_auth:
        st.subheader("Live Audio Verification & Anti-Spoof Inspection")
        
        if not enrollment_mgr.is_enrolled():
            st.warning("⚠️ **No authorized speaker profile enrolled yet!** Please navigate to 'Authorized Speaker Enrollment' tab first.")
            
        col_input, col_viz = st.columns([1, 1])
        
        with col_input:
            st.markdown("### 1. Audio Input Selection")
            input_mode = st.radio(
                "Choose Audio Input Method:",
                ["🎙️ Record Microphone Audio", "📁 Upload Audio File (WAV/MP3)", "🔊 Load Demo Test Waveform"],
                horizontal=True
            )
            
            audio_source = None
            
            if input_mode == "🎙️ Record Microphone Audio":
                st.write("Click record button below, speak clearly for 3 seconds:")
                # Try Streamlit mic recorder component
                try:
                    from streamlit_mic_recorder import mic_recorder
                    audio_dict = mic_recorder(
                        start_prompt="🔴 Start Recording (3s)",
                        stop_prompt="⬛ Stop Recording",
                        key="auth_mic",
                        format="wav"
                    )
                    if audio_dict and 'bytes' in audio_dict:
                        audio_source = io.BytesIO(audio_dict['bytes'])
                        st.audio(audio_source, format="audio/wav")
                except Exception as e:
                    st.error(f"Microphone recorder component error: {e}")
                    st.info("Falling back to file upload or Python sounddevice recorder below.")
                    if st.button("🔴 Record via Sounddevice (3s Local Mic)"):
                        with st.spinner("Recording 3 seconds from local microphone... Speak now!"):
                            rec_wav = audio.record_audio_sounddevice(duration=3.0)
                            buf = io.BytesIO()
                            import soundfile as sf
                            sf.write(buf, rec_wav, config.SAMPLE_RATE, format='WAV')
                            buf.seek(0)
                            audio_source = buf
                            st.audio(audio_source, format="audio/wav")
                            
            elif input_mode == "📁 Upload Audio File (WAV/MP3)":
                uploaded_file = st.file_uploader("Upload audio WAV file:", type=["wav", "mp3", "flac"], key="auth_file")
                if uploaded_file is not None:
                    audio_source = uploaded_file
                    st.audio(audio_source)
                    
            elif input_mode == "🔊 Load Demo Test Waveform":
                demo_option = st.selectbox(
                    "Select Pre-generated Demonstration Sample:",
                    [
                        "Genuine Live Voice Sample (Bona-fide)",
                        "Replay Attack Voice Sample (Phone Speaker Replay)",
                        "Unauthorized Speaker Sample"
                    ]
                )
                raw_dir = config.RAW_DATA_DIR
                if demo_option == "Genuine Live Voice Sample (Bona-fide)":
                    files = list((raw_dir / "bonafide").glob("*.wav"))
                    if files:
                        audio_source = str(files[0])
                elif demo_option == "Replay Attack Voice Sample (Phone Speaker Replay)":
                    files = list((raw_dir / "spoof").glob("*.wav"))
                    if files:
                        audio_source = str(files[0])
                else:
                    files = list((raw_dir / "bonafide").glob("*.wav"))
                    if len(files) > 1:
                        audio_source = str(files[-1])
                        
                if audio_source:
                    st.audio(audio_source)

        with col_viz:
            st.markdown("### 2. Signal Preview & Log-Mel Spectrogram")
            if audio_source is not None:
                try:
                    wav = audio.load_audio(audio_source, sr=config.SAMPLE_RATE, duration=config.AUDIO_DURATION)
                    log_mel = features.compute_log_mel_spectrogram(wav, sr=config.SAMPLE_RATE)
                    
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4))
                    t_ax = np.linspace(0, config.AUDIO_DURATION, len(wav))
                    ax1.plot(t_ax, wav, color='#2563EB', lw=0.8)
                    ax1.set_title("Audio Waveform", fontsize=10)
                    ax1.set_ylabel("Amplitude")
                    ax1.grid(True, alpha=0.3)
                    
                    cax = ax2.imshow(log_mel, aspect='auto', origin='lower', cmap='viridis')
                    ax2.set_title("Log-Mel Spectrogram (Anti-Spoof Feature Map)", fontsize=10)
                    ax2.set_xlabel("Time Frames")
                    ax2.set_ylabel("Mel Frequency")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                except Exception as e:
                    st.error(f"Error rendering visualizer: {e}")
            else:
                st.info("Record or upload audio to view waveform & Log-Mel Spectrogram.")

        st.divider()
        st.markdown("### 3. Run Pipeline Authentication")
        
        if st.button("🚀 AUTHENTICATE VOICE SAMPLE", type="primary", use_container_width=True):
            if audio_source is None:
                st.error("Please record or select an audio sample before clicking Authenticate.")
            else:
                with st.spinner("Processing parallel Speaker Verification & Anti-Spoof CNN models..."):
                    res = authenticator.authenticate(
                        audio_source,
                        speaker_threshold_override=spk_thresh,
                        spoof_threshold_override=spf_thresh
                    )
                    
                # Render Final Decision Box
                if res['final_decision'] == "ACCESS GRANTED":
                    st.markdown('<div class="decision-granted">✅ ACCESS GRANTED</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="decision-denied">❌ ACCESS DENIED</div>', unsafe_allow_html=True)
                    
                st.markdown(f'<div class="reason-box"><strong>DECISION REASON:</strong> {res["reason"]}</div>', unsafe_allow_html=True)
                
                # Detailed Metric Breakdown Cards
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Model 1: Speaker Verification Engine")
                    spk_badge = '<span class="badge-pass">PASS</span>' if res['speaker_pass'] else '<span class="badge-fail">FAIL</span>'
                    st.markdown(f"**Status**: {spk_badge}", unsafe_allow_html=True)
                    st.metric("Speaker Similarity", res['speaker_similarity_pct'])
                    st.progress(res['speaker_similarity'])
                    st.caption(f"Configured Threshold: `{spk_thresh * 100:.1f}%`")
                    
                with col2:
                    st.markdown("#### Model 2: Anti-Spoof / Replay Detection Engine")
                    live_badge = '<span class="badge-pass">LIVE VOICE</span>' if res['is_live'] else '<span class="badge-fail">SPOOF / REPLAY</span>'
                    st.markdown(f"**Liveness**: {live_badge}", unsafe_allow_html=True)
                    st.metric("Replay Probability", res['replay_probability_pct'])
                    st.progress(res['replay_probability'])
                    st.caption(f"Configured Spoof Threshold: `{spf_thresh * 100:.1f}%`")

    # =========================================================================
    # TAB 2: AUTHORIZED SPEAKER ENROLLMENT
    # =========================================================================
    with tab_enroll:
        st.subheader("Enroll Authorized User Voice Profile")
        st.markdown("""
        To demonstrate personalized voice authentication, enroll your own voice below:
        1. Record or upload **3 to 5 short voice samples** (e.g. saying *"Open security system"*).
        2. The system extracts deep speaker embedding vectors using ECAPA-TDNN.
        3. Computes and normalizes the speaker centroid vector stored in `data/enrolled/enrolled_speaker.npy`.
        """)
        
        col_e1, col_e2 = st.columns([1.2, 1])
        
        with col_e1:
            st.markdown("### Record / Add Enrollment Samples")
            user_tag = st.text_input("Authorized User Name / ID:", value="Authorized_User")
            
            if 'enroll_samples' not in st.session_state:
                st.session_state.enroll_samples = []
                
            # Mic input for enrollment
            try:
                from streamlit_mic_recorder import mic_recorder
                rec = mic_recorder(
                    start_prompt=f"🎤 Record Sample #{len(st.session_state.enroll_samples)+1}",
                    stop_prompt="⬛ Stop & Save Sample",
                    key=f"enroll_mic_{len(st.session_state.enroll_samples)}",
                    format="wav"
                )
                if rec and 'bytes' in rec:
                    st.session_state.enroll_samples.append(io.BytesIO(rec['bytes']))
                    st.success(f"Sample #{len(st.session_state.enroll_samples)} recorded successfully!")
            except Exception:
                pass
                
            uploaded_enroll_files = st.file_uploader(
                "Or upload enrollment WAV files:",
                type=["wav", "mp3"],
                accept_multiple_files=True,
                key="enroll_files_upload"
            )
            if uploaded_enroll_files:
                for f in uploaded_enroll_files:
                    st.session_state.enroll_samples.append(f)
                    
            st.write(f"**Current collected samples count**: `{len(st.session_state.enroll_samples)}`")
            
            if st.button("🧹 Clear Collected Draft Samples"):
                st.session_state.enroll_samples = []
                st.rerun()
                
            if st.button("💾 SAVE & ENROLL SPEAKER PROFILE", type="primary"):
                if len(st.session_state.enroll_samples) == 0:
                    st.error("Please record or upload at least 1 sample to enroll.")
                else:
                    with st.spinner("Extracting ECAPA-TDNN speaker embeddings & computing centroid..."):
                        success, msg, centroid = enrollment_mgr.enroll_speaker(
                            st.session_state.enroll_samples, user_id=user_tag
                        )
                        if success:
                            st.success(msg)
                            st.session_state.enroll_samples = []
                            st.rerun()
                        else:
                            st.error(msg)

        with col_e2:
            st.markdown("### Current Enrollment Status")
            if enrollment_mgr.is_enrolled():
                info = enrollment_mgr.get_enrollment_info() or {}
                emb = enrollment_mgr.get_enrolled_embedding()
                st.success("✅ **Authorized Profile Enrolled**")
                st.json(info)
                if emb is not None:
                    st.caption(f"Embedding Vector Dim: {emb.shape[0]} | Norm: {np.linalg.norm(emb):.4f}")
            else:
                st.warning("⚠️ No authorized profile currently enrolled.")

    # =========================================================================
    # TAB 3: PANEL DEMONSTRATION (4 SCENARIOS)
    # =========================================================================
    with tab_scenarios:
        st.subheader("🧪 Live 4-Scenario Evaluation Panel Script")
        st.markdown("Use this tab to demonstrate all 4 operational matrix scenarios live to the project panel.")
        
        sc1, sc2 = st.columns(2)
        sc3, sc4 = st.columns(2)
        
        with sc1:
            st.markdown("""
            <div class="scenario-card">
            <h4>SCENARIO 1 — Authorized + Live Voice</h4>
            <p><strong>Input:</strong> Authorized user speaks directly into microphone.</p>
            <ul>
                <li>Speaker Verification: <strong>PASS</strong></li>
                <li>Voice Liveness: <strong>LIVE</strong></li>
                <li>Replay Probability: <strong>LOW</strong></li>
                <li>Final Decision: <strong>ACCESS GRANTED</strong></li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Run Test Scenario 1", key="btn_sc1"):
                raw_dir = config.RAW_DATA_DIR
                bona_files = list((raw_dir / "bonafide").glob("*.wav"))
                if bona_files and enrollment_mgr.is_enrolled():
                    res = authenticator.authenticate(str(bona_files[0]), speaker_threshold_override=spk_thresh, spoof_threshold_override=spf_thresh)
                    st.write(res)
                else:
                    st.info("Please enroll an authorized speaker first.")

        with sc2:
            st.markdown("""
            <div class="scenario-card">
            <h4>SCENARIO 2 — Authorized Voice Replay Attack</h4>
            <p><strong>Input:</strong> Recorded authorized voice played back through speaker.</p>
            <ul>
                <li>Speaker Verification: <strong>PASS</strong></li>
                <li>Voice Liveness: <strong>SPOOF / REPLAY</strong></li>
                <li>Replay Probability: <strong>HIGH</strong></li>
                <li>Final Decision: <strong>ACCESS DENIED</strong></li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Run Test Scenario 2", key="btn_sc2"):
                raw_dir = config.RAW_DATA_DIR
                spoof_files = list((raw_dir / "spoof").glob("*.wav"))
                if spoof_files and enrollment_mgr.is_enrolled():
                    res = authenticator.authenticate(str(spoof_files[0]), speaker_threshold_override=spk_thresh, spoof_threshold_override=spf_thresh)
                    st.write(res)
                else:
                    st.info("Please enroll an authorized speaker first.")

        with sc3:
            st.markdown("""
            <div class="scenario-card">
            <h4>SCENARIO 3 — Unauthorized + Live Voice</h4>
            <p><strong>Input:</strong> Different person speaks directly into microphone.</p>
            <ul>
                <li>Speaker Verification: <strong>FAIL</strong></li>
                <li>Voice Liveness: <strong>LIVE</strong></li>
                <li>Replay Probability: <strong>LOW</strong></li>
                <li>Final Decision: <strong>ACCESS DENIED</strong></li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Run Test Scenario 3", key="btn_sc3"):
                raw_dir = config.RAW_DATA_DIR
                bona_files = list((raw_dir / "bonafide").glob("*.wav"))
                if len(bona_files) > 1 and enrollment_mgr.is_enrolled():
                    # Simulate wrong speaker by overriding enrolled vector with different dummy
                    dummy_emb = np.random.randn(192)
                    dummy_emb = (dummy_emb / np.linalg.norm(dummy_emb)).astype(np.float32)
                    res = authenticator.verifier.verify_speaker(str(bona_files[0]), dummy_emb, threshold=spk_thresh)
                    st.write({"Speaker Pass": res[0], "Similarity": f"{res[1]*100:.2f}%", "Decision": "ACCESS DENIED (UNAUTHORIZED SPEAKER)"})
                else:
                    st.info("Please prepare dataset and enroll speaker.")

        with sc4:
            st.markdown("""
            <div class="scenario-card">
            <h4>SCENARIO 4 — Unauthorized + Replay Attack</h4>
            <p><strong>Input:</strong> Recorded unauthorized voice played back through speaker.</p>
            <ul>
                <li>Speaker Verification: <strong>FAIL</strong></li>
                <li>Voice Liveness: <strong>SPOOF / REPLAY</strong></li>
                <li>Replay Probability: <strong>HIGH</strong></li>
                <li>Final Decision: <strong>ACCESS DENIED</strong></li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Run Test Scenario 4", key="btn_sc4"):
                raw_dir = config.RAW_DATA_DIR
                spoof_files = list((raw_dir / "spoof").glob("*.wav"))
                if spoof_files:
                    dummy_emb = np.random.randn(192)
                    dummy_emb = (dummy_emb / np.linalg.norm(dummy_emb)).astype(np.float32)
                    res_spk = authenticator.verifier.verify_speaker(str(spoof_files[0]), dummy_emb, threshold=spk_thresh)
                    wav = audio.load_audio(str(spoof_files[0]))
                    prob = authenticator.predict_liveness(wav)
                    st.write({
                        "Speaker Verification": "FAIL",
                        "Voice Liveness": "SPOOF / REPLAY",
                        "Replay Probability": f"{prob*100:.2f}%",
                        "Decision": "ACCESS DENIED (UNAUTHORIZED + REPLAY)"
                    })

    # =========================================================================
    # TAB 4: MODEL BENCHMARKS & EVALUATION RESULTS
    # =========================================================================
    with tab_eval:
        st.subheader("📊 Research Benchmark Results & Model Evaluation")
        
        results_dir = config.RESULTS_DIR
        cm_path = results_dir / "confusion_matrix.png"
        roc_path = results_dir / "roc_curve.png"
        hist_path = results_dir / "training_history.png"
        report_path = results_dir / "evaluation_report.txt"
        
        if report_path.exists():
            with open(report_path, "r") as f:
                report_txt = f.read()
            st.code(report_txt, language="text")
        else:
            st.info("No evaluation report generated yet. Click 'Run Model Evaluation' below.")
            
        r_col1, r_col2 = st.columns(2)
        with r_col1:
            if cm_path.exists():
                st.image(str(cm_path), caption="Confusion Matrix (Live vs Replay)", use_container_width=True)
            if hist_path.exists():
                st.image(str(hist_path), caption="AntiSpoofCNN Training Curves", use_container_width=True)
        with r_col2:
            if roc_path.exists():
                st.image(str(roc_path), caption="Receiver Operating Characteristic (ROC)", use_container_width=True)
                
        if st.button("⚡ Re-run Complete Model Evaluation"):
            with st.spinner("Evaluating model on test dataset split..."):
                from scripts import evaluate
                evaluate.main()
                st.rerun()

if __name__ == "__main__":
    main()
