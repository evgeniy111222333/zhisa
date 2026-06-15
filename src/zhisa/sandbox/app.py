import streamlit as st
import pandas as pd
import numpy as np
import time

st.set_page_config(page_title="ZHISA AI Sandbox", layout="wide")

st.title("🧠 ZHISA AI Visual Sandbox")
st.markdown("Interactive dashboard to evaluate the AI's real-time decision making on synthetic and adversarial data.")

st.sidebar.header("Control Panel")
mode = st.sidebar.radio("Test Mode", ["Behavioral Patterns", "Adversarial Injection", "Counterfactuals"])

# Mock AI state for UI demonstration
if 'step' not in st.session_state:
    st.session_state.step = 0
if 'running' not in st.session_state:
    st.session_state.running = False

def toggle_run():
    st.session_state.running = not st.session_state.running

col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    st.button("Play / Pause", on_click=toggle_run)
with col2:
    if st.button("Step Forward"):
        st.session_state.step += 1
with col3:
    if st.button("Reset"):
        st.session_state.step = 0
        st.session_state.running = False

st.divider()

# Main visual area
col_chart, col_brain = st.columns([2, 1])

with col_chart:
    st.subheader("📈 Market View")
    st.markdown("*The raw price action the AI is observing.*")
    # Generate some mock chart data based on step
    x = np.arange(50)
    y = np.sin(x/5 + st.session_state.step*0.1) * 1000 + 50000
    df_chart = pd.DataFrame({'price': y})
    st.line_chart(df_chart, height=400)
    
with col_brain:
    st.subheader("🤖 AI Thoughts")
    
    # Mock probabilities
    p_long = max(0, min(1, 0.5 + np.sin(st.session_state.step*0.2)*0.4))
    p_short = max(0, min(1, 0.5 - np.sin(st.session_state.step*0.2)*0.4))
    p_wait = 1.0 - p_long - p_short
    
    st.markdown("### Action Probabilities")
    st.progress(p_long, text=f"LONG: {p_long:.1%}")
    st.progress(p_short, text=f"SHORT: {p_short:.1%}")
    st.progress(abs(p_wait), text=f"WAIT: {abs(p_wait):.1%}")
    
    st.markdown("### Internal State")
    st.metric("Regime Detected", "Trending Up" if p_long > p_short else "Trending Down")
    st.metric("Uncertainty (Entropy)", f"{np.random.uniform(0.1, 0.5):.2f}")
    st.metric("Suggested Leverage", f"{np.random.uniform(1.0, 3.0):.1f}x")

if st.session_state.running:
    time.sleep(0.5)
    st.session_state.step += 1
    st.rerun()
