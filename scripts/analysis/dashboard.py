import streamlit as st
import pandas as pd
import json
import time
import plotly.express as px
from pathlib import Path
import glob
import os

# Configuration
st.set_page_config(page_title="Grand Benchmark Monitor", layout="wide")
RESULTS_DIR = Path("_docs/grand_benchmark_results")
CHECKPOINT_DIR = Path("_docs/grand_benchmark_checkpoints")

MODELS = {
    "Model_A": "model-a:q8_0",
    "AEGIS_0.6": "aegis-adjusted-0.6",
    "AEGIS_0.8": "aegis-0.8"
}

def load_data():
    """Load all benchmark results"""
    all_data = []
    files = list(RESULTS_DIR.glob("*_results.jsonl"))
    
    for file in files:
        benchmark_name = file.name.replace("_results.jsonl", "")
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    item['benchmark'] = benchmark_name
                    # Flatten result
                    if 'result' in item:
                        item['correct'] = item['result'].get('correct', False)
                        item['prediction'] = item['result'].get('prediction')
                        item['response'] = item['result'].get('response', '')
                        item['prompt'] = item['result'].get('prompt', '')
                    all_data.append(item)
                except:
                    continue
    
    return pd.DataFrame(all_data)

def load_checkpoint():
    """Load latest checkpoint info"""
    ckpts = sorted(CHECKPOINT_DIR.glob("checkpoint_*.json"), key=os.path.getmtime)
    if not ckpts:
        return None
    
    with open(ckpts[-1], 'r', encoding='utf-8') as f:
        return json.load(f)

# Header
st.title("🚀 Grand Benchmark Monitor")
st.caption(f"Last updated: {time.strftime('%H:%M:%S')}")

# Auto-refresh
if st.checkbox("Auto-refresh (10s)", value=True):
    time.sleep(10)
    st.rerun()

# Load Data
df = load_data()
ckpt = load_checkpoint()

# Status Indicators
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Status")
    if ckpt:
        st.info(f"Current Benchmark Index: {ckpt.get('benchmark_idx')}")
        st.info(f"Current Item Index: {ckpt.get('item_idx')}")
        st.text(f"Last Checkpoint: {time.ctime(ckpt.get('timestamp', 0))}")
    else:
        st.warning("No checkpoints found")

with col2:
    st.subheader("Total Processed")
    if not df.empty:
        total_samples = len(df)
        st.metric("Total Inferences", total_samples)
        unique_benchmarks = df['benchmark'].unique()
        st.write(f"Benchmarks: {', '.join(unique_benchmarks)}")
    else:
        st.write("No results yet")

with col3:
    st.subheader("Models")
    if not df.empty:
        models = df['model'].unique()
        st.write(list(models))

st.divider()

# Performance Metrics
if not df.empty:
    st.header("📊 Performance Overview")
    
    # Calculate Accuracy per Model per Benchmark
    stats = df.groupby(['benchmark', 'model'])['correct'].agg(['count', 'sum']).reset_index()
    stats['accuracy'] = (stats['sum'] / stats['count']) * 100
    stats.columns = ['Benchmark', 'Model', 'Total', 'Correct', 'Accuracy']
    
    # Display Stats Table
    st.dataframe(stats.style.format({'Accuracy': '{:.2f}%'}), use_container_width=True)
    
    # Charts
    c1, c2 = st.columns(2)
    
    with c1:
        fig_bar = px.bar(stats, x='Benchmark', y='Accuracy', color='Model', barmode='group',
                         title="Accuracy by Benchmark", text_auto='.2s')
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with c2:
        # Overall Accuracy
        overall = df.groupby('model')['correct'].mean().reset_index()
        overall['accuracy'] = overall['correct'] * 100
        fig_pie = px.bar(overall, x='model', y='accuracy', color='model', 
                         title="Overall Win Rate", text_auto='.2f')
        st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    # Head-to-Head Comparison
    st.header("⚔️ Head-to-Head Comparison")
    st.caption("Compare responses for the same question side-by-side.")
    
    # Get recent question IDs
    recent_ids = df['id'].unique()
    recent_ids = sorted(recent_ids, reverse=True)[:5] # Last 5 questions
    
    for q_id in recent_ids:
        q_data = df[df['id'] == q_id]
        if q_data.empty:
            continue
            
        # Get the question prompt from the first available model's entry
        prompt = q_data.iloc[0]['prompt']
        benchmark = q_data.iloc[0]['benchmark']
        
        with st.expander(f"[{benchmark}] Question ID: {q_id}", expanded=True):
            st.markdown("**Prompt:**")
            st.code(prompt)
            
            cols = st.columns(len(MODELS))
            sorted_models = sorted(q_data['model'].unique())
            
            for i, model in enumerate(sorted_models):
                row = q_data[q_data['model'] == model].iloc[0]
                with cols[i]:
                    st.subheader(model)
                    status = "✅ Correct" if row['correct'] else "❌ Incorrect"
                    st.markdown(f"**Status:** {status}")
                    st.markdown(f"**Prediction:** `{row['prediction']}`")
                    st.markdown("**Response:**")
                    st.info(row['response'])

    # Recent Results Log
    st.header("📝 Raw Log")
    st.dataframe(df.tail(20)[['benchmark', 'model', 'id', 'correct', 'prediction', 'response']].sort_index(ascending=False), use_container_width=True)

else:
    st.info("Waiting for data...")
