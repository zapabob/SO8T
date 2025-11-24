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

    # Recent Results Log
    st.header("📝 Recent Results")
    
    recent_df = df.tail(20).sort_index(ascending=False)
    
    for i, row in recent_df.iterrows():
        with st.expander(f"[{row['benchmark']}] {row['model']} (ID: {row['id']}) - {'✅ Correct' if row['correct'] else '❌ Incorrect'}"):
            st.markdown(f"**Prediction:** `{row['prediction']}`")
            st.markdown("**Prompt:**")
            st.code(row['prompt'])
            st.markdown("**Response:**")
            st.code(row['response'])
            st.divider()

else:
    st.info("Waiting for data...")
