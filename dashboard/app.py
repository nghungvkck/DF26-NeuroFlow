"""
AUTOSCALING PIPELINE DASHBOARD
=============================
Interactive visualization of autoscaling results using Streamlit.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import json

# =====================================================================
# PAGE CONFIGURATION
# =====================================================================

st.set_page_config(
    page_title="Autoscaling Pipeline",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚀 Autoscaling Pipeline Dashboard")
st.markdown("""
**PHASE A:** Model Evaluation - Real data forecast accuracy
**PHASE B:** Autoscaling Tests - Synthetic scenario strategy testing
**PHASE C:** Anomaly & Cost Analysis - Advanced detection and cost optimization
""")

# =====================================================================
# SIDEBAR - MODE SELECTION
# =====================================================================

st.sidebar.header("📊 Visualization Mode")
visualization_mode = st.sidebar.radio(
    "Choose what to visualize",
    ["Autoscaling Tests", "Model Evaluation", "Anomaly & Cost Analysis"],
)

results_dir = Path("results")

if not results_dir.exists():
    st.error("⚠️ No results found. Run `python run_pipeline.py` first.")
    st.stop()

# =====================================================================
# MODE 1: AUTOSCALING TESTS
# =====================================================================

if visualization_mode == "Autoscaling Tests":
    csv_path = results_dir / "simulation_results.csv"
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        st.error("⚠️ Results file not found.")
        st.stop()
    
    scenarios = sorted(df['scenario'].unique())
    strategies = sorted(df['strategy'].unique())
    
    st.sidebar.header("🔧 Filters")
    selected_scenario = st.sidebar.selectbox("Select Scenario", scenarios)
    selected_strategies = st.sidebar.multiselect(
        "Select Strategies", strategies, default=strategies
    )
    
    filtered_df = df[
        (df['scenario'] == selected_scenario) &
        (df['strategy'].isin(selected_strategies))
    ].copy()
    
    if filtered_df.empty:
        st.error("No data available for selected filters.")
        st.stop()
    
    st.markdown(f"**Scenario:** {selected_scenario}")
    
    # Tabs for Autoscaling Tests
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Load & Forecast",
        "📈 Pod Timeline",
        "💰 Cost Analysis",
        "🚨 SLA Violations",
        "📋 Metrics Comparison",
        "🔴 Anomaly Detection",
        "🎯 Advanced Metrics"
    ])
    
    # =====================================================================
    # TAB 1: Load vs Forecast
    # =====================================================================
    
    with tab1:
        st.subheader("Load Pattern vs Forecast Accuracy")
        
        first_strategy = selected_strategies[0]
        scenario_data = filtered_df[filtered_df['strategy'] == first_strategy].sort_values('time')
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=scenario_data['time'],
            y=scenario_data['actual_requests'],
            name='Actual Load',
            mode='lines',
            line=dict(color='blue', width=2),
        ))
        fig.add_trace(go.Scatter(
            x=scenario_data['time'],
            y=scenario_data['forecast_requests'],
            name='Forecast',
            mode='lines',
            line=dict(color='orange', width=2, dash='dash'),
        ))
        fig.update_layout(
            title=f"Load Pattern - {selected_scenario}",
            xaxis_title="Time (steps)",
            yaxis_title="Request Rate (req/s)",
            hovermode='x unified',
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            mean_error = np.mean(np.abs(scenario_data['forecast_requests'] - scenario_data['actual_requests']))
            st.metric("Mean Forecast Error", f"{mean_error:.1f} req/s")
        with col2:
            max_error = np.max(np.abs(scenario_data['forecast_requests'] - scenario_data['actual_requests']))
            st.metric("Max Forecast Error", f"{max_error:.1f} req/s")
        with col3:
            rmse = np.sqrt(np.mean((scenario_data['forecast_requests'] - scenario_data['actual_requests'])**2))
            st.metric("RMSE", f"{rmse:.1f} req/s")
        with col4:
            anomalies = scenario_data['z_anomaly'].sum()
            st.metric("Anomalies Detected", int(anomalies))
    
    # =====================================================================
    # TAB 2: Pod Timeline
    # =====================================================================
    
    with tab2:
        st.subheader("Pod Count Over Time")
        
        fig = go.Figure()
        colors = {'REACTIVE': '#1f77b4', 'PREDICTIVE': '#ff7f0e',
                  'CPU_BASED': '#2ca02c', 'HYBRID': '#d62728'}
        
        for strategy in selected_strategies:
            strategy_data = filtered_df[filtered_df['strategy'] == strategy].sort_values('time')
            fig.add_trace(go.Scatter(
                x=strategy_data['time'],
                y=strategy_data['pods_after'],
                name=strategy,
                mode='lines+markers',
                line=dict(color=colors.get(strategy, 'gray'), width=2),
            ))
        
        fig.update_layout(
            title=f"Pod Timeline - {selected_scenario}",
            xaxis_title="Time (steps)",
            yaxis_title="Number of Pods",
            hovermode='x unified',
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        for i, strategy in enumerate(selected_strategies[:3]):
            strategy_data = filtered_df[filtered_df['strategy'] == strategy]
            events = (strategy_data['scaling_action'] != 0).sum()
            with [col1, col2, col3][i]:
                st.metric(f"{strategy} Scaling Events", int(events))
    
    # =====================================================================
    # TAB 3: Cost Analysis
    # =====================================================================
    
    with tab3:
        st.subheader("Cost Comparison")
        
        fig = go.Figure()
        for strategy in selected_strategies:
            strategy_data = filtered_df[filtered_df['strategy'] == strategy].sort_values('time')
            cost_per_step = strategy_data['pods_after'] * 0.05 * (5 / 60)
            cumulative_cost = cost_per_step.cumsum()
            fig.add_trace(go.Scatter(
                x=strategy_data['time'],
                y=cumulative_cost,
                name=strategy,
                mode='lines',
                line=dict(color=colors.get(strategy, 'gray'), width=2),
            ))
        
        fig.update_layout(
            title=f"Cumulative Cost - {selected_scenario}",
            xaxis_title="Time (steps)",
            yaxis_title="Total Cost ($)",
            hovermode='x unified',
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Cost Summary")
        cost_summary = []
        for strategy in selected_strategies:
            strategy_data = filtered_df[filtered_df['strategy'] == strategy]
            total_cost = (strategy_data['pods_after'] * 0.05 * (5 / 60)).sum()
            avg_pods = strategy_data['pods_after'].mean()
            cost_summary.append({
                'Strategy': strategy,
                'Total Cost ($)': f"{total_cost:.2f}",
                'Avg Pods': f"{avg_pods:.1f}",
                'Cost/Pod': f"{total_cost / len(strategy_data):.4f}"
            })
        st.dataframe(pd.DataFrame(cost_summary), use_container_width=True)
    
    # =====================================================================
    # TAB 4: SLA Violations
    # =====================================================================
    
    with tab4:
        st.subheader("SLA Violations")
        
        fig = go.Figure()
        for strategy in selected_strategies:
            strategy_data = filtered_df[filtered_df['strategy'] == strategy].sort_values('time')
            fig.add_trace(go.Scatter(
                x=strategy_data['time'],
                y=strategy_data['sla_breached_before_scaling'].astype(int),
                name=strategy,
                mode='lines',
                fill='tozeroy',
                line=dict(color=colors.get(strategy, 'gray'), width=1),
                opacity=0.6,
            ))
        
        fig.update_layout(
            title=f"SLA Violation Timeline - {selected_scenario}",
            xaxis_title="Time (steps)",
            yaxis_title="SLA Breached (1=Yes)",
            hovermode='x unified',
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("SLA Statistics")
        col1, col2, col3, col4 = st.columns(4)
        for i, strategy in enumerate(selected_strategies[:4]):
            strategy_data = filtered_df[filtered_df['strategy'] == strategy]
            violations = strategy_data['sla_breached_before_scaling'].sum()
            violation_rate = violations / len(strategy_data) * 100
            with [col1, col2, col3, col4][i]:
                st.metric(f"{strategy}", f"{violations} ({violation_rate:.1f}%)")
    
    # =====================================================================
    # TAB 5: Metrics Comparison
    # =====================================================================
    
    with tab5:
        st.subheader("Comprehensive Metrics Comparison")
        
        metrics_data = []
        for strategy in selected_strategies:
            strategy_data = filtered_df[filtered_df['strategy'] == strategy]
            
            total_cost = (strategy_data['pods_after'] * 0.05 * (5 / 60)).sum()
            avg_pods = strategy_data['pods_after'].mean()
            sla_violations = strategy_data['sla_breached_before_scaling'].sum()
            violation_rate = sla_violations / len(strategy_data) * 100
            scaling_events = (strategy_data['scaling_action'] != 0).sum()
            
            utilization = strategy_data['actual_requests'] / (strategy_data['pods_after'] * 500)
            mean_util = utilization.mean() * 100
            max_util = utilization.max() * 100
            
            metrics_data.append({
                'Strategy': strategy,
                'Total Cost': f"${total_cost:.2f}",
                'Avg Pods': f"{avg_pods:.1f}",
                'SLA Violations': f"{sla_violations}",
                'Violation Rate': f"{violation_rate:.1f}%",
                'Scaling Events': f"{scaling_events}",
                'Mean Utilization': f"{mean_util:.1f}%",
                'Max Utilization': f"{max_util:.1f}%",
            })
        
        st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)
    
    # =====================================================================
    # TAB 6: Anomaly Detection
    # =====================================================================
    
    with tab6:
        st.subheader("🔴 Anomaly Detection Analysis")
        
        if 'is_anomaly' not in filtered_df.columns:
            st.warning("⚠️ Anomaly detection data not available. Run simulation with enable_advanced_metrics=True.")
        else:
            # Anomaly timeline
            st.markdown("**Anomaly Timeline**")
            
            fig = go.Figure()
            
            for strategy in selected_strategies:
                strategy_data = filtered_df[filtered_df['strategy'] == strategy].sort_values('time')
                
                # Plot requests with anomalies highlighted
                anomalies = strategy_data[strategy_data['is_anomaly'] == True]
                
                fig.add_trace(go.Scatter(
                    x=strategy_data['time'],
                    y=strategy_data['actual_requests'],
                    mode='lines',
                    name=f'{strategy} - Requests',
                    line=dict(width=2)
                ))
                
                if not anomalies.empty:
                    fig.add_trace(go.Scatter(
                        x=anomalies['time'],
                        y=anomalies['actual_requests'],
                        mode='markers',
                        name=f'{strategy} - Anomalies',
                        marker=dict(size=12, symbol='x', line=dict(width=2)),
                        hovertemplate='<b>Anomaly</b><br>Time: %{x}<br>Requests: %{y}<br>Reason: %{customdata}<extra></extra>',
                        customdata=anomalies['anomaly_reason'] if 'anomaly_reason' in anomalies.columns else ['Detected'] * len(anomalies)
                    ))
            
            fig.update_layout(
                title="Traffic Pattern with Anomalies",
                xaxis_title="Time",
                yaxis_title="Requests",
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Anomaly statistics
            st.markdown("**Anomaly Statistics by Strategy**")
            
            anomaly_stats = []
            for strategy in selected_strategies:
                strategy_data = filtered_df[filtered_df['strategy'] == strategy]
                
                total_anomalies = strategy_data['is_anomaly'].sum() if 'is_anomaly' in strategy_data.columns else 0
                anomaly_rate = total_anomalies / len(strategy_data) * 100 if len(strategy_data) > 0 else 0
                
                # Avg confidence
                avg_confidence = strategy_data['anomaly_confidence'].mean() * 100 if 'anomaly_confidence' in strategy_data.columns else 0
                
                # Scaling response to anomalies
                if 'is_anomaly' in strategy_data.columns:
                    anomaly_rows = strategy_data[strategy_data['is_anomaly'] == True]
                    scaling_response = (anomaly_rows['scaling_action'] != 0).sum()
                    response_rate = scaling_response / max(total_anomalies, 1) * 100
                else:
                    response_rate = 0
                
                anomaly_stats.append({
                    'Strategy': strategy,
                    'Total Anomalies': int(total_anomalies),
                    'Anomaly Rate': f"{anomaly_rate:.1f}%",
                    'Avg Confidence': f"{avg_confidence:.1f}%",
                    'Scaling Response': f"{response_rate:.1f}%"
                })
            
            st.dataframe(pd.DataFrame(anomaly_stats), use_container_width=True)
            
            # Anomaly types breakdown
            if 'anomaly_reason' in filtered_df.columns:
                st.markdown("**Anomaly Types Distribution**")
                
                anomaly_types = filtered_df[filtered_df['is_anomaly'] == True]['anomaly_reason'].value_counts()
                
                if not anomaly_types.empty:
                    fig = go.Figure(data=[
                        go.Pie(
                            labels=anomaly_types.index,
                            values=anomaly_types.values,
                            hole=0.3
                        )
                    ])
                    fig.update_layout(title="Anomaly Detection Methods", height=400)
                    st.plotly_chart(fig, use_container_width=True)
    
    # =====================================================================
    # TAB 7: Advanced Metrics (K8s HPA, AWS Auto Scaling)
    # =====================================================================
    
    with tab7:
        st.subheader("🎯 Advanced Platform Metrics")
        
        # Load metrics summary if available
        metrics_path = results_dir / "metrics_summary.json"
        
        try:
            with open(metrics_path, 'r') as f:
                metrics_summary = json.load(f)
        except FileNotFoundError:
            st.warning("⚠️ Advanced metrics not available.")
            metrics_summary = None
        
        if metrics_summary:
            # Find metrics for selected scenario
            scenario_metrics = {}
            for key, value in metrics_summary.items():
                if selected_scenario in key:
                    scenario_metrics[key] = value
            
            if scenario_metrics:
                st.markdown("**Kubernetes HPA Metrics**")
                
                k8s_data = []
                for strategy in selected_strategies:
                    strategy_key = f"{selected_scenario}_{strategy}"
                    if strategy_key in scenario_metrics:
                        m = scenario_metrics[strategy_key]
                        
                        k8s_data.append({
                            'Strategy': strategy,
                            'Avg CPU Utilization': f"{m.get('k8s_avg_cpu_utilization', 0):.1%}",
                            'Max CPU Utilization': f"{m.get('k8s_max_cpu_utilization', 0):.1%}",
                            'CPU Target Breaches': int(m.get('k8s_cpu_target_breaches', 0)),
                            'HPA Trigger Rate': f"{m.get('k8s_hpa_trigger_rate', 0):.1%}"
                        })
                
                if k8s_data:
                    st.dataframe(pd.DataFrame(k8s_data), use_container_width=True)
                
                st.markdown("---")
                st.markdown("**AWS Auto Scaling Metrics**")
                
                aws_data = []
                for strategy in selected_strategies:
                    strategy_key = f"{selected_scenario}_{strategy}"
                    if strategy_key in scenario_metrics:
                        m = scenario_metrics[strategy_key]
                        
                        aws_data.append({
                            'Strategy': strategy,
                            'Warm-up Time Ratio': f"{m.get('aws_warm_up_time_ratio', 0):.1%}",
                            'Cooldown Time Ratio': f"{m.get('aws_cooldown_time_ratio', 0):.1%}",
                            'Cooldown Effectiveness': f"{m.get('aws_cooldown_effectiveness', 0):.1%}",
                            'Target Tracking Breaches': int(m.get('aws_target_tracking_breaches', 0))
                        })
                
                if aws_data:
                    st.dataframe(pd.DataFrame(aws_data), use_container_width=True)
                
                st.markdown("---")
                st.markdown("**Cost Model Comparison**")
                
                # Show cost breakdown if available
                cost_breakdown_path = results_dir / "cost_breakdown.json"
                try:
                    with open(cost_breakdown_path, 'r') as f:
                        cost_data = json.load(f)
                    
                    if 'cost_models' in cost_data:
                        cost_models = cost_data['cost_models']
                        
                        cost_comparison = []
                        for model_name, model_data in cost_models.items():
                            cost_comparison.append({
                                'Cost Model': model_name.replace('_', ' ').title(),
                                'Total Cost': f"${model_data['total_cost']:.4f}",
                                'Model Type': model_data.get('model', 'N/A'),
                                'Savings': f"{model_data.get('savings_vs_simple', 0):.1f}%" if 'savings_vs_simple' in model_data else 'N/A'
                            })
                        
                        st.dataframe(pd.DataFrame(cost_comparison), use_container_width=True)
                        
                        # Visualization
                        fig = go.Figure(data=[
                            go.Bar(
                                x=[c['Cost Model'] for c in cost_comparison],
                                y=[float(c['Total Cost'].replace('$', '')) for c in cost_comparison],
                                text=[c['Total Cost'] for c in cost_comparison],
                                textposition='auto',
                            )
                        ])
                        fig.update_layout(
                            title="Cost Model Comparison",
                            xaxis_title="Cost Model",
                            yaxis_title="Total Cost ($)",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                except FileNotFoundError:
                    st.info("Cost breakdown not available. Run Phase C analysis.")
            else:
                st.warning(f"No advanced metrics found for scenario: {selected_scenario}")
        else:
            st.info("🔹 Run `python run_pipeline.py` to generate advanced metrics.")

# =====================================================================
# MODE 2: MODEL EVALUATION
# =====================================================================

elif visualization_mode == "Model Evaluation":
    model_eval_path = results_dir / "model_evaluation.json"
    
    try:
        with open(model_eval_path, 'r') as f:
            model_evaluation = json.load(f)
    except FileNotFoundError:
        st.error("⚠️ Model evaluation results not found.")
        st.stop()
    
    st.markdown("**Analysis:** Model Forecast Accuracy on Real Historical Data")
    st.subheader("📈 Model Performance Metrics")
    
    if "best_model_per_timeframe" not in model_evaluation:
        st.error("Invalid model evaluation data format.")
        st.stop()
    
    best_per_tf = model_evaluation.get("best_model_per_timeframe", {})
    metrics_by_model = model_evaluation.get("metrics_by_model", {})
    
    # Display best models
    cols = st.columns(3)
    timeframes = ["1m", "5m", "15m"]
    for i, tf in enumerate(timeframes):
        with cols[i]:
            if tf in best_per_tf:
                best = best_per_tf[tf]
                st.metric(
                    f"Best Model ({tf})",
                    best["model"],
                    delta=f"MAPE: {best['mape']:.2%}"
                )
    
    st.markdown("---")
    
    st.subheader("📊 Detailed Metrics by Model")
    
    for model_name in sorted(metrics_by_model.keys()):
        with st.expander(f"**{model_name.upper()}** Model Performance"):
            model_metrics = metrics_by_model[model_name]
            
            rows = []
            for tf in ["1m", "5m", "15m"]:
                if tf in model_metrics:
                    m = model_metrics[tf]
                    rows.append({
                        "Timeframe": tf.upper(),
                        "MAE": f"{m.get('mae', 0):.2f}",
                        "RMSE": f"{m.get('rmse', 0):.2f}",
                        "MAPE": f"{m.get('mape', 0):.2%}",
                        "Files": int(m.get('files', 0))
                    })
            
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

# =====================================================================
# MODE 3: ANOMALY & COST ANALYSIS
# =====================================================================

else:
    st.markdown("**Analysis:** Advanced Anomaly Detection and Cost Optimization")
    
    # Load Phase C results
    anomaly_path = results_dir / "anomaly_analysis.json"
    cost_path = results_dir / "cost_breakdown.json"
    
    anomaly_data = None
    cost_data = None
    
    try:
        with open(anomaly_path, 'r') as f:
            anomaly_data = json.load(f)
    except FileNotFoundError:
        pass
    
    try:
        with open(cost_path, 'r') as f:
            cost_data = json.load(f)
    except FileNotFoundError:
        pass
    
    if not anomaly_data and not cost_data:
        st.error("⚠️ Phase C results not found. Run `python run_pipeline.py --phase-c-only` first.")
        st.stop()
    
    # Create tabs for Phase C
    c_tab1, c_tab2, c_tab3 = st.tabs([
        "🔴 Anomaly Detection",
        "💰 Cost Models",
        "📊 Platform Metrics"
    ])
    
    # =====================================================================
    # C_TAB 1: Anomaly Detection Performance
    # =====================================================================
    
    with c_tab1:
        st.subheader("🔴 Anomaly Detection Performance")
        
        if anomaly_data and 'anomaly_detection' in anomaly_data:
            anomaly_results = anomaly_data['anomaly_detection']
            
            # Metrics cards
            st.markdown("**Detection Performance by Anomaly Type**")
            
            cols = st.columns(len(anomaly_results))
            for i, (anomaly_type, metrics) in enumerate(anomaly_results.items()):
                with cols[i]:
                    st.metric(
                        anomaly_type.replace('_', ' ').title(),
                        f"F1: {metrics['f1_score']:.3f}",
                        delta=f"Precision: {metrics['precision']:.3f}"
                    )
            
            st.markdown("---")
            
            # Detailed table
            st.markdown("**Detailed Metrics**")
            
            detail_data = []
            for anomaly_type, metrics in anomaly_results.items():
                detail_data.append({
                    'Anomaly Type': anomaly_type.replace('_', ' ').title(),
                    'F1 Score': f"{metrics['f1_score']:.3f}",
                    'Precision': f"{metrics['precision']:.3f}",
                    'Recall': f"{metrics['recall']:.3f}",
                    'True Positives': metrics['true_positives'],
                    'False Positives': metrics['false_positives'],
                    'Detection Rate': f"{metrics['detection_rate']:.1%}"
                })
            
            st.dataframe(pd.DataFrame(detail_data), use_container_width=True)
            
            # Visualization - F1 scores comparison
            st.markdown("**F1 Score Comparison**")
            
            fig = go.Figure()
            
            anomaly_types = [k.replace('_', ' ').title() for k in anomaly_results.keys()]
            f1_scores = [v['f1_score'] for v in anomaly_results.values()]
            precisions = [v['precision'] for v in anomaly_results.values()]
            recalls = [v['recall'] for v in anomaly_results.values()]
            
            fig.add_trace(go.Bar(name='F1 Score', x=anomaly_types, y=f1_scores))
            fig.add_trace(go.Bar(name='Precision', x=anomaly_types, y=precisions))
            fig.add_trace(go.Bar(name='Recall', x=anomaly_types, y=recalls))
            
            fig.update_layout(
                barmode='group',
                title="Anomaly Detection Metrics by Type",
                xaxis_title="Anomaly Type",
                yaxis_title="Score",
                yaxis_range=[0, 1],
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.markdown("**📌 Key Insights**")
            
            best_f1 = max(anomaly_results.items(), key=lambda x: x[1]['f1_score'])
            worst_f1 = min(anomaly_results.items(), key=lambda x: x[1]['f1_score'])
            avg_f1 = np.mean([v['f1_score'] for v in anomaly_results.values()])
            
            st.info(f"""
            - **Best Detection**: {best_f1[0].replace('_', ' ').title()} (F1: {best_f1[1]['f1_score']:.3f})
            - **Needs Improvement**: {worst_f1[0].replace('_', ' ').title()} (F1: {worst_f1[1]['f1_score']:.3f})
            - **Average F1 Score**: {avg_f1:.3f}
            - **Recommendation**: {'Excellent detection across all types' if avg_f1 > 0.8 else 'Consider ensemble methods for better accuracy'}
            """)
        else:
            st.warning("No anomaly detection data available.")
    
    # =====================================================================
    # C_TAB 2: Cost Model Comparison
    # =====================================================================
    
    with c_tab2:
        st.subheader("💰 Cost Model Comparison")
        
        if cost_data and 'cost_models' in cost_data:
            cost_models = cost_data['cost_models']
            
            # Cost comparison
            st.markdown("**Total Cost by Model**")
            
            cost_comparison = []
            for model_name, model_info in cost_models.items():
                cost_comparison.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'Total Cost': f"${model_info['total_cost']:.4f}",
                    'Model Type': model_info.get('model', 'N/A'),
                    'Savings vs Simple': f"{model_info.get('savings_vs_simple', 0):.1f}%" if 'savings_vs_simple' in model_info else 'N/A',
                    'Priority': model_info.get('priority', 'N/A').title() if 'priority' in model_info else 'N/A'
                })
            
            st.dataframe(pd.DataFrame(cost_comparison), use_container_width=True)
            
            # Bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=[c['Model'] for c in cost_comparison],
                    y=[float(c['Total Cost'].replace('$', '')) for c in cost_comparison],
                    text=[c['Total Cost'] for c in cost_comparison],
                    textposition='auto',
                    marker=dict(
                        color=[float(c['Total Cost'].replace('$', '')) for c in cost_comparison],
                        colorscale='RdYlGn_r',
                        showscale=True
                    )
                )
            ])
            
            fig.update_layout(
                title="Cost Model Comparison",
                xaxis_title="Cost Model",
                yaxis_title="Total Cost ($)",
                height=450
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Breakdown for cloud model
            if 'cloud_mixed' in cost_models and 'breakdown' in cost_models['cloud_mixed']:
                st.markdown("**Cloud Cost Breakdown (AWS/GCP/Azure Style)**")
                
                breakdown = cost_models['cloud_mixed']['breakdown']
                
                breakdown_df = pd.DataFrame([
                    {'Component': k.replace('_', ' ').title(), 'Cost': f"${v:.4f}"} 
                    for k, v in breakdown.items() if k != 'total'
                ])
                
                st.dataframe(breakdown_df, use_container_width=True)
                
                # Pie chart
                fig = go.Figure(data=[
                    go.Pie(
                        labels=[k.replace('_', ' ').title() for k in breakdown.keys() if k != 'total'],
                        values=[v for k, v in breakdown.items() if k != 'total'],
                        hole=0.3
                    )
                ])
                
                fig.update_layout(title="Cost Breakdown by Component", height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Kubernetes efficiency
            if 'kubernetes' in cost_models:
                st.markdown("**Kubernetes Efficiency Metrics**")
                
                k8s = cost_models['kubernetes']
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Packing Efficiency", f"{k8s.get('packing_efficiency', 0):.1%}")
                with col2:
                    st.metric("Wasted Capacity", f"{k8s.get('wasted_capacity_pct', 0):.1f}%")
                with col3:
                    st.metric("Avg Nodes", f"{k8s.get('avg_nodes', 0):.1f}")
            
            # Recommendations
            st.markdown("**📌 Cost Optimization Recommendations**")
            
            cheapest = min(cost_models.items(), key=lambda x: x[1]['total_cost'])
            most_expensive = max(cost_models.items(), key=lambda x: x[1]['total_cost'])
            
            savings = (most_expensive[1]['total_cost'] - cheapest[1]['total_cost']) / most_expensive[1]['total_cost'] * 100
            
            st.success(f"""
            - **Most Cost-Effective**: {cheapest[0].replace('_', ' ').title()} (${cheapest[1]['total_cost']:.4f})
            - **Highest Cost**: {most_expensive[0].replace('_', ' ').title()} (${most_expensive[1]['total_cost']:.4f})
            - **Potential Savings**: {savings:.1f}% by switching to optimal model
            - **Recommendation**: Use mixed instance types (spot + reserved) for best cost optimization
            """)
        else:
            st.warning("No cost model data available.")
    
    # =====================================================================
    # C_TAB 3: Platform-Specific Metrics
    # =====================================================================
    
    with c_tab3:
        st.subheader("📊 Platform-Specific Metrics")
        
        if cost_data and 'platform_metrics' in cost_data:
            platform_metrics = cost_data['platform_metrics']
            
            # Kubernetes HPA
            if 'kubernetes_hpa' in platform_metrics:
                st.markdown("**Kubernetes HPA Metrics**")
                
                k8s_hpa = platform_metrics['kubernetes_hpa']
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Avg CPU Utilization", f"{k8s_hpa.get('avg_cpu_utilization', 0):.1%}")
                with col2:
                    st.metric("CPU Target Breaches", k8s_hpa.get('cpu_target_breaches', 0))
                with col3:
                    st.metric("HPA Trigger Rate", f"{k8s_hpa.get('hpa_trigger_rate', 0):.1%}")
            
            st.markdown("---")
            
            # AWS Auto Scaling
            if 'aws_auto_scaling' in platform_metrics:
                st.markdown("**AWS Auto Scaling Metrics**")
                
                aws = platform_metrics['aws_auto_scaling']
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Warm-up Time Ratio", f"{aws.get('warm_up_time_ratio', 0):.1%}")
                with col2:
                    st.metric("Cooldown Effectiveness", f"{aws.get('cooldown_effectiveness', 0):.1%}")
                with col3:
                    st.metric("Target Tracking Breaches", aws.get('target_tracking_breaches', 0))
            
            st.markdown("---")
            
            # Summary
            st.markdown("**📌 Platform Insights**")
            
            st.info("""
            **Kubernetes HPA**: Monitors CPU/memory utilization and scales based on target thresholds (default 70%).
            
            **AWS Auto Scaling**: Uses warm-up periods to prevent premature scale-in and cooldown to avoid flapping.
            
            **Best Practice**: Combine HPA for immediate response with predictive scaling for proactive capacity planning.
            """)
        else:
            st.warning("No platform metrics data available.")
    
    # Run Phase C button
    st.markdown("---")
    if st.button("🔄 Re-run Phase C Analysis"):
        st.info("Run: `python run_pipeline.py --phase-c-only` in terminal")

