import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from pandas.plotting import lag_plot
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import seaborn as sns
import calplot

st.set_page_config(layout="wide")
st.title("📈 Time Series Analysis & Anomaly Detection")

# --- File upload ---
file = st.file_uploader("Upload your time series CSV file", type=["csv"])

if file:
    date_format = st.radio("Does your date use day-first format?", options=[True, False], format_func=lambda x: "Day First (DD/MM/YYYY)" if x else "Month First (MM/DD/YYYY)")
    df = pd.read_csv(file)
    st.success("File successfully uploaded!")

    # Detect date column
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=date_format)
    df = df.sort_values(by=date_col)
    df.set_index(date_col, inplace=True)

    st.sidebar.header("Options")
    all_columns = df.columns.tolist()

    # Detect categorical
    categorical_col = None
    for col in all_columns:
        if df[col].dtype == 'object' or df[col].nunique() < 20:
            categorical_col = col
            break

    hue = None
    if categorical_col:
        use_hue = st.sidebar.checkbox(f"Use '{categorical_col}' as hue (long format)?")
        if use_hue:
            hue = categorical_col
            df = df.pivot_table(index=df.index, columns=hue, values=[col for col in df.columns if col != hue])
            df.columns = ['_'.join(map(str, col)).strip() for col in df.columns.values]

    selected_col = st.sidebar.selectbox("Select a variable to analyze", [col for col in df.columns if np.issubdtype(df[col].dtype, np.number)])

    analysis_section = st.sidebar.radio("Choose Analysis Section", ["Time Series Visualization", "Decomposition & ACF/PACF", "Anomaly Detection", "Temporal Profiles", "Calendar Heatmap", "Monthly Calendar"])

    df_selected = df[[selected_col]].copy()

    if analysis_section == "Time Series Visualization":
        st.subheader("📈 Time Series Visualization")
        time_granularity = st.selectbox("Select summarization frequency", ["None", "Daily", "Weekly", "Monthly", "Quarterly", "Yearly"])
        window = st.number_input("Rolling Window (set 0 for no smoothing)", min_value=0, max_value=60, value=0)

        resample_map = {
            "None": None,
            "Daily": "D",
            "Weekly": "W",
            "Monthly": "M",
            "Quarterly": "Q",
            "Yearly": "Y"
        }
        resample_freq = resample_map[time_granularity]

        if resample_freq:
            resampled = df_selected.resample(resample_freq).agg(['mean', 'std'])
            resampled.columns = ['mean', 'std']
            if window > 0:
                resampled['mean'] = resampled['mean'].rolling(window).mean()
                resampled['std'] = resampled['std'].rolling(window).mean()
            df_summary = resampled
        else:
            df_summary = df_selected.copy()
            df_summary['mean'] = df_summary[selected_col].rolling(window).mean() if window > 0 else df_summary[selected_col]
            df_summary['std'] = df_summary[selected_col].rolling(window).std() if window > 0 else df_summary[selected_col].std()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_summary.index, y=df_summary['mean'], mode='lines', name='Mean'))
        fig.add_trace(go.Scatter(x=df_summary.index, y=df_summary['mean'] + df_summary['std'], mode='lines', name='Upper Band', line=dict(dash='dot')))
        fig.add_trace(go.Scatter(x=df_summary.index, y=df_summary['mean'] - df_summary['std'], mode='lines', name='Lower Band', line=dict(dash='dot')))
        fig.update_layout(title="Summarized Time Series", xaxis_title="Date", yaxis_title=selected_col)
        st.plotly_chart(fig, use_container_width=True)

    elif analysis_section == "Decomposition & ACF/PACF":
        st.subheader("📉 Seasonal Decomposition")
        window = st.slider("Seasonal window for decomposition", 2, 60, 12)
        try:
            result = seasonal_decompose(df[selected_col], model='additive', period=window)
            fig = result.plot()
            st.pyplot(fig)
        except:
            st.warning("Decomposition failed — try adjusting the window size.")

        st.subheader("🔁 ACF and PACF")
        fig_acf, ax = plt.subplots(1, 2, figsize=(12, 4))
        plot_acf(df[selected_col].dropna(), ax=ax[0])
        plot_pacf(df[selected_col].dropna(), ax=ax[1])
        ax[0].set_title("ACF")
        ax[1].set_title("PACF")
        st.pyplot(fig_acf)

    elif analysis_section == "Anomaly Detection":
        st.subheader("🚨 Anomaly Detection (Isolation Forest)")
        contamination = st.slider("Contamination (expected anomaly fraction)", 0.01, 0.5, 0.05, 0.01)
        df_anom = df_selected.copy()
        df_anom['anomaly'] = IsolationForest(contamination=contamination, random_state=42).fit_predict(df_anom[[selected_col]])
        df_anom['anomaly'] = df_anom['anomaly'].map({1: 0, -1: 1})

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_anom.index, y=df_anom[selected_col], mode='lines', name='Value'))
        fig.add_trace(go.Scatter(x=df_anom[df_anom['anomaly'] == 1].index,
                                 y=df_anom[df_anom['anomaly'] == 1][selected_col],
                                 mode='markers', name='Anomaly', marker=dict(color='red', size=8)))
        fig.update_layout(title="Anomaly Detection", xaxis_title="Date", yaxis_title=selected_col)
        st.plotly_chart(fig, use_container_width=True)

    elif analysis_section == "Temporal Profiles":
        st.subheader("🕒 Temporal Profiles")
        profile_unit = st.selectbox("Profile by", ["Day of Week", "Month", "Quarter", "Year"])
        df_profile = df_selected.copy()

        if profile_unit == "Day of Week":
            df_profile['profile'] = df_profile.index.dayofweek
        elif profile_unit == "Month":
            df_profile['profile'] = df_profile.index.month
        elif profile_unit == "Quarter":
            df_profile['profile'] = df_profile.index.quarter
        else:
            df_profile['profile'] = df_profile.index.year

        plot_type = st.radio("Plot type", ["Boxplot", "Lineplot"], horizontal=True)
        if plot_type == "Boxplot":
            fig = px.box(df_profile, x='profile', y=selected_col, title=f"Boxplot by {profile_unit}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            profile_summary = df_profile.groupby('profile')[selected_col].agg(['mean', 'std'])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=profile_summary.index, y=profile_summary['mean'], mode='lines+markers', name='Mean'))
            fig.add_trace(go.Scatter(x=profile_summary.index, y=profile_summary['mean'] + profile_summary['std'], mode='lines', name='Upper', line=dict(dash='dot')))
            fig.add_trace(go.Scatter(x=profile_summary.index, y=profile_summary['mean'] - profile_summary['std'], mode='lines', name='Lower', line=dict(dash='dot')))
            fig.update_layout(title=f"Profile by {profile_unit}", xaxis_title=profile_unit, yaxis_title=selected_col)
            st.plotly_chart(fig, use_container_width=True)

    elif analysis_section == "Calendar Heatmap":
        st.subheader("📆 Calendar Heatmap (Daily)")
        fig, ax = calplot.calplot(df_selected[selected_col])
        st.pyplot(fig)

    elif analysis_section == "Monthly Calendar":
        st.subheader("📅 Monthly Calendar Heatmap")
        df_month = df_selected.copy()
        df_month['month'] = df_month.index.month
        df_month['year'] = df_month.index.year
        heatmap_data = df_month.groupby(['year', 'month'])[selected_col].mean().unstack()

        fig = px.imshow(heatmap_data,
                        labels=dict(x="Month", y="Year", color=selected_col),
                        x=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                        y=heatmap_data.index,
                        aspect="auto",
                        color_continuous_scale="Viridis")
        fig.update_layout(title="Monthly Heatmap", xaxis_title="Month", yaxis_title="Year")
        st.plotly_chart(fig, use_container_width=True)
