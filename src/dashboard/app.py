"""Streamlit dashboard for recommendation engine visualization.

Displays ranking metrics, model comparison, A/B test results,
and recommendation quality analysis using synthetic demo data.

Run with: streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def generate_ranking_metrics(seed: int = 42) -> pd.DataFrame:
    """Generate synthetic ranking metrics at different K values."""
    rng = np.random.default_rng(seed)
    models = ["Collaborative (SVD)", "Content-Based", "Hybrid"]
    k_values = [5, 10, 20]
    rows = []
    for model in models:
        base = {"Collaborative (SVD)": 0.30, "Content-Based": 0.25, "Hybrid": 0.35}[
            model
        ]
        for k in k_values:
            scale = 1.0 + (k - 5) * 0.03
            rows.append(
                {
                    "model": model,
                    "k": k,
                    "precision_at_k": round(base * scale + rng.uniform(-0.03, 0.03), 4),
                    "recall_at_k": round(
                        (base - 0.05) * scale + rng.uniform(-0.02, 0.04), 4
                    ),
                    "ndcg_at_k": round(
                        (base + 0.1) * scale + rng.uniform(-0.02, 0.02), 4
                    ),
                }
            )
    return pd.DataFrame(rows)


def generate_rating_accuracy(seed: int = 42) -> pd.DataFrame:
    """Generate synthetic rating prediction accuracy."""
    rng = np.random.default_rng(seed)
    models = ["Collaborative (SVD)", "Content-Based", "Hybrid", "NMF"]
    rows = []
    for model in models:
        rows.append(
            {
                "model": model,
                "rmse": round(rng.uniform(0.85, 1.15), 4),
                "mae": round(rng.uniform(0.65, 0.90), 4),
            }
        )
    return pd.DataFrame(rows)


def generate_ab_test_data(seed: int = 42) -> pd.DataFrame:
    """Generate synthetic A/B test results."""
    rng = np.random.default_rng(seed)
    days = pd.date_range("2024-11-01", periods=28, freq="D")
    rows = []
    for day in days:
        for variant in ["Control (SVD)", "Treatment (Hybrid)"]:
            base_ctr = 0.12 if "Hybrid" in variant else 0.10
            rows.append(
                {
                    "date": day,
                    "variant": variant,
                    "ctr": round(base_ctr + rng.uniform(-0.02, 0.02), 4),
                    "impressions": int(rng.integers(500, 2000)),
                    "clicks": int(rng.integers(50, 250)),
                }
            )
    return pd.DataFrame(rows)


def generate_coverage_stats(seed: int = 42) -> dict:
    """Generate synthetic recommendation coverage statistics."""
    rng = np.random.default_rng(seed)
    return {
        "user_coverage": round(rng.uniform(0.85, 0.98), 4),
        "item_coverage": round(rng.uniform(0.45, 0.75), 4),
        "cold_start_users": int(rng.integers(50, 200)),
        "cold_start_items": int(rng.integers(100, 500)),
        "total_users": int(rng.integers(900, 1100)),
        "total_items": int(rng.integers(1500, 1800)),
    }


def render_header() -> None:
    """Render the dashboard header."""
    st.title("Recommendation Engine Dashboard")
    st.caption(
        "Hybrid recommendation system with collaborative filtering, "
        "content-based, and cold-start strategies on MovieLens data"
    )


def render_summary_metrics(rating_df: pd.DataFrame, coverage: dict) -> None:
    """Render top-level summary metric cards."""
    col1, col2, col3, col4 = st.columns(4)
    best = rating_df.loc[rating_df["rmse"].idxmin()]
    col1.metric("Best Model", best["model"].split("(")[0].strip())
    col2.metric("Best RMSE", f"{best['rmse']:.4f}")
    col3.metric("User Coverage", f"{coverage['user_coverage']:.0%}")
    col4.metric("Item Coverage", f"{coverage['item_coverage']:.0%}")


def render_ranking_comparison(rank_df: pd.DataFrame) -> None:
    """Render ranking metrics comparison."""
    st.subheader("Ranking Metrics by Model")
    metric = st.selectbox(
        "Select metric:",
        ["precision_at_k", "recall_at_k", "ndcg_at_k"],
    )
    fig = px.bar(
        rank_df,
        x="k",
        y=metric,
        color="model",
        barmode="group",
        text=metric,
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="auto")
    fig.update_layout(
        xaxis_title="K",
        yaxis_title=metric.replace("_", " ").title(),
        height=400,
        margin={"l": 40, "r": 20, "t": 30, "b": 40},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_rating_accuracy(rating_df: pd.DataFrame) -> None:
    """Render RMSE/MAE comparison."""
    st.subheader("Rating Prediction Accuracy")
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="RMSE",
            x=rating_df["model"],
            y=rating_df["rmse"],
            text=rating_df["rmse"].apply(lambda x: f"{x:.3f}"),
            textposition="auto",
        )
    )
    fig.add_trace(
        go.Bar(
            name="MAE",
            x=rating_df["model"],
            y=rating_df["mae"],
            text=rating_df["mae"].apply(lambda x: f"{x:.3f}"),
            textposition="auto",
        )
    )
    fig.update_layout(
        barmode="group",
        yaxis_title="Error",
        height=350,
        margin={"l": 40, "r": 20, "t": 30, "b": 40},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_ab_test(ab_df: pd.DataFrame) -> None:
    """Render A/B test CTR trend."""
    st.subheader("A/B Test: Click-Through Rate")
    fig = px.line(
        ab_df,
        x="date",
        y="ctr",
        color="variant",
        markers=True,
    )
    fig.update_layout(
        yaxis_title="CTR",
        yaxis={"tickformat": ".1%"},
        height=350,
        margin={"l": 40, "r": 20, "t": 30, "b": 40},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_coverage(coverage: dict) -> None:
    """Render coverage and cold-start statistics."""
    st.subheader("Coverage & Cold-Start Analysis")
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=coverage["user_coverage"] * 100,
                title={"text": "User Coverage %"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#2196F3"},
                    "steps": [
                        {"range": [0, 60], "color": "#FFCDD2"},
                        {"range": [60, 80], "color": "#FFF9C4"},
                        {"range": [80, 100], "color": "#C8E6C9"},
                    ],
                },
            )
        )
        fig.update_layout(height=250, margin={"l": 20, "r": 20, "t": 40, "b": 20})
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.metric("Cold-Start Users", f"{coverage['cold_start_users']:,}")
        st.metric("Cold-Start Items", f"{coverage['cold_start_items']:,}")
        st.metric("Total Users", f"{coverage['total_users']:,}")
        st.metric("Total Items", f"{coverage['total_items']:,}")


def main() -> None:
    """Main dashboard entry point."""
    render_header()

    rank_df = generate_ranking_metrics()
    rating_df = generate_rating_accuracy()
    ab_df = generate_ab_test_data()
    coverage = generate_coverage_stats()

    render_summary_metrics(rating_df, coverage)
    st.markdown("---")

    render_ranking_comparison(rank_df)

    col_left, col_right = st.columns(2)
    with col_left:
        render_rating_accuracy(rating_df)
    with col_right:
        render_ab_test(ab_df)

    st.markdown("---")
    render_coverage(coverage)


if __name__ == "__main__":
    main()
