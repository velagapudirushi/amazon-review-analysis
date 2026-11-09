"""
Visualization Module for Amazon Review Sentiment Analysis using Plotly.
Produces clean, publication-grade interactive charts.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


AMAZON_ORANGE = "#FF9900"
AMAZON_NAVY = "#146EB4"
AMAZON_DARK = "#232F3E"
SENTIMENT_COLORS = {
    "Positive": "#28a745",
    "Neutral": "#ffc107",
    "Negative": "#dc3545"
}


def plot_rating_distribution(df: pd.DataFrame) -> go.Figure:
    """Creates a stylized bar chart of review ratings (1 to 5)."""
    counts = df["Score"].value_counts().sort_index().reset_index()
    counts.columns = ["Rating", "Count"]
    counts["Percentage"] = (counts["Count"] / counts["Count"].sum() * 100).round(1)

    fig = px.bar(
        counts,
        x="Rating",
        y="Count",
        text=counts["Percentage"].apply(lambda x: f"{x}%"),
        color="Rating",
        color_discrete_sequence=["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#27ae60"],
        title="Distribution of Review Ratings (Scores 1 to 5)"
    )
    fig.update_traces(textposition="outside", hovertemplate="<b>Rating %{x} Stars</b><br>Reviews: %{y:,}<br>Share: %{text}<extra></extra>")
    fig.update_layout(
        template="plotly_white",
        xaxis=dict(title="Star Rating", tickmode="linear", tick0=1, dtick=1),
        yaxis=dict(title="Number of Reviews"),
        showlegend=False,
        height=400
    )
    return fig


def plot_rating_pie(df: pd.DataFrame) -> go.Figure:
    """Creates a donut chart of rating proportions."""
    counts = df["Score"].value_counts().sort_index().reset_index()
    counts.columns = ["Rating", "Count"]
    labels = [f"{r} Star{'s' if r > 1 else ''}" for r in counts["Rating"]]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=counts["Count"],
        hole=0.45,
        marker=dict(colors=["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#27ae60"]),
        textinfo="label+percent",
        insidetextorientation="radial"
    )])
    fig.update_layout(
        title="Rating Percentage Share",
        template="plotly_white",
        height=400,
        showlegend=True
    )
    return fig


def plot_top_words_comparison(positive_series: pd.Series, negative_series: pd.Series, top_n: int = 10) -> go.Figure:
    """Side-by-side comparison of top frequent terms in positive vs negative reviews."""
    pos_top = positive_series.head(top_n).reset_index()
    pos_top.columns = ["Word", "Count"]

    neg_top = negative_series.head(top_n).reset_index()
    neg_top.columns = ["Word", "Count"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=pos_top["Word"],
        y=pos_top["Count"],
        name="Positive Reviews",
        marker_color="#2ecc71"
    ))
    fig.add_trace(go.Bar(
        x=neg_top["Word"],
        y=neg_top["Count"],
        name="Negative Reviews",
        marker_color="#e74c3c"
    ))
    fig.update_layout(
        title=f"Top {top_n} Frequent Words: Positive vs. Negative Reviews",
        barmode="group",
        template="plotly_white",
        xaxis_title="Word / Token",
        yaxis_title="Frequency",
        height=420
    )
    return fig


def plot_review_length_distribution(df: pd.DataFrame) -> go.Figure:
    """Histogram of review word counts with KDE approximation."""
    if "ReviewLength" not in df.columns:
        lengths = df["Text"].astype(str).apply(lambda x: len(x.split()))
    else:
        lengths = df["ReviewLength"]

    # Cap display at 99th percentile for clean visualization
    cap = int(np.percentile(lengths, 99)) if len(lengths) > 0 else 300

    fig = px.histogram(
        x=lengths[lengths <= cap],
        nbins=40,
        title="Review Length Distribution (Word Count)",
        color_discrete_sequence=[AMAZON_NAVY],
        marginal="box"
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Review Length (Words)",
        yaxis_title="Count of Reviews",
        height=400
    )
    return fig


def plot_avg_length_by_rating(df: pd.DataFrame) -> go.Figure:
    """Bar chart showing average review length across 1-5 ratings."""
    if "ReviewLength" not in df.columns:
        df = df.copy()
        df["ReviewLength"] = df["Text"].astype(str).apply(lambda x: len(x.split()))

    avg_df = df.groupby("Score")["ReviewLength"].mean().reset_index()
    avg_df["ReviewLength"] = avg_df["ReviewLength"].round(1)

    fig = px.bar(
        avg_df,
        x="Score",
        y="ReviewLength",
        text="ReviewLength",
        color="Score",
        color_discrete_sequence=["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#27ae60"],
        title="Average Review Word Length by Star Rating"
    )
    fig.update_traces(textposition="outside", hovertemplate="<b>Score %{x}</b><br>Avg Words: %{y}<extra></extra>")
    fig.update_layout(
        template="plotly_white",
        xaxis=dict(title="Star Rating", tickmode="linear", tick0=1, dtick=1),
        yaxis=dict(title="Average Words per Review"),
        showlegend=False,
        height=400
    )
    return fig


def plot_helpfulness_by_rating(df: pd.DataFrame) -> go.Figure:
    """Box plot of helpfulness ratio across star ratings."""
    if "HelpfulnessRatio" not in df.columns and "HelpfulnessNumerator" in df.columns and "HelpfulnessDenominator" in df.columns:
        df = df.copy()
        denom = df["HelpfulnessDenominator"].replace(0, float("nan"))
        df["HelpfulnessRatio"] = df["HelpfulnessNumerator"] / denom

    filtered = df.dropna(subset=["HelpfulnessRatio", "Score"])
    fig = px.box(
        filtered,
        x="Score",
        y="HelpfulnessRatio",
        color="Score",
        color_discrete_sequence=["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#27ae60"],
        title="Helpfulness Ratio Distribution by Rating Score"
    )
    fig.update_layout(
        template="plotly_white",
        xaxis=dict(title="Star Rating", tickmode="linear", tick0=1, dtick=1),
        yaxis=dict(title="Helpfulness Ratio (0.0 to 1.0)"),
        showlegend=False,
        height=400
    )
    return fig


def plot_prediction_probabilities(probabilities: Dict[str, float]) -> go.Figure:
    """Bar chart of model prediction confidence across 1-5 star classes."""
    classes = list(probabilities.keys())
    scores = [v * 100 for v in probabilities.values()]

    colors = []
    for cls in classes:
        if "1" in cls or "2" in cls:
            colors.append("#dc3545")
        elif "3" in cls:
            colors.append("#ffc107")
        else:
            colors.append("#28a745")

    fig = go.Figure(go.Bar(
        x=classes,
        y=scores,
        text=[f"{s:.1f}%" for s in scores],
        textposition="outside",
        marker_color=colors
    ))
    fig.update_layout(
        title="Class Confidence Probability Distribution",
        template="plotly_white",
        xaxis_title="Predicted Rating",
        yaxis_title="Probability (%)",
        yaxis=dict(range=[0, max(scores) * 1.25 if scores else 100]),
        height=320,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig


def plot_confusion_matrix(cm: List[List[int]], classes: List[int]) -> go.Figure:
    """Heatmap of confusion matrix."""
    z = np.array(cm)
    labels = [f"{c} Star" for c in classes]

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=labels,
        y=labels,
        colorscale="Blues",
        text=z,
        texttemplate="%{text}",
        textfont={"size": 14},
        hoverongaps=False
    ))
    fig.update_layout(
        title="Model Confusion Matrix",
        xaxis_title="Predicted Class",
        yaxis_title="True Class",
        template="plotly_white",
        height=380
    )
    return fig
