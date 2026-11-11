"""
Utilidades para la Actividad 04: Clustering Lineal y No Lineal (scikit-learn).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    davies_bouldin_score,
    homogeneity_score,
    silhouette_score,
    v_measure_score,
)

warnings.filterwarnings("ignore")

PALETTE_CLUSTERS = [
    "#0F172A",  # Slate
    "#0D9488",  # Teal
    "#E11D48",  # Rose
    "#D97706",  # Amber
    "#4F46E5",  # Indigo
]
COLOR_NOISE = "#94A3B8"
COLOR_CENTROID = "#DC2626"
COLOR_GRID = "#E2E8F0"
COLOR_TEXT = "#1E293B"


def set_minimalist_style() -> None:
    """Estilo gráfico minimalista limpio."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#CBD5E1",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": COLOR_GRID,
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Liberation Sans", "Arial"],
            "font.size": 9.5,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "text.color": COLOR_TEXT,
            "axes.labelcolor": COLOR_TEXT,
            "legend.frameon": True,
            "legend.facecolor": "white",
            "legend.edgecolor": "#E2E8F0",
            "legend.fontsize": 8,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def evaluate_clustering(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    dataset_name: str,
    inertia: Optional[float] = None,
) -> Dict[str, float | int | str | None]:
    """Calcula métricas externas (ARI, V-Measure) e internas (Silhouette, Davies-Bouldin, WCSS)."""
    unique_labels = set(y_pred)
    n_clusters = len(unique_labels - {-1})
    n_noise = int(np.sum(y_pred == -1))
    noise_pct = float(100.0 * n_noise / len(y_pred))

    ari = float(adjusted_rand_score(y_true, y_pred))
    homogeneity = float(homogeneity_score(y_true, y_pred))
    completeness = float(completeness_score(y_true, y_pred))
    v_measure = float(v_measure_score(y_true, y_pred))

    mask_valid = y_pred != -1
    if n_clusters >= 2 and np.sum(mask_valid) > n_clusters:
        sil_score = float(silhouette_score(X[mask_valid], y_pred[mask_valid]))
        db_index = float(davies_bouldin_score(X[mask_valid], y_pred[mask_valid]))
    else:
        sil_score = float("nan")
        db_index = float("nan")

    return {
        "Dataset": dataset_name,
        "Modelo": model_name,
        "Clusters_Detectados": n_clusters,
        "Puntos_Ruido": n_noise,
        "Ruido_Pct": round(noise_pct, 2),
        "ARI": round(ari, 4),
        "V_Measure": round(v_measure, 4),
        "Homogeneidad": round(homogeneity, 4),
        "Completitud": round(completeness, 4),
        "Silhouette": round(sil_score, 4) if not np.isnan(sil_score) else None,
        "Davies_Bouldin": round(db_index, 4) if not np.isnan(db_index) else None,
        "Inercia_WCSS": round(float(inertia), 2) if inertia is not None else None,
    }


def compute_elbow_and_internal_curves(
    X: np.ndarray, k_range: range = range(1, 10), random_state: int = 42
) -> pd.DataFrame:
    """Evalúa Inercia (WCSS), Silhouette y Davies-Bouldin para k en k_range."""
    records = []
    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=random_state)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels) if k >= 2 else np.nan
        dbi = davies_bouldin_score(X, labels) if k >= 2 else np.nan
        records.append(
            {
                "k": int(k),
                "Inercia_WCSS": float(km.inertia_),
                "Silhouette": float(sil) if not np.isnan(sil) else None,
                "Davies_Bouldin": float(dbi) if not np.isnan(dbi) else None,
            }
        )
    return pd.DataFrame(records)


def plot_scatter_minimal(
    ax: plt.Axes,
    X_2d: np.ndarray,
    labels: np.ndarray,
    title: str,
    centroids_2d: Optional[np.ndarray] = None,
    xlabel: str = "X1",
    ylabel: str = "X2",
    show_legend: bool = False,
) -> None:
    """Gráfico de dispersión minimalista con mínimo texto."""
    unique_labels = sorted(list(set(labels)))
    has_noise = -1 in unique_labels

    for lbl in unique_labels:
        mask = labels == lbl
        if lbl == -1:
            ax.scatter(
                X_2d[mask, 0],
                X_2d[mask, 1],
                c=COLOR_NOISE,
                marker="x",
                s=26,
                alpha=0.8,
                linewidths=1.0,
                label="Ruido",
                zorder=2,
            )
        else:
            color = PALETTE_CLUSTERS[int(lbl) % len(PALETTE_CLUSTERS)]
            ax.scatter(
                X_2d[mask, 0],
                X_2d[mask, 1],
                c=color,
                s=28,
                alpha=0.85,
                edgecolors="white",
                linewidths=0.35,
                label=f"C{lbl}" if show_legend else None,
                zorder=3,
            )

    if centroids_2d is not None:
        ax.scatter(
            centroids_2d[:, 0],
            centroids_2d[:, 1],
            c=COLOR_CENTROID,
            marker="X",
            s=110,
            edgecolors="white",
            linewidths=1.0,
            label="Centroides",
            zorder=5,
        )

    ax.set_title(title, pad=6, fontsize=9.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if show_legend or has_noise or centroids_2d is not None:
        ax.legend(loc="best", frameon=True, framealpha=0.9)


def plot_elbow_and_validation_panel(
    df_curves: pd.DataFrame,
    optimal_k: int,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Panel minimalista: Elbow (WCSS), Silhouette y Davies-Bouldin vs k."""
    set_minimalist_style()
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))

    # 1. Elbow
    axes[0].plot(df_curves["k"], df_curves["Inercia_WCSS"], marker="o", color="#0F172A", lw=1.8, ms=5)
    axes[0].axvline(optimal_k, color="#E11D48", ls="--", lw=1.1, label=f"k={optimal_k}")
    axes[0].set_title("Método del Codo")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("WCSS")
    axes[0].set_xticks(df_curves["k"])
    axes[0].legend()

    # 2. Silhouette
    df_v = df_curves.dropna(subset=["Silhouette"])
    axes[1].plot(df_v["k"], df_v["Silhouette"], marker="s", color="#0D9488", lw=1.8, ms=5)
    axes[1].axvline(optimal_k, color="#E11D48", ls="--", lw=1.1, label=f"k={optimal_k}")
    axes[1].set_title("Silhouette")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Score")
    axes[1].set_xticks(df_v["k"])
    axes[1].legend()

    # 3. Davies-Bouldin
    axes[2].plot(df_v["k"], df_v["Davies_Bouldin"], marker="^", color="#D97706", lw=1.8, ms=5)
    axes[2].axvline(optimal_k, color="#E11D48", ls="--", lw=1.1, label=f"k={optimal_k}")
    axes[2].set_title("Davies-Bouldin")
    axes[2].set_xlabel("k")
    axes[2].set_ylabel("DBI")
    axes[2].set_xticks(df_v["k"])
    axes[2].legend()

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300)
    return fig


def plot_metrics_bar_comparison(
    df_metrics: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Gráfico de barras minimalista de métricas externas e internas."""
    set_minimalist_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4))

    models = df_metrics["Modelo"].tolist()
    x = np.arange(len(models))
    width = 0.25

    ax0 = axes[0]
    b1 = ax0.bar(x - width, df_metrics["ARI"].fillna(0), width, label="ARI", color="#0F172A")
    b2 = ax0.bar(x, df_metrics["V_Measure"].fillna(0), width, label="V-Measure", color="#0D9488")
    b3 = ax0.bar(x + width, df_metrics["Silhouette"].fillna(0), width, label="Silhouette", color="#E11D48")

    ax0.set_title("ARI, V-Measure y Silhouette")
    ax0.set_xticks(x)
    ax0.set_xticklabels(models)
    ax0.set_ylim(-0.08, 1.18)
    ax0.axhline(0, color="#94A3B8", lw=0.7)
    ax0.legend(loc="upper left", ncol=3)

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax0.annotate(
                f"{h:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, max(h, 0)),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

    ax1 = axes[1]
    db_vals = df_metrics["Davies_Bouldin"].fillna(0).values
    b_db = ax1.bar(x, db_vals, width=0.38, color="#D97706", label="DBI")
    ax1.set_title("Davies-Bouldin")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.set_ylim(0, max(db_vals) * 1.22 + 0.1)

    for bar in b_db:
        h = bar.get_height()
        ax1.annotate(
            f"{h:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300)
    return fig
