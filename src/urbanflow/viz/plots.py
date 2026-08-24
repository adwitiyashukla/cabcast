from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from urbanflow.viz.style import (
    AQUA,
    AXIS,
    BLUE,
    DIVERGING_CMAP,
    GRID,
    NEUTRAL,
    ORANGE,
    RED,
    SEQ_CMAP,
    SERIES,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    YELLOW,
    apply_style,
    finish,
    title_block,
)

DOW_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def demand_choropleth(zones_gdf, panel: pd.DataFrame, out: Path) -> Path:
    apply_style()
    mean_demand = panel.groupby("zone_id", observed=True)["trips"].mean()
    gdf = zones_gdf.copy()
    gdf["mean_trips"] = gdf["zone_id"].map(mean_demand)

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.6))
    modelled = gdf[gdf["mean_trips"].notna()]
    excluded = gdf[gdf["mean_trips"].isna()]

    for ax, use_log in zip(axes, (False, True), strict=True):
        values = modelled["mean_trips"]
        plot_values = np.log10(values + 1.0) if use_log else values
        excluded.plot(ax=ax, color="#efeeea", edgecolor="#dedcd6", linewidth=0.4)
        modelled.assign(v=plot_values).plot(
            column="v", ax=ax, cmap=SEQ_CMAP, edgecolor=NEUTRAL, linewidth=0.35, legend=True,
            legend_kwds={"shrink": 0.62, "pad": 0.01},
        )
        ax.set_axis_off()
        label = "log10(mean trips + 1)" if use_log else "mean trips per hour"
        ax.set_title(
            f"Mean hourly pickups by zone, {label}", color=TEXT_PRIMARY, fontsize=11.5,
            loc="left", pad=10,
        )

    fig.suptitle(
        "Demand is extremely concentrated: a linear scale shows only Midtown, a log scale reveals the full gradient",
        fontsize=12.5, color=TEXT_PRIMARY, x=0.012, ha="left", y=1.02, weight="semibold",
    )
    return finish(
        fig, out,
        f"{len(modelled)} modelled zones; grey zones fall below the minimum-demand threshold. "
        "Sequential single-hue ramp, light to dark.",
    )


def demand_heatmap(panel: pd.DataFrame, out: Path, top_n: int = 28) -> Path:
    apply_style()
    totals = panel.groupby("zone_id", observed=True)["trips"].sum().nlargest(top_n)
    sub = panel[panel["zone_id"].isin(totals.index)].copy()
    sub["hour_of_week"] = sub["hour_ts"].dt.dayofweek * 24 + sub["hour_ts"].dt.hour

    grid = sub.pivot_table(
        index="zone_id", columns="hour_of_week", values="trips", aggfunc="mean", observed=True
    ).reindex(index=totals.index, columns=range(168)).fillna(0.0)
    normed = grid.div(grid.max(axis=1).replace(0, 1), axis=0)

    fig, ax = plt.subplots(figsize=(14.2, 7.0))
    im = ax.imshow(normed.to_numpy(), aspect="auto", cmap=SEQ_CMAP, interpolation="nearest")
    ax.set_yticks(range(len(totals)))
    ax.set_yticklabels([f"zone {z}" for z in totals.index], fontsize=8)
    ax.set_xticks([d * 24 + 12 for d in range(7)])
    ax.set_xticklabels(DOW_LABELS)
    for d in range(1, 7):
        ax.axvline(d * 24 - 0.5, color="#ffffff", linewidth=1.2, alpha=0.85)
    ax.grid(False)
    ax.set_xlabel("hour of week")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.012)
    cbar.set_label("demand relative to each zone's own weekly peak", color=TEXT_SECONDARY, fontsize=9.5)
    title_block(
        ax,
        "Every zone has its own weekly rhythm",
        f"Top {top_n} zones by volume, each row scaled to its own maximum so shape is comparable across zones",
    )
    return finish(
        fig, out,
        "Commuter zones show twin weekday peaks; nightlife zones peak after midnight on Friday and Saturday.",
    )


def zone_network(zones_gdf, graph, out: Path) -> Path:
    apply_style()
    xy = np.c_[zones_gdf["centroid_lon"].to_numpy(), zones_gdf["centroid_lat"].to_numpy()]
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.4))

    rows, cols = graph.adjacency.nonzero()
    mask = rows < cols
    for ax, values, label in zip(
        axes,
        (graph.centrality["graph_betweenness"].to_numpy(), graph.eigenmaps[:, 0]),
        ("betweenness centrality", "1st non-trivial Laplacian eigenvector"),
        strict=True,
    ):
        for r, c in zip(rows[mask], cols[mask], strict=True):
            ax.plot(xy[[r, c], 0], xy[[r, c], 1], color=GRID, linewidth=0.6, zorder=1)
        cmap = SEQ_CMAP if "betweenness" in label else DIVERGING_CMAP
        vmax = np.abs(values).max() if "eigen" in label else None
        sc = ax.scatter(
            xy[:, 0], xy[:, 1], c=values, cmap=cmap, s=42, zorder=2,
            edgecolor="#ffffff", linewidth=0.7,
            vmin=-vmax if vmax else None, vmax=vmax if vmax else None,
        )
        fig.colorbar(sc, ax=ax, shrink=0.72, pad=0.012)
        ax.set_title(f"Zones coloured by {label}", color=TEXT_PRIMARY, fontsize=11.5, loc="left", pad=10)
        ax.set_axis_off()

    fig.suptitle(
        "The zone adjacency graph: bridges score high on betweenness, and the leading eigenvector splits the city geographically",
        fontsize=12.5, color=TEXT_PRIMARY, x=0.012, ha="left", y=1.02, weight="semibold",
    )
    return finish(
        fig, out,
        f"{graph.n} nodes, {int(graph.adjacency.nnz // 2)} edges. Eigenvector uses a diverging ramp because sign carries meaning.",
    )


def rebalancing_map(zones_gdf, zone_ids, moves, supply, demand, out: Path, top_flows: int = 90) -> Path:
    apply_style()
    lookup = zones_gdf.set_index("zone_id")
    xy = np.c_[
        lookup.loc[zone_ids, "centroid_lon"].to_numpy(),
        lookup.loc[zone_ids, "centroid_lat"].to_numpy(),
    ]
    flows = moves.copy()
    np.fill_diagonal(flows, 0.0)
    flat = np.dstack(np.unravel_index(np.argsort(flows, axis=None)[::-1], flows.shape))[0][:top_flows]

    fig, ax = plt.subplots(figsize=(9.0, 8.4))
    zones_gdf.plot(ax=ax, color="#f4f3ef", edgecolor="#dcdad3", linewidth=0.5)

    gap = demand / demand.sum() - supply / supply.sum()
    scale = np.abs(gap).max()
    ax.scatter(
        xy[:, 0], xy[:, 1], c=gap, cmap=DIVERGING_CMAP, vmin=-scale, vmax=scale,
        s=70, zorder=3, edgecolor="#ffffff", linewidth=0.8,
    )
    peak = flows[flat[:, 0], flat[:, 1]].max()
    for i, j in flat:
        w = flows[i, j] / peak
        if w < 0.06:
            continue
        ax.annotate(
            "", xy=(xy[j, 0], xy[j, 1]), xytext=(xy[i, 0], xy[i, 1]),
            arrowprops={
                "arrowstyle": "-|>", "color": ORANGE, "alpha": 0.30 + 0.55 * w,
                "linewidth": 0.7 + 2.6 * w, "shrinkA": 5, "shrinkB": 5,
                "connectionstyle": "arc3,rad=0.13",
            },
            zorder=2,
        )
    active = np.unique(np.concatenate([flat[:, 0], flat[:, 1]]))
    pad_x = max((xy[active, 0].max() - xy[active, 0].min()) * 0.28, 0.02)
    pad_y = max((xy[active, 1].max() - xy[active, 1].min()) * 0.16, 0.02)
    ax.set_xlim(xy[active, 0].min() - pad_x, xy[active, 0].max() + pad_x)
    ax.set_ylim(xy[active, 1].min() - pad_y, xy[active, 1].max() + pad_y)

    ax.set_axis_off()
    handles = [
        Line2D([], [], color=ORANGE, linewidth=2.4, label="recommended vehicle flow"),
        Line2D([], [], marker="o", color="none", markerfacecolor="#e34948", markersize=9, label="demand exceeds idle supply"),
        Line2D([], [], marker="o", color="none", markerfacecolor="#2a78d6", markersize=9, label="idle supply exceeds demand"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=9.5, framealpha=0.0)
    ax.set_title(
        "Optimal-transport rebalancing plan for the peak forecast hour",
        color=TEXT_PRIMARY, fontsize=12.5, loc="left", pad=14,
    )
    return finish(
        fig, out,
        f"Zoomed to the zones carrying flow. Top {top_flows} moves from the entropic-regularised "
        "Sinkhorn coupling; arrow width is proportional to vehicles moved.",
    )


def forecast_intervals(frame: pd.DataFrame, out: Path, n_zones: int = 3, days: int = 7) -> Path:
    apply_style()
    volumes = frame.groupby("zone_id", observed=True)["y"].sum().nlargest(n_zones)
    end = frame["hour_ts"].max()
    start = end - pd.Timedelta(days=days)

    fig, axes = plt.subplots(n_zones, 1, figsize=(13.6, 3.05 * n_zones), sharex=True)
    axes = np.atleast_1d(axes)

    for ax, zone in zip(axes, volumes.index, strict=True):
        sub = frame[(frame["zone_id"] == zone) & (frame["hour_ts"] > start)].sort_values("hour_ts")
        ax.fill_between(
            sub["hour_ts"], sub["lower"], sub["upper"], color=BLUE, alpha=0.16, linewidth=0,
            label="90% conformal interval",
        )
        ax.plot(sub["hour_ts"], sub["y"], color=TEXT_PRIMARY, linewidth=1.7, label="actual")
        ax.plot(sub["hour_ts"], sub["yhat"], color=ORANGE, linewidth=1.7, linestyle="--", label="forecast")
        missed = sub[(sub["y"] < sub["lower"]) | (sub["y"] > sub["upper"])]
        ax.scatter(missed["hour_ts"], missed["y"], color=RED, s=26, zorder=4, label="outside interval")
        cov = 1.0 - len(missed) / max(len(sub), 1)
        ax.set_ylabel("trips per hour")
        ax.text(
            0.996, 0.93, f"zone {zone}   empirical coverage {cov:.1%}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9.5, color=TEXT_MUTED,
        )
    axes[0].legend(loc="upper left", ncol=4)
    axes[0].set_title(
        "Forecasts with calibrated uncertainty on held-out test data",
        color=TEXT_PRIMARY, fontsize=12.5, loc="left",
    )
    axes[-1].set_xlabel("")
    return finish(
        fig, out,
        "The band is a Mondrian conformalised quantile interval: its width adapts to the demand level rather than being constant.",
    )


def model_comparison(point_metrics: dict, out: Path) -> Path:
    apply_style()
    names = list(point_metrics)
    mae = [point_metrics[n]["mae"] for n in names]
    order = np.argsort(mae)[::-1]
    names = [names[i].replace("_", " ") for i in order]
    mae = [mae[i] for i in order]
    best = int(np.argmin(mae))
    colors = ["#c7d9f0"] * len(mae)
    colors[best] = BLUE

    fig, ax = plt.subplots(figsize=(10.4, 0.72 * len(names) + 2.3))
    bars = ax.barh(names, mae, color=colors, height=0.62)
    for bar, value in zip(bars, mae, strict=True):
        ax.text(
            bar.get_width() + max(mae) * 0.012, bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}", va="center", fontsize=10, color=TEXT_PRIMARY,
        )
    lift = (mae[0] - mae[best]) / mae[0] * 100.0
    ax.set_xlim(0, max(mae) * 1.16)
    ax.set_xlabel("test MAE, trips per zone-hour (lower is better)")
    ax.grid(axis="y", visible=False)
    title_block(
        ax, "LightGBM against three transparent baselines",
        f"Measured on the untouched final test block. Best model cuts MAE {lift:.1f}% versus the weakest baseline.",
    )
    return finish(fig, out, "Single-hue magnitude encoding with the winner highlighted; every bar carries its value.")


def backtest_folds(per_fold: list[dict], out: Path) -> Path:
    apply_style()
    df = pd.DataFrame(per_fold)
    models = ["seasonal_naive", "drifted_seasonal_naive", "historical_mean", "lightgbm"]
    colors = [SERIES[3], SERIES[2], SERIES[1], SERIES[0]]

    fig, ax = plt.subplots(figsize=(11.4, 5.8))
    x = np.arange(len(df))
    for model, color in zip(models, colors, strict=True):
        col = f"{model}.mae"
        if col not in df.columns:
            continue
        ax.plot(x, df[col], marker="o", color=color, label=model.replace("_", " "), linewidth=2.0)
        ax.annotate(
            f"{df[col].iloc[-1]:.3f}", (x[-1], df[col].iloc[-1]), xytext=(9, 0),
            textcoords="offset points", va="center", fontsize=9.5, color=color, weight="semibold",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(df["fold"].str.replace("_", " "))
    ax.set_ylabel("MAE, trips per zone-hour")
    ax.set_xlim(-0.25, len(df) - 0.45)
    ax.legend(ncol=4, loc="upper left")
    title_block(
        ax, "Rolling-origin backtest: the ranking is stable across folds",
        "Each fold trains only on data preceding its validation window, with an embargo gap equal to the longest feature lag",
    )
    return finish(fig, out, "Four series on an adjacent-pair validated palette, each direct-labelled at its final value.")


def coverage_calibration(curves: dict[str, pd.DataFrame], out: Path) -> Path:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 6.4), sharey=True)
    panels = [
        ("empirical", "All zone-hours", "Most cells are zero demand, so any interval covers them"),
        ("empirical_active", "Cells with non-zero demand", "The cells where the interval has to do real work"),
    ]

    floor = min(c[col].min() for c in curves.values() for col in ("empirical", "empirical_active"))
    for ax, (column, heading, note) in zip(axes, panels, strict=True):
        ax.plot([0.66, 0.99], [0.66, 0.99], color=AXIS, linewidth=1.3, linestyle=":", zorder=1)
        for (name, curve), color in zip(curves.items(), (SERIES[0], SERIES[1], SERIES[2]), strict=False):
            ax.plot(
                curve["target"], curve[column], marker="o", color=color,
                label=name.replace("_", " "), linewidth=2.0, zorder=3,
            )
            ax.annotate(
                f"{curve[column].iloc[-1]:.3f}",
                (curve["target"].iloc[-1], curve[column].iloc[-1]),
                xytext=(8, -2), textcoords="offset points", fontsize=9, color=color, weight="semibold",
            )
        ax.set_xlabel("nominal coverage")
        ax.set_ylim(floor - 0.03, 1.01)
        ax.set_title(heading, color=TEXT_PRIMARY, fontsize=11.5, loc="left", pad=26)
        ax.text(0.0, 1.012, note, transform=ax.transAxes, fontsize=9.2, color=TEXT_MUTED, va="bottom")

    axes[0].set_ylabel("empirical coverage on test data")
    axes[0].legend(loc="upper left")
    fig.suptitle(
        "Calibration: the dotted diagonal is perfect coverage",
        fontsize=12.5, color=TEXT_PRIMARY, x=0.012, ha="left", y=1.035, weight="semibold",
    )
    return finish(
        fig, out,
        "Split conformal and CQR carry distribution-free finite-sample guarantees; the raw quantile model carries none.",
    )


def conditional_coverage(records: list[dict], target: float, out: Path) -> Path:
    apply_style()
    df = pd.DataFrame(records)
    x = np.arange(len(df))
    width = 0.38

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(10.8, 8.2), sharex=True, gridspec_kw={"height_ratios": [2.1, 1]}
    )
    ax.bar(x - width / 2, df["coverage_cqr"], width, color=SERIES[1], label="global CQR")
    ax.bar(x + width / 2, df["coverage_mondrian"], width, color=SERIES[0], label="Mondrian CQR")
    ax.axhline(target, color=TEXT_PRIMARY, linewidth=1.4, linestyle="--")
    ax.text(len(df) - 0.42, target + 0.004, f"target {target:.0%}", fontsize=9.5, color=TEXT_PRIMARY, ha="right")

    for xi, (a, b) in enumerate(zip(df["coverage_cqr"], df["coverage_mondrian"], strict=True)):
        ax.text(xi - width / 2, a + 0.004, f"{a:.3f}", ha="center", fontsize=8.6, color=TEXT_SECONDARY)
        ax.text(xi + width / 2, b + 0.004, f"{b:.3f}", ha="center", fontsize=8.6, color=TEXT_SECONDARY)

    lo = min(df["coverage_cqr"].min(), df["coverage_mondrian"].min(), target) - 0.035
    hi = max(df["coverage_cqr"].max(), df["coverage_mondrian"].max(), target) + 0.028
    ax.set_ylim(lo, hi)
    ax.set_ylabel("empirical coverage")
    ax.legend(loc="lower right", ncol=2)
    ax.grid(axis="x", visible=False)
    title_block(
        ax, "Marginal coverage can be right while conditional coverage is wrong",
        "Coverage within each predicted-demand quintile: a single global correction over-covers quiet cells and under-covers busy ones",
    )

    ax2.bar(x - width / 2, df["mean_width_cqr"], width, color=SERIES[1])
    ax2.bar(x + width / 2, df["mean_width_mondrian"], width, color=SERIES[0])
    ax2.set_ylabel("mean interval width")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Q{i + 1}\n(mean actual {m:.1f})" for i, m in enumerate(df["mean_actual"])])
    ax2.set_xlabel("predicted-demand quintile")
    ax2.grid(axis="x", visible=False)
    return finish(fig, out, "Mondrian conditioning trades a little width in busy strata for coverage that holds everywhere.")


def feature_importance(records: list[dict], out: Path, top_n: int = 20) -> Path:
    apply_style()
    df = pd.DataFrame(records).head(top_n).iloc[::-1]

    def family(name: str) -> str:
        if name.startswith(("lag_", "roll_", "dev_")):
            return "temporal history"
        if name.startswith("spatial_"):
            return "spatial diffusion"
        if name.startswith("lap_eig") or name.startswith("graph_"):
            return "graph structure"
        if name.startswith("fourier_"):
            return "Fourier seasonality"
        if name in {"temperature_2m", "precipitation", "snowfall", "wind_speed_10m",
                    "relative_humidity_2m", "cloud_cover", "is_raining", "is_snowing",
                    "is_freezing", "rain_x_peak", "temp_sq", "rain_x_crz"}:
            return "weather"
        return "calendar and zone"

    families = ["temporal history", "spatial diffusion", "graph structure", "Fourier seasonality", "weather", "calendar and zone"]
    palette = dict(zip(families, [SERIES[0], SERIES[1], SERIES[2], SERIES[3], SERIES[4], "#9a9993"], strict=True))
    df["family"] = df["feature"].map(family)

    fig, ax = plt.subplots(figsize=(11.0, 0.44 * len(df) + 2.6))
    ax.barh(df["feature"], df["importance_pct"], color=df["family"].map(palette), height=0.68)
    for y, v in enumerate(df["importance_pct"]):
        ax.text(v + df["importance_pct"].max() * 0.012, y, f"{v:.1f}%", va="center", fontsize=9, color=TEXT_PRIMARY)
    ax.set_xlabel("share of total split gain")
    ax.set_xlim(0, df["importance_pct"].max() * 1.14)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=9)
    handles = [Line2D([], [], marker="s", color="none", markerfacecolor=palette[f], markersize=9, label=f) for f in families if f in set(df["family"])]
    ax.legend(handles=handles, loc="lower right", ncol=2)
    title_block(
        ax, f"Top {top_n} features by split gain",
        "Colour groups features by family, which shows how much the engineered spatial and graph terms actually earn",
    )
    return finish(fig, out, "Single metric across many categories, so bars share one scale and every bar is labelled.")


def event_study(
    table: pd.DataFrame,
    out: Path,
    treatment_label: str = "congestion pricing",
    parallel_trends_holds: bool = True,
    pretrend_p: float = float("nan"),
    pretrend_slope: float = float("nan"),
) -> Path:
    apply_style()
    df = table.sort_values("event_time")
    fig, ax = plt.subplots(figsize=(11.6, 6.4))

    ax.axhline(0, color=AXIS, linewidth=1.1)
    ax.axvspan(-0.5, df["event_time"].max() + 0.6, color=NEUTRAL, alpha=0.75, zorder=0)
    ax.axvline(-0.5, color=TEXT_PRIMARY, linewidth=1.5, linestyle="--", zorder=2)

    pre = df[df["event_time"] < 0]
    post = df[df["event_time"] >= 0]
    for part, color in ((pre, TEXT_MUTED), (post, BLUE)):
        ax.errorbar(
            part["event_time"], part["coef_pct"],
            yerr=[part["coef_pct"] - part["ci_low_pct"], part["ci_high_pct"] - part["coef_pct"]],
            fmt="o", color=color, ecolor=color, elinewidth=1.5, capsize=3.5, markersize=6.5, zorder=3,
        )
    span = df["ci_high_pct"].max() - df["ci_low_pct"].min()
    ax.set_ylim(df["ci_low_pct"].min() - span * 0.06, df["ci_high_pct"].max() + span * 0.14)
    ax.annotate(
        f"{treatment_label} begins",
        xy=(-0.5, 1.0), xycoords=("data", "axes fraction"),
        xytext=(6, -12), textcoords="offset points",
        fontsize=10, color=TEXT_PRIMARY, va="top", ha="left", weight="semibold",
    )
    ax.set_xlabel("weeks relative to treatment")
    ax.set_ylabel("effect on treated-zone trips (%)")
    ax.set_xticks(df["event_time"][::2])
    handles = [
        Line2D([], [], marker="o", color=TEXT_MUTED, linestyle="none", markersize=7, label="pre-treatment (parallel-trends check)"),
        Line2D([], [], marker="o", color=BLUE, linestyle="none", markersize=7, label="post-treatment effect"),
    ]
    ax.legend(handles=handles, loc="lower left")
    if parallel_trends_holds:
        heading = "Event study: no pre-trend, then a persistent level shift"
        note = "Flat, insignificant pre-treatment coefficients are what licenses a causal reading of the post-treatment ones."
    else:
        heading = "Event study: parallel trends FAILS, so this is not a causal estimate"
        note = (
            "Treated zones were already drifting away from controls before treatment, so the "
            "post-treatment gap cannot be attributed to the policy."
        )
        pre = df[df["event_time"] < -1]
        if len(pre) > 3:
            fit = np.polyfit(pre["event_time"], pre["coef_pct"], 1)
            xs = np.array([pre["event_time"].min(), -1])
            ax.plot(xs, np.polyval(fit, xs), color=RED, linewidth=2.0, linestyle="-.", zorder=4)
            ax.annotate(
                f"pre-trend {pretrend_slope:+.2f}% per week\nF-test p = {pretrend_p:.1e}",
                xy=(xs.mean(), np.polyval(fit, xs.mean())),
                xytext=(0, 26), textcoords="offset points",
                fontsize=9.5, color=RED, weight="semibold", ha="center",
            )

    title_block(
        ax, heading,
        "Two-way fixed-effects coefficients with 95% confidence intervals, standard errors clustered by zone. Week -1 is the reference.",
    )
    return finish(fig, out, note)


def drift_panel(table: pd.DataFrame, psi_warn: float, psi_alert: float, out: Path, top_n: int = 18) -> Path:
    apply_style()
    df = table.head(top_n).iloc[::-1]
    colors = [RED if s == "alert" else (YELLOW if s == "warn" else AQUA) for s in df["status"]]

    fig, ax = plt.subplots(figsize=(10.8, 0.44 * len(df) + 2.6))
    ax.barh(df["feature"], df["psi"], color=colors, height=0.66)
    ax.axvline(psi_warn, color=YELLOW, linewidth=1.4, linestyle="--")
    ax.axvline(psi_alert, color=RED, linewidth=1.4, linestyle="--")
    ax.text(psi_warn, len(df) - 0.2, " warn", color=YELLOW, fontsize=9.5, va="top")
    ax.text(psi_alert, len(df) - 0.2, " alert", color=RED, fontsize=9.5, va="top")
    for y, v in enumerate(df["psi"]):
        ax.text(v + df["psi"].max() * 0.015, y, f"{v:.3f}", va="center", fontsize=9, color=TEXT_PRIMARY)
    ax.set_xlabel("population stability index, train reference against test window")
    ax.set_xlim(0, max(df["psi"].max() * 1.2, psi_alert * 1.25))
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=9)
    title_block(
        ax, "Feature drift between training and serving windows",
        "PSI above 0.25 is the alert threshold that would gate a production deployment",
    )
    return finish(fig, out, "Status colours are reserved for state and always ship with a labelled threshold line.")


def residual_diagnostics(frame: pd.DataFrame, out: Path) -> Path:
    apply_style()
    resid = frame["y"] - frame["yhat"]
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.9))

    bins = pd.qcut(frame["yhat"], q=18, duplicates="drop")
    grouped = resid.groupby(bins, observed=True)
    centres = [iv.mid for iv in grouped.groups]
    axes[0].axhline(0, color=AXIS, linewidth=1.2)
    axes[0].plot(centres, grouped.mean(), marker="o", color=BLUE, label="mean residual")
    axes[0].fill_between(
        centres, grouped.quantile(0.1), grouped.quantile(0.9), color=BLUE, alpha=0.15,
        linewidth=0, label="10th to 90th percentile",
    )
    axes[0].set_xlabel("predicted trips")
    axes[0].set_ylabel("residual (actual - predicted)")
    axes[0].legend(loc="upper left")
    axes[0].set_title("Residuals stay centred across the demand range", fontsize=11, loc="left", color=TEXT_PRIMARY)

    hourly = resid.groupby(frame["hour_ts"].dt.hour, observed=True).mean()
    axes[1].axhline(0, color=AXIS, linewidth=1.2)
    axes[1].plot(hourly.index, hourly.to_numpy(), marker="o", color=ORANGE)
    axes[1].set_xlabel("hour of day")
    axes[1].set_ylabel("mean residual")
    axes[1].set_xticks(range(0, 24, 3))
    axes[1].set_title("No systematic bias by time of day", fontsize=11, loc="left", color=TEXT_PRIMARY)

    axes[2].hist(resid, bins=70, color=AQUA, edgecolor="#ffffff", linewidth=0.35)
    axes[2].axvline(0, color=TEXT_PRIMARY, linewidth=1.3, linestyle="--")
    axes[2].set_xlabel("residual")
    axes[2].set_ylabel("count")
    axes[2].set_title(f"Residual distribution, mean {resid.mean():+.3f}", fontsize=11, loc="left", color=TEXT_PRIMARY)
    axes[2].grid(axis="x", visible=False)

    fig.suptitle(
        "Residual diagnostics on held-out test data",
        fontsize=12.5, color=TEXT_PRIMARY, x=0.012, ha="left", y=1.03, weight="semibold",
    )
    return finish(fig, out, "Each panel uses one axis and one series; nothing is stacked on a second scale.")


def rebalance_frontier(records: list[dict], chosen: dict, out: Path) -> Path:
    apply_style()
    df = pd.DataFrame(records)
    horizons = sorted(df["horizon_minutes"].unique())
    colors = [SERIES[0], SERIES[1], SERIES[2], SERIES[3]]

    fig, ax = plt.subplots(figsize=(11.2, 6.6))
    for horizon, color in zip(horizons, colors, strict=False):
        sub = df[df["horizon_minutes"] == horizon].sort_values("minutes_per_extra_trip")
        ax.plot(
            sub["minutes_per_extra_trip"], sub["unmet_reduction_pct"], marker="o",
            color=color, label=f"{horizon:.0f} min arrival horizon", linewidth=2.0,
        )
        for _, r in sub.iterrows():
            ax.annotate(
                f"{r['reposition_share']:.0%}",
                (r["minutes_per_extra_trip"], r["unmet_reduction_pct"]),
                xytext=(0, 9), textcoords="offset points", fontsize=8.4,
                color=TEXT_MUTED, ha="center",
            )

    ax.scatter(
        [chosen["minutes_per_extra_trip"]], [chosen["unmet_reduction_pct"]],
        s=190, facecolor="none", edgecolor=TEXT_PRIMARY, linewidth=2.0, zorder=5,
    )
    ax.annotate(
        "configured operating point",
        (chosen["minutes_per_extra_trip"], chosen["unmet_reduction_pct"]),
        xytext=(14, -20), textcoords="offset points", fontsize=9.5,
        color=TEXT_PRIMARY, weight="semibold",
        arrowprops={"arrowstyle": "-", "color": TEXT_PRIMARY, "linewidth": 1.0},
    )
    ax.set_xlabel("vehicle-minutes spent per additional trip served (lower is better)")
    ax.set_ylabel("unmet demand removed (%)")
    ax.legend(loc="lower right")
    title_block(
        ax, "The rebalancing policy has sharply diminishing returns",
        "Each point is a repositioning budget, labelled by the share of the idle fleet allowed to move",
    )
    return finish(
        fig, out,
        "Serving the last few percent of unmet demand costs more than twice as much per trip as the first half.",
    )
