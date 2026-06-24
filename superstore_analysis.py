from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots


def _prepare_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], errors="coerce")

    for col in ["Sales", "Quantity", "Discount", "Profit", "Shipping Cost"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Ship Days"] = (df["Ship Date"] - df["Order Date"]).dt.days
    df["Profit Margin"] = df["Profit"] / df["Sales"].replace(0, pd.NA)
    df["Order Month"] = df["Order Date"].dt.to_period("M").astype(str)
    return df


def _kpis(df: pd.DataFrame) -> dict[str, float]:
    return {
        "total_sales": float(df["Sales"].sum()),
        "total_profit": float(df["Profit"].sum()),
        "profit_margin_pct": float((df["Profit"].sum() / df["Sales"].sum()) * 100),
        "avg_discount_pct": float(df["Discount"].mean() * 100),
        "avg_ship_days": float(df["Ship Days"].mean()),
        "order_count": int(df["Order ID"].nunique()),
    }


def _write_analytical_report(df: pd.DataFrame, output_path: Path) -> None:
    kpi = _kpis(df)
    by_market = df.groupby("Market", as_index=False)[["Sales", "Profit"]].sum().sort_values("Sales", ascending=False)
    by_category = (
        df.groupby("Category", as_index=False)[["Sales", "Profit"]]
        .sum()
        .assign(MarginPct=lambda x: (x["Profit"] / x["Sales"] * 100).round(2))
        .sort_values("Sales", ascending=False)
    )
    worst_subcats = (
        df.groupby("Sub-Category", as_index=False)["Profit"]
        .sum()
        .sort_values("Profit", ascending=True)
        .head(5)
    )
    best_subcats = (
        df.groupby("Sub-Category", as_index=False)["Profit"]
        .sum()
        .sort_values("Profit", ascending=False)
        .head(5)
    )

    lines = [
        "# Analytical Report – Global Superstore",
        "",
        "## Executive KPI Summary",
        f"- Total Sales: **${kpi['total_sales']:,.2f}**",
        f"- Total Profit: **${kpi['total_profit']:,.2f}**",
        f"- Profit Margin: **{kpi['profit_margin_pct']:.2f}%**",
        f"- Distinct Orders: **{kpi['order_count']:,}**",
        f"- Average Discount: **{kpi['avg_discount_pct']:.2f}%**",
        f"- Average Shipping Lead Time: **{kpi['avg_ship_days']:.2f} days**",
        "",
        "## Sales and Profitability Insights",
        "### Market Performance",
    ]
    for _, r in by_market.head(8).iterrows():
        lines.append(f"- {r['Market']}: Sales ${r['Sales']:,.2f}, Profit ${r['Profit']:,.2f}")

    lines.extend(["", "### Category Performance"])
    for _, r in by_category.iterrows():
        lines.append(
            f"- {r['Category']}: Sales ${r['Sales']:,.2f}, Profit ${r['Profit']:,.2f}, Margin {r['MarginPct']:.2f}%"
        )

    lines.extend(["", "### Most Profitable Sub-Categories"])
    for _, r in best_subcats.iterrows():
        lines.append(f"- {r['Sub-Category']}: Profit ${r['Profit']:,.2f}")

    lines.extend(["", "### Most Loss-Making Sub-Categories"])
    for _, r in worst_subcats.iterrows():
        lines.append(f"- {r['Sub-Category']}: Profit ${r['Profit']:,.2f}")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_data_mining_findings(df: pd.DataFrame, output_path: Path) -> None:
    # Slightly negative lower bound ensures exact 0 discounts are included in the "0%" bucket.
    discount_bins = pd.cut(
        df["Discount"], bins=[-0.001, 0, 0.1, 0.2, 0.3, 1.0], labels=["0%", "1-10%", "11-20%", "21-30%", ">30%"]
    )
    discount_effect = (
        df.assign(DiscountBand=discount_bins)
        .groupby("DiscountBand", observed=True, as_index=False)[["Sales", "Profit"]]
        .sum()
        .assign(MarginPct=lambda x: (x["Profit"] / x["Sales"] * 100).round(2))
    )

    product_sales = df.groupby("Product Name", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False)
    product_sales["cum_sales_pct"] = product_sales["Sales"].cumsum() / product_sales["Sales"].sum() * 100
    pareto_product_count = int((product_sales["cum_sales_pct"] <= 80).sum())

    segment_profitability = (
        df.groupby("Segment", as_index=False)[["Sales", "Profit"]]
        .sum()
        .assign(MarginPct=lambda x: (x["Profit"] / x["Sales"] * 100).round(2))
    )
    corr = df[["Discount", "Profit"]].corr().loc["Discount", "Profit"]

    lines = [
        "# Data Mining Findings – Global Superstore",
        "",
        "## 1) Discount-to-Profit Relationship",
        f"- Correlation between Discount and Profit: **{corr:.4f}**",
        "- Profitability by discount bands:",
    ]
    for _, r in discount_effect.iterrows():
        lines.append(
            f"  - {r['DiscountBand']}: Sales ${r['Sales']:,.2f}, Profit ${r['Profit']:,.2f}, Margin {r['MarginPct']:.2f}%"
        )

    lines.extend(
        [
            "",
            "## 2) Pareto Concentration (Sales)",
            f"- Number of products contributing ~80% of sales: **{pareto_product_count}**",
            "",
            "## 3) Segment Profitability Mining",
        ]
    )
    for _, r in segment_profitability.iterrows():
        lines.append(
            f"- {r['Segment']}: Sales ${r['Sales']:,.2f}, Profit ${r['Profit']:,.2f}, Margin {r['MarginPct']:.2f}%"
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_recommendations(df: pd.DataFrame, output_path: Path) -> None:
    high_discount_loss = df[(df["Discount"] >= 0.2) & (df["Profit"] < 0)]
    hd_share = (high_discount_loss["Sales"].sum() / df["Sales"].sum()) * 100
    avg_ship = df.groupby("Ship Mode", as_index=False)["Ship Days"].mean().sort_values("Ship Days")
    weakest_market = df.groupby("Market", as_index=False)["Profit"].sum().sort_values("Profit").head(3)

    lines = [
        "# Strategic Recommendations – Global Superstore",
        "",
        "1. **Tighten discount governance**",
        f"   - High-discount loss-making transactions represent **{hd_share:.2f}%** of sales.",
        "   - Introduce discount approval thresholds by category/sub-category.",
        "",
        "2. **Rebalance portfolio toward high-margin lines**",
        "   - Prioritize profitable categories/sub-categories in campaigns and inventory allocation.",
        "   - Review persistent loss-making SKUs for repricing, bundling, or discontinuation.",
        "",
        "3. **Market-level turnaround plans**",
        "   - Focus remediation playbooks for lowest-profit markets first (pricing, product mix, and fulfillment).",
    ]
    for _, r in weakest_market.iterrows():
        lines.append(f"   - {r['Market']}: Profit ${r['Profit']:,.2f}")

    lines.extend(["", "4. **Optimize fulfillment and service levels**", "   - Average shipping lead-time by mode:"])
    for _, r in avg_ship.iterrows():
        lines.append(f"   - {r['Ship Mode']}: {r['Ship Days']:.2f} days")

    lines.extend(
        [
            "",
            "5. **Operationalize BI cadence**",
            "   - Track weekly KPI scorecards (Sales, Profit, Margin, Discount, Ship Days).",
            "   - Trigger proactive alerts for margin deterioration and loss-making discount spikes.",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_dashboard(df: pd.DataFrame, output_path: Path) -> None:
    monthly = df.groupby("Order Month", as_index=False)[["Sales", "Profit"]].sum().sort_values("Order Month")
    by_market = df.groupby("Market", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False).head(10)
    by_category = df.groupby("Category", as_index=False)["Profit"].sum().sort_values("Profit", ascending=False)
    by_segment = (
        df.groupby("Segment", as_index=False)[["Sales", "Profit"]]
        .sum()
        .assign(MarginPct=lambda x: x["Profit"] / x["Sales"] * 100)
    )
    discount_scatter = df[["Discount", "Profit", "Sales", "Category"]].dropna()
    subcat_profit = (
        df.groupby("Sub-Category", as_index=False)["Profit"].sum().sort_values("Profit", ascending=True).head(10)
    )

    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "Monthly Sales vs Profit",
            "Top Markets by Sales",
            "Profit by Category",
            "Segment Margin (%)",
            "Discount vs Profit",
            "Bottom 10 Sub-Categories by Profit",
        ),
        specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "xy"}]],
    )

    line_sales = px.line(monthly, x="Order Month", y="Sales").data[0]
    line_profit = px.line(monthly, x="Order Month", y="Profit").data[0]
    line_profit.name = "Profit"
    line_sales.name = "Sales"
    fig.add_trace(line_sales, row=1, col=1)
    fig.add_trace(line_profit, row=1, col=1)

    bar_market = px.bar(by_market, x="Market", y="Sales").data[0]
    fig.add_trace(bar_market, row=1, col=2)

    bar_category = px.bar(by_category, x="Category", y="Profit").data[0]
    fig.add_trace(bar_category, row=2, col=1)

    bar_segment = px.bar(by_segment, x="Segment", y="MarginPct").data[0]
    fig.add_trace(bar_segment, row=2, col=2)

    scatter = px.scatter(
        discount_scatter,
        x="Discount",
        y="Profit",
        size="Sales",
        color="Category",
        opacity=0.7,
    ).data
    for trace in scatter:
        fig.add_trace(trace, row=3, col=1)

    bar_subcat = px.bar(subcat_profit, x="Sub-Category", y="Profit").data[0]
    fig.add_trace(bar_subcat, row=3, col=2)

    fig.update_layout(height=1200, width=1500, title_text="Global Superstore – Interactive Dashboard", showlegend=True)
    fig.write_html(output_path, include_plotlyjs="cdn")


def run(input_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = _prepare_data(input_path)

    _write_dashboard(df, output_dir / "interactive_dashboard.html")
    _write_analytical_report(df, output_dir / "analytical_report.md")
    _write_data_mining_findings(df, output_dir / "data_mining_findings.md")
    _write_recommendations(df, output_dir / "strategic_recommendations.md")

    print(f"Generated outputs in: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Global Superstore BI and data mining analysis")
    parser.add_argument("--input", required=True, type=Path, help="Path to Global-Superstore.csv")
    parser.add_argument("--output-dir", default=Path("output"), type=Path, help="Output directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.input, args.output_dir)
