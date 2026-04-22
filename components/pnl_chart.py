"""
PnL Equity Curve + R-Multiple Distribution Chart
Renders via Plotly (bundled with Streamlit, no extra install needed).
Data source: closed_trades list — each trade has closed_at + pnl + r_multiple.
Chart is rebuilt from scratch on every render → always reflects latest data.
"""
import streamlit as st
from datetime import datetime

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ── R-band definitions ────────────────────────────────────────────────────────
R_BANDS = [
    ("< 1:1",      lambda r: r < 1.0),
    ("1:1 – 1:1.5", lambda r: 1.0 <= r < 1.5),
    ("1:1.5 – 1:2.5", lambda r: 1.5 <= r < 2.5),
    ("1:2.5 – 1:3", lambda r: 2.5 <= r < 3.0),
    ("≥ 1:3",      lambda r: r >= 3.0),
]
R_COLORS = ["#da3633", "#e3b341", "#58a6ff", "#3fb950", "#7ee787"]

_CHART_BG   = "#0d1117"
_GRID_COLOR = "#21262d"
_TEXT_COLOR = "#8b949e"
_FONT_FMLY  = "DM Sans, sans-serif"


def render_pnl_chart(data: dict):
    closed = data.get("closed_trades", [])

    if not closed:
        st.markdown("""
        <div style="text-align:center;padding:3rem;color:#30363d;
             border:1px dashed #21262d;border-radius:10px;margin-bottom:1rem">
            <div style="font-size:2rem;margin-bottom:0.5rem">📈</div>
            <div style="font-size:0.95rem;color:#484f58">
                Kapalı işlem olmadan grafik oluşturulamaz
            </div>
        </div>""", unsafe_allow_html=True)
        return

    if not HAS_PLOTLY:
        st.warning("Plotly yüklü değil. `pip install plotly` çalıştırın.")
        return

    # ── Sort trades by close date ─────────────────────────────────────────────
    def _parse_dt(t):
        raw = t.get("closed_at") or t.get("created_at") or ""
        try:
            return datetime.fromisoformat(raw)
        except Exception:
            return datetime.min

    sorted_trades = sorted(closed, key=_parse_dt)

    # ── Build cumulative PnL series ───────────────────────────────────────────
    dates    = []
    cum_pnl  = []
    labels   = []
    colors   = []
    running  = 0.0

    for t in sorted_trades:
        pnl      = float(t.get("pnl", 0))
        running += pnl
        sym      = t.get("symbol", "?")
        res      = t.get("result", "?")
        rr_d     = t.get("rr_display", "?")
        sign     = "+" if pnl >= 0 else ""

        dt = _parse_dt(t)
        dates.append(dt.strftime("%d %b %Y %H:%M") if dt != datetime.min else "?")
        cum_pnl.append(round(running, 2))
        labels.append(f"#{t.get('id','?')} {sym} {res}<br>PnL: {sign}${pnl:,.2f}<br>Kümülatif: ${running:,.2f}<br>RR: {rr_d}")
        colors.append("#3fb950" if pnl >= 0 else "#da3633")

    # ── Build R-distribution bar data ─────────────────────────────────────────
    r_counts = []
    for label, fn in R_BANDS:
        cnt = sum(1 for t in closed
                  if t.get("r_multiple") is not None and fn(float(t.get("r_multiple", 0))))
        r_counts.append(cnt)

    # ── Figure with 2 rows ────────────────────────────────────────────────────
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.12,
        subplot_titles=["📈 Kümülatif PnL (Equity Curve)", "📊 R Multiple Dağılımı"],
    )

    # ── Row 1: Equity curve ───────────────────────────────────────────────────
    # Area fill
    fig.add_trace(go.Scatter(
        x=list(range(len(dates))),
        y=cum_pnl,
        mode="none",
        fill="tozeroy",
        fillcolor="rgba(63,185,80,0.08)" if cum_pnl[-1] >= 0 else "rgba(218,54,51,0.08)",
        showlegend=False,
        hoverinfo="skip",
    ), row=1, col=1)

    # Line
    fig.add_trace(go.Scatter(
        x=list(range(len(dates))),
        y=cum_pnl,
        mode="lines+markers",
        line=dict(color="#58a6ff", width=2.5, shape="spline", smoothing=0.6),
        marker=dict(size=7, color=colors, line=dict(width=1.5, color="#0d1117")),
        customdata=labels,
        hovertemplate="%{customdata}<extra></extra>",
        showlegend=False,
    ), row=1, col=1)

    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="#30363d", line_width=1, row=1, col=1)

    # X-axis tick labels (trade index + symbol)
    tick_labels = [f"#{t.get('id','?')} {t.get('symbol','?')}" for t in sorted_trades]

    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(len(dates))),
        ticktext=tick_labels,
        tickangle=-40,
        tickfont=dict(size=11, color=_TEXT_COLOR, family=_FONT_FMLY),
        gridcolor=_GRID_COLOR,
        row=1, col=1,
    )
    fig.update_yaxes(
        tickprefix="$",
        tickfont=dict(size=12, color=_TEXT_COLOR, family=_FONT_FMLY),
        gridcolor=_GRID_COLOR,
        row=1, col=1,
    )

    # ── Row 2: R distribution bars ────────────────────────────────────────────
    fig.add_trace(go.Bar(
        x=[b[0] for b in R_BANDS],
        y=r_counts,
        marker_color=R_COLORS,
        marker_line_color="#0d1117",
        marker_line_width=1.5,
        text=r_counts,
        textposition="outside",
        textfont=dict(size=13, color="#c9d1d9", family=_FONT_FMLY),
        hovertemplate="%{x}: %{y} işlem<extra></extra>",
        showlegend=False,
    ), row=2, col=1)

    fig.update_xaxes(
        tickfont=dict(size=12, color=_TEXT_COLOR, family=_FONT_FMLY),
        gridcolor=_GRID_COLOR,
        row=2, col=1,
    )
    fig.update_yaxes(
        tickfont=dict(size=12, color=_TEXT_COLOR, family=_FONT_FMLY),
        gridcolor=_GRID_COLOR,
        dtick=1,
        row=2, col=1,
    )

    # ── Global layout ─────────────────────────────────────────────────────────
    fig.update_layout(
        height=620,
        paper_bgcolor=_CHART_BG,
        plot_bgcolor=_CHART_BG,
        font=dict(family=_FONT_FMLY, size=13, color=_TEXT_COLOR),
        margin=dict(l=10, r=10, t=55, b=10),
        hoverlabel=dict(
            bgcolor="#161b22",
            bordercolor="#30363d",
            font=dict(size=13, color="#c9d1d9", family=_FONT_FMLY),
        ),
    )
    # Subplot titles styling
    for ann in fig.layout.annotations:
        ann.font.size   = 13
        ann.font.color  = "#c9d1d9"
        ann.font.family = _FONT_FMLY

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Summary row below chart ───────────────────────────────────────────────
    final_pnl = cum_pnl[-1] if cum_pnl else 0
    peak      = max(cum_pnl) if cum_pnl else 0
    trough    = min(cum_pnl) if cum_pnl else 0
    # Max drawdown: biggest drop from a peak
    max_dd = 0.0
    running_peak = cum_pnl[0] if cum_pnl else 0
    for v in cum_pnl:
        if v > running_peak:
            running_peak = v
        dd = running_peak - v
        if dd > max_dd:
            max_dd = dd

    fc    = "#3fb950" if final_pnl >= 0 else "#da3633"
    sign  = "+" if final_pnl >= 0 else ""
    st.markdown(f"""
    <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));
                gap:0.5rem;margin-top:0.5rem">
      <div class="detail-item">
        <div class="detail-label">Net PnL</div>
        <div class="detail-value" style="color:{fc}">{sign}${final_pnl:,.2f}</div>
      </div>
      <div class="detail-item">
        <div class="detail-label">En Yüksek</div>
        <div class="detail-value" style="color:#3fb950">+${peak:,.2f}</div>
      </div>
      <div class="detail-item">
        <div class="detail-label">En Düşük</div>
        <div class="detail-value" style="color:#da3633">${trough:,.2f}</div>
      </div>
      <div class="detail-item">
        <div class="detail-label">Max Drawdown</div>
        <div class="detail-value" style="color:#e3b341">-${max_dd:,.2f}</div>
      </div>
      <div class="detail-item">
        <div class="detail-label">Toplam İşlem</div>
        <div class="detail-value">{len(closed)}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
