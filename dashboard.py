"""
Section 6 — the dashboard. Built LAST, per the master prompt's explicit
build order, because it's a view onto the pipeline/harness, not the point
of the pilot itself.

IMPORTANT — what this app does and does NOT do:
  - It reads from storage.py (the SQLite DB) and accuracy_harness.py's
    results CSV. It does NOT call RailRadar/OpenWeatherMap itself.
  - The actual live polling is pipeline.py's job, run as a SEPARATE
    long-lived process (`python pipeline.py --loop`), per Section 7. Run it
    alongside this dashboard, not instead of it.
  - Requires: pip install dash dash-leaflet plotly
    (not pre-installed in every environment — see requirements.txt)

Run:
    python dashboard.py
Colab:
    app.run(jupyter_mode="inline")   # see bottom of this file
"""

from datetime import datetime, timedelta

import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_leaflet as dl
import plotly.graph_objects as go

import config
import storage
import accuracy_harness
from timeutils import now_ist, IST

# ---------------------------------------------------------------------------
# IRCTC-style theme
# ---------------------------------------------------------------------------
COLORS = {
    "header_blue": "#0B3D62",
    "saffron": "#FF9933",
    "white": "#FFFFFF",
    "green": "#138808",
    "card_bg": "#FFFFFF",
    "page_bg": "#F2F4F7",
    "text": "#1A1A1A",
    "muted": "#6B7280",
    "good": "#138808",
    "warn": "#B45309",
    "bad": "#B91C1C",
}

FONT_LINK = html.Link(
    rel="stylesheet",
    href="https://fonts.googleapis.com/css2?family=Noto+Sans:wght@400;600;700&family=Roboto+Mono&display=swap",
)

TRICOLOR_STRIP = html.Div(style={"display": "flex", "height": "6px"}, children=[
    html.Div(style={"flex": 1, "backgroundColor": COLORS["saffron"]}),
    html.Div(style={"flex": 1, "backgroundColor": COLORS["white"]}),
    html.Div(style={"flex": 1, "backgroundColor": COLORS["green"]}),
])

HEADER = html.Div(
    style={"backgroundColor": COLORS["header_blue"], "padding": "18px 28px", "color": "white"},
    children=[
        html.Div("PRAVAAH", style={"fontFamily": "Noto Sans", "fontWeight": 700, "fontSize": "26px", "letterSpacing": "1px"}),
        html.Div("Lucknow \u2194 New Delhi ETA Accuracy Pilot", style={"fontFamily": "Noto Sans", "fontSize": "14px", "opacity": 0.9}),
    ],
)

DISCLAIMER = html.Div(
    "Prototype \u2014 not an official Indian Railways or IRCTC service. "
    "Predictions and confidence intervals are experimental and may be wrong.",
    style={"textAlign": "center", "fontSize": "12px", "color": COLORS["muted"],
           "padding": "10px", "fontFamily": "Noto Sans"},
)

CARD_STYLE = {
    "backgroundColor": COLORS["card_bg"], "borderRadius": "10px", "padding": "18px",
    "boxShadow": "0 1px 3px rgba(0,0,0,0.12)", "margin": "10px", "fontFamily": "Noto Sans",
}

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "PRAVAAH \u2014 LKO-NDLS Pilot"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def freshness_label(api_last_update_ist_str: str) -> tuple:
    """Section 6: freshness/provenance indicator, driven off the API's own
    last-update timestamp, never a hardcoded 'live' claim."""
    if not api_last_update_ist_str:
        return "no recent signal \u2014 estimated from schedule", COLORS["bad"]
    try:
        last = datetime.fromisoformat(api_last_update_ist_str)
    except ValueError:
        return "no recent signal \u2014 estimated from schedule", COLORS["bad"]
    age_s = (now_ist() - last.astimezone(IST)).total_seconds()
    if age_s < 0:
        age_s = 0
    if age_s < 20 * 60:
        return f"GPS confirmed {int(age_s)}s ago", COLORS["good"]
    elif age_s < 60 * 60:
        return f"GPS confirmed {int(age_s // 60)} min ago", COLORS["warn"]
    else:
        return "no recent signal \u2014 estimated from schedule", COLORS["bad"]


def train_card(row) -> html.Div:
    r = dict(row)
    tn = r["train_number"]
    info = config.TRAIN_ROSTER.get(tn, {"name": tn})
    fresh_text, fresh_color = freshness_label(r.get("api_last_update_ist"))

    eta_field = r.get("model_predicted_arrival_ist") or r.get("naive_predicted_arrival_ist")
    eta_source = "model" if r.get("model_predicted_arrival_ist") else "naive (no trained model yet)"
    conf_text = ""
    if r.get("confidence_low_ist") and r.get("confidence_high_ist"):
        low = datetime.fromisoformat(r["confidence_low_ist"]).strftime("%H:%M")
        high = datetime.fromisoformat(r["confidence_high_ist"]).strftime("%H:%M")
        conf_text = f"90% interval: {low} \u2013 {high} IST"

    eta_display = "\u2014"
    if eta_field:
        eta_display = datetime.fromisoformat(eta_field).strftime("%H:%M IST")

    return html.Div(style=CARD_STYLE, children=[
        html.Div([
            html.Span(f"{tn}", style={"fontFamily": "Roboto Mono", "fontWeight": 700, "fontSize": "18px"}),
            html.Span(f"  {info.get('name', '')}", style={"fontSize": "16px", "marginLeft": "6px"}),
        ]),
        html.Div(f"Last seen: {r.get('last_station_code', '\u2014')} \u00b7 "
                  f"{r.get('distance_remaining_km', '?')} km to NDLS",
                  style={"color": COLORS["muted"], "marginTop": "4px"}),
        html.Div(f"Predicted arrival: {eta_display}", style={"fontSize": "20px", "fontWeight": 700, "marginTop": "8px"}),
        html.Div(f"({eta_source})", style={"fontSize": "12px", "color": COLORS["muted"]}),
        html.Div(conf_text, style={"fontSize": "13px", "color": COLORS["muted"], "marginTop": "2px"}),
        html.Div(fresh_text, style={"fontSize": "13px", "color": fresh_color, "marginTop": "8px", "fontWeight": 600}),
        html.Button("I'm on this train", id={"type": "crowdsource-btn", "train": tn},
                    style={"marginTop": "12px", "padding": "8px 14px", "borderRadius": "6px",
                           "border": f"1px solid {COLORS['header_blue']}", "backgroundColor": "white",
                           "color": COLORS["header_blue"], "cursor": "pointer"}),
    ])


def build_route_map(latest_rows) -> dl.Map:
    station_markers = [
        dl.Marker(position=(s["lat"], s["lon"]), children=dl.Tooltip(f"{code} \u2014 {s['name']}"))
        for code, s in config.STATIONS.items()
    ]
    route_line = dl.Polyline(
        positions=[(config.STATIONS[c]["lat"], config.STATIONS[c]["lon"]) for c in config.ROUTE_ORDER],
        color=COLORS["header_blue"], weight=3, opacity=0.6,
    )
    train_markers = []
    for row in latest_rows:
        r = dict(row)
        st = config.STATIONS.get(r.get("last_station_code"))
        if st:
            train_markers.append(
                dl.CircleMarker(center=(st["lat"], st["lon"]), radius=9, color=COLORS["saffron"],
                                 fillColor=COLORS["saffron"], fillOpacity=0.9,
                                 children=dl.Tooltip(f"Train {r['train_number']} near {r.get('last_station_code')}"))
            )
    center = (config.STATIONS["BE"]["lat"], config.STATIONS["BE"]["lon"])
    return dl.Map(center=center, zoom=7, style={"height": "380px", "width": "100%", "borderRadius": "10px"},
                   children=[dl.TileLayer(), route_line, *station_markers, *train_markers])


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
def passenger_view():
    latest_rows = storage.fetch_latest_per_train()
    cards = [train_card(r) for r in latest_rows] or [
        html.Div("No polls logged yet \u2014 start pipeline.py to begin tracking.", style=CARD_STYLE)
    ]
    return html.Div([
        html.Div(cards, style={"display": "flex", "flexWrap": "wrap"}),
        html.Div(style=CARD_STYLE, children=[
            html.Div("Route map", style={"fontWeight": 700, "marginBottom": "8px"}),
            build_route_map(latest_rows),
        ]),
        html.Div(id="crowdsource-modal-area"),
    ])


def control_room_view():
    rows = storage.fetch_latest_per_train()
    if not rows:
        return html.Div("No data logged yet.", style=CARD_STYLE)
    records = []
    for row in rows:
        r = dict(row)
        fresh_text, _ = freshness_label(r.get("api_last_update_ist"))
        records.append({
            "Train": r["train_number"],
            "Last station": r.get("last_station_code"),
            "Dist. remaining (km)": r.get("distance_remaining_km"),
            "Reported delay (min)": r.get("reported_delay_min"),
            "Visibility (km)": r.get("weather_visibility_km"),
            "Precip (mm)": r.get("weather_precip_mm"),
            "Freshness": fresh_text,
            "Last poll (IST)": r.get("poll_timestamp_ist"),
        })
    return html.Div(style=CARD_STYLE, children=[
        dash_table.DataTable(
            data=records,
            columns=[{"name": c, "id": c} for c in records[0].keys()],
            style_cell={"fontFamily": "Noto Sans", "padding": "8px", "textAlign": "left"},
            style_header={"backgroundColor": COLORS["header_blue"], "color": "white", "fontWeight": 700},
            style_data_conditional=[
                {"if": {"filter_query": "{Visibility (km)} < 1.5"}, "backgroundColor": "#FFF4E5"},
            ],
        )
    ])


def model_performance_view():
    latest = accuracy_harness.latest_result()
    hist = accuracy_harness.results_history()

    if not latest or latest.get("status") in (None, "NO_RUNS_YET"):
        return html.Div(style=CARD_STYLE, children=[
            html.Div("No harness runs yet.", style={"fontWeight": 700}),
            html.Div("Run `python accuracy_harness.py` once real (or demo) logged data with "
                     "joined actual arrivals exists."),
        ])

    status = latest.get("status")
    body = [html.Div(f"Status: {status}", style={"fontWeight": 700, "marginBottom": "10px"})]

    if status == "OK":
        naive_mae = float(latest["naive_mae_min"])
        model_mae = float(latest["model_mae_min"])
        improvement = float(latest["improvement_pct"])
        target_cov = float(latest["conformal_target_coverage"]) * 100
        obs_cov = float(latest["conformal_observed_coverage"]) * 100

        bar_fig = go.Figure(data=[go.Bar(
            x=["Naive baseline", "Correction model"], y=[naive_mae, model_mae],
            marker_color=[COLORS["muted"], COLORS["header_blue"]],
            text=[f"{naive_mae:.1f} min", f"{model_mae:.1f} min"], textposition="outside",
        )])
        bar_fig.update_layout(title="MAE on logged holdout data (lower is better)",
                               yaxis_title="Minutes", template="plotly_white", height=320)

        cov_fig = go.Figure(data=[go.Bar(
            x=["Target coverage", "Observed coverage"], y=[target_cov, obs_cov],
            marker_color=[COLORS["muted"], COLORS["green"]],
            text=[f"{target_cov:.0f}%", f"{obs_cov:.1f}%"], textposition="outside",
        )])
        cov_fig.update_layout(title="90% conformal interval \u2014 target vs. observed coverage",
                               yaxis_title="%", yaxis_range=[0, 100], template="plotly_white", height=320)

        body += [
            html.Div(style={"display": "flex", "gap": "18px", "flexWrap": "wrap"}, children=[
                html.Div(dcc.Graph(figure=bar_fig), style={"flex": "1 1 400px"}),
                html.Div(dcc.Graph(figure=cov_fig), style={"flex": "1 1 400px"}),
            ]),
            html.Div(f"Improvement over naive: {improvement:.1f}%  "
                     f"(scored on {int(latest['n_holdout_rows'])} held-out rows, "
                     f"{int(latest['n_scored_rows'])} total scored rows in the DB)",
                     style={"fontSize": "16px", "fontWeight": 600, "marginTop": "6px"}),
            html.Div("Published Indian-Railways ML studies report roughly 20\u201330% error reduction "
                     "over this exact naive baseline as a realistic range \u2014 shown here as a "
                     "comparison point, not a target.", style={"fontSize": "12px", "color": COLORS["muted"]}),
        ]

        if not hist.empty and len(hist) > 1:
            trend_fig = go.Figure()
            trend_fig.add_trace(go.Scatter(x=hist["run_timestamp"], y=hist["naive_mae_min"],
                                            mode="lines+markers", name="Naive MAE"))
            trend_fig.add_trace(go.Scatter(x=hist["run_timestamp"], y=hist["model_mae_min"],
                                            mode="lines+markers", name="Model MAE"))
            trend_fig.update_layout(title="MAE over successive harness runs, as more data comes in",
                                     yaxis_title="Minutes", template="plotly_white", height=320)
            body.append(dcc.Graph(figure=trend_fig))
    else:
        body.append(html.Div(
            "Not enough scored data yet for a fair model-vs-naive comparison. "
            "This is expected early in the pilot \u2014 the harness will not report an "
            "improvement % or coverage number until it can compute one honestly.",
            style={"color": COLORS["warn"]},
        ))
        if latest.get("naive_mae_min"):
            body.append(html.Div(f"Naive MAE so far: {float(latest['naive_mae_min']):.1f} min"))

    return html.Div(style=CARD_STYLE, children=body)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
app.layout = html.Div(style={"backgroundColor": COLORS["page_bg"], "minHeight": "100vh", "fontFamily": "Noto Sans"}, children=[
    FONT_LINK,
    HEADER,
    TRICOLOR_STRIP,
    dcc.Tabs(id="tabs", value="passenger", children=[
        dcc.Tab(label="Passenger view", value="passenger"),
        dcc.Tab(label="Control room view", value="control"),
        dcc.Tab(label="Model performance", value="model"),
    ]),
    html.Div(id="tab-content", style={"padding": "10px"}),
    dcc.Interval(id="refresh-interval", interval=60 * 1000, n_intervals=0),
    DISCLAIMER,
])


@app.callback(Output("tab-content", "children"), Input("tabs", "value"), Input("refresh-interval", "n_intervals"))
def render_tab(tab, _n):
    if tab == "passenger":
        return passenger_view()
    elif tab == "control":
        return control_room_view()
    elif tab == "model":
        return model_performance_view()
    return html.Div()


# --- Crowdsource "I'm on this train" modal (Section 3.3) ---------------------
@app.callback(
    Output("crowdsource-modal-area", "children"),
    Input({"type": "crowdsource-btn", "train": dash.ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def open_crowdsource_modal(n_clicks_list):
    ctx = dash.callback_context
    if not ctx.triggered or not any(n_clicks_list):
        return dash.no_update
    triggered_id = ctx.triggered_id
    train_number = triggered_id["train"]
    return html.Div(style={**CARD_STYLE, "border": f"2px solid {COLORS['header_blue']}"}, children=[
        html.Div(f"Report your position \u2014 Train {train_number}", style={"fontWeight": 700}),
        html.Div("By submitting, you consent to sharing an approximate position while on this train.",
                  style={"fontSize": "12px", "color": COLORS["muted"], "marginBottom": "8px"}),
        dcc.Input(id="cs-lat", type="number", placeholder="Latitude", style={"marginRight": "8px"}),
        dcc.Input(id="cs-lon", type="number", placeholder="Longitude", style={"marginRight": "8px"}),
        html.Button("Submit", id="cs-submit", n_clicks=0),
        html.Div(id="cs-confirm", style={"marginTop": "8px", "color": COLORS["good"]}),
        dcc.Store(id="cs-train-store", data=train_number),
    ])


@app.callback(
    Output("cs-confirm", "children"),
    Input("cs-submit", "n_clicks"),
    State("cs-lat", "value"), State("cs-lon", "value"), State("cs-train-store", "data"),
    prevent_initial_call=True,
)
def submit_crowdsource(n_clicks, lat, lon, train_number):
    if not n_clicks or lat is None or lon is None:
        return dash.no_update
    storage.insert_crowdsource_report({
        "report_timestamp_ist": now_ist().isoformat(),
        "train_number": train_number,
        "service_date": now_ist().strftime("%Y-%m-%d"),
        "user_id": "anonymous-demo-user",  # replace with real session/user id in production
        "reported_lat": lat,
        "reported_lon": lon,
        "nearest_station_code": None,
    })
    return "Thanks \u2014 position recorded."

<<<<<<< HEAD

=======
server = app.server
>>>>>>> 9fb5063 (Fix Render deployment)
if __name__ == "__main__":
    storage.init_db()
    app.run(debug=True)
    # Colab:
    # app.run(jupyter_mode="inline")
