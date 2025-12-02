import dash
from dash import dcc, html, Input, Output, State, ALL, MATCH, ctx
import plotly.graph_objects as go
import numpy as np
import os
import sys
from pathlib import Path

# --- Data Loading / Mocking Layer ---

try:
    import utils
    import models

    # 1. Load data exactly as in your script
    script_dir = Path(__file__).resolve().parent
    default_output = (script_dir.parent / "output").resolve()
    output_dir = Path(os.environ.get("ANAMOL_OUTPUT_DIR", str(default_output)))

    print(f"Attempting to load data from: {output_dir}")
    data = utils.load_all_throughputs(output_dir=str(output_dir))

    if not data:
        raise RuntimeError("No data found, switching to mock mode.")

    def get_param_names(res_key):
        try:
            return models.get_resource_param_spec(res_key)
        except:
            return None

except (ImportError, RuntimeError) as e:
    print(f"NOTICE: {e}")
    print("Generating MOCK DATA for demonstration purposes...")

    # --- MOCK DATA GENERATOR ---
    data = {}

    # Mock 1: ROB (Reorder Buffer)
    rob_params = np.array([[32, 4], [32, 8], [64, 4], [64, 8], [128, 4], [128, 8]])
    rob_thr = np.abs(np.cumsum(np.random.randn(6, 100), axis=1) + 10)
    data["ROB (Mock)"] = (rob_params, rob_thr)

    # Mock 2: Store Queue
    sq_params = np.arange(16, 33).reshape(-1, 1)
    sq_thr = np.abs(np.cumsum(np.random.randn(len(sq_params), 100), axis=1) + 5)
    data["Store Queue (Mock)"] = (sq_params, sq_thr)

    # Mock 3: L1 Cache (Geometric progression example)
    l1_params = np.array([2**i for i in range(10)]).reshape(-1, 1)  # 1, 2, 4... 512
    l1_thr = np.abs(np.cumsum(np.random.randn(len(l1_params), 100), axis=1) + 20)
    data["L1 Cache (Mock)"] = (l1_params, l1_thr)

    def get_param_names(res_key):
        if "ROB" in res_key:
            return ["ROB Size", "Issue Width"]
        elif "Store" in res_key:
            return ["Entries"]
        elif "Cache" in res_key:
            return ["Cache Size (KB)"]
        return None


# --- Core Logic ---


def _find_best_index_for_params(params: np.ndarray, sel_vals):
    """Find index of the parameter combination matching sel_vals."""
    sel = np.asarray(sel_vals).reshape(-1)
    if params.shape[1] == 1:
        sel = sel.reshape(1)
        mask = params[:, 0] == sel[0]
    else:
        mask = np.all(params == sel, axis=1)

    if np.any(mask):
        return int(np.where(mask)[0][0])

    # Fallback: Nearest Neighbor
    diffs = params.astype(float) - sel.astype(float)
    d2 = np.sum(diffs * diffs, axis=1)
    return int(np.argmin(d2))


# --- CDF helper (use utils if available, otherwise local fallback) ---
# Prefer the canonical implementation from utils when present, otherwise provide a small fallback.
try:
    import utils  # already imported elsewhere in file; this is safe

    compute_cdf_features_safe = utils.compute_cdf_features
except Exception:

    def compute_cdf_features_safe(samples: np.ndarray, num_points: int = 50):
        samples = np.asarray(samples).reshape(-1)
        if samples.size == 0:
            return np.array([]), np.array([]), 0.0
        ps = np.linspace(1, 99, num_points)
        cdf_raw = np.percentile(samples, ps)
        w = np.clip(samples, 0, None)
        if w.sum() == 0:
            cdf_weighted = cdf_raw.copy()
        else:
            w = w / w.sum()
            target_count = 10_000
            counts = np.round(w * target_count).astype(int)
            mask = counts > 0
            if mask.sum() == 0:
                cdf_weighted = cdf_raw.copy()
            else:
                expanded = np.repeat(samples[mask], counts[mask])
                cdf_weighted = np.percentile(expanded, ps)
        mean_val = float(np.mean(samples))
        return cdf_raw, cdf_weighted, mean_val


# --- Dash Application Layout ---

app = dash.Dash(
    __name__, external_stylesheets=["https://codepen.io/chriddyp/pen/bWLwgP.css"]
)
app.title = "Throughput Visualizer"

resource_keys = sorted(data.keys())
# Default to the first resource if available
default_val = [resource_keys[0]] if resource_keys else []

app.layout = html.Div(
    [
        html.Div(
            [
                html.H2("Resource Throughput Explorer", style={"marginBottom": "10px"}),
                html.P(
                    "Select multiple resources to compare. Adjust parameters for each below.",
                    style={"color": "#666"},
                ),
            ],
            style={"textAlign": "center", "padding": "20px"},
        ),
        html.Hr(),
        html.Div(
            [
                # Left Column: Controls
                html.Div(
                    [
                        html.Label(
                            "Select Resources (Multi-select):",
                            style={"fontWeight": "bold", "fontSize": "1.1em"},
                        ),
                        dcc.Dropdown(
                            id="resource-dropdown",
                            options=[{"label": k, "value": k} for k in resource_keys],
                            value=default_val,
                            multi=True,  # ENABLED MULTI-SELECT
                            placeholder="Select resources...",
                            style={"marginBottom": "20px"},
                        ),
                        # Dynamic controls container - Scrollable sidebar
                        html.Div(
                            id="controls-container",
                            style={
                                "maxHeight": "75vh",
                                "overflowY": "auto",
                                "paddingRight": "10px",
                            },
                        ),
                    ],
                    className="four columns",
                ),  # 4/12 columns width
                # Right Column: Graph + View Tabs
                html.Div(
                    [
                        dcc.Tabs(
                            id="view-tabs",
                            value="throughput",
                            children=[
                                dcc.Tab(label="Throughput", value="throughput"),
                                dcc.Tab(label="CDFs", value="cdfs"),
                            ],
                            style={"marginBottom": "10px"},
                        ),
                        dcc.Graph(
                            id="throughput-graph",
                            style={"height": "75vh"},
                            config={"displayModeBar": True},
                        ),
                    ],
                    className="eight columns",
                ),  # 8/12 columns width
            ],
            className="row",
            style={"maxWidth": "100%", "margin": "20px"},
        ),
    ]
)


# --- Callbacks ---


@app.callback(
    Output("controls-container", "children"), Input("resource-dropdown", "value")
)
def update_controls(selected_resources):
    """
    Generates a control card for EACH selected resource.
    Uses 'Index Mapping' for sliders to prevent crowding.
    """
    if not selected_resources:
        return html.Div(
            "Please select a resource.", style={"color": "#888", "fontStyle": "italic"}
        )

    # Ensure it's a list (Dropdown can return string if multi=False, but we set multi=True)
    if not isinstance(selected_resources, list):
        selected_resources = [selected_resources]

    all_controls = []

    for res_key in selected_resources:
        if res_key not in data:
            continue

        params, _ = data[res_key]
        params = np.atleast_2d(params)

        param_names = get_param_names(res_key)
        if not param_names or len(param_names) != params.shape[1]:
            param_names = [f"Param {i}" for i in range(params.shape[1])]

        # Card for this resource
        card_content = [
            html.H5(
                res_key,
                style={
                    "borderBottom": "1px solid #ccc",
                    "paddingBottom": "5px",
                    "marginTop": "0",
                },
            ),
        ]

        for col_idx, name in enumerate(param_names):
            col_vals = params[:, col_idx].astype(int)
            unique_vals = sorted(list(np.unique(col_vals)))

            # --- IMPROVED SLIDER LOGIC ---
            # Instead of using value=actual_value, we use value=index.
            # This ensures even spacing regardless of whether values are [1,2,3] or [1, 2, 4, 8, 16...]

            # Create readable marks
            count = len(unique_vals)
            step_size = 1
            if count > 10:
                step_size = count // 5  # Show roughly 5-6 labels max

            marks = {}
            for i, val in enumerate(unique_vals):
                if i % step_size == 0 or i == count - 1:
                    marks[i] = str(val)

            control_id = {
                "type": "param-control",
                "resource": res_key,
                "index": col_idx,
            }

            # Label ID for dynamic updates
            label_id = {"type": "param-label", "resource": res_key, "index": col_idx}

            # Initialize label with current value (index 0)
            initial_val = unique_vals[0]
            label_text = f"{name}: {initial_val}"

            label = html.Label(
                label_text,
                id=label_id,
                style={"fontWeight": "bold", "fontSize": "0.9em", "marginTop": "10px"},
            )

            slider = dcc.Slider(
                id=control_id,
                min=0,
                max=len(unique_vals) - 1,
                step=1,
                value=0,  # Default to first index
                marks=marks,
                # Disabled default tooltip so it doesn't show the index "0, 1, 2"
                tooltip=None,
            )

            card_content.append(html.Div([label, slider]))

        card = html.Div(
            card_content,
            style={
                "backgroundColor": "white",
                "padding": "15px",
                "marginBottom": "15px",
                "borderRadius": "5px",
                "boxShadow": "0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24)",
            },
        )
        all_controls.append(card)

    return all_controls


@app.callback(
    Output({"type": "param-label", "resource": MATCH, "index": MATCH}, "children"),
    [
        Input(
            {"type": "param-control", "resource": MATCH, "index": MATCH}, "drag_value"
        ),
        Input({"type": "param-control", "resource": MATCH, "index": MATCH}, "value"),
    ],
    State({"type": "param-control", "resource": MATCH, "index": MATCH}, "id"),
)
def update_dynamic_label(drag_val, set_val, slider_id):
    """
    Updates the label text in real-time while dragging.
    Uses drag_value for immediate feedback without triggering heavy graph updates.
    """
    ctx_triggered = ctx.triggered_id

    # Determine which value to use (drag takes precedence if active)
    idx = set_val  # Default
    if (
        ctx.triggered
        and "drag_value" in ctx.triggered[0]["prop_id"]
        and drag_val is not None
    ):
        idx = drag_val
    elif set_val is not None:
        idx = set_val

    # Recover Data
    res_key = slider_id["resource"]
    col_idx = slider_id["index"]

    if res_key not in data:
        return "Unknown"

    params, _ = data[res_key]
    params = np.atleast_2d(params)

    # Get param name
    param_names = get_param_names(res_key)
    if not param_names or len(param_names) != params.shape[1]:
        param_name = f"Param {col_idx}"
    else:
        param_name = param_names[col_idx]

    # Get real value from index
    col_vals = params[:, col_idx].astype(int)
    unique_vals = sorted(list(np.unique(col_vals)))

    # Safety clamp
    if idx >= len(unique_vals):
        idx = len(unique_vals) - 1
    if idx < 0:
        idx = 0

    real_val = unique_vals[int(idx)]

    return f"{param_name}: {real_val}"


@app.callback(
    Output("throughput-graph", "figure"),
    [
        Input("view-tabs", "value"),
        Input("resource-dropdown", "value"),
        Input({"type": "param-control", "resource": ALL, "index": ALL}, "value"),
    ],
    [State({"type": "param-control", "resource": ALL, "index": ALL}, "id")],
)
def update_graph(view, selected_resources, param_indices, param_ids):
    """
    Reconstructs the parameters from the slider indices and plots either:
      - view == "throughput": time-series throughput lines (existing behavior)
      - view == "cdfs": percentile CDFs (raw + weighted) and a horizontal mean line
    """
    fig = go.Figure()

    if not selected_resources:
        fig.update_layout(
            template="plotly_white",
            xaxis_title="Window ID" if view == "throughput" else "Percentile",
            yaxis_title="Throughput",
            annotations=[
                dict(
                    text="Select a resource to begin",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                )
            ],
        )
        return fig

    if not isinstance(selected_resources, list):
        selected_resources = [selected_resources]

    # Map the flat list of inputs back to structured data: {resource_key: {col_idx: slider_index_value}}
    current_settings = {}
    if param_ids and param_indices:
        for val, id_dict in zip(param_indices, param_ids):
            res = id_dict["resource"]
            idx = id_dict["index"]
            if res not in current_settings:
                current_settings[res] = {}
            current_settings[res][idx] = val

    # Color palette cycle for different resources
    colors = [
        "#EF553B",
        "#636EFA",
        "#00CC96",
        "#AB63FA",
        "#FFA15A",
        "#19D3F3",
        "#FF6692",
        "#B6E880",
    ]

    for i, res_key in enumerate(selected_resources):
        if res_key not in data:
            continue

        params, thr = data[res_key]
        params = np.atleast_2d(params)
        num_combos, num_windows = thr.shape
        x_axis = np.arange(num_windows)

        # Determine Color for this resource
        color = colors[i % len(colors)]

        slider_vals_for_res = current_settings.get(res_key, {})
        actual_param_vals = []
        for col_idx in range(params.shape[1]):
            col_data = params[:, col_idx].astype(int)
            unique_options = sorted(list(np.unique(col_data)))
            slider_idx = slider_vals_for_res.get(col_idx, 0)
            if slider_idx >= len(unique_options):
                slider_idx = len(unique_options) - 1
            val = unique_options[slider_idx]
            actual_param_vals.append(val)

        best_idx = _find_best_index_for_params(params, actual_param_vals)

        if view == "throughput":
            # Existing throughput plotting behavior
            if len(selected_resources) == 1:
                indices_to_plot = range(num_combos)
                if num_combos > 200:
                    indices_to_plot = np.linspace(0, num_combos - 1, 200, dtype=int)
                for ctx_i in indices_to_plot:
                    fig.add_trace(
                        go.Scatter(
                            x=x_axis,
                            y=thr[ctx_i],
                            mode="lines",
                            line=dict(color="lightgray", width=1),
                            opacity=0.2,
                            hoverinfo="skip",
                            showlegend=False,
                        )
                    )

            label_parts = [
                f"{n}={v}"
                for n, v in zip(get_param_names(res_key) or [], actual_param_vals)
            ]
            label = f"{res_key} ({', '.join(label_parts)})"
            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=thr[best_idx],
                    mode="lines",
                    line=dict(color=color, width=2.5),
                    name=label,
                    showlegend=True,
                )
            )

        elif view == "cdfs":
            # CDF view: use canonical compute_cdf_features output (50 raw, 50 weighted, mean)
            num_points = 50
            cdf_raw, cdf_weighted, mean_val = compute_cdf_features_safe(
                thr[best_idx], num_points=num_points
            )

            # X layout: use the same percentile axis for both curves (overlayed)
            percentiles = np.linspace(1, 99, num_points)

            fig.add_trace(
                go.Scatter(
                    x=percentiles,
                    y=cdf_raw,
                    mode="lines",
                    line=dict(color=color, width=2),
                    name=f"{res_key} (raw)",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=percentiles,
                    y=cdf_weighted,
                    mode="lines",
                    line=dict(color=color, width=2, dash="dash"),
                    name=f"{res_key} (weighted)",
                )
            )
            # Mean as a single marker (centered on the percentile axis) and labeled
            mean_x = float(np.mean(percentiles))
            fig.add_trace(
                go.Scatter(
                    x=[mean_x],
                    y=[mean_val],
                    mode="markers+text",
                    marker=dict(color=color, size=9, symbol="diamond"),
                    text=[f"mean: {mean_val:.2f}"],
                    textposition="top center",
                    hovertemplate=f"{res_key} mean: {{y:.2f}}<extra></extra>",
                    name=f"{res_key} (mean)",
                    showlegend=True,
                )
            )

    # Final layout adjustments depending on view
    if view == "throughput":
        fig.update_layout(
            title="Throughput Comparison",
            xaxis_title="Window ID",
            yaxis_title="Throughput",
            template="plotly_white",
            hovermode="x unified",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            margin=dict(l=40, r=40, t=80, b=40),
        )
    else:
        fig.update_layout(
            title="Throughput CDFs (percentile → throughput)",
            xaxis_title="Percentile",
            yaxis_title="Throughput",
            template="plotly_white",
            hovermode="x unified",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            margin=dict(l=40, r=40, t=80, b=40),
        )

    return fig


if __name__ == "__main__":
    app.run(debug=True)
