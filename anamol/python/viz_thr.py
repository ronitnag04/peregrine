import dash
from dash import dcc, html, Input, Output, State, ALL, ctx
import plotly.graph_objects as go
import numpy as np
import os
import sys
from pathlib import Path

# --- Data Loading / Mocking Layer ---

# We attempt to import the modules referenced in your script.
# If they exist, we use them. If not, we generate mock data so the app runs.
try:
    import utils
    import models

    # 1. Load data exactly as in your script
    # Use the directory of this code file so default_output is ../output relative to the file
    script_dir = Path(__file__).resolve().parent
    default_output = (script_dir.parent / "output").resolve()
    output_dir = Path(os.environ.get("ANAMOL_OUTPUT_DIR", str(default_output)))

    print(f"Attempting to load data from: {output_dir}")
    data = utils.load_all_throughputs(output_dir=str(output_dir))

    if not data:
        raise RuntimeError("No data found, switching to mock mode.")

    # Helper to get param names safely using your models module
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
    # Params: Size (32, 64, 128), Width (4, 8) -> 6 combinations
    rob_params = np.array([[32, 4], [32, 8], [64, 4], [64, 8], [128, 4], [128, 8]])
    # Generate random throughput walks
    rob_thr = np.abs(np.cumsum(np.random.randn(6, 50), axis=1) + 10)
    data["ROB (Mock)"] = (rob_params, rob_thr)

    # Mock 2: Store Queue
    # Params: Entries (16 to 32) -> 17 combinations
    sq_params = np.arange(16, 33).reshape(-1, 1)
    sq_thr = np.abs(np.cumsum(np.random.randn(len(sq_params), 50), axis=1) + 5)
    data["Store Queue (Mock)"] = (sq_params, sq_thr)

    def get_param_names(res_key):
        if "ROB" in res_key:
            return ["ROB Size", "Issue Width"]
        elif "Store" in res_key:
            return ["Entries"]
        return None


# --- Core Logic Adapted from visualize_thr.py ---


def _find_best_index_for_params(params: np.ndarray, sel_vals):
    """
    Find index of the parameter combination matching sel_vals.
    Exact match preferred, else nearest neighbor (L2).
    """
    sel = np.asarray(sel_vals).reshape(-1)

    # Handle scalar param case (N, 1)
    if params.shape[1] == 1:
        sel = sel.reshape(1)
        mask = params[:, 0] == sel[0]
    else:
        # Handle multi-dim param case (N, M)
        # Check if all columns match
        mask = np.all(params == sel, axis=1)

    if np.any(mask):
        return int(np.where(mask)[0][0])

    # Fallback: Nearest Neighbor (L2 distance)
    diffs = params.astype(float) - sel.astype(float)
    d2 = np.sum(diffs * diffs, axis=1)
    return int(np.argmin(d2))


# --- Dash Application Layout ---

app = dash.Dash(
    __name__, external_stylesheets=["https://codepen.io/chriddyp/pen/bWLwgP.css"]
)
app.title = "Throughput Visualizer"

# Prepare Dropdown Options
resource_options = [{"label": k, "value": k} for k in sorted(data.keys())]
default_resource = resource_options[0]["value"] if resource_options else None

app.layout = html.Div(
    [
        # Header
        html.Div(
            [
                html.H2("Resource Throughput Explorer", style={"marginBottom": "10px"}),
                html.P(
                    "Select a resource and adjust parameters to compare throughput traces.",
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
                            "Select Resource:",
                            style={"fontWeight": "bold", "fontSize": "1.1em"},
                        ),
                        dcc.Dropdown(
                            id="resource-dropdown",
                            options=resource_options,
                            value=default_resource,
                            clearable=False,
                            style={"marginBottom": "30px"},
                        ),
                        html.Div(
                            html.Label(
                                "Parameters:",
                                style={"fontWeight": "bold", "fontSize": "1.1em"},
                            )
                        ),
                        # This container will be populated dynamically by the callback
                        html.Div(
                            id="controls-container",
                            style={
                                "backgroundColor": "#f8f9fa",
                                "padding": "20px",
                                "borderRadius": "8px",
                                "border": "1px solid #ddd",
                            },
                        ),
                    ],
                    className="four columns",
                ),
                # Right Column: Graph
                html.Div(
                    [
                        dcc.Graph(
                            id="throughput-graph",
                            style={"height": "70vh"},
                            config={"displayModeBar": True},
                        )
                    ],
                    className="eight columns",
                ),
            ],
            className="row",
            style={"margin": "20px"},
        ),
    ]
)


# --- Callbacks ---


@app.callback(
    Output("controls-container", "children"), Input("resource-dropdown", "value")
)
def update_controls(res_key):
    """
    Creates sliders or dropdowns dynamically based on the number and range
    of parameters for the selected resource.
    """
    if not res_key or res_key not in data:
        return html.Div("No data available.")

    params, _ = data[res_key]
    params = np.atleast_2d(params)

    # Retrieve parameter names (e.g., "Size", "Width")
    param_names = get_param_names(res_key)
    # Fallback names if none provided
    if not param_names or len(param_names) != params.shape[1]:
        param_names = [f"Param {i}" for i in range(params.shape[1])]

    controls = []

    # Iterate over each parameter column to create a widget
    for col_idx, name in enumerate(param_names):
        col_vals = params[:, col_idx].astype(int)
        unique_vals = np.unique(col_vals)
        unique_vals_list = sorted(list(unique_vals))

        # We assign a Pattern Matching ID to the input
        # 'index': col_idx ensures we can reconstruct the order later
        control_id = {"type": "param-control", "index": col_idx}

        label = html.Label(f"{name}", style={"fontWeight": "bold", "marginTop": "15px"})

        # Logic from visualize_thr.py: Choose widget based on cardinality
        if len(unique_vals) <= 12:
            # Small set: Discrete Slider
            # map values to string labels
            marks = {int(v): str(v) for v in unique_vals}
            widget = dcc.Slider(
                id=control_id,
                min=min(unique_vals_list),
                max=max(unique_vals_list),
                step=None,  # Snaps to marks only
                marks=marks,
                value=unique_vals_list[0],
            )
        elif len(unique_vals) <= 50:
            # Medium set: Dropdown
            widget = dcc.Dropdown(
                id=control_id,
                options=[{"label": str(v), "value": int(v)} for v in unique_vals_list],
                value=unique_vals_list[0],
                clearable=False,
            )
        else:
            # Large set: Numeric Slider (continuous-ish)
            widget = dcc.Slider(
                id=control_id,
                min=min(unique_vals_list),
                max=max(unique_vals_list),
                step=1,
                value=unique_vals_list[0],
                tooltip={"placement": "bottom", "always_visible": True},
            )

        controls.append(html.Div([label, widget]))

    return controls


@app.callback(
    Output("throughput-graph", "figure"),
    [
        Input({"type": "param-control", "index": ALL}, "value"),
        Input("resource-dropdown", "value"),
    ],
)
def update_graph(param_values, res_key):
    """
    Redraws the graph.
    Inputs:
       param_values: List of values from all dynamic sliders.
       res_key: Selected resource name.
    """
    if not res_key or res_key not in data:
        return go.Figure()

    params, thr = data[res_key]
    params = np.atleast_2d(params)
    num_combos, num_windows = thr.shape
    x_axis = np.arange(num_windows)

    # 1. Determine which combo is selected
    # Note: param_values list order corresponds to the creation order (params columns)

    # Guard against partial updates (e.g. while controls are being built)
    if not param_values or len(param_values) != params.shape[1]:
        # Default to index 0 if inputs aren't ready
        best_idx = 0
    else:
        best_idx = _find_best_index_for_params(params, param_values)

    # 2. Build the Figure
    fig = go.Figure()

    # A. Add Faint Context Traces (Gray Lines)
    # Performance check: If > 500 traces, sample them to keep the browser responsive
    indices_to_plot = range(num_combos)
    if num_combos > 500:
        indices_to_plot = np.linspace(0, num_combos - 1, 500, dtype=int)

    for i in indices_to_plot:
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=thr[i],
                mode="lines",
                line=dict(color="lightgray", width=1),
                opacity=0.3,
                hoverinfo="skip",  # Disable hover for background lines for performance
                showlegend=False,
            )
        )

    # B. Add Highlighted Trace (Crimson Line)
    selected_y = thr[best_idx]

    # Generate title string
    param_names = get_param_names(res_key)
    if not param_names:
        param_names = [f"p{i}" for i in range(params.shape[1])]

    # Get actual values of the selected combo
    actual_vals = params[best_idx]
    title_parts = [f"{n}={v}" for n, v in zip(param_names, actual_vals)]
    title_str = f"<b>{res_key}</b><br>Combo {best_idx}: ({', '.join(title_parts)})"

    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=selected_y,
            mode="lines+markers",
            line=dict(color="crimson", width=3),
            name="Selected",
            showlegend=True,
        )
    )

    fig.update_layout(
        title=title_str,
        xaxis_title="Window ID",
        yaxis_title="Throughput",
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=80, b=40),
    )

    return fig


if __name__ == "__main__":
    app.run(debug=True)
