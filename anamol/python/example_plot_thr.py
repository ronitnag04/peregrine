# %%
# Plotly interactive plot for ROB throughputs

import numpy as np
import plotly.graph_objs as go
import ipywidgets as widgets
from IPython.display import display, HTML
import utils
import models
from pathlib import Path
import os

# %%
# load throughput data (output dir defaults to ../output relative to this file)
script_dir = Path(__file__).resolve().parent
default_output = (script_dir / ".." / "output").resolve()
output_dir = Path(os.environ.get("ANAMOL_OUTPUT_DIR", str(default_output)))
data = utils.load_all_throughputs(output_dir=str(output_dir))
if not data:
    raise RuntimeError(f"No throughput files found in {output_dir}")

# pick the resource that refers to ROB (prefer enum, fallback to name scan)
rob_key = models.resource_key(models.Resource.ROB)
if rob_key not in data:
    rob_key = None
    for k in data.keys():
        if "rob" in k.lower():
            rob_key = k
            break
if rob_key is None:
    raise RuntimeError(
        "No resource with 'rob' found in loaded data. Keys: "
        + ", ".join(sorted(data.keys()))
    )


def _find_best_index_for_params(params: np.ndarray, sel_vals):
    # sel_vals: list/tuple/array of length param_cols
    sel = np.asarray(sel_vals).reshape(-1)
    if params.shape[1] == 1:
        sel = sel.reshape(1)
    # exact match
    if params.shape[1] == 1:
        mask = params[:, 0] == sel[0]
    else:
        mask = np.all(params == sel, axis=1)
    if np.any(mask):
        return int(np.where(mask)[0][0])
    # fallback to nearest (L2)
    diffs = params.astype(float) - sel.astype(float)
    d2 = np.sum(diffs * diffs, axis=1)
    return int(np.argmin(d2))


def _make_param_sliders(params: np.ndarray, param_names):
    """
    Return (sliders_list, container_widget, get_selected_vals)
    get_selected_vals() -> tuple of selected slider values

    Use discrete widgets for discrete parameter values:
      - SelectionSlider when <= 12 unique options (good for small ordered sets)
      - Dropdown when between 13 and 50 options
      - IntSlider fallback when values are many/dense
    """
    sliders = []
    for col, name in enumerate(param_names):
        col_vals = params[:, col].astype(int)
        unique_vals = np.unique(col_vals)
        init = int(col_vals[0])

        if unique_vals.size <= 12:
            # nice slider that shows all discrete choices
            widget = widgets.SelectionSlider(
                options=list(unique_vals),
                value=init,
                description=name + ":",
                continuous_update=False,
                orientation="horizontal",
                layout=widgets.Layout(width="300px"),
            )
        elif unique_vals.size <= 50:
            # dropdown for larger choice sets
            widget = widgets.Dropdown(
                options=list(unique_vals),
                value=init,
                description=name + ":",
                layout=widgets.Layout(width="220px"),
            )
        else:
            # fallback to numeric slider for dense ranges
            widget = widgets.IntSlider(
                value=init,
                min=int(unique_vals.min()),
                max=int(unique_vals.max()),
                step=1,
                description=name + ":",
            )

        sliders.append(widget)

    if len(sliders) == 1:
        container = sliders[0]
    else:
        container = widgets.HBox(sliders)

    def get_selected_vals():
        return tuple(int(s.value) for s in sliders)

    return sliders, container, get_selected_vals


def plot_resource(data, resource):
    """
    Create and display an interactive throughput plot for a given resource.

    resource: either a resource key (str), a models.Resource enum, or an integer index into data.keys().
    Returns: (fig, sliders_list, info_widget)
    """
    # resolve resource key by index, enum, exact key, or case-insensitive substring
    if isinstance(resource, int):
        keys = list(data.keys())
        if resource < 0 or resource >= len(keys):
            raise IndexError(f"resource index {resource} out of range")
        res_key = keys[resource]
    elif isinstance(resource, models.Resource):
        res_key = models.resource_key(resource)
    else:
        if resource in data:
            res_key = resource
        else:
            matches = [k for k in data.keys() if str(resource).lower() in k.lower()]
            if not matches:
                raise KeyError(
                    f"No resource matching '{resource}' in data. Keys: {', '.join(sorted(data.keys()))}"
                )
            res_key = matches[0]

    params, thr = data[res_key]  # thr shape: (num_combos, num_windows)
    num_combos, num_windows = thr.shape
    x = np.arange(num_windows)

    # param names and sliders
    param_names = models.get_resource_param_spec(res_key)
    # ensure params has shape (N, param_cols)
    params = np.atleast_2d(params)
    if params.shape[1] != len(param_names):
        # fallback: trim or pad names
        param_names = [f"param{i}" for i in range(params.shape[1])]

    sliders, sliders_container, get_selected_vals = _make_param_sliders(
        params, param_names
    )

    # initial selected index based on slider initial values
    init_idx = _find_best_index_for_params(params, get_selected_vals())

    # create FigureWidget with initial trace
    init_vals = tuple(int(x) for x in np.atleast_1d(params[init_idx]))
    init_pairs = ", ".join(f"{n}={v}" for n, v in zip(param_names, init_vals))
    fig = go.FigureWidget(
        layout=dict(
            title=f"{res_key} — combo {init_idx} ({init_pairs})",
            xaxis_title="window id",
            yaxis_title="throughput",
        )
    )

    # faint traces for all combos
    for i in range(num_combos):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=thr[i],
                mode="lines",
                line=dict(color="gray"),
                opacity=0.12,
                name=f"combo {i}",
                showlegend=False,
            )
        )

    # highlighted trace (last trace index)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=thr[init_idx],
            mode="lines+markers",
            line=dict(color="crimson", width=2),
            name="selected",
            showlegend=True,
        )
    )

    idx_display = HTML(
        f"<b>Resource:</b> {res_key} &nbsp; <b>combos:</b> {num_combos} &nbsp; <b>windows:</b> {num_windows}"
    )

    def _update_highlighted_trace():
        sel_vals = get_selected_vals()
        idx = _find_best_index_for_params(params, sel_vals)
        fig.data[-1].y = thr[idx]
        param_pairs = ", ".join(f"{n}={v}" for n, v in zip(param_names, sel_vals))
        fig.layout.title = f"{res_key} — combo {idx} ({param_pairs})"

    for s in sliders if isinstance(sliders, list) else [sliders]:
        s.observe(lambda change: _update_highlighted_trace(), names="value")

    display(idx_display, sliders_container, fig)
    # return list of sliders (normalize), figure, info widget
    if isinstance(sliders, list):
        return fig, sliders, idx_display
    else:
        return fig, [sliders], idx_display


# use the helper to show the ROB resource plot (calls display)
fig_obj, slider_objs, info_obj = plot_resource(data, models.Resource.ROB)

# %%
res = models.resource_key(models.Resource.ROB)  # string key for indexing data
params, thr = data[res]  # params: (num_combos, 1 or 2), thr: (num_combos, num_windows)

param_names = models.get_resource_param_spec(res)
for i, row in enumerate(params):
    vals = [int(v) for v in row]
    combo_dict = dict(zip(param_names, vals))
    print(f"combo {i}: {combo_dict}  → throughput shape: {thr[i].shape}")

# %%
fig_obj, slider_objs, info_obj = plot_resource(
    data, models.Resource.LOAD_LS_PIPES_LOWER
)

# %%
fig_obj, slider_objs, info_obj = plot_resource(data, models.Resource.STORE_QUEUE)

# %%
# show in browser via renderer
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"  # opens fig.show() in your default browser
fig = go.Figure(data=go.Scatter(y=[1, 3, 2]))
fig.show()

if __name__ == "__main__":
    # If running as a script (not in a notebook), build a browser-friendly Plotly figure
    # with a native Plotly slider that toggles the highlighted trace per parameter combo.
    import plotly.io as pio

    pio.renderers.default = "browser"

    # Use the last resource plotted (res, params, thr should be defined above)
    try:
        params  # ensure variables exist
        thr
    except NameError:
        raise RuntimeError(
            "params/thr not available - run plot_resource(...) earlier in the script"
        )

    params = np.atleast_2d(params)
    num_combos, num_windows = thr.shape
    x = np.arange(num_windows)
    param_names = models.get_resource_param_spec(res)
    if params.shape[1] != len(param_names):
        param_names = [f"param{i}" for i in range(params.shape[1])]

    # build figure: faint traces for all combos + one highlighted trace per combo (only one visible at a time)
    fig_browser = go.Figure()
    # faint traces
    for i in range(num_combos):
        fig_browser.add_trace(
            go.Scatter(
                x=x,
                y=thr[i],
                mode="lines",
                line=dict(color="gray"),
                opacity=0.12,
                name=f"combo {i}",
                showlegend=False,
            )
        )
    # highlighted traces (one per combo, only the initial one visible)
    visible_init = [False] * num_combos
    init_idx = 0
    visible_init[init_idx] = True
    for i in range(num_combos):
        fig_browser.add_trace(
            go.Scatter(
                x=x,
                y=thr[i],
                mode="lines+markers",
                line=dict(color="crimson", width=2),
                name="selected",
                showlegend=False,
                visible=(i == init_idx),
            )
        )

    # slider steps: each step makes exactly one highlighted trace visible (faint traces stay visible)
    steps = []
    for i in range(num_combos):
        vals = tuple(int(x) for x in np.atleast_1d(params[i]))
        label = ", ".join(f"{n}={v}" for n, v in zip(param_names, vals))
        vis = [True] * num_combos + [j == i for j in range(num_combos)]
        step = dict(
            method="restyle",
            args=["visible", vis],
            label=label,
        )
        steps.append(step)

    slider = dict(
        active=init_idx,
        currentvalue={"prefix": f"{res} — "},
        pad={"t": 50},
        steps=steps,
    )

    fig_browser.update_layout(
        title=f"{res} — combo {init_idx} ("
        + ", ".join(
            f"{n}={v}"
            for n, v in zip(
                param_names, tuple(int(x) for x in np.atleast_1d(params[init_idx]))
            )
        )
        + ")",
        xaxis_title="window id",
        yaxis_title="throughput",
        sliders=[slider],
    )

    # open interactive plot in browser
    fig_browser.show()
