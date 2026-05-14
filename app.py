import streamlit as st
import streamlit.components.v1 as components
import networkx as nx
import pandas as pd
import numpy as np
import random
import plotly.graph_objects as go
import plotly.express as px
from pyvis.network import Network
import tempfile
import os

from centrality.classical.degree import DegreeCentrality, WeightedDegreeCentrality
from centrality.classical.closeness import ClosenessCentrality
from centrality.classical.betweenness import BetweennessCentrality
from centrality.classical.eigenvector import EigenvectorCentrality
from centrality.weighted.weighted_closeness import WeightedClosenessCentrality
from centrality.weighted.weighted_betweenness import WeightedBetweennessCentrality
from centrality.weighted.weighted_eigenvector import WeightedEigenvectorCentrality
from centrality.directed.directed_degree import (
    InDegreeCentrality,
    OutDegreeCentrality,
    WeightedInDegreeCentrality,
    WeightedOutDegreeCentrality,
)
from centrality.directed.directed_betweenness import DirectedBetweennessCentrality
from centrality.directed.pagerank import PageRankCentrality
from centrality.directed.directed_closeness import DirectedClosenessCentrality
from centrality.directed.directed_weighted_closeness import DirectedWeightedClosenessCentrality
from centrality.directed.directed_weighted_betweenness import DirectedWeightedBetweennessCentrality
from centrality.directed.directed_eigenvector import (
    DirectedEigenvectorCentrality,
    DirectedWeightedEigenvectorCentrality,
)
from centrality.directed.weighted_pagerank import WeightedPageRankCentrality
from centrality.quota_based import BundleIndexCentrality, PivotalIndexCentrality
from utils.comparator import CentralityComparator

# Page config

st.set_page_config(
    page_title="Network Centrality Analyzer",
    page_icon="🔵",
    layout="wide"
)

st.title("Network Centrality Analyzer")
st.markdown("Compute and compare centrality indices for network structures.")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Analysis", "Compare Networks", "Node Impact", "Data Laboratory", "About"
])

# Sidebar

st.sidebar.header("Settings")

GRAPH_OPTIONS = [
    "Karate Club",
    "Les Misérables",
    "Florentine Families",
    "Davis Women's Club",
    "Petersen Graph",
    "Random Graph",
    "Upload CSV",
]

RANDOM_GRAPH_MODELS = [
    "Erdős-Rényi",
    "Barabási-Albert",
    "Watts-Strogatz",
    "Scale-Free Directed",
]

graph_option = st.sidebar.selectbox(
    "Select network",
    GRAPH_OPTIONS
)


def random_graph_controls(container, key_prefix):
    params = {}
    with container.expander("Random graph parameters", expanded=True):
        params["model"] = st.selectbox(
            "Model",
            RANDOM_GRAPH_MODELS,
            key=f"{key_prefix}_random_model",
        )
        params["n"] = st.slider(
            "Number of nodes",
            min_value=5,
            max_value=300,
            value=50,
            step=5,
            key=f"{key_prefix}_random_n",
        )
        params["seed"] = st.number_input(
            "Seed",
            min_value=0,
            max_value=1_000_000,
            value=42,
            step=1,
            key=f"{key_prefix}_random_seed",
        )

        if params["model"] == "Erdős-Rényi":
            params["p"] = st.slider(
                "Edge probability",
                min_value=0.01,
                max_value=1.0,
                value=0.08,
                step=0.01,
                key=f"{key_prefix}_er_p",
            )
        elif params["model"] == "Barabási-Albert":
            max_m = max(1, min(20, params["n"] - 1))
            params["m"] = st.slider(
                "Edges per new node",
                min_value=1,
                max_value=max_m,
                value=min(2, max_m),
                step=1,
                key=f"{key_prefix}_ba_m",
            )
        elif params["model"] == "Watts-Strogatz":
            max_k = max(2, min(30, params["n"] - 1))
            if max_k % 2 == 1:
                max_k -= 1
            params["k"] = st.slider(
                "Nearest neighbors",
                min_value=2,
                max_value=max_k,
                value=min(4, max_k),
                step=2,
                key=f"{key_prefix}_ws_k",
            )
            params["p"] = st.slider(
                "Rewiring probability",
                min_value=0.0,
                max_value=1.0,
                value=0.20,
                step=0.01,
                key=f"{key_prefix}_ws_p",
            )
        else:
            params["alpha"] = st.slider(
                "Preferential attachment probability",
                min_value=0.05,
                max_value=0.90,
                value=0.41,
                step=0.01,
                key=f"{key_prefix}_sf_alpha",
            )
            params["beta"] = st.slider(
                "Internal edge probability",
                min_value=0.05,
                max_value=0.90,
                value=0.54,
                step=0.01,
                key=f"{key_prefix}_sf_beta",
            )
            params["gamma"] = max(0.01, 1.0 - params["alpha"] - params["beta"])
            st.caption(f"New-node incoming probability: {params['gamma']:.2f}")

    return params


analysis_random_params = None
if graph_option == "Random Graph":
    analysis_random_params = random_graph_controls(st.sidebar, "analysis")

st.sidebar.header("Centrality Indices")

INDEX_GROUPS = {
    "Classical": ["Degree", "Closeness", "Betweenness", "Eigenvector"],
    "Weighted": [
        "Weighted Degree", "Weighted Closeness",
        "Weighted Betweenness", "Weighted Eigenvector"
    ],
    "Directed": [
        "In-Degree", "Out-Degree", "Directed Closeness",
        "Directed Betweenness", "Directed Eigenvector", "PageRank"
    ],
    "Directed Weighted": [
        "Weighted In-Degree", "Weighted Out-Degree",
        "Directed Weighted Closeness", "Directed Weighted Betweenness",
        "Directed Weighted Eigenvector", "Weighted PageRank"
    ],
    "Quota-based": ["Bundle Index", "Pivotal Index"],
}

DIRECTED_INDICES = set(INDEX_GROUPS["Directed"] + INDEX_GROUPS["Directed Weighted"])
UNDIRECTED_INDICES = set(INDEX_GROUPS["Classical"] + INDEX_GROUPS["Weighted"])
QUOTA_BASED_INDICES = set(INDEX_GROUPS["Quota-based"])

selected = []
for group_name, group_indices in INDEX_GROUPS.items():
    default_group = ["Degree", "Closeness", "Betweenness"] if group_name == "Classical" else []
    with st.sidebar.expander(group_name, expanded=group_name == "Classical"):
        selected.extend(
            st.multiselect(
                f"{group_name} indices",
                group_indices,
                default=default_group,
                key=f"indices_{group_name.lower().replace(' ', '_')}",
                label_visibility="collapsed",
            )
        )

node_attributes_df = None
node_id_column = None
quota_column = None
quota_indices_selected = any(name in QUOTA_BASED_INDICES for name in selected)
quota_max_group_size = 3
node_attributes_file = None

if quota_indices_selected:
    st.sidebar.header("Node Attributes")
    node_attributes_file = st.sidebar.file_uploader(
        "Upload node attributes CSV",
        type=["csv"],
        help="Required for Bundle Index and Pivotal Index. Include one node id column and one numeric quota column.",
    )
    quota_max_group_size = st.sidebar.slider(
        "Max group size for quota indices",
        min_value=1,
        max_value=6,
        value=3,
        step=1,
        help="Limits the number of incoming neighbors considered together. Larger values can become slow.",
    )
    with st.sidebar.expander("Node attributes CSV example"):
        st.code(
            "node,quota,gdp,population\n"
            "v1,1,100,20\n"
            "v2,1,250,35\n"
            "v3,1,180,28",
            language="text",
        )

    if node_attributes_file is not None:
        node_attributes_file.seek(0)
        try:
            node_attributes_df = pd.read_csv(node_attributes_file)
            if not node_attributes_df.empty:
                columns = list(node_attributes_df.columns)
                node_id_column = st.sidebar.selectbox(
                    "Node id column",
                    columns,
                    key="node_id_column",
                )
                numeric_columns = [
                    col for col in columns
                    if col != node_id_column and pd.api.types.is_numeric_dtype(node_attributes_df[col])
                ]
                quota_options = numeric_columns or [col for col in columns if col != node_id_column]
                if quota_options:
                    quota_column = st.sidebar.selectbox(
                        "Quota column",
                        quota_options,
                        key="quota_column",
                    )
                else:
                    st.sidebar.warning("Attributes CSV needs at least one quota column.")
        except Exception as exc:
            st.sidebar.warning(f"Could not read node attributes CSV: {exc}")

# Helpers

DIRECTED_DATASETS = {"Scale-Free Directed (Barabási-Albert)"}

INDEX_MAP = {
    "Degree": DegreeCentrality,
    "Closeness": ClosenessCentrality,
    "Betweenness": BetweennessCentrality,
    "Eigenvector": EigenvectorCentrality,
    "Weighted Degree": WeightedDegreeCentrality,
    "Weighted Closeness": WeightedClosenessCentrality,
    "Weighted Betweenness": WeightedBetweennessCentrality,
    "Weighted Eigenvector": WeightedEigenvectorCentrality,
    "In-Degree": InDegreeCentrality,
    "Out-Degree": OutDegreeCentrality,
    "Weighted In-Degree": WeightedInDegreeCentrality,
    "Weighted Out-Degree": WeightedOutDegreeCentrality,
    "Directed Closeness": DirectedClosenessCentrality,
    "Directed Betweenness": DirectedBetweennessCentrality,
    "Directed Eigenvector": DirectedEigenvectorCentrality,
    "Directed Weighted Closeness": DirectedWeightedClosenessCentrality,
    "Directed Weighted Betweenness": DirectedWeightedBetweennessCentrality,
    "Directed Weighted Eigenvector": DirectedWeightedEigenvectorCentrality,
    "PageRank": PageRankCentrality,
    "Weighted PageRank": WeightedPageRankCentrality,
    "Bundle Index": BundleIndexCentrality,
    "Pivotal Index": PivotalIndexCentrality,
}


def _coerce_like_node(value, node):
    try:
        return type(node)(value)
    except (TypeError, ValueError):
        return value


def build_quota_map(graph, attributes_df, node_col, quota_col):
    if attributes_df is None or node_col is None or quota_col is None:
        return {}

    raw = {}
    raw_by_text = {}
    for _, row in attributes_df.iterrows():
        try:
            node_id = row[node_col]
            quota = float(row[quota_col])
            raw[node_id] = quota
            raw_by_text[str(node_id)] = quota
        except (TypeError, ValueError):
            continue

    quotas = {}
    for node in graph.nodes():
        candidates = [
            node,
            str(node),
            _coerce_like_node(node, next(iter(raw), node)),
        ]
        for candidate in candidates:
            if candidate in raw:
                quotas[node] = raw[candidate]
                break
            if str(candidate) in raw_by_text:
                quotas[node] = raw_by_text[str(candidate)]
                break

    return quotas

def create_random_graph(params):
    model = params.get("model", "Barabási-Albert")
    n = int(params.get("n", 50))
    seed = int(params.get("seed", 42))

    if model == "Erdős-Rényi":
        return nx.erdos_renyi_graph(n, params.get("p", 0.08), seed=seed)
    if model == "Barabási-Albert":
        m = min(int(params.get("m", 2)), n - 1)
        return nx.barabasi_albert_graph(n, m, seed=seed)
    if model == "Watts-Strogatz":
        k = min(int(params.get("k", 4)), n - 1)
        if k % 2 == 1:
            k -= 1
        return nx.watts_strogatz_graph(n, max(2, k), params.get("p", 0.20), seed=seed)

    alpha = float(params.get("alpha", 0.41))
    beta = float(params.get("beta", 0.54))
    gamma = float(params.get("gamma", 0.05))
    total = alpha + beta + gamma
    alpha, beta, gamma = alpha / total, beta / total, gamma / total
    return nx.scale_free_graph(n, alpha=alpha, beta=beta, gamma=gamma, seed=seed)


@st.cache_data
def load_graph(option, uploaded_file=None, random_params=None):
    if option == "Karate Club":
        return nx.karate_club_graph()
    elif option == "Les Misérables":
        return nx.les_miserables_graph()
    elif option == "Florentine Families":
        return nx.florentine_families_graph()
    elif option == "Davis Women's Club":
        return nx.davis_southern_women_graph()
    elif option == "Petersen Graph":
        return nx.petersen_graph()
    elif option == "Random Graph":
        return create_random_graph(random_params or {})
    elif option == "Upload CSV" and uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        create_using = nx.DiGraph() if st.session_state.get("csv_directed", False) else nx.Graph()
        if df.shape[1] >= 3:
            return nx.from_pandas_edgelist(
                df, source=df.columns[0],
                target=df.columns[1],
                edge_attr=df.columns[2],
                create_using=create_using
            )
        else:
            return nx.from_pandas_edgelist(
                df, source=df.columns[0],
                target=df.columns[1],
                create_using=create_using
            )
    return nx.karate_club_graph()

def add_weights(G):
    G = G.copy()
    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        for u, v, k, data in G.edges(data=True, keys=True):
            if 'weight' not in data:
                G[u][v][k]['weight'] = random.uniform(0.5, 5.0)
    else:
        for u, v, data in G.edges(data=True):
            if 'weight' not in data:
                G[u][v]['weight'] = random.uniform(0.5, 5.0)
    return G

def check_compatibility(G, selected):
    is_directed = G.is_directed()
    warnings = []

    directed_selected = [s for s in selected if s in DIRECTED_INDICES]
    undirected_selected = [s for s in selected if s in UNDIRECTED_INDICES]

    if not is_directed and directed_selected:
        warnings.append(
            f"⚠️ **{', '.join(directed_selected)}** require a **directed** graph. "
            f"Please select **Random Graph → Scale-Free Directed** or upload a directed CSV, "
            f"or remove these indices."
        )
    if is_directed and undirected_selected:
        warnings.append(
            f"ℹ️ **{', '.join(undirected_selected)}** are designed for **undirected** graphs. "
            f"They will be computed on the directed graph but results may be less meaningful."
        )
    return warnings

def compute_all(G, selected, quotas=None, max_group_size=3):
    comp = CentralityComparator()
    comp.failures = {}
    for name in selected:
        try:
            if name in QUOTA_BASED_INDICES:
                instance = INDEX_MAP[name](
                    G,
                    quotas=quotas or {},
                    max_group_size=max_group_size,
                )
            else:
                instance = INDEX_MAP[name](G)
            instance.compute()
            comp.add(name, instance)
        except Exception as exc:
            comp.failures[name] = str(exc)
    return comp


def available_indices(selected, comp):
    return [name for name in selected if name in comp.results]


def show_compute_failures(comp):
    if getattr(comp, "failures", None):
        failed = ", ".join(comp.failures)
        st.warning(f"Could not compute: {failed}. Please check graph type and edge weights.")

def score_to_color(score, max_s):
    ratio = score / max_s if max_s > 0 else 0
    r = int(255 * ratio)
    g = int(80 * ratio)
    b = int(200 * (1 - ratio))
    return f"#{r:02x}{g:02x}{b:02x}"


def layout_positions(G, layout_name):
    if layout_name == "Physics simulation":
        return None

    simple_graph = nx.Graph(G)
    try:
        if layout_name == "Spring layout":
            return nx.spring_layout(simple_graph, seed=42, weight="weight")
        if layout_name == "Kamada-Kawai layout":
            return nx.kamada_kawai_layout(simple_graph, weight="weight")
        if layout_name == "Circular layout":
            return nx.circular_layout(simple_graph)
        if layout_name == "Shell layout":
            return nx.shell_layout(simple_graph)
        if layout_name == "Spectral layout":
            return nx.spectral_layout(simple_graph, weight="weight")
    except Exception:
        return nx.spring_layout(simple_graph, seed=42, weight="weight")

    return None


def build_pyvis(G, scores, selected, comp, highlight=None, visual_options=None):
    visual_options = visual_options or {}
    node_count = G.number_of_nodes()
    edge_count = G.number_of_edges()
    large_mode = visual_options.get("large_mode", node_count > 100 or edge_count > 300)
    show_labels = visual_options.get("show_labels", not large_mode)
    show_edges = visual_options.get("show_edges", True)
    node_scale = visual_options.get("node_scale", 0.65 if large_mode else 1.0)
    edge_opacity = visual_options.get("edge_opacity", 0.08 if large_mode else 0.22)
    edge_width = visual_options.get("edge_width", 0.5 if large_mode else 1.0)
    layout_name = visual_options.get("layout", "Physics simulation")
    positions = layout_positions(G, layout_name)
    physics_enabled = positions is None
    stabilization_iterations = 400 if large_mode else 180
    gravitational_constant = -9000 if large_mode else -3500
    spring_length = 220 if large_mode else 150
    spring_constant = 0.035 if large_mode else 0.05

    max_score = max(scores.values()) if scores else 1.0
    net = Network(height="620px", width="100%", bgcolor="#0f1117", font_color="white")
    
    arrows_option = '"arrows": {"to": {"enabled": true, "scaleFactor": 0.5}}' if G.is_directed() else '"arrows": {"to": {"enabled": false}}'
    
    net.set_options(f"""
    {{
      "physics": {{
        "enabled": {str(physics_enabled).lower()},
        "barnesHut": {{
          "gravitationalConstant": {gravitational_constant},
          "centralGravity": 0.12,
          "springLength": {spring_length},
          "springConstant": {spring_constant},
          "avoidOverlap": 0.35
        }},
        "minVelocity": 0.75,
        "solver": "barnesHut",
        "stabilization": {{
          "enabled": true,
          "iterations": {stabilization_iterations},
          "updateInterval": 25,
          "fit": true
        }}
      }},
      "interaction": {{
        "zoomView": true,
        "dragView": true,
        "hover": true,
        "tooltipDelay": 100,
        "navigationButtons": true,
        "keyboard": false
      }},
      "edges": {{
        "smooth": false,
        "color": {{"color": "#9ca3af", "opacity": {edge_opacity}}},
        "width": {edge_width},
        "selectionWidth": 1,
        "hoverWidth": 1,
        {arrows_option}
      }},
      "nodes": {{
        "shape": "dot",
        "shadow": true
      }}
    }}
    """)
    
    for node in G.nodes():
        score = scores.get(node, 0.0)
        size = (8 + score * 42) * node_scale
        color = score_to_color(score, max_score)
        is_highlighted = highlight is not None and str(node) == str(highlight)
        tooltip_lines = [f"Node {node}"]
        for name in selected:
            if name in comp.results:
                tooltip_lines.append(f"{name}: {comp.results[name].get(node, 0.0):.4f}")
        tooltip = "\n".join(tooltip_lines)
        node_kwargs = {}
        label_text = str(node) if show_labels or is_highlighted else " "
        font_options = (
            {"size": 14 if is_highlighted else 10, "color": "white"}
            if show_labels or is_highlighted
            else {"size": 0, "color": "rgba(255,255,255,0)"}
        )
        if positions is not None and node in positions:
            x, y = positions[node]
            node_kwargs.update({
                "x": float(x) * 650,
                "y": float(y) * 650,
                "fixed": {"x": True, "y": True},
            })

        net.add_node(
            str(node),
            label=label_text,
            size=size * 1.8 if is_highlighted else size,
            color={
                "background": "#ffdd00" if is_highlighted else color,
                "border": "#ff8800" if is_highlighted else "#ffffff44",
                "highlight": {"background": "#ffdd00", "border": "#ff8800"}
            },
            title=tooltip,
            font=font_options,
            **node_kwargs,
        )
    if show_edges:
        for u, v, data in G.edges(data=True):
            net.add_edge(str(u), str(v), color=f"rgba(156, 163, 175, {edge_opacity})", width=edge_width)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode='w') as f:
        net.save_graph(f.name)
        tmp_path = f.name
    with open(tmp_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    os.unlink(tmp_path)
    graph_style = """
    <style>
      html, body {
        margin: 0 !important;
        padding: 0 !important;
        background: #0f1117 !important;
        overflow: hidden !important;
      }
      #mynetwork {
        border: 0 !important;
        outline: 0 !important;
        box-shadow: none !important;
        background: #0f1117 !important;
      }
      .vis-network {
        border: 0 !important;
        outline: 0 !important;
        background: #0f1117 !important;
      }
      canvas {
        border: 0 !important;
        outline: 0 !important;
      }
    </style>
    """
    fit_script = """
    <script>
      (function () {
        var userInteracted = false;
        var initialFitFinished = false;
        var autoFitting = false;

        function markUserInteracted() {
          if (!autoFitting) {
            userInteracted = true;
            initialFitFinished = true;
          }
        }

        function fitNetwork(force) {
          if (userInteracted && !force) return;
          if (initialFitFinished && !force) return;
          if (typeof network === "undefined") return;
          var container = document.getElementById("mynetwork");
          if (!container || container.offsetWidth === 0 || container.offsetHeight === 0) return;
          autoFitting = true;
          network.fit({
            animation: {
              duration: 350,
              easingFunction: "easeInOutQuad"
            }
          });
          setTimeout(function () {
            autoFitting = false;
          }, 450);
        }

        if (typeof network !== "undefined") {
          network.on("zoom", markUserInteracted);
          network.on("dragStart", markUserInteracted);
          network.on("selectNode", markUserInteracted);

          network.once("stabilizationIterationsDone", function () {
            fitNetwork(true);
            setTimeout(function () {
              network.setOptions({ physics: false });
              initialFitFinished = true;
            }, 900);
          });
          network.on("afterDrawing", function () {
            if (!window.__centralityInitialFitDone) {
              window.__centralityInitialFitDone = true;
              setTimeout(function () { fitNetwork(false); }, 250);
            }
          });
        }

        window.addEventListener("load", function () {
          setTimeout(function () {
            fitNetwork(false);
            initialFitFinished = true;
          }, 1200);
        });
        window.addEventListener("resize", function () {
          if (!userInteracted) {
            setTimeout(function () { fitNetwork(false); }, 150);
          }
        });
        document.addEventListener("mouseenter", function () {
          if (!initialFitFinished && !userInteracted) {
            setTimeout(function () { fitNetwork(false); }, 100);
          }
        });
      })();
    </script>
    """
    html_content = html_content.replace("</head>", graph_style + "\n</head>")
    html_content = html_content.replace("</body>", fit_script + "\n</body>")
    return html_content

# TAB 1 - ANALYSIS

with tab1:
    uploaded_file = None
    if graph_option == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader(
            "Upload edge list CSV (columns: source, target)",
            type=["csv"]
        )
        st.session_state["csv_directed"] = st.sidebar.checkbox(
            "Treat as directed graph",
            value=False
        )
        with st.sidebar.expander("CSV format examples"):
            st.markdown("**Undirected, no weights:**")
            st.code("source,target\n0,1\n1,2\n2,3", language="text")
            st.markdown("**Undirected, with weights:**")
            st.code("source,target,weight\n0,1,2.5\n1,2,0.8\n2,3,4.1", language="text")
            st.markdown("**Directed (check box above):**")
            st.code("source,target\n0,1\n1,2\n2,0", language="text")
            st.markdown("**Directed with weights:**")
            st.code("source,target,weight\n0,1,3.0\n1,2,1.5\n2,0,2.0", language="text")
            st.markdown("*Column names can be anything - order matters.*")

    if graph_option == "Upload CSV" and uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)
        create_using = nx.DiGraph() if st.session_state.get("csv_directed", False) else nx.Graph()
        if df_upload.shape[1] >= 3:
            G = add_weights(nx.from_pandas_edgelist(df_upload, source=df_upload.columns[0], target=df_upload.columns[1], edge_attr=df_upload.columns[2], create_using=create_using))
        else:
            G = add_weights(nx.from_pandas_edgelist(df_upload, source=df_upload.columns[0], target=df_upload.columns[1], create_using=create_using))
    else:
        G = add_weights(load_graph(graph_option, random_params=analysis_random_params))

    quota_values = build_quota_map(G, node_attributes_df, node_id_column, quota_column)
    if selected and any(name in QUOTA_BASED_INDICES for name in selected):
        missing_quota_count = G.number_of_nodes() - len(quota_values)
        if missing_quota_count:
            if node_attributes_df is not None and node_id_column is not None:
                st.warning(
                    "Bundle Index and Pivotal Index require node quotas. "
                    f"Matched {len(quota_values)} of {G.number_of_nodes()} graph nodes. "
                    f"Check that values in `{node_id_column}` match graph node ids."
                )
            else:
                st.warning(
                    "Bundle Index and Pivotal Index require node quotas. "
                    "Upload a node attributes CSV and select a quota column in the sidebar."
                )

    graph_type = "Directed" if G.is_directed() else "Undirected"
    st.sidebar.markdown(f"**Graph type:** {graph_type}")

    if not selected:
        st.warning("Please select at least one centrality index.")
    else:
        warnings = check_compatibility(G, selected)
        for w in warnings:
            st.warning(w)

        valid_selected = []
        for s in selected:
            if s in DIRECTED_INDICES and not G.is_directed():
                continue
            valid_selected.append(s)

        if not valid_selected:
            st.error("No compatible indices selected for this graph type. Please adjust your selection.")
        else:
            comp = compute_all(
                G,
                valid_selected,
                quotas=quota_values,
                max_group_size=quota_max_group_size,
            )
            show_compute_failures(comp)
            valid_selected = available_indices(valid_selected, comp)

            if not valid_selected:
                st.error("No selected indices could be computed for this graph.")
            else:

                st.subheader("Network Overview")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Nodes", G.number_of_nodes())
                col2.metric("Edges", G.number_of_edges())
                col3.metric("Density", f"{nx.density(G):.4f}")
                col4.metric("Type", graph_type)

                st.subheader("Top Nodes")
                total_nodes = G.number_of_nodes()

                col_empty, col_sl, col_dec = st.columns([3, 2, 2])
                with col_sl:
                    top_n = st.slider(
                        "Number of top nodes to display",
                        min_value=1,
                        max_value=total_nodes,
                        value=total_nodes,
                        step=1
                    )
                with col_dec:
                    decimal_places = st.selectbox(
                        "Decimals", [2, 3, 4, 5, 6], index=2
                    )

                first_scores = comp.results[valid_selected[0]]
                top_nodes = sorted(first_scores.items(), key=lambda x: -x[1])[:top_n]
                top_node_ids = [n for n, _ in top_nodes]

                rows = []
                for node in top_node_ids:
                    row = {"Node": str(node)}
                    for name in valid_selected:
                        row[name] = round(comp.results[name].get(node, 0.0), 4)
                    rows.append(row)

                df_results = pd.DataFrame(rows)
                df_display = df_results.copy()
                for col in df_display.columns:
                    if col != "Node":
                        df_display[col] = df_display[col].apply(lambda x: round(float(x), decimal_places))

                df_display.index = range(1, len(df_display) + 1)
                st.dataframe(df_display, use_container_width=True)

                csv = df_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download results as CSV",
                    data=csv,
                    file_name="centrality_results.csv",
                    mime="text/csv"
                )

                st.subheader("Distribution Chart")
                distribution_options = ["Degree"]
                if G.is_directed():
                    distribution_options.extend(["In-degree", "Out-degree"])
                distribution_options.extend(valid_selected)
                distribution_options = list(dict.fromkeys(distribution_options))
                distribution_choice = st.selectbox(
                    "Select distribution to display",
                    distribution_options,
                    key="analysis_distribution_choice",
                )

                if distribution_choice == "Degree":
                    distribution_values = [degree for _, degree in G.degree()]
                    x_title = "Degree"
                elif distribution_choice == "In-degree":
                    distribution_values = [degree for _, degree in G.in_degree()]
                    x_title = "In-degree"
                elif distribution_choice == "Out-degree":
                    distribution_values = [degree for _, degree in G.out_degree()]
                    x_title = "Out-degree"
                else:
                    distribution_values = list(comp.results[distribution_choice].values())
                    x_title = "Centrality score"

                fig_dist = go.Figure(data=[
                    go.Histogram(
                        x=distribution_values,
                        marker_color="#636EFA",
                        opacity=0.85,
                        hovertemplate=f"{x_title}: %{{x}}<br>Count: %{{y}}<extra></extra>",
                    )
                ])
                fig_dist.update_layout(
                    title=f"{distribution_choice} Distribution",
                    xaxis_title=x_title,
                    yaxis_title="Number of nodes",
                    height=420,
                    bargap=0.05,
                )
                st.plotly_chart(fig_dist, use_container_width=True)

                st.subheader("Interactive Network Graph")
                large_graph_default = G.number_of_nodes() > 100 or G.number_of_edges() > 300
                col_graph1, col_graph2, col_graph3 = st.columns([3, 2, 2])
                with col_graph1:
                    st.markdown(f"Node size and color reflect **{valid_selected[0]}** centrality.")
                with col_graph2:
                    filter_top = st.slider(
                        "Show top N nodes",
                        min_value=1,
                        max_value=G.number_of_nodes(),
                        value=min(G.number_of_nodes(), 80) if large_graph_default else G.number_of_nodes(),
                        step=1
                    )
                with col_graph3:
                    highlight_node = st.text_input(
                        "Highlight node",
                        value="",
                        placeholder="e.g. 0"
                    )

                with st.expander("Graph display settings", expanded=large_graph_default):
                    col_layout, col_v1, col_v2 = st.columns([2, 1, 1])
                    with col_layout:
                        graph_layout = st.selectbox(
                            "Layout",
                            [
                                "Physics simulation",
                                "Spring layout",
                                "Kamada-Kawai layout",
                                "Circular layout",
                                "Shell layout",
                                "Spectral layout",
                            ],
                            index=0,
                        )
                    with col_v1:
                        large_mode = st.checkbox(
                            "Large graph mode",
                            value=large_graph_default,
                            help="Reduces node size, edge opacity, and freezes physics after layout stabilization.",
                        )
                    with col_v2:
                        show_labels = st.checkbox(
                            "Show node labels",
                            value=not large_graph_default,
                        )
                    col_v3, col_v4 = st.columns(2)
                    with col_v3:
                        show_edges = st.checkbox(
                            "Show edges",
                            value=True,
                        )
                    with col_v4:
                        node_scale = st.slider(
                            "Node size",
                            min_value=0.4,
                            max_value=1.4,
                            value=0.65 if large_graph_default else 1.0,
                            step=0.05,
                        )
                    edge_opacity = st.slider(
                        "Edge opacity",
                        min_value=0.02,
                        max_value=0.35,
                        value=0.08 if large_graph_default else 0.22,
                        step=0.01,
                    )
                    if G.number_of_nodes() > 120 and filter_top == G.number_of_nodes():
                        st.info("For large graphs, filtering to top nodes or hiding labels usually makes the network easier to read.")

                visual_options = {
                    "large_mode": large_mode,
                    "show_labels": show_labels,
                    "show_edges": show_edges,
                    "node_scale": node_scale,
                    "edge_opacity": edge_opacity,
                    "layout": graph_layout,
                }

                scores = comp.results[valid_selected[0]]
                top_nodes_filter = set(
                    node for node, _ in sorted(scores.items(), key=lambda x: -x[1])[:filter_top]
                )
                G_filtered = G.subgraph(top_nodes_filter).copy()
                scores_filtered = {node: scores[node] for node in top_nodes_filter}

                comp_filtered = CentralityComparator()
                for name in valid_selected:
                    try:
                        filtered_quotas = {
                            node: quota_values[node]
                            for node in G_filtered.nodes()
                            if node in quota_values
                        }
                        if name in QUOTA_BASED_INDICES:
                            instance = INDEX_MAP[name](
                                G_filtered,
                                quotas=filtered_quotas,
                                max_group_size=quota_max_group_size,
                            )
                        else:
                            instance = INDEX_MAP[name](G_filtered)
                        instance.compute()
                        comp_filtered.add(name, instance)
                    except Exception:
                        pass

                st.markdown(f"Showing **{filter_top}** of **{G.number_of_nodes()}** nodes.")

                try:
                    hl = int(highlight_node) if highlight_node else None
                except ValueError:
                    hl = highlight_node if highlight_node else None

                html_content = build_pyvis(
                    G_filtered,
                    scores_filtered,
                    valid_selected,
                    comp_filtered,
                    highlight=hl,
                    visual_options=visual_options,
                )
                components.html(html_content, height=640, scrolling=False)

                if len(valid_selected) > 1:
                    st.subheader("Correlation Matrix")
                    nodes_list = list(G.nodes())
                    matrix = np.array([
                        [comp.results[name][node] for node in nodes_list]
                        for name in valid_selected
                    ])
                    corr = np.corrcoef(matrix)
                    fig_corr = go.Figure(data=go.Heatmap(
                        z=corr,
                        x=valid_selected,
                        y=valid_selected,
                        colorscale="RdBu",
                        zmin=-1, zmax=1,
                        text=[[f"{corr[i][j]:.2f}" for j in range(len(valid_selected))]
                            for i in range(len(valid_selected))],
                        texttemplate="%{text}",
                        textfont={"size": 14},
                        hovertemplate="<b>%{y} vs %{x}</b><br>Correlation: %{z:.3f}<extra></extra>"
                    ))
                    fig_corr.update_layout(
                        title="Pearson Correlation between Centrality Indices",
                        height=420,
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)

                st.subheader("Node Inspector")
                selected_node = st.selectbox(
                    "Select a node to inspect",
                    options=sorted(G.nodes())
                )

                if selected_node is not None:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Node {selected_node} - Centrality Scores**")
                        rows_inspector = []
                        for name in valid_selected:
                            score = comp.results[name].get(selected_node, 0.0)
                            all_scores = list(comp.results[name].values())
                            avg = np.mean(all_scores)
                            diff_pct = ((score - avg) / avg * 100) if avg > 0 else 0
                            percentile = int(np.sum(np.array(all_scores) <= score) / len(all_scores) * 100)
                            rows_inspector.append({
                                "Index": name,
                                "Score": round(score, 4),
                                "Avg": round(avg, 4),
                                "vs Avg": f"{diff_pct:+.1f}%",
                                "Percentile": f"{percentile}th"
                            })
                        st.dataframe(
                            pd.DataFrame(rows_inspector),
                            use_container_width=True,
                            hide_index=True
                        )
                        st.markdown("**Neighbors**")
                        neighbors = list(G.neighbors(selected_node))
                        st.metric("Degree", len(neighbors))
                        st.write(f"Connected to: {neighbors}")

                    with col2:
                        if len(valid_selected) >= 3:
                            categories = valid_selected
                            values = [comp.results[name].get(selected_node, 0.0) for name in valid_selected]
                            avg_values = [np.mean(list(comp.results[name].values())) for name in valid_selected]

                            fig_radar = go.Figure()
                            fig_radar.add_trace(go.Scatterpolar(
                                r=avg_values + [avg_values[0]],
                                theta=categories + [categories[0]],
                                fill='toself',
                                fillcolor='rgba(150,150,150,0.15)',
                                line=dict(color='#888888', width=1, dash='dash'),
                                name='Network Average'
                            ))
                            fig_radar.add_trace(go.Scatterpolar(
                                r=values + [values[0]],
                                theta=categories + [categories[0]],
                                fill='toself',
                                fillcolor='rgba(99, 110, 250, 0.3)',
                                line=dict(color='#636EFA', width=2),
                                marker=dict(size=6),
                                name=f"Node {selected_node}"
                            ))
                            fig_radar.update_layout(
                                polar=dict(
                                    radialaxis=dict(
                                        visible=True,
                                        range=[0, max(
                                            max(comp.results[name].values())
                                            for name in valid_selected
                                        )],
                                        gridcolor="#333333",
                                        linecolor="#333333",
                                    ),
                                    angularaxis=dict(gridcolor="#333333"),
                                    bgcolor="#1a1a2e"
                                ),
                                showlegend=True,
                                legend=dict(x=0.8, y=1.1),
                                title=f"Node {selected_node} vs Network Average",
                                height=380,
                            )
                            st.plotly_chart(fig_radar, use_container_width=True)
                        else:
                            st.info("Select 3 or more indices to see the radar chart.")

                st.markdown("---")
                st.markdown("*Network Centrality Analyzer - Bachelor's Thesis Project*")

# TAB 2 - COMPARE NETWORKS

with tab2:
    st.subheader("Compare Two Networks")
    st.markdown("Select two networks and compare how centrality indices differ between them.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### Network A")
        graph_a = st.selectbox(
            "Select Network A",
            GRAPH_OPTIONS,
            key="graph_a"
        )
        if graph_a == "Upload CSV":
            uploaded_a = st.file_uploader("Upload CSV for Network A", type=["csv"], key="upload_a")
            csv_directed_a = st.checkbox("Directed graph (A)", key="dir_a")
        else:
            uploaded_a = None
            csv_directed_a = False
        random_params_a = random_graph_controls(st, "compare_a") if graph_a == "Random Graph" else None

    with col_b:
        st.markdown("### Network B")
        graph_b = st.selectbox(
            "Select Network B",
            GRAPH_OPTIONS,
            index=1,
            key="graph_b"
        )
        if graph_b == "Upload CSV":
            uploaded_b = st.file_uploader("Upload CSV for Network B", type=["csv"], key="upload_b")
            csv_directed_b = st.checkbox("Directed graph (B)", key="dir_b")
        else:
            uploaded_b = None
            csv_directed_b = False
        random_params_b = random_graph_controls(st, "compare_b") if graph_b == "Random Graph" else None

    if not selected:
        st.warning("Please select at least one centrality index in the sidebar.")
    else:
        compare_ready = True

        if graph_a == "Upload CSV":
            if uploaded_a is not None:
                df_a = pd.read_csv(uploaded_a)
                create_a = nx.DiGraph() if csv_directed_a else nx.Graph()
                if df_a.shape[1] >= 3:
                    GA = add_weights(nx.from_pandas_edgelist(df_a, source=df_a.columns[0], target=df_a.columns[1], edge_attr=df_a.columns[2], create_using=create_a))
                else:
                    GA = add_weights(nx.from_pandas_edgelist(df_a, source=df_a.columns[0], target=df_a.columns[1], create_using=create_a))
            else:
                st.warning("Please upload a CSV file for Network A.")
                compare_ready = False
        else:
            GA = add_weights(load_graph(graph_a, random_params=random_params_a))

        if graph_b == "Upload CSV":
            if uploaded_b is not None:
                df_b = pd.read_csv(uploaded_b)
                create_b = nx.DiGraph() if csv_directed_b else nx.Graph()
                if df_b.shape[1] >= 3:
                    GB = add_weights(nx.from_pandas_edgelist(df_b, source=df_b.columns[0], target=df_b.columns[1], edge_attr=df_b.columns[2], create_using=create_b))
                else:
                    GB = add_weights(nx.from_pandas_edgelist(df_b, source=df_b.columns[0], target=df_b.columns[1], create_using=create_b))
            else:
                st.warning("Please upload a CSV file for Network B.")
                compare_ready = False
        else:
            GB = add_weights(load_graph(graph_b, random_params=random_params_b))

        if compare_ready:
            warnings_a = check_compatibility(GA, selected)
            warnings_b = check_compatibility(GB, selected)
            for w in warnings_a:
                st.warning(f"Network A: {w}")
            for w in warnings_b:
                st.warning(f"Network B: {w}")

            valid_a = [s for s in selected if not (s in DIRECTED_INDICES and not GA.is_directed())]
            valid_b = [s for s in selected if not (s in DIRECTED_INDICES and not GB.is_directed())]

            if not valid_a:
                st.warning("Network A: no compatible indices for this graph type. Please adjust your selection.")
                compare_ready = False
            if not valid_b:
                st.warning("Network B: no compatible indices for this graph type. Please adjust your selection.")
                compare_ready = False

        if compare_ready:
            quotas_a = build_quota_map(GA, node_attributes_df, node_id_column, quota_column)
            quotas_b = build_quota_map(GB, node_attributes_df, node_id_column, quota_column)
            comp_a = compute_all(
                GA,
                valid_a,
                quotas=quotas_a,
                max_group_size=quota_max_group_size,
            )
            comp_b = compute_all(
                GB,
                valid_b,
                quotas=quotas_b,
                max_group_size=quota_max_group_size,
            )
            show_compute_failures(comp_a)
            show_compute_failures(comp_b)
            valid_a = available_indices(valid_a, comp_a)
            valid_b = available_indices(valid_b, comp_b)

            if not valid_a:
                st.warning("Network A: no selected indices could be computed.")
                compare_ready = False
            if not valid_b:
                st.warning("Network B: no selected indices could be computed.")
                compare_ready = False

        if compare_ready:
            st.subheader("Network Overview")
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("A - Nodes", GA.number_of_nodes())
            c2.metric("A - Edges", GA.number_of_edges())
            c3.metric("A - Density", f"{nx.density(GA):.4f}")
            c4.metric("B - Nodes", GB.number_of_nodes())
            c5.metric("B - Edges", GB.number_of_edges())
            c6.metric("B - Density", f"{nx.density(GB):.4f}")

            max_top = min(GA.number_of_nodes(), GB.number_of_nodes())
            top_n_compare = st.slider(
                "Top N nodes",
                min_value=1,
                max_value=max_top,
                value=min(10, max_top),
                step=1,
            )
            st.subheader("Top Nodes Comparison")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**{graph_a}** ({'Directed' if GA.is_directed() else 'Undirected'})")
                scores_a = comp_a.results[valid_a[0]]
                top_a = sorted(scores_a.items(), key=lambda x: -x[1])[:top_n_compare]
                rows_a = []
                for node, _ in top_a:
                    row = {"Node": str(node)}
                    for name in valid_a:
                        row[name] = round(comp_a.results[name].get(node, 0.0), 4)
                    rows_a.append(row)
                st.dataframe(pd.DataFrame(rows_a), use_container_width=True)
            with col2:
                st.markdown(f"**{graph_b}** ({'Directed' if GB.is_directed() else 'Undirected'})")
                scores_b = comp_b.results[valid_b[0]]
                top_b = sorted(scores_b.items(), key=lambda x: -x[1])[:top_n_compare]
                rows_b = []
                for node, _ in top_b:
                    row = {"Node": str(node)}
                    for name in valid_b:
                        row[name] = round(comp_b.results[name].get(node, 0.0), 4)
                    rows_b.append(row)
                st.dataframe(pd.DataFrame(rows_b), use_container_width=True)

            common_valid = [s for s in selected if s in valid_a and s in valid_b]
            if common_valid:
                st.subheader("Score Distribution Comparison")
                dist_index = st.selectbox(
                    "Select index to compare distributions",
                    common_valid,
                    key="dist_index"
                )
                vals_a = list(comp_a.results[dist_index].values())
                vals_b = list(comp_b.results[dist_index].values())

                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(
                    x=vals_a, name=graph_a,
                    opacity=0.7, marker_color='#636EFA',
                    hovertemplate="Score: %{x:.3f}<br>Count: %{y}<extra></extra>"
                ))
                fig_dist.add_trace(go.Histogram(
                    x=vals_b, name=graph_b,
                    opacity=0.7, marker_color='#EF553B',
                    hovertemplate="Score: %{x:.3f}<br>Count: %{y}<extra></extra>"
                ))
                fig_dist.update_layout(
                    barmode='overlay',
                    title=f"{dist_index} Centrality - Score Distribution",
                    xaxis_title="Score",
                    yaxis_title="Count",
                    height=400,
                    legend_title="Network"
                )
                st.plotly_chart(fig_dist, use_container_width=True)
            else:
                st.info("No common compatible indices are available for both selected networks.")

            st.subheader("Network Graphs")
            st.markdown(f"**{graph_a}**")
            html_a = build_pyvis(GA, comp_a.results[valid_a[0]], valid_a, comp_a)
            components.html(html_a, height=620, scrolling=False)

            st.markdown(f"**{graph_b}**")
            html_b = build_pyvis(GB, comp_b.results[valid_b[0]], valid_b, comp_b)
            components.html(html_b, height=620, scrolling=False)

# TAB 3 - NODE IMPACT

with tab3:
    st.subheader("Node Impact Analysis")
    st.markdown("""
    Select a node to remove and see how centrality scores change across the network.
    This shows which nodes are most critical to the network structure.
    """)

    if not selected:
        st.warning("Please select at least one centrality index in the sidebar.")
    else:
        G_impact = G.copy()

        warnings_impact = check_compatibility(G_impact, selected)
        for w in warnings_impact:
            st.warning(w)

        valid_impact = [s for s in selected if not (s in DIRECTED_INDICES and not G_impact.is_directed())]

        if not valid_impact:
            st.error("No compatible indices for this graph type.")
        else:
            impact_quotas = {
                node: quota_values[node]
                for node in G_impact.nodes()
                if node in quota_values
            }
            comp_full = compute_all(
                G_impact,
                valid_impact,
                quotas=impact_quotas,
                max_group_size=quota_max_group_size,
            )
            show_compute_failures(comp_full)
            valid_impact = available_indices(valid_impact, comp_full)

            if not valid_impact:
                st.error("No selected indices could be computed for this graph.")
            else:

                col1, col2 = st.columns([2, 1])
                with col1:
                    remove_node = st.selectbox(
                        "Select node to remove",
                        options=sorted(G_impact.nodes()),
                        key="remove_node"
                    )
                with col2:
                    impact_index = st.selectbox(
                        "Index to analyze",
                        valid_impact,
                        key="impact_index"
                    )

                G_removed = G_impact.copy()
                G_removed.remove_node(remove_node)
                removed_quotas = {
                    node: impact_quotas[node]
                    for node in G_removed.nodes()
                    if node in impact_quotas
                }
                comp_removed = compute_all(
                    G_removed,
                    valid_impact,
                    quotas=removed_quotas,
                    max_group_size=quota_max_group_size,
                )

                common_nodes = [n for n in G_impact.nodes() if n != remove_node]
                changes = []
                for node in common_nodes:
                    score_before = comp_full.results[impact_index].get(node, 0.0)
                    score_after = comp_removed.results[impact_index].get(node, 0.0)
                    diff = score_after - score_before
                    changes.append({
                        "Node": str(node),
                        "Before": round(score_before, 4),
                        "After": round(score_after, 4),
                        "Change": round(diff, 4),
                    })

                df_changes = pd.DataFrame(changes)
                df_changes = df_changes.sort_values("Change", ascending=False)

                st.subheader("Network Impact Summary")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Nodes before", G_impact.number_of_nodes())
                c2.metric("Nodes after", G_removed.number_of_nodes())
                c3.metric("Edges before", G_impact.number_of_edges())
                c4.metric("Edges after", G_removed.number_of_edges())

                st.subheader(f"{impact_index} Score Changes After Removing Node {remove_node}")
                top_changed = df_changes.head(15)
                colors_bar = ['#00CC96' if x >= 0 else '#EF553B' for x in top_changed['Change']]

                fig_impact = go.Figure()
                fig_impact.add_trace(go.Bar(
                    x=top_changed['Node'],
                    y=top_changed['Change'],
                    marker_color=colors_bar,
                    hovertemplate="Node: %{x}<br>Change: %{y:+.4f}<extra></extra>"
                ))
                fig_impact.update_layout(
                    title=f"Score change after removing node {remove_node} (top 15 most affected)",
                    xaxis_title="Node",
                    yaxis_title="Score Change",
                    height=400,
                )
                fig_impact.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
                st.plotly_chart(fig_impact, use_container_width=True)

                st.subheader("Network Before vs After")
                st.markdown(f"**Before - Node {remove_node} present**")
                html_before = build_pyvis(G_impact, comp_full.results[impact_index], valid_impact, comp_full)
                components.html(html_before, height=620, scrolling=False)

                st.markdown(f"**After - Node {remove_node} removed**")
                html_after = build_pyvis(G_removed, comp_removed.results[impact_index], valid_impact, comp_removed)
                components.html(html_after, height=620, scrolling=False)

                st.subheader("Full Change Table")
                st.dataframe(
                    df_changes.style.background_gradient(subset=['Change'], cmap='RdYlGn'),
                    use_container_width=True
                )

                most_affected = df_changes.iloc[0]
                most_dropped = df_changes.sort_values("Change").iloc[0]
                col1, col2 = st.columns(2)
                col1.success(f"Most gained: Node {most_affected['Node']} (+{most_affected['Change']:.4f})")
                col2.error(f"Most dropped: Node {most_dropped['Node']} ({most_dropped['Change']:.4f})")

# TAB 4 - DATA LABORATORY

with tab4:
    st.subheader("Data Laboratory")
    st.markdown("Inspect edge data and node attributes used by the current network.")

    data_view = st.selectbox(
        "Data table",
        ["Nodes", "Edges"],
        key="data_laboratory_view",
    )

    if data_view == "Nodes":
        node_rows = []
        for node in G.nodes():
            row = {"Node": node}
            row.update(G.nodes[node])
            if node in quota_values:
                row["Selected quota"] = quota_values[node]
            node_rows.append(row)

        nodes_df = pd.DataFrame(node_rows)
        if node_attributes_df is not None and node_id_column is not None:
            nodes_for_merge = nodes_df.copy()
            attrs_for_merge = node_attributes_df.copy()
            nodes_for_merge["_node_key"] = nodes_for_merge["Node"].astype(str)
            attrs_for_merge["_node_key"] = attrs_for_merge[node_id_column].astype(str)
            display_df = nodes_for_merge.merge(
                attrs_for_merge,
                how="left",
                on="_node_key",
            )
            display_df = display_df.drop(columns=["_node_key"])
            if node_id_column in display_df.columns and node_id_column != "Node":
                display_df = display_df.drop(columns=[node_id_column])
            matched_rows = display_df[quota_column].notna().sum() if quota_column in display_df.columns else 0
            if matched_rows == 0:
                st.warning(
                    "The attributes file is loaded, but its node ids do not match the current graph nodes. "
                    "For Karate Club, node ids are 0, 1, 2, ..., 33. The paper example uses v1, v2, ..., v5."
                )
        else:
            display_df = nodes_df
            if any(name in QUOTA_BASED_INDICES for name in selected):
                st.info("Upload a node attributes CSV in the sidebar to use quotas for Bundle/Pivotal indices.")

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        if quota_values:
            st.success(f"Quota column selected: {quota_column}. Matched {len(quota_values)} of {G.number_of_nodes()} nodes.")
        elif any(name in QUOTA_BASED_INDICES for name in selected):
            st.warning("Quota-based indices are selected, but no usable quota values are loaded.")

    else:
        edge_rows = []
        if G.is_multigraph():
            for source, target, key, data in G.edges(keys=True, data=True):
                row = {"Source": source, "Target": target, "Key": key}
                row.update(data)
                edge_rows.append(row)
        else:
            for source, target, data in G.edges(data=True):
                row = {"Source": source, "Target": target}
                row.update(data)
                edge_rows.append(row)

        st.dataframe(pd.DataFrame(edge_rows), use_container_width=True, hide_index=True)
        with st.expander("Edge CSV example"):
            st.code("source,target,weight\n0,1,2.5\n2,1,1.0\n3,2,4.1", language="text")

# TAB 5 - ABOUT

with tab5:
    st.subheader("About This Tool")
    st.markdown("""
    This tool provides an interactive interface for computing and comparing
    centrality indices in network structures. It supports classical, weighted,
    directed, directed weighted, and quota-based indices.
    """)

    st.subheader("Quota-Based Indices")
    st.markdown("""
    Bundle Index and Pivotal Index use node quotas. A quota is a threshold of
    incoming influence for a node. For example, in a financial network it can
    represent the amount of incoming loss or pressure needed to make a node
    critical.

    Bundle Index counts groups of incoming neighbors whose total edge weight
    reaches the quota of the target node. Pivotal Index counts nodes that are
    decisive inside these critical groups. The parameter `k` limits the maximum
    size of a group, because checking all possible groups can become too slow
    for nodes with many incoming links.
    """)

    st.subheader("Data Laboratory")
    st.markdown("""
    The Data Laboratory tab shows the data used by the current network.
    `Edges` contains the source node, target node, and edge attributes such as
    weight. `Nodes` contains graph nodes and, when uploaded, node attributes
    such as quota, GDP, population, or any other user-defined columns.

    Node attributes are currently required for Bundle Index and Pivotal Index,
    because these indices need a selected quota column.
    """)

    st.subheader("Implemented Indices")

    indices_info = {
        "Degree Centrality": {
            "formula": "C(v) = deg(v) / (N - 1)",
            "description": "Measures the number of direct connections a node has, normalized by the maximum possible connections.",
            "use_case": "Finding the most connected individuals in social networks.",
            "type": "Undirected"
        },
        "Closeness Centrality": {
            "formula": "C(v) = (N-1) / Σ d(v,u)",
            "description": "Measures how close a node is to all other nodes based on shortest path lengths.",
            "use_case": "Identifying nodes that can quickly spread information across a network.",
            "type": "Undirected"
        },
        "Betweenness Centrality": {
            "formula": "C(v) = Σ σ(s,t|v) / σ(s,t)",
            "description": "Measures how often a node lies on the shortest path between two other nodes.",
            "use_case": "Finding bottlenecks and bridge nodes in communication networks.",
            "type": "Undirected"
        },
        "Eigenvector Centrality": {
            "formula": "C(v) = (1/λ) Σ C(u) for u in N(v)",
            "description": "A node is important if it is connected to other important nodes.",
            "use_case": "Ranking web pages, finding influential users in social networks.",
            "type": "Undirected"
        },
        "Weighted Degree Centrality": {
            "formula": "C(v) = Σ w(e) / (N - 1)",
            "description": "Extends Degree Centrality by summing edge weights instead of counting connections.",
            "use_case": "Transport networks where connection strength matters.",
            "type": "Undirected (weighted)"
        },
        "Weighted Closeness Centrality": {
            "formula": "C(v) = (N-1) / Σ d_w(v,u)",
            "description": "Uses weighted shortest paths where stronger connections mean shorter distances.",
            "use_case": "Logistics and supply chain optimization.",
            "type": "Undirected (weighted)"
        },
        "Weighted Betweenness Centrality": {
            "formula": "C(v) = Σ σ_w(s,t|v) / σ_w(s,t)",
            "description": "Betweenness computed on weighted shortest paths.",
            "use_case": "Finding critical nodes in weighted infrastructure networks.",
            "type": "Undirected (weighted)"
        },
        "Weighted Eigenvector Centrality": {
            "formula": "C(v) = (1/λ) Σ w(u,v)C(u)",
            "description": "Eigenvector centrality where stronger links contribute more to node importance.",
            "use_case": "Finding influential nodes in weighted social or transport networks.",
            "type": "Undirected (weighted)"
        },
        "In-Degree Centrality": {
            "formula": "C(v) = in_deg(v) / (N - 1)",
            "description": "Normalized number of incoming edges. High in-degree means many nodes point to this node.",
            "use_case": "Citation networks, finding authoritative pages on the web.",
            "type": "Directed"
        },
        "Out-Degree Centrality": {
            "formula": "C(v) = out_deg(v) / (N - 1)",
            "description": "Normalized number of outgoing edges. High out-degree means this node points to many others.",
            "use_case": "Social networks, finding active broadcasters.",
            "type": "Directed"
        },
        "Weighted In-Degree Centrality": {
            "formula": "C(v) = Σ w(u,v) / (N - 1)",
            "description": "Sums weights of incoming edges to measure weighted authority or inflow.",
            "use_case": "Citation, trade, traffic, and influence networks with weighted incoming links.",
            "type": "Directed (weighted)"
        },
        "Weighted Out-Degree Centrality": {
            "formula": "C(v) = Σ w(v,u) / (N - 1)",
            "description": "Sums weights of outgoing edges to measure weighted activity or outflow.",
            "use_case": "Networks where outgoing volume or distribution strength matters.",
            "type": "Directed (weighted)"
        },
        "Directed Closeness Centrality": {
            "formula": "C(v) = (R/(N-1)) · R / Σ d(u,v)",
            "description": "Measures reachability in directed graphs using incoming paths by default.",
            "use_case": "Finding nodes that can be reached efficiently by many other nodes.",
            "type": "Directed"
        },
        "Directed Betweenness Centrality": {
            "formula": "C(v) = Σ σ(s,t|v) / σ(s,t)",
            "description": "Counts how often a node lies on directed shortest paths.",
            "use_case": "Finding brokers and bottlenecks in directed communication or dependency networks.",
            "type": "Directed"
        },
        "Directed Eigenvector Centrality": {
            "formula": "C(v) = (1/λ) Σ C(u), for u → v",
            "description": "A node is important if important nodes point to it.",
            "use_case": "Authority-like ranking in directed networks.",
            "type": "Directed"
        },
        "Directed Weighted Closeness Centrality": {
            "formula": "C(v) = (R/(N-1)) · R / Σ d_w(u,v)",
            "description": "Directed closeness using weighted shortest paths where stronger links are shorter.",
            "use_case": "Weighted directed flow, dependency, or transport networks.",
            "type": "Directed (weighted)"
        },
        "Directed Weighted Betweenness Centrality": {
            "formula": "C(v) = Σ σ_w(s,t|v) / σ_w(s,t)",
            "description": "Directed betweenness computed on weighted shortest paths.",
            "use_case": "Finding critical intermediaries in weighted directed infrastructure.",
            "type": "Directed (weighted)"
        },
        "Directed Weighted Eigenvector Centrality": {
            "formula": "C(v) = (1/λ) Σ w(u,v)C(u), for u → v",
            "description": "Directed eigenvector centrality that also accounts for edge weights.",
            "use_case": "Authority-like ranking when incoming links have different strengths.",
            "type": "Directed (weighted)"
        },
        "PageRank": {
            "formula": "PR(v) = (1-d)/N + d · Σ PR(u)/out_deg(u)",
            "description": "Models a random walker following edges with damping factor d=0.85. Nodes pointed to by important nodes get higher scores.",
            "use_case": "Web search ranking, finding influential nodes in directed networks.",
            "type": "Directed"
        },
        "Weighted PageRank": {
            "formula": "PR(v) = (1-d)/N + d · Σ PR(u)w(u,v)/Σw(u,*)",
            "description": "PageRank variant where stronger outgoing links pass a larger share of rank.",
            "use_case": "Ranking in directed networks with weighted links.",
            "type": "Directed (weighted)"
        },
        "Bundle Index": {
            "formula": "BI(i) = Σ 1[Σ w(j,i) ≥ q_i], |S| ≤ k",
            "description": "Counts incoming groups of nodes whose total edge weight reaches the quota of the target node.",
            "use_case": "Modeling group influence when several connected actors can jointly affect a node.",
            "type": "Quota-based"
        },
        "Pivotal Index": {
            "formula": "PI(i) = Σ pivotal nodes in critical groups S",
            "description": "Counts nodes that are decisive inside critical incoming groups: without such a node, the group no longer reaches the quota.",
            "use_case": "Finding nodes whose participation is crucial for group influence.",
            "type": "Quota-based"
        },
    }

    about_groups = {
        "Classical": [
            "Degree Centrality", "Closeness Centrality",
            "Betweenness Centrality", "Eigenvector Centrality"
        ],
        "Weighted": [
            "Weighted Degree Centrality", "Weighted Closeness Centrality",
            "Weighted Betweenness Centrality", "Weighted Eigenvector Centrality"
        ],
        "Directed": [
            "In-Degree Centrality", "Out-Degree Centrality",
            "Directed Closeness Centrality", "Directed Betweenness Centrality",
            "Directed Eigenvector Centrality", "PageRank"
        ],
        "Directed Weighted": [
            "Weighted In-Degree Centrality", "Weighted Out-Degree Centrality",
            "Directed Weighted Closeness Centrality",
            "Directed Weighted Betweenness Centrality",
            "Directed Weighted Eigenvector Centrality",
            "Weighted PageRank"
        ],
        "Quota-based": ["Bundle Index", "Pivotal Index"],
    }

    about_tabs = st.tabs(list(about_groups.keys()))
    for tab, (_, names) in zip(about_tabs, about_groups.items()):
        with tab:
            for name in names:
                info = indices_info[name]
                with st.expander(f"{name} - {info['type']}"):
                    st.markdown(f"**Formula:** `{info['formula']}`")
                    st.markdown(f"**Description:** {info['description']}")
                    st.markdown(f"**Use case:** {info['use_case']}")

    st.subheader("Supported Graph Types")
    st.markdown("""
    | Index | Undirected | Directed |
    |-------|-----------|---------|
    | Degree, Closeness, Betweenness, Eigenvector | ✅ | ❌ |
    | Weighted Degree, Weighted Closeness, Weighted Betweenness, Weighted Eigenvector | ✅ | ❌ |
    | In-Degree, Out-Degree, Directed Closeness, Directed Betweenness, Directed Eigenvector, PageRank | ❌ | ✅ |
    | Weighted In/Out-Degree, Directed Weighted Closeness, Directed Weighted Betweenness, Directed Weighted Eigenvector, Weighted PageRank | ❌ | ✅ |
    | Bundle Index, Pivotal Index | ✅* | ✅ |

    `*` Undirected graphs are converted to reciprocal directed edges with default weight 1 when needed.
    """)

    st.subheader("Random Graph Models")
    st.markdown("""
    | Model | Main parameters | Typical purpose |
    |-------|-----------------|-----------------|
    | Erdős-Rényi | number of nodes, edge probability, seed | Baseline random network with independently sampled edges |
    | Barabási-Albert | number of nodes, edges per new node, seed | Scale-free network with preferential attachment |
    | Watts-Strogatz | number of nodes, nearest neighbors, rewiring probability, seed | Small-world network with local clustering |
    | Scale-Free Directed | number of nodes, attachment probabilities, seed | Directed scale-free network for directed centrality indices |
    """)

    st.subheader("Tech Stack")
    st.markdown("""
    - **Python 3.9+**
    - **NetworkX** - graph analysis
    - **NumPy** - matrix operations
    - **Streamlit** - web interface
    - **Plotly** - interactive charts
    - **Pyvis** - interactive network visualization
    """)

    st.subheader("Author")
    st.markdown("""
    Developed as part of a Bachelor's Thesis in Data Science and Business Analytics.
    Repository: [github.com/ekozlovskii/centrality-repo](https://github.com/ekozlovskii/centrality-repo)
    """)
