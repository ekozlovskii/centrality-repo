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
from utils.comparator import CentralityComparator

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Network Centrality Analyzer",
    page_icon="🔵",
    layout="wide"
)

st.title(" Network Centrality Analyzer")
st.markdown("Compute and compare centrality indices for network structures.")

tab1, tab2, tab3, tab4 = st.tabs(["Analysis", "Compare Networks", "Node Impact", "About"])

# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════

st.sidebar.header("Settings")

graph_option = st.sidebar.selectbox(
    "Select network",
    ["Karate Club", "Les Misérables", "Random (Barabási–Albert)",
     "Florentine Families", "Davis Women's Club", "Petersen Graph", "Upload CSV"]
)

st.sidebar.header("Centrality Indices")
selected = st.sidebar.multiselect(
    "Select indices to compute",
    ["Degree", "Closeness", "Betweenness", "Eigenvector",
     "Weighted Degree", "Weighted Closeness", "Weighted Betweenness"],
    default=["Degree", "Closeness", "Betweenness"]
)

top_n = st.sidebar.slider("Top N nodes to display", 5, 20, 10)

# ── Helpers ──────────────────────────────────────────────────────────────────

INDEX_MAP = {
    "Degree": DegreeCentrality,
    "Closeness": ClosenessCentrality,
    "Betweenness": BetweennessCentrality,
    "Eigenvector": EigenvectorCentrality,
    "Weighted Degree": WeightedDegreeCentrality,
    "Weighted Closeness": WeightedClosenessCentrality,
    "Weighted Betweenness": WeightedBetweennessCentrality,
}

@st.cache_data
def load_graph(option, uploaded_file=None):
    if option == "Karate Club":
        return nx.karate_club_graph()
    elif option == "Les Misérables":
        return nx.les_miserables_graph()
    elif option == "Random (Barabási–Albert)":
        return nx.barabasi_albert_graph(50, 2, seed=42)
    elif option == "Florentine Families":
        return nx.florentine_families_graph()
    elif option == "Davis Women's Club":
        return nx.davis_southern_women_graph()
    elif option == "Petersen Graph":
        return nx.petersen_graph()
    elif option == "Upload CSV" and uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if df.shape[1] >= 3:
            return nx.from_pandas_edgelist(
                df, source=df.columns[0],
                target=df.columns[1],
                edge_attr=df.columns[2]
            )
        else:
            return nx.from_pandas_edgelist(
                df, source=df.columns[0],
                target=df.columns[1]
            )
    return nx.karate_club_graph()

def add_weights(G):
    G = G.copy()
    for u, v in G.edges():
        if 'weight' not in G[u][v]:
            G[u][v]['weight'] = random.uniform(0.5, 5.0)
    return G

def compute_all(G, selected):
    comp = CentralityComparator()
    for name in selected:
        instance = INDEX_MAP[name](G)
        instance.compute()
        comp.add(name, instance)
    return comp

def score_to_color(score, max_s):
    ratio = score / max_s if max_s > 0 else 0
    r = int(255 * ratio)
    g = int(80 * ratio)
    b = int(200 * (1 - ratio))
    return f"#{r:02x}{g:02x}{b:02x}"

def build_pyvis(G, scores, selected, comp, highlight=None):
    max_score = max(scores.values()) if scores else 1.0
    net = Network(height="620px", width="100%", bgcolor="#0f1117", font_color="white")
    net.set_options("""
    {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -3000,
          "centralGravity": 0.2,
          "springLength": 150,
          "springConstant": 0.05
        },
        "stabilization": {"iterations": 150}
      },
      "interaction": {
        "zoomView": false,
        "dragView": true,
        "hover": true,
        "tooltipDelay": 100,
        "navigationButtons": true,
        "keyboard": false
      },
      "edges": {
        "smooth": false,
        "color": {"opacity": 0.3}
      },
      "nodes": {
        "shape": "dot",
        "shadow": true
      }
    }
    """)
    
    for node in G.nodes():
        score = scores.get(node, 0.0)
        size = 12 + score * 55
        color = score_to_color(score, max_score)
        is_highlighted = highlight is not None and str(node) == str(highlight)
        tooltip_lines = [f"Node {node}"]
        for name in selected:
            tooltip_lines.append(f"{name}: {comp.results[name].get(node, 0.0):.4f}")
        tooltip = "\n".join(tooltip_lines)
        net.add_node(
            str(node),
            label=str(node),
            size=size * 1.8 if is_highlighted else size,
            color={
                "background": "#ffdd00" if is_highlighted else color,
                "border": "#ff8800" if is_highlighted else "#ffffff44",
                "highlight": {"background": "#ffdd00", "border": "#ff8800"}
            },
            title=tooltip,
            font={"size": 14 if is_highlighted else 11, "color": "white"}
        )
    for u, v, data in G.edges(data=True):
        w = data.get('weight', 1.0)
        net.add_edge(str(u), str(v), value=w, color="#aaaaaa40", width=1)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode='w') as f:
        net.save_graph(f.name)
        tmp_path = f.name
    with open(tmp_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    os.unlink(tmp_path)
    return html_content

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

with tab1:
    uploaded_file = None
    if graph_option == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader(
            "Upload edge list CSV (columns: source, target)",
            type=["csv"]
        )

    G = add_weights(load_graph(graph_option, uploaded_file))

    if not selected:
        st.warning("Please select at least one centrality index.")
        st.stop()

    comp = compute_all(G, selected)

    # Network overview
    st.subheader("Network Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Nodes", G.number_of_nodes())
    col2.metric("Edges", G.number_of_edges())
    col3.metric("Density", f"{nx.density(G):.4f}")

    # Top nodes table
    st.subheader("Top Nodes")
    first_scores = comp.results[selected[0]]
    top_nodes = sorted(first_scores.items(), key=lambda x: -x[1])[:top_n]
    top_node_ids = [n for n, _ in top_nodes]

    rows = []
    for node in top_node_ids:
        row = {"Node": str(node)}
        for name in selected:
            row[name] = round(comp.results[name].get(node, 0.0), 4)
        rows.append(row)

    df_results = pd.DataFrame(rows)
    st.dataframe(df_results, use_container_width=True)

    csv = df_results.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download results as CSV",
        data=csv,
        file_name="centrality_results.csv",
        mime="text/csv"
    )

    # Bar chart
    st.subheader("Centrality Comparison Chart")
    fig_bar = go.Figure()
    colors = px.colors.qualitative.Plotly
    for i, name in enumerate(selected):
        values = [comp.results[name].get(node, 0.0) for node in top_node_ids]
        fig_bar.add_trace(go.Bar(
            name=name,
            x=[str(n) for n in top_node_ids],
            y=values,
            marker_color=colors[i % len(colors)],
            hovertemplate=f"<b>{name}</b><br>Node: %{{x}}<br>Score: %{{y:.4f}}<extra></extra>"
        ))
    fig_bar.update_layout(
        barmode='group',
        xaxis_title="Node",
        yaxis_title="Centrality Score",
        title="Centrality Indices Comparison",
        legend_title="Index",
        hovermode="x unified",
        height=450,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Network graph
    st.subheader("Interactive Network Graph")
    col_graph1, col_graph2 = st.columns([3, 1])
    with col_graph1:
        st.markdown(f"Node size and color reflect **{selected[0]}** centrality.")
    with col_graph2:
        filter_top = st.slider(
            "Show top N nodes",
            min_value=5,
            max_value=G.number_of_nodes(),
            value=G.number_of_nodes(),
            step=1
        )
        highlight_node = st.text_input(
            "Highlight node",
            value="",
            placeholder="e.g. 0"
        )

    scores = comp.results[selected[0]]
    top_nodes_filter = set(
        node for node, _ in sorted(scores.items(), key=lambda x: -x[1])[:filter_top]
    )
    G_filtered = G.subgraph(top_nodes_filter).copy()
    scores_filtered = {node: scores[node] for node in top_nodes_filter}

    comp_filtered = CentralityComparator()
    for name in selected:
        instance = INDEX_MAP[name](G_filtered)
        instance.compute()
        comp_filtered.add(name, instance)

    st.markdown(f"Showing **{filter_top}** of **{G.number_of_nodes()}** nodes.")

    try:
        hl = int(highlight_node) if highlight_node else None
    except ValueError:
        hl = highlight_node if highlight_node else None

    html_content = build_pyvis(G_filtered, scores_filtered, selected, comp_filtered, highlight=hl)
    components.html(html_content, height=640, scrolling=False)

    # Correlation matrix
    if len(selected) > 1:
        st.subheader("Correlation Matrix")
        nodes_list = list(G.nodes())
        matrix = np.array([
            [comp.results[name][node] for node in nodes_list]
            for name in selected
        ])
        corr = np.corrcoef(matrix)
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr,
            x=selected,
            y=selected,
            colorscale="RdBu",
            zmin=-1, zmax=1,
            text=[[f"{corr[i][j]:.2f}" for j in range(len(selected))]
                  for i in range(len(selected))],
            texttemplate="%{text}",
            textfont={"size": 14},
            hovertemplate="<b>%{y} vs %{x}</b><br>Correlation: %{z:.3f}<extra></extra>"
        ))
        fig_corr.update_layout(
            title="Pearson Correlation between Centrality Indices",
            height=420,
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    # Node inspector
    st.subheader("Node Inspector")
    selected_node = st.selectbox(
        "Select a node to inspect",
        options=sorted(G.nodes())
    )

    if selected_node is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Node {selected_node} — Centrality Scores**")
            rows_inspector = []
            for name in selected:
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
            if len(selected) >= 3:
                categories = selected
                values = [comp.results[name].get(selected_node, 0.0) for name in selected]
                avg_values = [np.mean(list(comp.results[name].values())) for name in selected]

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
                                for name in selected
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
    st.markdown("*Network Centrality Analyzer — Bachelor's Thesis Project*")

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — COMPARE NETWORKS
# ════════════════════════════════════════════════════════════════════════════

with tab2:
    st.subheader("Compare Two Networks")
    st.markdown("Select two networks and compare how centrality indices differ between them.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### Network A")
        graph_a = st.selectbox(
            "Select Network A",
            ["Karate Club", "Les Misérables", "Random (Barabási–Albert)",
             "Florentine Families", "Davis Women's Club", "Petersen Graph"],
            key="graph_a"
        )
    with col_b:
        st.markdown("### Network B")
        graph_b = st.selectbox(
            "Select Network B",
            ["Karate Club", "Les Misérables", "Random (Barabási–Albert)",
             "Florentine Families", "Davis Women's Club", "Petersen Graph"],
            index=1,
            key="graph_b"
        )

    if not selected:
        st.warning("Please select at least one centrality index in the sidebar.")
    else:
        GA = add_weights(load_graph(graph_a))
        GB = add_weights(load_graph(graph_b))
        comp_a = compute_all(GA, selected)
        comp_b = compute_all(GB, selected)

        st.subheader("Network Overview")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("A — Nodes", GA.number_of_nodes())
        c2.metric("A — Edges", GA.number_of_edges())
        c3.metric("A — Density", f"{nx.density(GA):.4f}")
        c4.metric("B — Nodes", GB.number_of_nodes())
        c5.metric("B — Edges", GB.number_of_edges())
        c6.metric("B — Density", f"{nx.density(GB):.4f}")

        st.subheader("Top Nodes Comparison")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**{graph_a}**")
            scores_a = comp_a.results[selected[0]]
            top_a = sorted(scores_a.items(), key=lambda x: -x[1])[:top_n]
            rows_a = []
            for node, _ in top_a:
                row = {"Node": str(node)}
                for name in selected:
                    row[name] = round(comp_a.results[name].get(node, 0.0), 4)
                rows_a.append(row)
            st.dataframe(pd.DataFrame(rows_a), use_container_width=True)
        with col2:
            st.markdown(f"**{graph_b}**")
            scores_b = comp_b.results[selected[0]]
            top_b = sorted(scores_b.items(), key=lambda x: -x[1])[:top_n]
            rows_b = []
            for node, _ in top_b:
                row = {"Node": str(node)}
                for name in selected:
                    row[name] = round(comp_b.results[name].get(node, 0.0), 4)
                rows_b.append(row)
            st.dataframe(pd.DataFrame(rows_b), use_container_width=True)

        st.subheader("Score Distribution Comparison")
        dist_index = st.selectbox(
            "Select index to compare distributions",
            selected,
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
            title=f"{dist_index} Centrality — Score Distribution",
            xaxis_title="Score",
            yaxis_title="Count",
            height=400,
            legend_title="Network"
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        st.subheader("Network Graphs")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**{graph_a}**")
            html_a = build_pyvis(GA, comp_a.results[selected[0]], selected, comp_a)
            components.html(html_a, height=500, scrolling=False)
        with col2:
            st.markdown(f"**{graph_b}**")
            html_b = build_pyvis(GB, comp_b.results[selected[0]], selected, comp_b)
            components.html(html_b, height=500, scrolling=False)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — NODE IMPACT
# ════════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("Node Impact Analysis")
    st.markdown("""
    Select a node to remove and see how centrality scores change across the network.
    This shows which nodes are most critical to the network structure.
    """)

    if not selected:
        st.warning("Please select at least one centrality index in the sidebar.")
    else:
        G_impact = add_weights(load_graph(graph_option))
        comp_full = compute_all(G_impact, selected)

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
                selected,
                key="impact_index"
            )

        G_removed = G_impact.copy()
        G_removed.remove_node(remove_node)
        comp_removed = compute_all(G_removed, selected)

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
        colors_bar = ['#00CC96' if x >= 0 else '#EF553B'
                      for x in top_changed['Change']]

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
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Before — Node {remove_node} present**")
            html_before = build_pyvis(G_impact, comp_full.results[impact_index], selected, comp_full)
            components.html(html_before, height=450, scrolling=False)
        with col2:
            st.markdown(f"**After — Node {remove_node} removed**")
            html_after = build_pyvis(G_removed, comp_removed.results[impact_index], selected, comp_removed)
            components.html(html_after, height=450, scrolling=False)

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

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ════════════════════════════════════════════════════════════════════════════

with tab4:
    st.subheader("About This Tool")
    st.markdown("""
    This tool provides an interactive interface for computing and comparing
    centrality indices in network structures. It supports classical indices
    as well as weighted variants for more nuanced analysis.
    """)

    st.subheader("Implemented Indices")

    indices_info = {
        "Degree Centrality": {
            "formula": "C(v) = deg(v) / (N - 1)",
            "description": "Measures the number of direct connections a node has, normalized by the maximum possible connections.",
            "use_case": "Finding the most connected individuals in social networks."
        },
        "Closeness Centrality": {
            "formula": "C(v) = (N-1) / Σ d(v,u)",
            "description": "Measures how close a node is to all other nodes based on shortest path lengths.",
            "use_case": "Identifying nodes that can quickly spread information across a network."
        },
        "Betweenness Centrality": {
            "formula": "C(v) = Σ σ(s,t|v) / σ(s,t)",
            "description": "Measures how often a node lies on the shortest path between two other nodes.",
            "use_case": "Finding bottlenecks and bridge nodes in communication networks."
        },
        "Eigenvector Centrality": {
            "formula": "C(v) = (1/λ) Σ C(u) for u in N(v)",
            "description": "A node is important if it is connected to other important nodes. Basis for Google PageRank.",
            "use_case": "Ranking web pages, finding influential users in social networks."
        },
        "Weighted Degree Centrality": {
            "formula": "C(v) = Σ w(e) / (N - 1)",
            "description": "Extends Degree Centrality by summing edge weights instead of counting connections.",
            "use_case": "Transport networks where connection strength matters."
        },
        "Weighted Closeness Centrality": {
            "formula": "C(v) = (N-1) / Σ d_w(v,u)",
            "description": "Uses weighted shortest paths where stronger connections mean shorter distances.",
            "use_case": "Logistics and supply chain optimization."
        },
        "Weighted Betweenness Centrality": {
            "formula": "C(v) = Σ σ_w(s,t|v) / σ_w(s,t)",
            "description": "Betweenness computed on weighted shortest paths.",
            "use_case": "Finding critical nodes in weighted infrastructure networks."
        },
    }

    for name, info in indices_info.items():
        with st.expander(f"{name}"):
            st.markdown(f"**Formula:** `{info['formula']}`")
            st.markdown(f"**Description:** {info['description']}")
            st.markdown(f"**Use case:** {info['use_case']}")

    st.subheader("Tech Stack")
    st.markdown("""
    - **Python 3.9+**
    - **NetworkX** — graph analysis
    - **NumPy** — matrix operations
    - **Streamlit** — web interface
    - **Plotly** — interactive charts
    - **Pyvis** — interactive network visualization
    """)

    st.subheader("Author")
    st.markdown("""
    Developed as part of a Bachelor's Thesis in Data Science and Business Analytics.
    Repository: [github.com/ekozlovskii/centrality-repo](https://github.com/ekozlovskii/centrality-repo)
    """)