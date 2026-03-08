# utils/comparator.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class CentralityComparator:
    """
    Compares multiple centrality indices on the same graph.
    Provides correlation analysis and visualization.
    """

    def __init__(self):
        self.results = {}  # { 'IndexName': {node: score} }

    def add(self, name: str, centrality_instance):
        """Add a computed centrality index."""
        if not centrality_instance.scores:
            centrality_instance.compute()
        self.results[name] = centrality_instance.scores

    def top_nodes(self, n: int = 10):
        """Print comparison table of top-n nodes sorted by first index."""
        if not self.results:
            print("No indices added yet.")
            return

        names = list(self.results.keys())
        first = self.results[names[0]]
        top = sorted(first.items(), key=lambda x: -x[1])[:n]

        # Header
        header = f"{'Node':<8}" + "".join(f"{name:<14}" for name in names)
        print(header)
        print("-" * len(header))

        for node, _ in top:
            row = f"{node:<8}"
            for name in names:
                score = self.results[name].get(node, 0.0)
                row += f"{score:<14.4f}"
            print(row)

    def correlation_matrix(self):
        """Compute Pearson correlation between all index pairs."""
        names = list(self.results.keys())
        nodes = list(list(self.results.values())[0].keys())
        n = len(names)

        # Строим матрицу значений
        matrix = np.array([
            [self.results[name][node] for node in nodes]
            for name in names
        ])

        corr = np.corrcoef(matrix)

        print("\n=== Correlation Matrix ===")
        print(f"{'':16}" + "".join(f"{name:<14}" for name in names))
        print("-" * (16 + 14 * n))
        for i, name in enumerate(names):
            row = f"{name:<16}"
            for j in range(n):
                row += f"{corr[i][j]:<14.3f}"
            print(row)

        return corr, names

    def plot_comparison(self, top_n: int = 15):
        """Bar chart comparing all indices for top nodes."""
        if not self.results:
            return

        names = list(self.results.keys())
        first = self.results[names[0]]
        top_nodes = [node for node, _ in sorted(
            first.items(), key=lambda x: -x[1])[:top_n]]

        x = np.arange(len(top_nodes))
        width = 0.8 / len(names)
        colors = list(mcolors.TABLEAU_COLORS.values())

        fig, ax = plt.subplots(figsize=(14, 6))

        for i, name in enumerate(names):
            values = [self.results[name].get(node, 0.0) for node in top_nodes]
            ax.bar(x + i * width, values, width, label=name, color=colors[i % len(colors)])

        ax.set_xlabel('Node')
        ax.set_ylabel('Centrality Score')
        ax.set_title('Centrality Indices Comparison')
        ax.set_xticks(x + width * (len(names) - 1) / 2)
        ax.set_xticklabels([str(n) for n in top_nodes])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig('comparison.png', dpi=150)
        plt.show()
        print("Saved to comparison.png")