import math
from typing import Dict
from collections import defaultdict

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from absl import app, flags

import matplotlib.patheffects as path_effects


flags.DEFINE_string('input_file', 'taxonomy_sanitized.csv', 'The input file to read')
flags.DEFINE_string('output_file', 'taxonomy_out.csv', 'The output file to write')
flags.DEFINE_integer('absorb_fewer_than', 250, 'Absorb subtreees with fewer than `N` total counts into parent')
flags.DEFINE_integer('merge_fewer_than', 2000, 'Merge nodes with fewer than `N` total counts')
FLAGS = flags.FLAGS

"""

This script processes taxonomy data from a CSV file to construct a tree from a FPT taxonomy CSV with `fpt` and `count` columns, optionally combine subtrees based on thresholds, visualize the tree radially in an SVG and output node data to a CSV.

Usage:
    python tree_utils.py --input_file=<path> --output_file=<path> [--absorb_fewer_than=<N>] [--merge_fewer_than=<N>]

- input_file: CSV with 'fpt' (path like "Cat > Subcat") and 'count' columns.
- output_file: Where to save the output CSV.
- absorb_fewer_than=N: Absorb subtrees with fewer than N node `total`.
- merge_fewer_than=N: Merge subtrees with fewer than N node `total`.

"""

class Node:
    """Class representing a node in the tree.

    Attributes:
        id (str): Unique identifier.
        parent_id (str): Parent's identifier.
        count (int): Direct count.
        total (int): Total count including descendants.
    """

    def __init__(self, id: str, parent_id: str, count: int):
        self.id = id
        self.parent_id = parent_id
        self.count = count
        self.total = count


class Tree:
    """Class managing the tree structure."""

    root_id: str = "Root"

    def __init__(self):
        """Initializes the tree with a root node."""

        self.adj = defaultdict(set)
        self.node_dict = {}

        # Set root node
        self.adj[self.root_id]
        self.node_dict[self.root_id] = Node(id=self.root_id, parent_id=self.root_id, count=0)

    def set_child(self, node: Node) -> Node:
        """Adds or updates a child node and propagates count updates to ancestors."""

        parent_id = node.parent_id
        if parent_id not in self.node_dict:
            raise ValueError(f"Parent node with id {parent_id} does not exist")
        
        if node.id in self.node_dict:
            previous_node = self.node_dict[node.id]
            offset = node.count - previous_node.count
            node.total = previous_node.total + offset
        else:
            offset = node.count
        self.adj[parent_id].add(node.id)

        # Update the total for all parents
        while parent_id != self.root_id:
            parent = self.node_dict[parent_id]
            parent.total += offset
            parent_id = parent.parent_id
        self.node_dict[self.root_id].total += offset

        self.node_dict[node.id] = node
        return self.node_dict[node.id]

    def check_subtree(self, node_id: str) -> None:
        """Recursively verifies that node totals match subtree sums."""

        node = self.node_dict[node_id]
        subtotal = node.count
        for child_id in self.adj[node_id]:
            self.check_subtree(child_id)
            subtotal += self.node_dict[child_id].total
        assert node.total == subtotal, f"Node {node_id} has total {node.total} which is not equal to subtotal {subtotal}"
    
    def absorb_node(self, node_id: str) -> None:
        """Absorbs a node into its parent."""

        if node_id == self.root_id:
            return

        if node_id not in self.node_dict:
            raise ValueError(f"Node with id {node_id} does not exist")

        node = self.node_dict[node_id]
        for child_id in self.adj[node_id]:
            self.absorb_node(child_id)
            self.node_dict.pop(child_id)
        self.adj[node_id].clear()
        self.node_dict.pop(node_id)

        parent = node.parent_id
        self.node_dict[parent].count += node.count
        self.adj[parent].remove(node_id)
    
    def absorb_nodes_fewer_than(self, node_id: str, n: int) -> None:
        """Absorbs subtrees with fewer than n node `total`."""

        for child_id in list(self.adj[node_id]):
            self.absorb_nodes_fewer_than(child_id, n)

        if self.node_dict[node_id].total < n:
            self.absorb_node(node_id)

    def merge_nodes_fewer_than(self, node_id: str, n: int) -> None:
        """Merges subtrees with fewer than n node `total`."""

        for child_id in list(self.adj[node_id]):
            self.merge_nodes_fewer_than(child_id, n)

        if self.node_dict[node_id].total < n:
            for child_id in list(self.adj[node_id]):
                self.absorb_node(child_id)

    def get_num_leaves(self) -> Dict[str, int]:
        """Computes the number of leaves under each node recursively."""

        num_leaves = {}
        def _num_leaves(node_id: str, num_leaves: Dict[str, int]):
            if not self.adj[node_id]:
                num_leaves[node_id] = 1
                return
            
            n = 0
            for child_id in self.adj[node_id]:
                if child_id not in num_leaves:
                    _num_leaves(child_id, num_leaves)
                n += num_leaves[child_id]
            num_leaves[node_id] = n
            return n

        _num_leaves(self.root_id, num_leaves)
        return num_leaves

    def save_tree(self) -> None:
        """Generates a radial visualization of the tree and saves it to 'tree_graph.svg'."""

        G = nx.Graph()

        for node_id in self.node_dict:
            G.add_node(node_id)
        
        for node_id in self.adj:
            for child_id in self.adj[node_id]:
                G.add_edge(node_id, child_id, weight=self.node_dict[child_id].total)

        # Recursive function to compute radial positions and angles
        angles = {}
        pos = {}
        num_leaves = self.get_num_leaves()
        def layout_radial(node, angle_start, angle_end, radius):
            # Calculate midpoint angle and position the node
            angle = (angle_start + angle_end) / 2
            pos[node] = (radius * math.cos(angle), radius * math.sin(angle))

            deg_angle = math.degrees(angle)
            if deg_angle > 90:
                deg_angle -= 180
            elif deg_angle < -90:
                deg_angle += 180
            angles[node] = deg_angle
            
            # Get and sort children
            children = sorted(list(self.adj[node]), key=lambda c: self.node_dict[c].total, reverse=True)
            if not children:
                return
            
            # Divide the angle span among children
            angle_span = angle_end - angle_start
            child_span = angle_span / num_leaves[node]
            current_start = angle_start
        
            # Recurse for each child with updated angle and increased radius
            for child in children:
                current_end = current_start + child_span * num_leaves[child]
                layout_radial(child, current_start, current_end, radius + 100 + math.log(self.node_dict[child].total))
                current_start = current_end
        
        # Start layout from root with full circle (0 to 2π) at radius 0
        layout_radial(self.root_id, -math.pi, math.pi, 0)
        
        # Early return if no positions (empty tree)
        if not pos:
            return
        
        # Compute bounding box from positions
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        width = max_x - min_x + 1e-6
        height = max_y - min_y + 1e-6
        
        # Determine number of nodes for scaling
        N = len(self.node_dict)
        
        # Set figure size dynamically
        fig_width = 10 + width / 20
        fig_height = 10 + height / 20
        plt.figure(figsize=(fig_width, fig_height))
        
        # Draw the graph with computed positions and edge widths
        totals = [self.node_dict[node].total for node in G.nodes()]
        max_total = max(totals)
        min_size = 1
        max_size = 5e4
        node_sizes = [min_size + (max_size - min_size) * (total / max_total) if max_total > 0 else min_size for total in totals]

        edge_widths = [G[u][v]['weight'] for u, v in G.edges()]
        max_edge = max(edge_widths) if edge_widths else 1
        min_width = 0.1
        max_width = 1
        edge_widths = [min_width + (max_width - min_width) * (w / max_edge) if max_edge > 0 else min_width for w in edge_widths]
        nx.draw(G, pos=pos, with_labels=False, node_size=node_sizes, width=edge_widths)

        # Draw the labels
        labels = {node: f"{node.split(' > ')[-1]}: {self.node_dict[node].total}" for node in G.nodes()}
        min_font = 1
        max_font = 40
        font_sizes = {}
        for node, nsize in zip(G.nodes(), node_sizes):
            if max_size > min_size:
                font_sizes[node] = min_font + (max_font - min_font) * (nsize - min_size) / (max_size - min_size)
            else:
                font_sizes[node] = (min_font + max_font) / 2

        text_objects = nx.draw_networkx_labels(G, pos, labels)
        for node, text in text_objects.items():
            text.set_fontsize(font_sizes[node])
            text.set_rotation(angles[node])
            text.set_path_effects([path_effects.withStroke(linewidth=1, foreground='white')])
        
        # Add margins and set plot limits
        margin = max(width, height) * 0.05
        plt.xlim(min_x - margin, max_x + margin)
        plt.ylim(min_y - margin, max_y + margin)
        
        # Save the figure to file with tight bounding box
        plt.savefig('tree_graph.svg', bbox_inches='tight')

    def to_pandas(self) -> pd.DataFrame:
        """Converts tree nodes to a Pandas DataFrame with id, count, total."""

        return pd.DataFrame([{
            'id': node.id,
            'count': node.count,
            'total': node.total
        } for node in self.node_dict.values()])


def get_prefixes(path: str) -> list[str]:
    """Generates list of cumulative path prefixes from a ' > ' separated string."""

    subpaths = path.split(' > ')
    if len(subpaths) <= 1:
        return subpaths

    prefixes = [subpaths[0]]
    for subpath in subpaths[1:]:
        prefixes.append(prefixes[-1] + ' > ' + subpath)
    return prefixes


def main(argv):
    """Main function: loads data, builds tree, merges if specified, visualizes, and saves output."""

    df = pd.read_csv(FLAGS.input_file)
    df['path_parts'] = df['fpt'].apply(get_prefixes)

    tree = Tree()
    for index, row in df.iterrows():
        count = int(row['count'])
        parts = row['path_parts']
        current_id = "Root"
        for part in parts:
            if part not in tree.node_dict:
                new_node = Node(id=part, parent_id=current_id, count=count)
                tree.set_child(new_node)
            current_id = part

    tree.check_subtree(tree.root_id)

    if FLAGS.absorb_fewer_than is not None:
        tree.absorb_nodes_fewer_than(tree.root_id, FLAGS.absorb_fewer_than)
        tree.check_subtree(tree.root_id)

    if FLAGS.merge_fewer_than is not None:
        tree.merge_nodes_fewer_than(tree.root_id, FLAGS.merge_fewer_than)
        tree.check_subtree(tree.root_id)

    tree.save_tree()
    tree.to_pandas().to_csv(FLAGS.output_file, index=False)

if __name__ == "__main__":
    app.run(main)