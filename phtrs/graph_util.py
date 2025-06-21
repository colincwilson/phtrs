# Graph utilities.
import re, sys
import igraph
from nltk.tree import *


def string_to_tree(syntax_str):
    """ Converted bracketed string to tree. """
    tree = Tree.fromstring(syntax_str)
    tree.pretty_print()
    return tree


def string_to_graph(syntax_str):
    """ Convert bracketed string to graph. """
    tree = string_to_tree(syntax_str)
    graph = tree_to_graph(tree)
    return graph


def tree_to_graph(tree):
    """
    Convert tree to graph.
    Returns: igraph.Graph, root node index
    source: CoPilot (GPT-4.1)
    """
    graph = igraph.Graph(directed=True)
    node_ids = {}

    def add_nodes(t, parent_id=None):
        node_id = len(graph.vs)
        label = t.label() if hasattr(t, "label") else str(t)
        graph.add_vertex(name=label, label=label)
        node_ids[id(t)] = node_id
        if parent_id is not None:
            graph.add_edge(parent_id, node_id)
        if hasattr(t, "height") and t.height() > 1:
            for child in t:
                add_nodes(child, node_id)
        elif isinstance(t, str):
            # Leaf node.
            pass

    add_nodes(tree)
    return graph, 0  # root node index is 0


if __name__ == "__main__":
    # Example usage of the bracketed_to_igraph function.
    tree_str = "(S (NP (DT The) (NN dog)) (VP (VBD barked)))"
    graph, _ = string_to_graph(tree_str)
    print(list(graph.vs))
    graph.write('fig/syntax_graph.dot', format='dot')
