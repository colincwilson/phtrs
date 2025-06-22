# Graph utilities.
# todo: index nodes/vertices
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
    orig version: CoPilot (GPT-4.1)
    """
    graph = igraph.Graph(directed=True)
    node_ids = {}

    def add_nodes(t, parent_id=None):
        node_id = len(graph.vs)
        label = t.label() if hasattr(t, 'label') else str(t)
        graph.add_vertex(name=label, label=label)
        node_ids[id(t)] = node_id
        if parent_id is not None:
            graph.add_edge(parent_id, node_id)
        if hasattr(t, 'height') and t.height() > 1:
            for child in t:
                add_nodes(child, node_id)
        elif isinstance(t, str):
            # Leaf node.
            pass

    add_nodes(tree)
    print(node_ids)
    return graph, 0  # root node index is 0


def draw_layered_graph(graph, source=None):
    """
    Layout graph in GraphViz/dot format, arranging vertices
    in rows by 'type' attribute.
    orig version: CoPilot (GPT-4.1)
    """
    # Group vertices by type
    type_to_nodes = {}
    for v in graph.vs:
        t = v.attributes().get('type', 'undef')
        type_to_nodes.setdefault(t, []).append(v.index)

    lines = ['digraph G {', 'rankdir=LR;', 'node [shape=box]']
    # Create subgraphs for each type to enforce same rank (row).
    types = type_to_nodes.keys()
    for t in types:
        nodes = type_to_nodes[t]
        lines.append(f'  subgraph cluster_{t} {{')
        lines.append('    rank=same;')
        for idx in nodes:
            label = graph.vs[idx]['label'] \
                if 'label' in graph.vs[idx].attributes() \
                else str(idx)
            lines.append(f'    {idx} [label="{label}"];')
        lines.append('  }')

    # Add edges.
    for e in graph.es:
        src, tgt = e.source, e.target
        lines.append(f'  {src} -> {tgt};')

    lines.append('}')
    ret = '\n'.join(lines)

    # Write to file.
    if source:
        with open(source, 'w') as f:
            f.write(ret)
    return ret


if __name__ == "__main__":
    # Test.
    tree_str = "(S (NP (DT The) (NN dog)) (VP (VBD barked)))"
    graph, _ = string_to_graph(tree_str)
    print(list(graph.vs))
    graph.write('fig/syntax_graph.dot', format='dot')
