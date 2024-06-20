import networkx as nx
import numpy as np
import pytest
from napari_skeleton_curator.code.skeleton import *
from napari_skeleton_curator.code.utils import *

def generate_toy_skeleton_graph(num_nodes, angle, edge_length):
    # Create a toy graph
    graph = nx.Graph()

    # Convert angle to radians
    angle_rad = np.radians(angle/2)

    # Add node positions
    node_pos_dic = {0: np.array([0, 0])}
    parent_nodes = [0]  # Start with the root node
    i = 1

    while i < num_nodes:
        new_parents = []
        for parent_node in parent_nodes:
            if i < num_nodes:
                # Add the first child
                m = edge_length * np.cos(angle_rad)
                n = edge_length * np.sin(angle_rad)
                node_pos_dic[i] = node_pos_dic[parent_node] + np.array([m, n, 0]) 
                graph.add_node(i)
                graph.add_edge(parent_node, i)
                new_parents.append(i)
                i += 1

            if i < num_nodes:
                # Add the second child and rotate in the other direction
                m = edge_length * np.cos(-1*angle_rad)
                n = edge_length * np.sin(-1*angle_rad)
                node_pos_dic[i] = node_pos_dic[parent_node] + np.array([m, n, 0]) 
                graph.add_node(i)
                graph.add_edge(parent_node, i)
                new_parents.append(i)
                i += 1

        parent_nodes = new_parents

    nx.set_node_attributes(graph, node_pos_dic, 'pos')
    return graph, node_pos_dic


def test_compute_branch_length():
    # Generate a toy graph
    graph, _ = generate_toy_skeleton_graph(7, 60, 1)

    # Compute the branch lengths
    compute_branch_length(graph)
    branch_lengths = nx.get_edge_attributes(graph, 'length')

    # Check the branch lengths
    assert branch_lengths == {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}

test_compute_branch_length()
