import networkx as nx
import numpy as np
import pytest
from napari_skeleton_curator.skeleton import *
# from napari_skeleton_curator.utils import *
from napari_skeleton_curator.utils import generate_toy_skeleton_graph

def test_compute_branch_length():
    # Generate a toy graph
    num_nodes = 10
    edge_length = 10
    graph, _ = generate_toy_skeleton_graph(num_nodes, 60, edge_length)
    # Compute the branch lengths
    compute_branch_length(graph)
    branch_lengths = nx.get_edge_attributes(graph, 'length')
    num_edges = len(graph.edges)
    sum_length = np.round(sum(branch_lengths.values()),2)

    # Check the branch lengths
    assert sum_length == num_edges * edge_length


def test_compute_start_end_node():
    # Generate a toy graph
    num_nodes = 10
    edge_length = 10
    origin = -1
    graph, _ = generate_toy_skeleton_graph(num_nodes, 60, edge_length)
    # Compute the start and end nodes
    compute_start_end_node(graph, origin)
    start_nodes = nx.get_edge_attributes(graph, 'start_node')
    end_nodes = nx.get_edge_attributes(graph, 'end_node')

    for edge in graph.edges:
        assert (start_nodes[edge] in list(edge) and
                 end_nodes[edge] in list(edge) and 
                 start_nodes[edge] < end_nodes[edge])


def test_compute_level():
    # Generate a toy graph
    num_nodes = 8
    edge_length = 10
    graph, _ = generate_toy_skeleton_graph(num_nodes, 60, edge_length)
    origin = -1
    # Compute the levels
    compute_start_end_node(graph, origin)

    compute_level(graph, origin)

    levels = nx.get_edge_attributes(graph, 'level')
    end_nodes = nx.get_edge_attributes(graph, 'end_node')
    # Check the levels

    for edge in graph.edges:
        if end_nodes[edge] <1:
            assert levels[edge] == 0
        elif end_nodes[edge] < 3:
            assert levels[edge] == 1
        elif end_nodes[edge] < 7:
            assert levels[edge] == 2



