import numpy as np
import networkx as nx
import pytest
from napari_skeleton_curator.skeleton import *
# from napari_skeleton_curator.utils import *

from napari_skeleton_curator.utils import generate_toy_skeleton_graph
from napari_skeleton_curator.measurements import *


def test_compute_midline_branch_angles():
    # Generate a toy graph
    num_nodes = 10
    edge_length = 10
    angle = 60
    origin = -1
    graph, _ = generate_toy_skeleton_graph(num_nodes, angle, edge_length)
    # Compute the branch lengths
    compute_branch_length(graph)
    compute_start_end_node(graph, origin)
    compute_level(graph, origin)
    angle_df,_,_,_,_,_ = compute_midline_branch_angles(graph, origin = -1)
    branch_angles = angle_df['angle']
    num_edges = len(graph.edges)
    sum_angle = np.round(sum(branch_angles.values),2)


    assert sum_angle == num_edges * angle/2

test_compute_midline_branch_angles()