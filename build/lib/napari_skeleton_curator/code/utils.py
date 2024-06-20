import numpy as np
import networkx as nx
from scipy.spatial.transform import Rotation as R

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return np.squeeze(np.asarray(vector / np.linalg.norm(vector)))


def ensure_same_normal_direction(normals:dict, reference_direction):
    """Ensure that all normals have the same direction."""
    for key, normal in normals.items():
        if np.sign(normal[0]) != reference_direction:
            print('flip')
            normals[key] = -normal  # Reverse the direction of the normal
    return normals


def count_number_of_tips_connected_to_edge(graph, start_node, end_node):
    # Get highlighted edge
    NODE_COORDINATE_KEY = 'node_coordinate'
    
    #generate diGraph
    if type(graph) != nx.DiGraph:
        graph = nx.DiGraph(graph)
        graph.remove_edges_from(graph.edges - nx.bfs_edges(graph, start_node))


    # Perform a breadth-first search starting from the end_node
    subtree = nx.bfs_tree(graph, end_node)
    
    # Initialize count of endpoints
    num_endpoints = 0
    
    # Iterate through nodes in the subtree
    for node in subtree.nodes:
        # Check if the node is a leaf node (degree 1) and is not the start_node
        if subtree.degree(node) == 1 and node != start_node:
            num_endpoints += 1
            
    return num_endpoints

def rotation_matrix_from_vectors(a, b):
    """
    Compute the rotation matrix that rotates unit vector a onto unit vector b.
    
    Parameters:
        a (numpy.ndarray): The initial unit vector.
        b (numpy.ndarray): The target unit vector.
        
    Returns:
        numpy.ndarray: The rotation matrix.
    """
    # Compute the cross product and its magnitude
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    
    # Compute the dot product
    c = np.dot(a, b)
    
    # Skew-symmetric cross-product matrix
    v_cross = np.array([[0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0]])
    
    # Rotation matrix
    if s != 0:
        rm = np.eye(3) + v_cross + np.dot(v_cross, v_cross) * ((1 - c) / (s ** 2))
    else:
        rm = np.eye(3)
    
    return R.from_matrix(rm)

