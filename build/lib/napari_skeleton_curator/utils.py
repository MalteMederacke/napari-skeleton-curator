import numpy as np
import networkx as nx
from scipy.spatial.transform import Rotation as R
from morphosamplers.spline import Spline3D

EDGE_COORDINATES_KEY = "edge_coordinates"
NODE_COORDINATE_KEY = "node_coordinate"
EDGE_SPLINE_KEY = "edge_spline"

def generate_toy_skeleton_graph(num_nodes, angle, edge_length):
    # Create a toy graph
    graph = nx.Graph()

    # Convert angle to radians
    angle_rad = np.radians(angle/2)

    # Add node positions
    node_pos_dic = {0: np.array([0, 0, 0])}
    parent_nodes = [0]  # Start with the root node
    i = 1
    #add trachea
    trachea_pos = np.array([-edge_length, 0, 0])
    node_pos_dic[-1] = trachea_pos
    graph.add_node(-1, node_coordinate = trachea_pos)
    graph.add_edge(-1, 0, edge_coordinates=np.linspace(trachea_pos, np.array([0, 0, 0]), 5+edge_length))
    while i < num_nodes:
        new_parents = []
        for parent_node in parent_nodes:
            if i < num_nodes:
                # Add the first child
                m = edge_length * np.cos(angle_rad)
                n = edge_length * np.sin(angle_rad)
                new_pos = node_pos_dic[parent_node] + np.array([m, n, 0])
                node_pos_dic[i] = new_pos
                edge_coordinates = np.linspace(node_pos_dic[parent_node], new_pos, 5+edge_length)
                graph.add_node(i)
                graph.add_edge(parent_node, i,edge_coordinates=edge_coordinates)
                new_parents.append(i)
                i += 1

            if i < num_nodes:
                # Add the second child and rotate in the other direction
                m = edge_length * np.cos(-1*angle_rad)
                n = edge_length * np.sin(-1*angle_rad)
                new_pos = node_pos_dic[parent_node] + np.array([m, n, 0])
                node_pos_dic[i] = new_pos                
                edge_coordinates = np.linspace(node_pos_dic[parent_node], new_pos, 5+edge_length)

                graph.add_node(i)
                graph.add_edge(parent_node, i, edge_coordinates=edge_coordinates)
                new_parents.append(i)
                i += 1

        parent_nodes = new_parents

    nx.set_node_attributes(graph, node_pos_dic, NODE_COORDINATE_KEY)
    #add splines
    for edge in graph.edges:
        edge_coordinates = graph.edges[edge][EDGE_COORDINATES_KEY]
        edge_spline = Spline3D(points = edge_coordinates)
        graph.edges[edge][EDGE_SPLINE_KEY] = edge_spline


    return graph, node_pos_dic



def generate_toy_skeleton_graph_symmetric_branch_angle(num_nodes, angle, edge_length):
    # Create a toy graph
    graph = nx.DiGraph()

    angle_rad = np.radians(angle/2)



    # Add node positions
    node_pos_dic = {0: np.array([0, 0, 0])}
    parent_nodes = [0]  # Start with the root node
    #add trachea
    trachea_pos = np.array([-edge_length, 0, 0])
    node_pos_dic[-1] = trachea_pos
    graph.add_node(-1, node_coordinate = trachea_pos)
    graph.add_edge(-1, 0, edge_coordinates=np.linspace(trachea_pos, np.array([0, 0, 0]), 5+edge_length))
    
    #initialize the first two branches 
    m = edge_length * np.cos(angle_rad)
    n = edge_length * np.sin(angle_rad)
    new_pos = node_pos_dic[0] + np.array([m, n, 0])
    node_pos_dic[1] = new_pos
    edge_coordinates = np.linspace(node_pos_dic[0], new_pos, 5+edge_length)
    graph.add_node(1)
    graph.add_edge(0, 1,edge_coordinates=edge_coordinates, side = 'left')
    m = edge_length * np.cos(-1*angle_rad)
    n = edge_length * np.sin(-1*angle_rad)
    new_pos = node_pos_dic[0] + np.array([m, n, 0])
    node_pos_dic[2] = new_pos
    edge_coordinates = np.linspace(node_pos_dic[0], new_pos, 5+edge_length)
    graph.add_node(2)
    graph.add_edge(0, 2,edge_coordinates=edge_coordinates, side = 'right')
    
    parent_nodes = [1, 2]  # Start with the root node
    i = 3
    while i < num_nodes:
        new_parents = []
        for parent_node in parent_nodes:
            if i < num_nodes:
                # Add the first child
                #if the parent is a left node, the child is a left node and 
                # needs to be rotated to the left
                    # Convert angle to radians
                angle_rad = np.radians(angle/2)

                #get the path to node 0 and count the number of left vs right edges
                path = nx.shortest_path(graph, 0, parent_node)
                edges = [(path[i], path[i+1]) for i in range(len(path)-1)]               
                #get sides 
                sides = [graph.edges[edge]['side'] for edge in edges]
                #count the number of left and right edges
                left_edges = sides.count('left')
                right_edges = sides.count('right')
                num_rotations = left_edges - right_edges

                angle_rad = angle_rad*(num_rotations+1)

                # if list(graph.in_edges(parent_node, data=True))[0][2]['side'] == 'right':
                #     angle_rad = angle_rad*-1

                m = edge_length * np.cos(angle_rad)
                n = edge_length * np.sin(angle_rad)
                side = 'left'



                new_pos = node_pos_dic[parent_node] + np.array([m, n, 0])


                    


                node_pos_dic[i] = new_pos
                edge_coordinates = np.linspace(node_pos_dic[parent_node], new_pos, 5+edge_length)
                graph.add_node(i)
                graph.add_edge(parent_node, i,edge_coordinates=edge_coordinates, side = side)
                new_parents.append(i)
                i += 1

            if i < num_nodes:
                # Add the second child and rotate in the other direction
                angle_rad = np.radians(angle)/2

                #get the path to node 0 and count the number of left vs right edges
                path = nx.shortest_path(graph, 0, parent_node)
                edges = [(path[i], path[i+1]) for i in range(len(path)-1)]               
                #get sides 
                sides = [graph.edges[edge]['side'] for edge in edges]
                #count the number of left and right edges
                left_edges = sides.count('left')
                right_edges = sides.count('right')
                num_rotations = left_edges - right_edges

                angle_rad = angle_rad*(num_rotations-1)
                


                # if list(graph.in_edges(parent_node, data=True))[0][2]['side'] == 'right':
                #     angle_rad = angle_rad*-1

                m = edge_length * np.cos(1* angle_rad)
                n = edge_length * np.sin(1* angle_rad)
                side = 'right'
                new_pos = node_pos_dic[parent_node] + np.array([m, n, 0])
                node_pos_dic[i] = new_pos                
                edge_coordinates = np.linspace(node_pos_dic[parent_node], new_pos, 5+edge_length)

                graph.add_node(i)
                graph.add_edge(parent_node, i, edge_coordinates=edge_coordinates, side = side)
                new_parents.append(i)
                i += 1

        parent_nodes = new_parents


        #add splines
        for edge in graph.edges:
            edge_coordinates = graph.edges[edge][EDGE_COORDINATES_KEY]
            edge_spline = Spline3D(points = edge_coordinates)
            graph.edges[edge][EDGE_SPLINE_KEY] = edge_spline



    nx.set_node_attributes(graph, node_pos_dic, NODE_COORDINATE_KEY)
    return graph, node_pos_dic


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

def add_parent_angles(graph,df, origin):
    tree = nx.DiGraph(graph)
    tree.remove_edges_from(tree.edges - nx.bfs_edges(tree, origin))

    edge_length = []
    parent_edges = []
    angle_edges = df['edge']
    for edge in angle_edges:
        edge = graph.edges[edge]

        edge_length.append(edge['length'])
        # parent_edges.append(e14_skeleton.graph.edges[edge]['parent'])
        parent_edge = tree.in_edges(edge['start_node'])
        parent_edges.append(list(parent_edge)[0])
    df['length'] = edge_length
    df['parent_edge'] = parent_edges

    parent_angle = []
    parent_nTips = []
    parent_radius = []
    for row in df.iterrows():
        parent_row = df.loc[df['edge'] == row[1]['parent_edge']]

        if len(parent_row) == 0:
            parent_angle.append(None)
            parent_nTips.append(None)
            parent_radius.append(None)
        else:
            parent_angle.append(parent_row['angle'].values[0])
            parent_nTips.append(parent_row['num_tips'].values[0])
            parent_radius.append(parent_row['radius'].values[0])
    df['parent_angle'] = parent_angle
    df['parent_nTips'] = parent_nTips
    df['parent_radius'] = parent_radius
    return df





