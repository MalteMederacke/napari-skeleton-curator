import numpy as np
import networkx as nx
from scipy.spatial.transform import Rotation as R
from morphosamplers.spline import Spline3D
from scipy.spatial import cKDTree
from collections import defaultdict, deque

import skan as sk
import networkx as nx

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



def simplify_graph(G):
    """
    The simplifyGraph function simplifies a given graph by removing nodes of degree 2 and fusing their incident edges.
    Source:  https://stackoverflow.com/questions/53353335/networkx-remove-node-and-reconnect-edges

    :param G: A NetworkX graph object to be simplified
    :return: A tuple consisting of the simplified NetworkX graph object, a list of positions of kept nodes, and a list of indices of kept nodes.
    """

    g = G.copy()

    while any(degree == 2 for _, degree in g.degree):

        keept_node_pos = []
        keept_node_idx = []
        g0 = g.copy()  # <- simply changing g itself would cause error `dictionary changed size during iteration`
        for node, degree in g.degree():
            if degree == 2:

                if g.is_directed():  # <-for directed graphs
                    a0, b0 = list(g0.in_edges(node))[0]
                    a1, b1 = list(g0.out_edges(node))[0]

                else:
                    edges = g0.edges(node)
                    edges = list(edges.__iter__())
                    if len(edges) != 2:
                        continue
                    a0, b0 = edges[0]
                    a1, b1 = edges[1]

                e0 = a0 if a0 != node else b0
                e1 = a1 if a1 != node else b1

                g0.remove_node(node)
                g0.add_edge(e0, e1)
            else:
                keept_node_pos.append(g.nodes[node]['node_coordinate'])
                keept_node_idx.append(node)
        g = g0
    return g, keept_node_pos, keept_node_idx




def skan_to_networkx(skeleton):
    skel_nx = nx.from_scipy_sparse_array(skeleton.graph)

    for i, node in enumerate(skel_nx.nodes):
        skel_nx.nodes[node]['node_coordinate'] = skeleton.coordinates[i]

    #remove nodes without edges
    skel_nx.remove_nodes_from(list(nx.isolates(skel_nx)))

    #reduce to topology 
    skeleton_graph_simple, kept_node_pos, kept_node_idx = simplify_graph(skel_nx)

    #add attributes

    path_dict = {}
    node_coords = nx.get_node_attributes(skel_nx, 'node_coordinate')
    for e1, e2 in skeleton_graph_simple.edges:
        shortest_path = nx.shortest_path(skel_nx, e1, e2)

        # shortest_path = shortest_path[1:-1]
        path_dict[(e1, e2)] = np.array([node_coords[i] for i in shortest_path])

    nx.set_edge_attributes(skeleton_graph_simple, path_dict, 'edge_coordinates')

    return skeleton_graph_simple

def networkx_to_skan(skeleton_graph):
    #create skan skeleton
    skel = sk.Skeleton(skeleton3D_to_skan(skeleton_graph))
    skel.graph = nx.to_scipy_sparse_array(skeleton_graph)

    return skel


def select_points_in_bounding_box(
    points: np.ndarray,
    lower_left_corner: np.ndarray,
    upper_right_corner: np.ndarray,
) -> np.ndarray:
    """From an array of points, select all points inside a specified
    axis-aligned bounding box.
    Parameters
    ----------
    points : np.ndarray
        The n x d array containing the n, d-dimensional points to check.
    lower_left_corner : np.ndarray
        The point corresponding to the corner of the bounding box
        with lowest coordinate values.
    upper_right_corner : np.ndarray
        The point corresponding to the corner of the bounding box
        with the highest coordinate values.
    Returns
    -------
    points_in_box : np.ndarray
        The n x d array containing the n points inside of the
        specified bounding box.
    """
    in_box_mask = np.all(
        np.logical_and(
            lower_left_corner <= points, upper_right_corner >= points
        ),
        axis=1,
    )
    return points[in_box_mask]

def draw_line_segment(
    start_point: np.ndarray,
    end_point: np.ndarray,
    skeleton_image: np.ndarray,
    fill_value: int = 1,
):
    """Draw a line segment in-place.
    Note: line will be clipped if it extends beyond the
    bounding box of the skeleton_image.
    Parameters
    ----------
    start_point : np.ndarray
        (d,) array containing the starting point of the line segment.
        Must be an integer index.
    end_point : np.ndarray
        (d,) array containing the end point of the line segment.
        Most be an integer index
    skeleton_image : np.ndarray
        The image in which to draw the line segment.
        Must be the same dimensionality as start_point and end_point.
    fill_value : int
        The value to use for the line segment.
        Default value is 1.
    """
    branch_length = np.linalg.norm(end_point - start_point)
    n_skeleton_points = int(2 * branch_length)
    skeleton_points = np.linspace(start_point, end_point, n_skeleton_points)

    # filter for points within the image
    image_bounds = np.asarray(skeleton_image.shape) - 1
    skeleton_points = select_points_in_bounding_box(
        points=skeleton_points,
        lower_left_corner=np.array([0, 0, 0]),
        upper_right_corner=image_bounds,
    ).astype(int)
    skeleton_image[
        skeleton_points[:, 0], skeleton_points[:, 1], skeleton_points[:, 2]
    ] = fill_value

def draw_points(
    points: np.ndarray,
    # end_point: np.ndarray,
    skeleton_image: np.ndarray,
    fill_value: int = 1,
):
    """Draw a line segment in-place.
    Note: line will be clipped if it extends beyond the
    bounding box of the skeleton_image.
    Parameters
    ----------
    start_point : np.ndarray
        (d,) array containing the starting point of the line segment.
        Must be an integer index.
    end_point : np.ndarray
        (d,) array containing the end point of the line segment.
        Most be an integer index
    skeleton_image : np.ndarray
        The image in which to draw the line segment.
        Must be the same dimensionality as start_point and end_point.
    fill_value : int
        The value to use for the line segment.
        Default value is 1.
    """
 
    skeleton_points = points

    # filter for points within the image
    image_bounds = np.asarray(skeleton_image.shape) - 1
    skeleton_points = select_points_in_bounding_box(
        points=skeleton_points,
        lower_left_corner=np.array([0, 0, 0]),
        upper_right_corner=image_bounds,
    ).astype(int)
    skeleton_image[
        skeleton_points[:, 0], skeleton_points[:, 1], skeleton_points[:, 2]
    ] = fill_value

def skeleton3D_to_skan(G,include_edge_points = True, branch_point_value = 1, edge_point_value =1, pad_size = 15):
    """transforms networkx digraph to image. 
    Returns three 3D images containing the voxelized graph, its branch point value and its endpoints,  respectivley.

    Args:
        G (nx.digraph): graph to be voxelized. Needs 3D euclidian coordinates in node attribute called 'edge_coordinates'.
        include_edge_points (bool, optional): If True, returns 2 extra label images with the branch points and endpoints. 
                                            CURRENTLY BROKEN. Defaults to True.
        branch_point_value (int, optional): value branch point label image gets. Defaults to 1.
        edge_point_value (int, optional): valuee branch point label image gets. Defaults to 1.

    Returns:
        The (x,y,z) Images containing the skeleton_image, branch_points_image, end_point_image: 
    """


    pos = nx.get_edge_attributes(G, 'edge_coordinates')
    pos = np.concatenate(list(pos.values()))


    x_offset = 0
    y_offset = 0
    z_offset = 0  
    x_coord = int(np.ceil(np.max(pos[:,0]) - x_offset))
    y_coord = int(np.ceil(np.max(pos[:,1]) - y_offset))
    z_coord = int(np.ceil(np.max(pos[:,2]) - z_offset))

    skeleton_image = np.zeros((x_coord + pad_size, y_coord + pad_size, z_coord + pad_size), dtype = 'uint16')
    skeleton_image[pos[:,0], pos[:,1], pos[:,2]] = 1

    return skeleton_image


def merge_two_graphs_with_overlap(graph1, graph2, distance_threshold=3):
    graph2 = graph2.copy()
    edges1 = nx.get_edge_attributes(graph1, 'edge_coordinates')
    edges1_coords = np.concatenate(list(edges1.values()))

    edges2 = nx.get_edge_attributes(graph2, 'edge_coordinates')
    edges2_coords =np.concatenate(list(edges2.values()))



    edges2_rev = {}
    for key, array in edges2.items():
        for coord in array:
            edges2_rev[tuple(coord)] = {'edge':key, 'distance':None}


    kdt = cKDTree(edges1_coords)
    dist, indices = kdt.query(edges2_coords)

    for i,coord in enumerate(edges2_coords): 
        distance = dist[i]
        edges2_rev[tuple(coord)]['distance'] = distance

    #summarize to get the avarage distance per edge...
    edge_distances = {}
    for key, value in edges2_rev.items():
        edge = value['edge']
        distance = value['distance']
        if edge not in edge_distances:
            edge_distances[edge] = []
        edge_distances[edge].append(distance)



    #get mean distance per edge
    mean_edge_distances = {}
    for key, value in edge_distances.items():
        mean_edge_distances[key] = np.mean(value)


    print(len({k for k,v in mean_edge_distances.items() if v < distance_threshold}), 'edges removed')

    #TODO: rename the edges of graph2 to be unique...
    graph2.remove_edges_from({k for k,v in mean_edge_distances.items() if v < distance_threshold})


    skel_graph_merged = nx.compose(graph1, graph2)
    #remove nodes without edges
    skel_graph_merged.remove_nodes_from(list(nx.isolates(skel_graph_merged)))

    return skel_graph_merged
    



def get_nth_generation_edges(graph:nx.DiGraph, start_node, n):
    # Initialize a queue for BFS
    queue = deque([(start_node, 0)])  # (current_node, current_level)
    result = defaultdict(list)

    while queue:
        current_node, current_level = queue.popleft()

        # If we have not yet reached the nth generation, continue to add children
        if current_level < n:
            for child in graph.successors(current_node):
                result[current_level + 1].append((current_node, child))
                queue.append((child, current_level + 1))

    # Convert result to a list of lists
    result_list = [result[level] for level in range(1, n + 1)]
    return result_list


