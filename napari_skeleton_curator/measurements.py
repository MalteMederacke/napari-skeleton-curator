
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
from .utils import (unit_vector, 
                   ensure_same_normal_direction, 
                   count_number_of_tips_connected_to_edge)
import open3d as o3d
from .skeleton import compute_start_end_node

def compute_midline_branch_angles(graph:nx.Graph, origin:int, sample_distance:float=0.01):

    """Computes the midline anlges for each branch in the graph and returns
    a dataframe with the angles and all metadata in the tree.

    Points are equal distantly sampled along each parent branch and the inverse
    mean vector is considered as the midline. Then the mean vector from the branch
    is taken and the minimal angle between these two computed. To compute
    the angle based on the branch nodes, see function "compute_midline_branch_angle_branch_nodes"

    Returns
    -------
    pd.Dataframe
        Dataframe containing midline angle measurements and all other edge data

    Raises
    ------
    ValueError
        Raises and error if the end point of the parent branch and the 
        start point of the daughter branch are not the same
    ValueError
        Raises and error if the length of the midline vector is != 1
    ValueError
        Raises and error if the length of the branch vector is != 1

    """

    tree = nx.DiGraph(graph)
    tree.remove_edges_from(tree.edges - nx.bfs_edges(tree, origin))

    compute_start_end_node(tree, origin)

    angle_dict = {}
    start_nodes = nx.get_edge_attributes(tree, 'start_node')
    end_nodes = nx.get_edge_attributes(tree, 'end_node')
    splines = nx.get_edge_attributes(tree, 'edge_spline')

    node_coordinates =nx.get_node_attributes(tree, 'node_coordinate')

    attr_dict = {}
    center_points = []
    midline_points = []

    for u,v,attr in tree.edges(data = True):
        edge = (u,v)
        # if len(tree.out_edges(edge[1])) != 2:
        #     continue
        
        #continue if no parent edge (eg. trachea)
        if not tree.in_edges(start_nodes[edge]):
            # angle_dict[edge] = 0
            attr_dict[edge] = attr

            attr_dict[edge]['edge'] = edge
     
            continue
        parent_edge = list(tree.in_edges(start_nodes[edge]))[0]
        parent_spline = splines[parent_edge]

        parent_ps = []
        #sampling along the whole spline
        i = sample_distance
        while i <1- sample_distance:
            parent_p = parent_spline.sample(i)[0]
            parent_ps.append(parent_p)
            i += sample_distance
        #take the average of the sampled points
        parent_point = np.nanmean(parent_ps, axis = 0)
    
        # parent_start_node_coordinates = node_coordinates[start_nodes[parent_edge]]
        parent_end_node_coordinates = node_coordinates[end_nodes[parent_edge]]

        parent_vector = unit_vector(parent_point - parent_end_node_coordinates)
        midline_vector = -parent_vector

        #same for edge
        start_node_coordinates = node_coordinates[start_nodes[edge]]

        if np.all(parent_end_node_coordinates != start_node_coordinates):
            raise ValueError('Branch point ill defined.')

        spline = splines[edge]
        edge_ps = []
        i = sample_distance
        while i < 1-sample_distance:
            edge_p = spline.sample(i)[0]
            edge_ps.append(edge_p)
            i+= sample_distance

        edge_point = np.nanmean(edge_ps, axis =0)
        
        branch_vector = unit_vector(edge_point - start_node_coordinates)

        if round(np.linalg.norm(midline_vector)) != 1:
            raise ValueError('Midline vector is not normalized. Its length is {}'.format(np.linalg.norm(midline_vector)))
        if round(np.linalg.norm(branch_vector)) != 1:
            raise ValueError('Branch vector is not normalized. Its length is {}'.format(np.linalg.norm(branch_vector)))


        dot = np.dot(midline_vector, branch_vector)
        angle = np.degrees(np.arccos(dot))

        angle_dict[edge] = angle
        attr_dict[edge] = attr
        attr_dict[edge]['angle'] = angle
        attr_dict[edge]['edge'] = edge
        
        #store for viz
        center_points.append(parent_end_node_coordinates)
        midline_points.append(parent_end_node_coordinates + (10 * midline_vector))



    angle_df = pd.DataFrame.from_dict(attr_dict, orient ='index').reset_index(drop=True)
        
    
    return angle_df, center_points, midline_points



def compute_midline_branch_angle_branch_nodes(graph:nx.digraph, origin:int):

    """Computes the midline anlges for each branch in the graph and returns
    a dataframe with the angles and all metadata in the tree.

    To compute the vectors, only the branch nodes are taken in consideration.
    Branches are simplified to a straight line
    Returns
    -------
    pd.Dataframe
        Dataframe containing midline angle measurements and all other edge data

    Raises
    ------
    ValueError
        Raises and error if the end point of the parent branch and the 
        start point of the daughter branch are not the same
    ValueError
        Raises and error if the length of the midline vector is != 1
    ValueError
        Raises and error if the length of the branch vector is != 1

    """
    tree = nx.DiGraph(graph)
    tree.remove_edges_from(tree.edges - nx.bfs_edges(tree, origin))

    compute_start_end_node(tree, origin)

    angle_dict = {}
    start_nodes = nx.get_edge_attributes(tree, 'start_node')
    end_nodes = nx.get_edge_attributes(tree, 'end_node')

    node_coordinates =nx.get_node_attributes(tree, 'node_coordinate')

    attr_dict = {}
    center_points = []
    midline_points = []

    for u,v,attr in tree.edges(data = True):
        edge = (u,v)
        # if len(tree.out_edges(edge[1])) != 2:
        #     continue
        
        if not tree.in_edges(start_nodes[edge]):
            # angle_dict[edge] = 0

            attr_dict[edge] = attr
            attr_dict[edge]['edge'] = edge
            continue

        parent_edge = list(tree.in_edges(start_nodes[edge]))[0]

        parent_start_node_coordinates = node_coordinates[start_nodes[parent_edge]]
        parent_end_node_coordinates = node_coordinates[end_nodes[parent_edge]]

        parent_vector = unit_vector(parent_start_node_coordinates - parent_end_node_coordinates)
        midline_vector = -parent_vector

        start_node_coordinates = node_coordinates[start_nodes[edge]]

        if np.all(parent_end_node_coordinates != start_node_coordinates):
            raise ValueError('Branch point ill defined.')

        end_node_coordinates = node_coordinates[end_nodes[edge]]
        branch_vector = unit_vector(end_node_coordinates - start_node_coordinates)

        if round(np.linalg.norm(midline_vector)) != 1:
            raise ValueError('Midline vector is not normalized. Its length is {}'.format(np.linalg.norm(midline_vector)))
        if round(np.linalg.norm(branch_vector)) != 1:
            raise ValueError('Branch vector is not normalized. Its length is {}'.format(np.linalg.norm(branch_vector)))




        dot = np.dot(midline_vector, branch_vector)
        angle = np.degrees(np.arccos(dot))

        angle_dict[edge] = angle
        attr_dict[edge] = attr
        attr_dict[edge]['angle'] = angle
        attr_dict[edge]['edge'] = edge

        


        #store for viz
        center_points.append(parent_end_node_coordinates)
        midline_points.append(parent_end_node_coordinates + (10 * midline_vector))


    angle_df = pd.DataFrame.from_dict(attr_dict, orient ='index').reset_index(drop=True)
        
    
    return angle_df, center_points, midline_points











def add_parent_angles_to_df(graph:nx.graph,df:pd.DataFrame, origin:int):
    tree = nx.DiGraph(graph)
    tree.remove_edges_from(tree.edges - nx.bfs_edges(tree, origin))

    edge_length = []
    parent_edges = []
    angle_edges = df['edge']
    for edge in angle_edges:
        edge = graph.edges[edge]

        edge_length.append(edge['length'])
        # parent_edges.append(e14_skeleton.graph.edges[edge]['parent'])
        if not tree.in_edges(edge['start_node']):
            parent_edges.append(None)

            continue

        parent_edge = tree.in_edges(edge['start_node'])
        parent_edges.append(list(parent_edge)[0])
    df['length'] = edge_length
    df['parent_edge'] = parent_edges

    parent_angle = []
    parent_nTips = []

    for row in df.iterrows():
        parent_row = df.loc[df['edge'] == row[1]['parent_edge']]

        if len(parent_row) == 0:
            parent_angle.append(None)
            parent_nTips.append(None)
        else:
            parent_angle.append(parent_row['angle'].values[0])
            # parent_nTips.append(parent_row['num_tips'].values[0])
            
    df['parent_angle'] = parent_angle
    # df['parent_nTips'] = parent_nTips
    return df


def compute_branch_orientation(graph:nx.Graph, origin:int) -> pd.DataFrame:
    """Computes in which orienation branch points (Nodes with degree 3) are oriented.
      Compare with trachea orientation to determine if branches are oriented in the same direction.

    Parameters
    ----------
    graph : nx.Graph
        Branching tree formatted as directonal graph with trachea as origin
    origin : int
        Node ID of origin

    Returns
    -------
    pd.DataFrame
        Dataframe with branch orientation and level
    """
    normals_branches = {}
    level_dic = {}

    tree = nx.DiGraph(graph)
    tree.remove_edges_from(tree.edges - nx.bfs_edges(tree, origin))

    # normals_to_plot_non_rot = []
    parent_tangent_dic = {}
    # normals_non_rot = {}
    # rotation = []
    for edge in tree.edges():
        if len(tree.out_edges(edge[1])) != 2:
            continue
        p_branch  = tree.nodes[edge[1]]['node_coordinate']
        parent_spline = tree.edges[edge]['edge_spline']
        p_parent = parent_spline.sample(0.9)[0]
        p_daughters = []
        for daughter_node in tree.successors(edge[1]):
            daughter = (edge[1], daughter_node)
            spline_daughter = tree.edges[daughter]['edge_spline']

            #sample a couple of points and take avarage
            dp = []
            for sample_distance in np.linspace(0.1,0.9, 20):
                dp.append(spline_daughter.sample(sample_distance)[0])
            # p_daughters.append(np.mean(dp, axis = 0))
            p_daughters.append(tree.nodes[daughter_node]['node_coordinate'])
        
        #compute vectors
        v1 = p_daughters[0] - p_parent
        v2 = p_daughters[1] - p_parent
        normal = np.cross(v1,v2)
        normal = unit_vector(normal)
        

        parent_tangent = p_parent - p_branch
        parent_tangent = unit_vector(parent_tangent)
        parent_tangent_dic[edge[1]] = parent_tangent
        # normals_non_rot[edge[1]] = normal

        normals_branches[edge[1]] = normal
        level_dic[edge[1]] = tree.nodes[edge[1]]['level']
    
    trachea_normal =normals_branches[list(tree.out_edges(origin))[0][1]]
    normals_branches_ordered  = ensure_same_normal_direction(normals_branches, np.sign(trachea_normal[0]))
    # normals_non_rot  = ensure_same_normal_direction(normals_non_rot, np.sign(trachea_normal[0]))

    #rotate normals
    # axis_to_rotate_to = np.array([0,1,1])
    # for node, normal in normals_branches_ordered.items():
    #     parent_tangent = parent_tangent_dic[node]
    #     tangent_rm = rotation_matrix_from_vectors(parent_tangent, 
    #                                         unit_vector(parent_tangent *axis_to_rotate_to))
    #     normal_rot = tangent_rm.apply(normal)
    #     normals_branches_ordered[node] = normal_rot

    #if normals need to be plotted
    normals_to_plot = {}
    for node, normal in normals_branches_ordered.items():
        normals_to_plot[node] = np.array([tree.nodes[node]['node_coordinate'], tree.nodes[node]['node_coordinate'] + 70*normal])
    # for node, normal in normals_non_rot.items():
    #     normals_to_plot_non_rot.append(np.array([tree.nodes[node]['node_coordinate'], tree.nodes[node]['node_coordinate'] + 70*normal]))

    #compare angle between trachea and branches
    # angles = []
    # levels = []
    # nodes = []
    # for node, normal in normals_branches_ordered.items():
    #     dot = np.dot(trachea_normal, normal)
    #     # norm = np.linalg.norm(trachea_normal) * np.linalg.norm(normal)
    #     # print(norm)
    #     levels.append(level_dic[node])
    #     angle = np.degrees(np.arccos(dot))
    #     angles.append(angle)
    #     nodes.append(node)


    #compare angle between parent and branch
    angles = []
    nodes = []
    levels = []
    for node, normal in normals_branches_ordered.items():
        parent = list(tree.in_edges(node))[0][0]
        
        if parent == origin:
            continue
        if parent not in list(normals_branches_ordered.keys()):
            continue
        parent_normal = unit_vector(normals_branches_ordered[parent])
        normal = unit_vector(normal)
        dot = np.dot(parent_normal, normal)
        
        angle = np.degrees(np.arccos(dot))
        angles.append(angle)
        # level_dic[node] = tree.nodes[node]['level']
        levels.append(level_dic[node])
        nodes.append(node)

    angle_df = pd.DataFrame({'angle': angles, 'level': levels, 'node': nodes})

    return angle_df,normals_branches_ordered,normals_to_plot

def count_branch_orientation_changes(graph, origin, angle_df, change_cut_off = 50):
    #check how often orientation changes per branch sequence.
    #get branch sequence
    branch_sequences = []
    for node, degree in graph.degree():
        if degree != 1:
            continue
        shortest_path = nx.shortest_path(graph, origin, node)
        if len(shortest_path) > 0:
            branch_sequences.append(shortest_path)


    angle_sequences = []
    path_level = []
    nodes = []
    for branch_sequence in branch_sequences:
        angle_sequence = []
        distance_from_origin = []
        nodes_in_path = []

        for node in branch_sequence:
            if node in list(angle_df['node']):
                # print(node, angles[angles['node'] == node]['angle'].values[0])
                angle_sequence.append(angle_df[angle_df['node'] == node]['angle'].values[0])

                distance_from_origin.append(graph.nodes[node]['level'])
                nodes_in_path.append(int(node))
        angle_sequences.append(angle_sequence)
        path_level.append(distance_from_origin)
        nodes.append(nodes_in_path)

    for i, angle_sequence in enumerate(angle_sequences):
        path_df = pd.DataFrame({'angle': angle_sequence, 'distance_from_origin': path_level[i], 'nodes': nodes[i]})
        #change distance and node to int
        path_df['distance_from_origin'] = path_df['distance_from_origin'].astype(int)
        path_df['nodes'] = path_df['nodes'].astype(int)
        path_df['path'] = int(i)
        if i == 0:
            angle_sequence_df = path_df
        else:
            angle_sequence_df = pd.concat([angle_sequence_df, path_df])

        #count how often the anlge changes by more than 33 degrees
    # angle_sequence_df['angle_change'] = angle_sequence_df.groupby('path')['angle'].diff()
    angle_sequence_df['angle_change'] = angle_sequence_df['angle']
    angle_sequence_df['angle_change'] = angle_sequence_df['angle_change'].abs()
    angle_sequence_df['angle_change_binary'] = (angle_sequence_df['angle_change'] > change_cut_off) & (angle_sequence_df['angle_change'] < 180 - change_cut_off)
    path_count_angle_changes =angle_sequence_df.groupby('path')['angle_change_binary'].sum()
    path_count_angle_changes = path_count_angle_changes.reset_index(name ='num_orientation_changes')

    return path_count_angle_changes, angle_sequence_df



def compute_dimension_of_lobes(graph, lobe):
    #get all edges of lobe
    lobe_edges = [edge for edge in graph.edges(data = True) if edge[2].get('lobe') == lobe]
    lobe_coord = [edge[2]['edge_coordinates'] for edge in lobe_edges]
    #flatten
    lobe_coord_flat = np.array([x for y in lobe_coord for x in y])

    #compute alpha shapes
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lobe_coord_flat)
    alpha = 300
    mesh= o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha = alpha)
    mesh.compute_vertex_normals()

    #sample n points from surface
    n_points = 100000
    points = mesh.sample_points_uniformly(number_of_points = n_points)
    m, cov = points.compute_mean_and_covariance()
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    #get principal axis
    principal_axis = eigenvectors[np.argmax(eigenvalues)]
    #get major and minor axis length
    # Sort eigenvectors based on eigenvalues
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Compute the lengths of the major and minor axes
    major_length = np.sqrt(eigenvalues[sorted_indices[0]])
    minor_length = np.sqrt(eigenvalues[sorted_indices[-1]])

    major_axis = sorted_eigenvectors[:, 0] 
    minor_axis = sorted_eigenvectors[:, -1]

    #compute length along major and minor axis
    vertices = np.array(points.points)
    projected_distances_major = np.dot(vertices,major_axis)
    min_distance = np.min(projected_distances_major)
    max_distance = np.max(projected_distances_major)
    major_axis_length = max_distance - min_distance

    projected_distances_minor = np.dot(vertices,minor_axis)
    min_distance = np.min(projected_distances_minor)
    max_distance = np.max(projected_distances_minor)
    minor_axis_length = max_distance - min_distance

    return major_axis_length, minor_axis_length, mesh


def compute_vertical_distance_to_carina(node_id, graph, origin):
    p1 = graph.nodes[node_id]['node_coordinate']


    #find carina
    carina_id = list(graph.edges(origin))[0][1]
    carina_coordinate = graph.nodes[carina_id]['node_coordinate']

    trachea_vector = graph.nodes[origin]['node_coordinate']-graph.nodes[carina_id]['node_coordinate']
    tracheal_unit_vector = unit_vector(trachea_vector)
    p1_carina  = carina_coordinate - p1
    carina_unit_vector = unit_vector(p1_carina)

    #distance to carina
    d = np.linalg.norm(carina_coordinate - p1)
    # #better use distance with sign
    # d = np.dot(p1_carina, tracheal_unit_vector)


    p1_trachea = p1+tracheal_unit_vector*1000
    p1_trachea = p1_trachea[0]
    #angle
    beta = np.arccos(np.dot(carina_unit_vector, tracheal_unit_vector))

    vertical_distance = d * np.cos(beta)
    return vertical_distance


def commulative_length_to_node(graph,node,origin):
    path_to_node = nx.shortest_path(graph, origin, node)
    edges_in_path = [(path_to_node[i], path_to_node[i+1]) for i in range(len(path_to_node)-1)]
    length = 0
    for edge in edges_in_path:
        length += graph.edges[edge]['length']

    return length