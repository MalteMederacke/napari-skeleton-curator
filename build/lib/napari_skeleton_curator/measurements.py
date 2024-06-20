
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
from .utils import (unit_vector, 
                   ensure_same_normal_direction, 
                   count_number_of_tips_connected_to_edge)
import open3d as o3d

def compute_midline_branch_angles(graph:nx.Graph, 
                                  origin:int, 
                                  sample_distance:float = 0.01):
    
    """Compute the angles between the midline and the daughter branches.
    Commonly called theta.

    Parameters
    ----------
    graph : nx.Graph
        Branching tree with branch coordinates stored in edge_spline,
        and node coordinates stored in node_coordinate
        generation of the edge stored in level
    origin : int
        inlet of the graph, usually the trachea
    sample_distance : float, optional
        Position on where to sample the spline to compute the midline.
          Relative, (1 is on the opposite end of the branch, 
          0 is at the branch point),
            by default 0.01

    Returns
    -------
    pd.DataFrame
        pd.DataFrame with the angles between the midline and the
          daughter branches and associated information
    list
        List of midline points for visualization
    list
        List of center points for visualization
    list
        List of parent points for visualization
    list
        List of daughter points for visualization
    list
        List of daughter edges for visualization


    Raises
    ------
    ValueError
        If the midline vector is not normalized
    ValueError
        If the daughter vector is not normalized
    """
    
    #compute directed graph
    tree = nx.DiGraph(graph)
    tree.remove_edges_from(tree.edges - nx.bfs_edges(tree, origin))


    mps = []
    pp = []
    dps = []
    center_points = []
    levels = []
    angles = []
    daughter_edge =[]
    numTips = []
    angle_dict = {}

    #angle only needed for branch with two daughters
    for edge in tree.edges:
        if len(tree.out_edges(edge[1])) != 2:
            continue
        #get the parent spline
        spline = tree.edges[edge]['edge_spline']
        #sample at a couple of different distances on the parent edge 
        # to account for local curvature
        parent_ps = []
        for i in [1,5,10]:
            parent_p = spline.sample(1-sample_distance*i)[0]
            parent_ps.append(parent_p)
        #take the average of the sampled points
        parent_p = np.mean(parent_ps, axis = 0)

        #center point of the three branches of interest
        center_point = tree.nodes[edge[1]]['node_coordinate']

        #get the midline vector
        mp_vector = - unit_vector(parent_p - center_point)

        #store midline points and center points for visualization
        if tree.edges[edge]['level'] <10:
            mps.append(center_point +(10*mp_vector))
            center_points.append(center_point)
            pp.append(parent_p)

        #measure the angle between the midline and the daughter branches
        for daughter_node in tree.successors(edge[1]):
            daughter = (edge[1], daughter_node)
            level  = tree.edges[daughter]['level']
            daughter_spline = tree.edges[daughter]['edge_spline']
            #sample three points on the daughter branch to account for local curvature
            daughter_ps = []
            for i in [1,5,10]:
                daughter_point = daughter_spline.sample(sample_distance*i)[0]
                daughter_ps.append(daughter_point)
            daughter_point = np.mean(daughter_ps, axis = 0)
            #get the vector of the daughter branch
            daughter_vector = daughter_point - center_point
            daughter_vector = unit_vector(daughter_vector)
            #store daughter points for visualization
            dps.append(center_point+(10*daughter_vector))

            #used unit vectors or normalize
            # throw error if length of vector != 1
            if np.linalg.norm(mp_vector) != 1:
                raise ValueError('Midline vector is not normalized')
            if np.linalg.norm(daughter_vector) != 1:
                raise ValueError('Daughter vector is not normalized')
            
            #compute the angle between the midline and the daughter branch
            dot = np.dot(mp_vector, daughter_vector)
            angle = np.degrees(np.arccos(dot))

            #store the angle, level and edge
            daughter_edge.append(daughter_spline.points)
            levels.append(level)
            angles.append(angle)
            numTips.append(count_number_of_tips_connected_to_edge(tree, edge[1], daughter_node))

            angle_dict[daughter] = angle

    #create a dataframe with the angles and associated information
    angle_df = pd.DataFrame({'level': levels, 
                             'angle': angles, 
                             'edge': list(angle_dict.keys()), 
                             'num_tips': numTips})
    
    return angle_df, mps, center_points, pp,dps,daughter_edge
             
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
