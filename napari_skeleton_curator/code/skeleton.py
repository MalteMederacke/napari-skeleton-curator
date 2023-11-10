from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings

from magicgui import magicgui
from morphosamplers.spline import Spline3D
from morphosamplers.sampler import sample_volume_at_coordinates, generate_2d_grid, place_sampling_grids, sample_subvolumes
import napari
from napari.layers import Shapes, Points, Image
from napari.types import LayerDataTuple
import networkx as nx
import numpy as np
import pandas as pd
from psygnal.containers import EventedSet
from scipy.spatial.transform import Rotation
from skimage.io import imsave

import pyqtgraph as pg
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QComboBox, QVBoxLayout, QWidget,QPushButton
from skimage.transform import rotate
from superqt.sliders import QLabeledSlider



NODE_COORDINATE_KEY = "node_coordinate"
EDGE_COORDINATES_KEY = "edge_coordinates"
EDGE_SPLINE_KEY = "edge_spline"
NODE_INDEX_KEY = "node_index"

EDGE_FEATURES_START_NODE_KEY = "start_node"
EDGE_FEATURES_END_NODE_KEY = "end_node"
EDGE_FEATURES_HIGHLIGHT_KEY = "highlight"



def update_edge_merged_to_first_node(skeleton, new_edge, new_node_index):
    """Clean up an edge that was merged to the start node.
    
    This updates the edge coordinates and spline. It adds a line
    connecting the new node to the merged node in the edge.
    
    Parameters
    ----------
    skeleton : nx.Graph
        The skeleton containing the merge edge.
    new_edge : Tuple[int, int]
        The edge key for the newly created edge that must be updated.
    new_node_index : int
        The index of the new node that was merged into the edge.
    
    Returns
    -------
    skeleton : nx.Graph
        The updated graph
    """
    print(new_edge, new_node_index)
    # get the edge attributes
    edge_attributes = skeleton.edges[new_edge]
    
    # update the edge coordinates
    original_edge_coordinates = edge_attributes["edge_coordinates"]
    new_node_coordinate = skeleton.nodes(data=True)[new_node_index][NODE_COORDINATE_KEY]
    
    if np.linalg.norm(new_node_coordinate - original_edge_coordinates[-1]) > np.linalg.norm(new_node_coordinate - original_edge_coordinates[0]):
        print("flip oc")
        original_edge_coordinates = np.flip(original_edge_coordinates, axis = 0)
    # update the spline
    n_points = int(np.linalg.norm(new_node_coordinate - original_edge_coordinates[0]))
    interpolated_coordinates = np.linspace(
        new_node_coordinate,
        original_edge_coordinates[0],
        n_points,
        endpoint=False
    )
    points1 =interpolated_coordinates
    points2 = original_edge_coordinates
    start_node = new_node_coordinate
    old_node = [x for x in new_edge if x != new_node_index][0]
    middle_node = skeleton.nodes(data=True)[old_node][NODE_COORDINATE_KEY]
    #start and end node coordinates


    if np.allclose(points1[0], start_node) & np.allclose(points2[0], middle_node):
        print('no flip')
        spline_points = np.vstack((points1, points2))
        
    elif np.allclose(points1[-1], start_node) & np.allclose(points2[0], middle_node):
        print('flip 1')
        spline_points = np.vstack((np.flip(points1, axis = 0), points2))
    elif np.allclose(points1[0], start_node) & np.allclose(points2[-1], middle_node):
        print('flip 2')
        spline_points = np.vstack((points1, np.flip(points2, axis = 0)))
    elif np.allclose(points1[-1], start_node) & np.allclose(points2[-1], middle_node):
        print('flip both')
        spline_points = np.vstack((np.flip(points1, axis = 0), np.flip(points2, axis = 0)))    
    else:
        warnings.warn('Warning: Edge splines not connected. Consider recomputing.')
        spline_points = np.vstack((points1, points2))

    _, idx = np.unique(spline_points, axis=0, return_index=True)
    spline_points = spline_points[np.sort(idx)]

    new_edge_coordinates = spline_points
    
    # new_edge_coordinates = np.concatenate(
    #     [interpolated_coordinates, original_edge_coordinates]
    # )


    new_spline = Spline3D(points=new_edge_coordinates)
    
    # add the edge attributes back
    new_edge_attributes = {
        "edge_coordinates": new_edge_coordinates,
        "edge_spline": new_spline
    }
    nx.set_edge_attributes(skeleton, {new_edge: new_edge_attributes})
    
    return skeleton


def update_edge_merged_to_last_node(skeleton, new_edge, new_node_index):
    """Clean up an edge that was merged to the end node.
    
    This updates the edge coordinates and spline. It adds a line
    connecting the new node to the merged node in the edge.
    
    Parameters
    ----------
    skeleton : nx.Graph
        The skeleton containing the merge edge.
    new_edge : Tuple[int, int]
        The edge key for the newly created edge that must be updated.
    new_node_index : int
        The index of the new node that was merged into the edge.
    
    Returns
    -------
    skeleton : nx.Graph
        The updated graph
    """

    # get the edge attributes
    edge_attributes = skeleton.edges[new_edge]
    
    # update the edge coordinates
    original_edge_coordinates = edge_attributes["edge_coordinates"]
    new_node_coordinate = skeleton.nodes(data=True)[new_node_index][NODE_COORDINATE_KEY]
    
    if np.linalg.norm(new_node_coordinate - original_edge_coordinates[-1]) > np.linalg.norm(new_node_coordinate - original_edge_coordinates[0]):
        print("flip oc")
        original_edge_coordinates = np.flip(original_edge_coordinates, axis = 0)
    # update the spline
    n_points = int(np.linalg.norm(new_node_coordinate - original_edge_coordinates[-1]))
    interpolated_coordinates = np.linspace(
        original_edge_coordinates[-1],
        new_node_coordinate,
        n_points,
    )
    
    points1 =interpolated_coordinates
    points2 = original_edge_coordinates
    start_node = new_node_coordinate
    old_node = [x for x in new_edge if x != new_node_index][0]
    middle_node = skeleton.nodes(data=True)[old_node][NODE_COORDINATE_KEY]
    #start and end node coordinates


    if np.allclose(points1[0], start_node) & np.allclose(points2[0], middle_node):
        print('no flip')
        spline_points = np.vstack((points1, points2))
        
    elif np.allclose(points1[-1], start_node) & np.allclose(points2[0], middle_node):
        print('flip 1')
        spline_points = np.vstack((np.flip(points1, axis = 0), points2))
    elif np.allclose(points1[0], start_node) & np.allclose(points2[-1], middle_node):
        print('flip 2')
        spline_points = np.vstack((points1, np.flip(points2, axis = 0)))
    elif np.allclose(points1[-1], start_node) & np.allclose(points2[-1], middle_node):
        print('flip both')
        spline_points = np.vstack((np.flip(points1, axis = 0), np.flip(points2, axis = 0)))    
    else:
        warnings.warn('Warning: Edge splines not connected. Consider recomputing.')
        spline_points = np.vstack((points1, points2))

    _, idx = np.unique(spline_points, axis=0, return_index=True)
    spline_points = spline_points[np.sort(idx)]

    new_edge_coordinates = spline_points
    # new_edge_coordinates = np.concatenate(
    #     [original_edge_coordinates, interpolated_coordinates[1::]]
    # )
    new_spline = Spline3D(points=new_edge_coordinates)
    # add the edge attributes back
    new_edge_attributes = {
        "edge_coordinates": new_edge_coordinates,
        "edge_spline": new_spline
    }
    nx.set_edge_attributes(skeleton, {new_edge: new_edge_attributes})
    
    print(skeleton, new_edge_attributes)
    return skeleton

def add_or_update_shapes_layer(
        viewer: napari.Viewer,
        layer_data_tuple: LayerDataTuple,
        edge_width: int = 3
):
    """Add a shapes layer to the viewer. If there
    is already a layer of the same name, update it instead.
    """

    layer_kwargs = layer_data_tuple[1]
    layer_name = layer_kwargs["name"]

    if (layer_name in viewer.layers) and isinstance(viewer.layers[layer_name], Shapes):
        # update because the layer is already there
        layer = viewer.layers[layer_name]
        layer.selected_data = set(range(layer.nshapes))
        layer.remove_selected()
        shape_type = layer_kwargs.pop("shape_type", "path")
        layer.add(
            data=layer_data_tuple[0],
            shape_type=shape_type
        )

        for parameter, value in layer_kwargs.items():
            # set the new kwargs
            setattr(layer, parameter, value)
    
    else:
        layer = viewer.add_shapes(layer_data_tuple[0], edge_width = edge_width, **layer_kwargs)

    return layer


def add_or_update_points_layer(
    viewer: napari.Viewer,
    layer_data_tuple: LayerDataTuple
) -> Points:
    layer_kwargs = layer_data_tuple[1]
    layer_name = layer_kwargs["name"]

    if (layer_name in viewer.layers) and isinstance(viewer.layers[layer_name], Points):
        layer = viewer.layers[layer_name]
        layer.data = layer_data_tuple[0]
        for parameter, value in layer_kwargs.items():
            # set the new kwargs
            setattr(layer, parameter, value)
    else:
        layer = viewer.add_points(
            layer_data_tuple[0],
            **layer_kwargs
        )


def add_or_update_image_layer(
    viewer: napari.Viewer,
    layer_data_tuple: LayerDataTuple
) -> Points:
    layer_kwargs = layer_data_tuple[1]
    layer_name = layer_kwargs["name"]

    if (layer_name in viewer.layers) and isinstance(viewer.layers[layer_name], Image):
        layer = viewer.layers[layer_name]
        layer.data = layer_data_tuple[0]
        for parameter, value in layer_kwargs.items():
            # set the new kwargs
            setattr(layer, parameter, value)
    else:
        layer = viewer.add_image(
            layer_data_tuple[0],
            **layer_kwargs
        )


def points_in_bounding_box(
    point_coordinates: np.ndarray,
    min_corner: np.ndarray,
    max_corner: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Find points contained within a bounding box."""
    above_min = np.all(point_coordinates > min_corner, axis=1)
    below_max = np.all(point_coordinates < max_corner, axis=1)
    within_bounding_box = np.logical_and(above_min, below_max)
    
    print(within_bounding_box.sum())

    return point_coordinates[within_bounding_box], within_bounding_box


def graph_nodes_to_napari(
    node_data,
    layer_name: str="nodes"
) -> LayerDataTuple:
    node_attributes = []
    node_coordinates = []
    for node_index, attributes in node_data.items():
        # get the node data
        node_coordinates.append(attributes.get(NODE_COORDINATE_KEY))
        attributes_for_features = {
            k: v for k, v in attributes.items()
            if k != NODE_COORDINATE_KEY
        }
        attributes_for_features.update({NODE_INDEX_KEY: node_index})
        node_attributes.append(attributes_for_features)

    # make the layer data
    node_features = pd.DataFrame(node_attributes)
    node_coordinates = np.stack(node_coordinates)
    points_kwargs = {"name": layer_name, "features": node_features}
    return (node_coordinates, points_kwargs, "Points")


def graph_edges_to_napari(
    edge_data,
    n_spline_points: int = 5,
    layer_name: str="edges"
) -> LayerDataTuple:
    """Create the Shapes LayerDataTuple for the graph edges"""
    edge_attributes = []
    spline_points = []
    for start_index, end_index, attributes in edge_data:
        spline = attributes.get(EDGE_SPLINE_KEY)

        attributes_for_table = {
            k: v for k, v in attributes.items()
            if k not in [EDGE_SPLINE_KEY, EDGE_COORDINATES_KEY]
        }

        # add the start and end indices
        attributes_for_table.update(
            {
                EDGE_FEATURES_START_NODE_KEY: start_index,
                EDGE_FEATURES_END_NODE_KEY: end_index,
            }
        )
        edge_attributes.append(attributes_for_table)

        # get the spline_points
        spline_coordinates = np.linspace(0, 1, n_spline_points, endpoint=True)
        spline_points.append(
            spline.sample(
                u=spline_coordinates,
                derivative_order=0
            )
        )

    spline_features = pd.DataFrame(edge_attributes)
    spline_features[EDGE_FEATURES_HIGHLIGHT_KEY] = False

    shapes_kwargs = {
        "shape_type": "path",
        "features": spline_features,
        "name": layer_name
    }

    return (spline_points, shapes_kwargs, "shapes")
def merge_nodes(skeleton: nx.Graph, node_to_keep: int, node_to_merge: int, copy: bool=True):
    """Merge two nodes in a graph.
    
    Parameters
    ----------
    skeleton : nx.Graph
        The skeleton in which to perform the merging.
    node_to_keep : int
        The index of the merged nodes to keep in the graph.
    node_to_merge : int
        The index of the merged nodes that should be removed.
    copy: bool
        If set to True, returns a copy of the graph object. Otherwise,
        performs modifications in place. Defaults to False.
    
    Returns
    -------
    skeleton : nx.Graph
        The skeleton with the merged edges.
    """
    
    
    # get the edges that will be merged
    merged_edges = list(skeleton.edges(node_to_merge))
    
    # get the new edges
    merge_to_first_node = []
    edges_to_fix = []
    for edge in merged_edges:
        edge_attributes = skeleton.edges[edge]
        edge_original_start_node = edge_attributes[EDGE_FEATURES_START_NODE_KEY]
        if edge_original_start_node == node_to_merge:
            # determine if the node being merged is the first or last node in the edge coordinates
            merge_to_first_node.append(True)
        else:
            merge_to_first_node.append(False)
        
        # get the new edge that will be created
        new_edge = tuple(node for node in edge if node != node_to_merge) + (node_to_keep,)
        edges_to_fix.append(new_edge)
        
    # print(edges_to_fix, skeleton)
    # perform the merge
    graph_before = skeleton.copy()
    skeleton = nx.contracted_nodes(skeleton, u=node_to_keep, v=node_to_merge, copy=copy)

    
    # fix the edges
    for first_node, edge in zip(merge_to_first_node, edges_to_fix):
        if first_node:
            skeleton = update_edge_merged_to_first_node(skeleton, new_edge=edge, new_node_index=node_to_keep)
        else:

            skeleton = update_edge_merged_to_last_node(skeleton, new_edge=edge, new_node_index=node_to_keep)

        # #detect all changes
        # changed_edges = set(graph_before.edges) - set(skeleton.edges)
        # print(changed_edges)
        # for edge in changed_edges:
        #     for node in edge:
        #         if len(skeleton.edges(node)) == 0:
        #             skeleton.remove_node(node)
        #         #merge edges if node has degree 2
        #         elif len(skeleton.edges(node)) == 2:
        #             #merge
        #             print('merge')
        #             #order of edges to merge is important. Merge incoming edge with outgoing edge
        #             if list(skeleton.edges(node, data = 'start_node'))[0][2]:
        #                 new_edge = [None] * 2
        #                 for u,v,attr in graph.edges(node, data = True):
        #                     if (u,v) not in edge:
        #                         if attr['start_node'] == node:
        #                             new_edge[1] = attr['end_node']
        #                         if attr['end_node'] == node:
        #                             new_edge[0] = attr['start_node']
        #                 Skeleton3D.merge_edge(new_edge[0], node, new_edge[1])

        #             #if no direction, merge however... might lead to funny splines
        #             else:
        #                 new_edge = [x for y in list(graph.edges(node)) for x in y if x not in edge]
        #                 Skeleton3D.merge_edge(new_edge[0], node, new_edge[1])

        # #check if graph is still connected, if not remove orphaned nodes
        # skeleton.remove_nodes_from(list(nx.isolates(skeleton)))

    return skeleton

def count_number_of_tips_connected_to_edge(graph, start_node, end_node):
    #add start and end node to graph based on breadth first search/distance to origin
    graph_copy = graph.copy()
    graph_copy.remove_edge(start_node, end_node)
    subtree = nx.bfs_tree(graph_copy, end_node)

    #count number of endpoints
    end_points = []

class Skeleton3D:
    def __init__(self, graph: nx.Graph):
        self.graph = graph

    def nodes(self, data: bool = True):
        """Passthrough for nx.Graph.nodes"""
        return self.graph.nodes(data=data)

    def edges(self, data: bool = True):
        """Passthrough for nx.Graph.edges"""
        return self.graph.edges(data=data)

    @property
    def node_coordinates(self) -> np.ndarray:
        """Coordinates of the nodes.

        Index matched to nx.Graph.nodes()
        """
        node_data = self.nodes(data=True)
        coordinates = [
            data[NODE_COORDINATE_KEY] for _, data in node_data
        ]
        return np.stack(coordinates)

    def sample_points_on_edge(
            self,
            start_node: int,
            end_node: int,
            u: List[float],
            derivative_order: int = 0
    ):
        spline = self.graph[start_node][end_node][EDGE_SPLINE_KEY]
        return spline.sample(
            u=u,
            derivative_order=derivative_order
        )

    def sample_slices_on_edge(
            self,
            image: np.ndarray,
            image_voxel_size: Tuple[float, float, float],
            start_node: int,
            end_node: int,
            slice_pixel_size: float,
            slice_width: int,
            slice_spacing: float,
            interpolation_order: int = 1
    ) -> np.ndarray:
        # get the spline object
        spline = self.graph[start_node][end_node][EDGE_SPLINE_KEY]

        # get the positions along the spline
        positions = spline.sample(separation=slice_spacing)
        orientations = spline.sample_orientations(separation=slice_spacing)

        # get the sampling coordinates
        sampling_shape = (slice_width, slice_width)
        grid = generate_2d_grid(grid_shape=sampling_shape, grid_spacing=(slice_pixel_size, slice_pixel_size))
        sampling_coords = place_sampling_grids(grid, positions, orientations)

        # convert the sampling coordinates into the image indices
        sampling_coords = sampling_coords / np.array(image_voxel_size)

        return sample_volume_at_coordinates(
            image, sampling_coords, interpolation_order=interpolation_order
        )

    def sample_image_around_node(
            self,
            node_index: int,
            image: np.ndarray,
            image_voxel_size: Tuple[float, float, float],
            bounding_box_shape: Union[float, Tuple[float, float, float]]=10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract an axis-aligned bounding box from an image around a node.

        Parameters
        ----------
        node_index : int
            The index of the node to sample around.
        image : np.ndarray
            The image to sample from.
        image_voxel_size : Tuple[float, float, float]
            Size of the image voxel in each axis. Should convert to the same
            scale as the skeleton graph.
        bounding_box_shape : Union[float, Tuple[float, float, float]]
            The shape of the bounding box to extract. Size should be specified
            in the coordinate system of the skeleton. If a single float is provided,
            a cube with edge-length bounding_box_shape will be extracted. Otherwise,
            provide a tuple with one element for each axis.

        Returns
        -------
        sub_volume : np.ndarray
            The extracted bounding box.
        bounding_box : np.ndarray
            (2, 3) array with the coordinates of the
            upper left and lower right hand corners of the bounding box.
        """

        # get the node coordinates
        node_coordinate = self.graph.nodes(data=NODE_COORDINATE_KEY)[node_index]

        # convert node coordinate to
        graph_to_image_factor = 1 / np.array(image_voxel_size)
        node_coordinate_image = node_coordinate * graph_to_image_factor

        # convert the bounding box to image coordinates
        if isinstance(bounding_box_shape, int) or isinstance(bounding_box_shape, float):
            bounding_box_shape = (bounding_box_shape, bounding_box_shape, bounding_box_shape)
        grid_shape = np.asarray(bounding_box_shape) * graph_to_image_factor
        bounding_box_min = np.clip(node_coordinate_image - (grid_shape / 2), a_min=[0, 0, 0], a_max=image.shape)
        bounding_box_max = np.clip(node_coordinate_image + (grid_shape / 2), a_min=[0, 0, 0], a_max=image.shape)
        bounding_box = np.stack([bounding_box_min, bounding_box_max]).astype(int)

        # sample the image
        sub_volume = image[
            bounding_box[0, 0]:bounding_box[1, 0],
            bounding_box[0, 1]:bounding_box[1, 1],
            bounding_box[0, 2]:bounding_box[1, 2],
        ]

        return np.asarray(sub_volume), bounding_box

    def shortest_path(
            self,
            start_node: int,
            end_node: int
    ) -> Optional[List[int]]:
        return nx.shortest_path(self.graph, source=start_node, target=end_node)

    @classmethod
    def parse(
            cls,
            graph: nx.Graph,
            edge_attributes: Optional[Dict[str, Any]]=None,
            node_attributes: Optional[Dict[str, Any]]=None,
            edge_coordinates_key: str = EDGE_COORDINATES_KEY,
            node_coordinate_key: str = NODE_COORDINATE_KEY,
            scale: Tuple[float, float, float] = (1, 1, 1)
    ):
        # make a copy of the graph so we don't clobber the original attributes
        graph = deepcopy(graph)

        scale = np.asarray(scale)
        if edge_attributes is None:
            edge_attributes = {}
        if node_attributes is None:
            node_attributes = {}

        # keys for the edge attributes holding the start and
        # end node for the edge coordinates
        node_index_keys = (
            EDGE_FEATURES_START_NODE_KEY,
            EDGE_FEATURES_END_NODE_KEY
        )

        # parse the edge attributes
        parsed_edge_attributes = {}
        for start_index, end_index, attributes in graph.edges(data=True):
            # remove attribute not specified
            keys_to_delete = [
                key for key in attributes if (
                        (key not in edge_attributes) and (key != edge_coordinates_key)
                        and (key not in node_index_keys)
                )
            ]
            for key in keys_to_delete:
                del attributes[key]

            for expected_key, default_value in edge_attributes.items():
                # add expected keys that are missing
                if expected_key not in attributes:
                    attributes.update({expected_key: default_value})

            # make the edge spline
            # coordinates = np.asarray(attributes[edge_coordinates_key]) * scale
            # spline = Spline3D(points=coordinates)

            #if we have directionality in the graph, we should construct the spline
            #from start node to end node.
            if (EDGE_FEATURES_START_NODE_KEY in attributes):
                if np.allclose(dict(graph.nodes(data =node_coordinate_key))[attributes[EDGE_FEATURES_START_NODE_KEY]],
                               np.asarray(attributes[edge_coordinates_key])[-1]):
                    coordinates = np.asarray(attributes[edge_coordinates_key]) * scale
                    coordinate = np.flip(coordinates, axis=0)
                    if len(np.unique(coordinates, axis=0)) != len(coordinates):
                        print('Warning: Duplicate coordinates in edge... Discarding')

                        _, idx = np.unique(coordinates, axis=0, return_index=True)
                        coordinates = coordinates[np.sort(idx)]
                    # print(coordinate.shape, coordinate.dtype,'flip')
                    # spline = Spline3D(points=coordinate)
                else :
                    coordinates = np.asarray(attributes[edge_coordinates_key]) * scale
                    # print(coordinate.shape, coordinate.dtype,'no flip')
                    if len(np.unique(coordinates, axis=0)) != len(coordinates):
                        print('Warning: Duplicate coordinates in edge... Discarding')

                        _, idx = np.unique(coordinates, axis=0, return_index=True)
                        coordinates = coordinates[np.sort(idx)]

                    # spline = Spline3D(points=coordinates)
            else :
                coordinates = np.asarray(attributes[edge_coordinates_key]) * scale
                if len(np.unique(coordinates, axis=0)) != len(coordinates):
                    print('Warning: Duplicate coordinates in edge {}{}... Discarding').format(start_index,end_index)
                    # print(coordinates)

                    _, idx = np.unique(coordinates, axis=0, return_index=True)
                    coordinates = coordinates[np.sort(idx)]
                # print(coordinate.shape, coordinate.dtype,'no dir')

            spline = Spline3D(points=coordinates)


            parsed_edge_attributes.update(
                {
                    (start_index, end_index): {
                        EDGE_COORDINATES_KEY: coordinates,
                        EDGE_SPLINE_KEY: spline,
                    }
                }
            )

            if (EDGE_FEATURES_START_NODE_KEY not in attributes) or \
                (EDGE_FEATURES_END_NODE_KEY not in attributes):
               warnings.warn(f"edge start/end node not provided.\
                             inferring from edge key: {(start_index, end_index)}")
               parsed_edge_attributes[(start_index, end_index)].update(
                {
                    EDGE_FEATURES_START_NODE_KEY: start_index,
                    EDGE_FEATURES_END_NODE_KEY: end_index,
                }
            ) 

        nx.set_edge_attributes(graph, parsed_edge_attributes)

        # parse the node attributes
        parsed_node_attributes = {}
        for node_index, attributes in graph.nodes(data=True):
            # remove attribute not specified
            keys_to_delete = [
                key for key in attributes if (
                        (key not in node_attributes) and (key != node_coordinate_key)
                )
            ]
            for key in keys_to_delete:
                del attributes[key]

            for expected_key, default_value in node_attributes.items():
                # add expected keys that are missing
                if expected_key not in attributes:
                    attributes.update({expected_key: default_value})
            # add the node coordinates
            coordinate = np.asarray(attributes[node_coordinate_key])
            coordinate = coordinate * scale

            parsed_node_attributes.update(
                {
                    node_index: {
                        NODE_COORDINATE_KEY: coordinate
                    }
                }
            )
        nx.set_node_attributes(graph, parsed_node_attributes)

        return cls(graph=graph)

    def merge_edge(self, n1:int, v1:int, n2:int):
        """merge edges in graph and add edge attributes. n1 is merged with n2. v1 is removed.
        MALTE"""
        graph = self.graph.copy()

        start_node = graph.nodes(data=True)[n1][NODE_COORDINATE_KEY]
        end_node = graph.nodes(data=True)[n2][NODE_COORDINATE_KEY]
        middle_node = graph.nodes(data=True)[v1][NODE_COORDINATE_KEY]


        edge_attributes1 = graph.get_edge_data(n1,v1)
        edge_attributes2 = graph.get_edge_data(v1,n2)
        graph.remove_edge(n1,v1)
        graph.remove_edge(v1,n2)
        graph.remove_node(v1)
        merge_edge = (n1,n2)
        merge_attributes = {}
        for key in edge_attributes1:
            if key == 'validated':
                if edge_attributes1[key] and edge_attributes2[key] == True:
                    merge_attributes[key] = True
                else:
                    merge_attributes[key] = False
            # if key == 'edge_coordinates':

            #     merge_attributes[key] = np.vstack((edge_attributes1[key],edge_attributes2[key]))
            if key == 'edge_spline':
                # points1 = edge_attributes1['edge_spline'].points
                # points2 = edge_attributes2['edge_spline'].points
                points1 = edge_attributes1['edge_coordinates']
                points2 = edge_attributes2['edge_coordinates']
                #start and end node coordinates


                if np.allclose(points1[0], start_node) & np.allclose(points2[0], middle_node):
                    print('no flip')
                    spline_points = np.vstack((points1, points2))
                    
                elif np.allclose(points1[-1], start_node) & np.allclose(points2[0], middle_node):
                    print('flip 1')
                    spline_points = np.vstack((np.flip(points1, axis = 0), points2))
                elif np.allclose(points1[0], start_node) & np.allclose(points2[-1], middle_node):
                    print('flip 2')
                    spline_points = np.vstack((points1, np.flip(points2, axis = 0)))
                elif np.allclose(points1[-1], start_node) & np.allclose(points2[-1], middle_node):
                    print('flip both')
                    spline_points = np.vstack((np.flip(points1, axis = 0), np.flip(points2, axis = 0)))    
                else:
                    warnings.warn('Warning: Edge splines not connected. Consider recomputing.')
                    spline_points = np.vstack((points1, points2))
                #sanity check
                if np.allclose(spline_points[-1], end_node):
                    print('sanity check passed')    

                    
                _, idx = np.unique(spline_points, axis=0, return_index=True)
                spline_points = spline_points[np.sort(idx)]
                spline = Spline3D(points = spline_points)
                merge_attributes[key] = spline
                merge_attributes['edge_coordinates'] = spline_points
            if key == 'start_node':
                merge_attributes[key] = n1
            if key == 'end_node':
                merge_attributes[key] = n2
            if key == 'level':
                merge_attributes[key] = edge_attributes1[key]

            if key not in  ['validated', 
                              'edge_coordinates', 
                              'edge_spline', 
                              'start_node', 
                              'end_node', 
                              'level']:
                warnings.warn('Warning: Attribute {} not merged. Consider recomputing.'.format(key))
                
        graph.add_edge(*merge_edge, **merge_attributes)
        self.graph = graph
    
    def merge_nodes(self, node_to_keep: int, node_to_merge: int) -> None:
        self.graph = merge_nodes(
            skeleton=self.graph,
            node_to_keep=node_to_keep,
            node_to_merge=node_to_merge,
            copy=True
        )



class SkeletonEdgeIterator:
    """This iterator returns the edges at subsequent distances
    away from the starting node with each iteration.
    """
    def __init__(
        self,
        skeleton: Skeleton3D,
        starting_node: int,
        max_distance: Optional[int]=None
    ):
        self.starting_node = starting_node
        self.max_distance = max_distance
        self.skeleton = skeleton
        
        # state variable to store the edges that were
        # traced in the previous iteration
        self.previous_edges: List[Tuple[int, int]] = []
            
        # state variable to store the nodes to start from in
        # the next iteration
        self.nodes_to_trace_from: List[int] = []
            
        # state variable to store the current level being iterated
        self.current_level_index: int = 0
        
    def __iter__(self):
        """Initialize and return the iterator"""
        # initialize the state variables
        self.previous_edges = []
        self.nodes_to_trace_from = [self.starting_node]
        self.current_level_index = 0
        
        return self
    
    def __next__(self) -> List[Tuple[int, int]]:
        """Perform the next iteration.
        
        Returns
        -------
        edge_list : List[Tuple[int, int]]
            The list of edges at the next level.
        """
        if (self.current_level_index == self.max_distance) or (len(self.nodes_to_trace_from) == 0):
            # stop iterating if the max number of levels
            # has been reached or there are no more nodes to trace
            # (i.e., end of the skeleton reached.
            raise StopIteration

        # since we have an undirected graph, our edge list must be symmetric
        reversed_edges = [(end_node, start_node) for start_node, end_node in self.previous_edges]
        full_previous_edge_list = self.previous_edges + reversed_edges
        
        # iterate through the nodes to trace from
        edge_list = []
        for start_node in self.nodes_to_trace_from:
            # get the edges going to the node
            edges_from_node = [
                edge for edge in self.skeleton.graph.edges(start_node)
                if (edge not in full_previous_edge_list)
            ]
            
            edge_list += edges_from_node
        
        
        if len(edge_list) == 0:
            # if no new edges were found, stop iterating
            raise StopIteration
        
        # update the state variables
        self.previous_edges = edge_list
        self._update_nodes_to_trace_from(edge_list)
        self.current_level_index += 1
        
        return edge_list
    
    def _update_nodes_to_trace_from(self, new_edges: List[Tuple[int, int]]):
        """Update the nodes to trace from for the next iteration.
        
        This takes all nodes in the edge list and then removes the ones
        that we traced from in the current interation.
        """
        new_nodes_set = set()
        for edge in new_edges:
            new_nodes_set = new_nodes_set.union(set(edge))
        
        # set the new nodes that are not in the old nodes to trace from
        previous_nodes_to_trace_from = set(self.nodes_to_trace_from)
        self.nodes_to_trace_from = new_nodes_set.difference(previous_nodes_to_trace_from)
        

class SkeletonBoundingBoxView:
    def __init__(
        self,
        skeleton: Skeleton3D,
        viewer: napari.Viewer,
        min_corner: np.ndarray,
        max_corner: np.ndarray,
        name: str = "view"
    ):
        self._skeleton = skeleton
        self._viewer = viewer
        self._min_corner = min_corner
        self._max_corner = max_corner
        self._name = name

        # update the view
        self._sub_graph = self.get_sub_graph()
        self.update_napari_view()

    @property
    def name(self) -> str:
        return self._name

    @property
    def min_corner(self) -> np.ndarray:
        return self._min_corner

    @property
    def max_corner(self) -> np.ndarray:
        return self._max_corner
    
    def get_sub_graph(self):
        # determine which nodes are in the bounding box
        coordinates = self._skeleton.node_coordinates
        _, within_bounding_box = points_in_bounding_box(
            point_coordinates=coordinates,
            min_corner=self.min_corner,
            max_corner=self.max_corner
        )
        
        # get the node indices in the bounding box
        print(self.min_corner, self.max_corner)
        if within_bounding_box.sum() == 0:
            raise ValueError("No nodes in the specified bounding box.")
        in_bounding_box_indices = np.squeeze(np.argwhere(within_bounding_box))
        print(in_bounding_box_indices.shape)

        node_list = list(self._skeleton.graph.nodes())
        nodes_in_box = [node_list[index] for index in in_bounding_box_indices]
        
        # get the edges with a node in the bounding box
        # out_edges_in_bounding_box = list(self._skeleton.graph.out_edges(nodes_in_box))
        # in_edges_in_bounding_box = list(self._skeleton.graph.in_edges(nodes_in_box))
        # edges_in_bounding_box = in_edges_in_bounding_box + out_edges_in_bounding_box
        edges_in_bounding_box = list(self._skeleton.graph.edges(nodes_in_box))
        
        # get the sub_graph
        return self._skeleton.graph.edge_subgraph(edges_in_bounding_box)

    def update_napari_view(self) -> List[LayerDataTuple]:

        # make nodes
        node_data = dict(self._sub_graph.nodes(data=True))
        node_layer_data = graph_nodes_to_napari(
            node_data=node_data,
            layer_name=f"nodes - {self.name}"
        )
        
        # make edges
        edge_data = list(self._sub_graph.edges(data=True))
        edge_layer_data = graph_edges_to_napari(
            edge_data=edge_data,
            layer_name=f"edges - {self.name}"
        )
        edge_width = self._viewer.layers['edges'].edge_width        

        # add the layers to the viewer
        add_or_update_points_layer(viewer=self._viewer, layer_data_tuple=node_layer_data)
        add_or_update_shapes_layer(viewer=self._viewer, layer_data_tuple=edge_layer_data, edge_width=edge_width)
        

    def refresh_view(self):
        # update the subgraph
        self._sub_graph = self.get_sub_graph()

        # update the layers in the napari viewer
        self.update_napari_view()
        
    

class SkeletonViewer:
    def __init__(self, skeleton: Skeleton3D,
                image: np.ndarray,
                image_voxel_size: float,
                viewer: Optional[napari.Viewer] = None,
                edge_width: int = 1,
                nodes_size: int = 3
                ):
        self.skeleton = skeleton

        if viewer is None:
            viewer = napari.Viewer()
        self.viewer = viewer

        # selection containers
        self.selected_edges = EventedSet()

        # models for bounding box views on the graph
        self.views: List[SkeletonBoundingBoxView] = []

        # set up defaults
        self.n_spline_points = 5
        
        self.edge_width = edge_width

        # self.nodes_size = nodes_size

        # store the image for sampling
        # we will likely factor this out later
        self.image = image

        #store voxel size (assume isotropic for now)
        self.image_voxel_size = image_voxel_size
        # initialize the viewer
        self._initialize_viewer(image=image, 
                                edge_width=edge_width,
                                nodes_size = nodes_size)

        # set the default highlight colors
        self.highlight_colormap: Dict[bool, np.ndarray] = {
            False: np.array([1, 1, 0, 0.5]),
            True: np.array([1, 0, 1, 1])
        }

    @property
    def highlight_colormap(self) -> Dict[bool, np.ndarray]:
        return self._highlight_colormap

    @highlight_colormap.setter
    def highlight_colormap(self, colormap: Dict[bool, np.ndarray]) -> None:
        self.edges_layer.edge_color_cycle_map = colormap
        self.edges_layer.refresh_colors(False)
        self._highlight_colormap = colormap

    @property
    def node_size(self) -> np.ndarray:
        """Size of the node points in the viewer."""
        return self.nodes_layer.size

    @node_size.setter
    def node_size(self, node_size: Union[float, np.ndarray]) -> None:
        self.nodes_layer.size = node_size

    @property
    def highlighted_edges(self):
        """Table of currently highlighted edges"""
        return self.edges_layer.features.loc[
            self.edges_layer.features[EDGE_FEATURES_HIGHLIGHT_KEY]
        ]

    def _initialize_viewer(self, image: Optional[np.ndarray] = None, edge_width: int = 1, nodes_size:int = 3):
        # setup the node layer
        print("initilize nodes")
        self.nodes_layer = self._initialize_nodes(nodes_size)
        # setup the edges layer
        print("initilize edges")
        self.edges_layer = self._initialize_edges(edge_width=edge_width)
        

        # add the widget for viewing around a node
        self.sub_volume_widget = magicgui(self.view_subvolume_around_node, 
                                        node_index={"max": 1E7, "min": 0}, 
                                        # image_voxel_size=dict(default = tuple([self.image_voxel_size]*3))
                                        )
        self.viewer.window.add_dock_widget(self.sub_volume_widget.native)

        # add widget for plot the image around sliced edge
        image_slice_widget = QtImageSliceWidget()
        image_slice_widget.viewer = self
        self.viewer.window.add_dock_widget(image_slice_widget, area='right')

        # add the widget for slicing edges
        self.slicing_widget = magicgui(self.slice_selected_edges)
        self.viewer.window.add_dock_widget(self.slicing_widget.native)

        # add the widget for merging edges
        # self.merge_edges_widget = magicgui(self.merge_highlighted_edge_with_parent, call_button='merge edge')
        # self.viewer.window.add_dock_widget(self.merge_edges_widget.native)
        
        # add the widget for merging nodes
        self.merge_nodes_widget = magicgui(self.merge_nodes, call_button="merge nodes", node_to_keep = {"max":2**20}, node_to_merge = {"max":2**20} )
        self.viewer.window.add_dock_widget(self.merge_nodes_widget.native)

        # add widget for rendering part of the tree
        self.render_subtree_widget = magicgui(self.render_part_of_tree, call_button="render subtree", depth = {"max":int(999)})
        self.viewer.window.add_dock_widget(self.render_subtree_widget.native)

        #add the widget for deleting edges
        self.delete_edges_widget = magicgui(self.delete_edge, call_button="delete edge")
        self.viewer.window.add_dock_widget(self.delete_edges_widget.native)
        #add widget to count number of tips connected to edge
        self.count_number_of_tips_connected_to_edge_widget = magicgui(self.count_number_of_tips_connected_to_edge,
                                                                      call_button="count tips",
                                                                      result_widget=True)
        self.viewer.window.add_dock_widget(self.count_number_of_tips_connected_to_edge_widget.native)

        #add widtet to highlight edges based on nodes
        self.highlight_edges_by_edge_nodes_widget = magicgui(self.highlight_edges_by_nodes,
                                                             call_button="highlight edge")
        self.viewer.window.add_dock_widget(self.highlight_edges_by_edge_nodes_widget.native)

        #add widget to view subgraph from highlighted edge
        self.view_subtree_widget = magicgui(self.view_subtree,
                                            max_depth = {"max":32})
        self.viewer.window.add_dock_widget(self.view_subtree_widget.native)



        # hook up the events
        self._connect_events()

    def _initialize_nodes(self, size) -> napari.layers.Points:
        # node_data = deepcopy(dict(self.skeleton.graph.nodes(data=True)))
        node_data = dict(self.skeleton.graph.nodes(data=True))

        node_coordinates, node_kwargs, _ = graph_nodes_to_napari(
            node_data=node_data,
            layer_name="nodes_coordinates"
        )

        return self.viewer.add_points(node_coordinates,size =size,  **node_kwargs)

    def _initialize_edges(self, edge_width = 1) -> napari.layers.Shapes:
        # edge_data = deepcopy(list(self.skeleton.graph.edges(data=True)))
        edge_data = list(self.skeleton.graph.edges(data=True))

        spline_data, spline_kwargs, _ = graph_edges_to_napari(
            edge_data=edge_data,
            n_spline_points=self.n_spline_points,
            layer_name="edges"
        )

        return self.viewer.add_shapes(
            spline_data,
            edge_width=edge_width,
            **spline_kwargs
        )

    def merge_highlighted_edge_with_parent(self):
        """merge highlighted edge with parent node. MALTE"""
        if type(self.skeleton.graph) != nx.DiGraph:
            raise TypeError('Graph needs to be directed. Use nx.DiGraph')
        v1, n2 = high_edge = (list(self.highlighted_edges['start_node'])[0], list(self.highlighted_edges['end_node'])[0])
        n1,v1 = list(self.skeleton.graph.in_edges(high_edge[0]))[0]
        self.skeleton.merge_edge(n1, v1, n2)
        #update layers
        self.nodes_layer.data = self.skeleton.node_coordinates
        self.nodes_layer.features = dict(self.skeleton.nodes(data=NODE_COORDINATE_KEY)).keys()
        self.update_edge_layer()
    
    def update_node_layer(self):
        """update nodes after changes eg. merge."""
        self.nodes_layer.data = self.skeleton.node_coordinates
        self.nodes_layer.features = dict(self.skeleton.nodes(data=NODE_COORDINATE_KEY)).keys()
        self.nodes_layer.refresh()
    
    def update_edge_layer(self):
        """update edges after changes eg. merge. 
        Maybe faster to do only on changes made for bigger trees. MALTE"""
        edge_data = list(self.skeleton.graph.edges(data=True))
        #get old edge features

        edge_attributes = []
        spline_points = []
        for start_index, end_index, attributes in edge_data:
            spline = attributes.get(EDGE_SPLINE_KEY)

            attributes_for_table = {
                k: v for k, v in attributes.items()
                if k not in [EDGE_SPLINE_KEY, EDGE_COORDINATES_KEY]
            }

            # add the start and end indices

            if (EDGE_FEATURES_START_NODE_KEY not in attributes) or \
                (EDGE_FEATURES_END_NODE_KEY not in attributes):
               warnings.warn(f"edge start/end node not provided.\
                             inferring from edge key: {(start_index, end_index)}")
               attributes_for_table[(start_index, end_index)].update(
                {
                    EDGE_FEATURES_START_NODE_KEY: start_index,
                    EDGE_FEATURES_END_NODE_KEY: end_index,
                }
               )


            edge_attributes.append(attributes_for_table)

            # get the spline_points
            spline_coordinates = np.linspace(0, 1, self.n_spline_points, endpoint=True)
            spline_points.append(
                spline.sample(
                    u=spline_coordinates,
                    derivative_order=0
                )
            )

        spline_features = pd.DataFrame(edge_attributes)
        spline_features[EDGE_FEATURES_HIGHLIGHT_KEY] = False

        self.edges_layer.data = np.array(spline_points)
        self.edges_layer.features = spline_features
        self.edges_layer.refresh()

        for view in self.views:
            # update the views
            view.refresh_view()
    


    def update(self) -> None:
        """Redraw the graph.
        
        This updates the full graph and will be slow for large graphs.
        
        todo: make incremental update method
        """
        self.update_node_layer()
        self.update_edge_layer()

    def _connect_events(self) -> None:
        self.viewer.mouse_drag_callbacks.append(self._select_edge_on_click)

    def highlight_edges_by_index(self, edge_indices: List[int], edges_layer: Optional[Shapes]=None):
        """Highlight edges by their index in the layer features table.

        The index value is the order the edges are returned by
        skeleton_graph.edges.
        """
        if edges_layer is None:
            edges_layer = self.edges_layer
        self._reset_edge_highlight(edges_layer=edges_layer)
    
        edge_features = edges_layer.features
        edge_features.loc[
            edge_features.index[edge_indices],
            EDGE_FEATURES_HIGHLIGHT_KEY
        ] = True

        # self.edges_layer.edge_color = EDGE_FEATURES_HIGHLIGHT_KEY
        edges_layer.edge_color = EDGE_FEATURES_HIGHLIGHT_KEY
        self.highlight_colormap = {
            False: np.array([1, 1, 0, 0.5]),
            True: np.array([1, 0, 1, 1])
        }
        edges_layer.edge_color_cycle_map = self.highlight_colormap

        edges_layer.refresh_colors(False)
        self._highlight_colormap = self.highlight_colormap

    def highlight_edges_by_nodes(
            self,
            edge_list: List[Tuple[int, int]],
            edges_layer: Optional[Shapes]=None 
    ):
        """Highlight edges by their start and end nodes.
        
        Parameters
        ----------
        edge_list
            A list of tuples where each tuple is an edge.
            The first element in the tuple is the start node.
            The second element in the tuple is the end node.
        edges_layer
            The layer to highlight.
        """
        if edges_layer is None:
            edges_layer = self.edges_layer
        edge_features = edges_layer.features 
        edge_masks = []
        for start_node, end_node in edge_list:
            edge_masks.append(
                (edge_features["start_node"] == start_node) &
                 (edge_features["end_node"] == end_node)
            )
        if len(edge_masks) > 0:
            # get the indices of each highlighted edge
            edge_masks = np.atleast_2d(np.column_stack(edge_masks))
            selected_edge_mask = np.any(edge_masks, axis=1)
            edge_indices = list(edge_features.index[selected_edge_mask])
        else:
            edge_indices = []

        self.highlight_edges_by_index(edge_indices=edge_indices, edges_layer=edges_layer)

    def edge_indices_from_nodes(self, edge_list: List[Tuple[int, int]]) -> List[int]:
        edge_features = self.edges_layer.features
        edge_indices = []
        for row_index, row in edge_features.iterrows():
            node_pair = (row["start_node"], row["end_node"])
            reversed_node_pair = (row["end_node"], row["start_node"])

            if (node_pair in edge_list) or (reversed_node_pair in edge_list):
                edge_list.append(row_index)
        return edge_indices

    def _reset_edge_highlight(self, refresh_colors: bool = True, edges_layer: Optional[Shapes]=None ) -> None:
        if edges_layer is None:
            edges_layer = self.edges_layer
        edges_layer.features[EDGE_FEATURES_HIGHLIGHT_KEY] = False
        if refresh_colors:
            edges_layer.refresh_colors(False)

    def _select_edge_on_click(self, layer, event=None) -> None:
        """Callback function to select edge upon mouse click."""
        edges_layer = None
        for layer in list(self.viewer.layers.selection):
            if isinstance(layer, Shapes):
                # take the first selected shapes layer as the edge layer
                edges_layer = layer
                break
        if edges_layer is None:
            # if no edge layer is present, return early
            return

        result = edges_layer.get_value(
            event.position,
            view_direction=event.view_direction,
            dims_displayed=event.dims_displayed,
            world=True
        )
        self.selected_edges.clear()

        if result is None:
            # no edge layer is selected
            selected_edges = []
        else:
            value = result[0]
            if value is None:
                selected_edges = []
            else:
                selected_edges = [value]

        # get the start and end nodes for the selected edges
        edge_features = edges_layer.features
        selected_rows = edge_features.loc[edge_features.index[selected_edges]]
        selected_edges_nodes = [
            (row["start_node"], row["end_node"]) for _, row in selected_rows.iterrows()
        ]

        if edges_layer is not self.edges_layer:
            print("highlighting view")
            # highlight the view if operating on the view
            self.highlight_edges_by_nodes(selected_edges_nodes, edges_layer=edges_layer)
        
        # highlight the main skeleton
        self.highlight_edges_by_nodes(selected_edges_nodes)
        self.selected_edges.update(selected_edges_nodes)

    def slice_selected_edges(
            self,
            file_base_name: str,
            image_voxel_size: Tuple[float, float, float] = (1, 1, 1),
            slice_spacing: int = 1,
            slice_pixel_size: float = 1,
            slice_width: int = 20,
            interpolation_order: int = 1
    ) -> None:
        selected_edge_indices = list(self.selected_edges)

        for edge_index in selected_edge_indices:
            edge_features = self.edges_layer.features
            start_node = edge_features.loc[
                edge_features.index[edge_index],
                EDGE_FEATURES_START_NODE_KEY
            ]
            end_node = edge_features.loc[
                edge_features.index[edge_index],
                EDGE_FEATURES_END_NODE_KEY
            ]

            image_slices = self.skeleton.sample_slices_on_edge(
                image=self.image,
                image_voxel_size=image_voxel_size,
                start_node=start_node,
                end_node=end_node,
                slice_width=slice_width,
                slice_pixel_size=slice_pixel_size,
                slice_spacing=slice_spacing,
                interpolation_order=interpolation_order
            )

            filename = f"{file_base_name}_{start_node}_{end_node}.tif"
            imsave(filename, image_slices, check_contrast=False)

    def view_subvolume_around_node(
            self,
            node_index: int,
            image_voxel_size: Tuple[float, float, float] = (1, 1, 1),
            bounding_box_width: float = 300
    ):
        sub_volume, bounding_box = self.skeleton.sample_image_around_node(
            node_index=node_index,
            image=self.image,
            image_voxel_size=image_voxel_size,
            bounding_box_shape=bounding_box_width,
        )
        min_corner = bounding_box[0] * np.array(image_voxel_size)
        max_corner = bounding_box[1] * np.array(image_voxel_size)

        # make the layers
        base_layer_name = f"node {node_index}"

        image_kwargs = {
            "translate": min_corner,
            "colormap": "bop blue",
            "scale": image_voxel_size,
            "name": f"{base_layer_name} - image"
        }
        add_or_update_image_layer(
            viewer=self.viewer,
            layer_data_tuple=(sub_volume, image_kwargs, "image")
        )
        # self.viewer.add_image(
        #     sub_volume,
        #     translate=min_corner,
        #     colormap="bop blue",
        #     scale=image_voxel_size,
        #     name=f"{base_layer_name} - image"
        # )

        
        bounding_box_view = SkeletonBoundingBoxView(
            self.skeleton,
            viewer=self.viewer,
            min_corner=min_corner,
            max_corner=max_corner,
            name=base_layer_name
        )
        self.views.append(bounding_box_view)
        
    def merge_nodes(self, node_to_keep: int, node_to_merge: int) -> None:
        # merge the nodes
        self.skeleton.merge_nodes(
            node_to_keep=node_to_keep,
            node_to_merge=node_to_merge
        )

        # redraw
        self.update()

    def delete_edge(self):
        """delete highlighted edge. MALTE"""

        
        high_edge = (list(self.highlighted_edges['start_node'])[0], 
                    list(self.highlighted_edges['end_node'])[0])
        #copy graph
        graph = self.skeleton.graph.copy()
        graph.remove_edge(*high_edge)

        # if not nx.is_connected(graph):
        #     components = nx.connected_components(graph)
        #     c_list = [c for c in sorted(components, key=len, reverse=False)]
        #     graph.remove_nodes_from(c_list[0])


        #detect all changes
        changed_edges = set(self.skeleton.graph.edges) - set(graph.edges)
        for edge in changed_edges:
            for node in edge:
                if len(graph.edges(node)) == 0:
                    graph.remove_node(node)
                #merge edges if node has degree 2
                elif len(graph.edges(node)) == 2:
                    #merge
                    print('merge')
                    #order of edges to merge is important. Merge incoming edge with outgoing edge
                    if list(graph.edges(node, data = 'start_node'))[0][2]:
                        new_edge = [None] * 2
                        for u,v,attr in graph.edges(node, data = True):
                            if (u,v) not in edge:
                                if attr['start_node'] == node:
                                    new_edge[1] = attr['end_node']
                                if attr['end_node'] == node:
                                    new_edge[0] = attr['start_node']
                        self.skeleton.merge_edge(new_edge[0], node, new_edge[1])

                    #if no direction, merge however... might lead to funny splines
                    else:
                        new_edge = [x for y in list(graph.edges(node)) for x in y if x not in edge]
                        self.skeleton.merge_edge(new_edge[0], node, new_edge[1])

        #check if graph is still connected, if not remove orphaned nodes
        self.skeleton.graph.remove_nodes_from(list(nx.isolates(self.skeleton.graph)))
        
 

        self.update()
    
    def count_number_of_tips_connected_to_edge(self, plot_tips:bool= True):
        #get highlighted edge

        feature_table = self.edges_layer.features
        #add start and end node to graph based on breadth first search/distance to origin
        graph = self.skeleton.graph.copy()


        start_node = feature_table.loc[feature_table['highlight'] == True]['start_node'].values[0]
        end_node = feature_table.loc[feature_table['highlight'] == True]['end_node'].values[0]
        graph.remove_edge(start_node, end_node)
        subtree = nx.bfs_tree(graph, end_node)

        #count number of endpoints
        end_points = []
        end_point_coordinates = []
        for node in subtree.nodes:
            if subtree.degree(node) == 1:
                end_points.append(node)
                end_point_coordinates.append(graph.nodes[node][NODE_COORDINATE_KEY])
        
        if plot_tips == True:
            self.viewer.add_points(end_point_coordinates,
                                   size = self.node_size, 
                                   face_color='red', 
                                   name = f'tips_of_{start_node}_{end_node}')
        return len(end_points)
    #view subtree
    def view_subtree(self, max_depth:int = 0):
        #copy graph
        g = self.skeleton.graph.copy()
        #get highlighted edge
        highlighted_edge = self.highlighted_edges
        highlighted_edge = highlighted_edge.start_node.values[0], highlighted_edge.end_node.values[0]
        #remove edge
        g.remove_edge(*highlighted_edge)
        #get subtree
        # if max_depth is None:
        subtree = nx.descendants(g, highlighted_edge[1])

        g2 = self.skeleton.graph.subgraph(subtree)

        if max_depth != 0:
            subtree = list(SkeletonEdgeIterator(self.skeleton, starting_node = highlighted_edge[1],max_distance = max_depth))
            subtree = set([node for edge in [edge for edges in subtree for edge in edges] for node in edge])
            subtree.add(highlighted_edge[1])
            g2 = g2.subgraph(subtree)
        spline_path = []

        for _,_,attr in g2.edges(data = True):
            spline_coordinates = np.linspace(0, 1, 5, endpoint=True)

            spline_path.append(attr['edge_spline'].sample(u = spline_coordinates,
                                                        derivative_order = 0))
        self.viewer.add_shapes(spline_path, 
                               shape_type='path', 
                               edge_width=4, 
                               edge_color='blue', 
                               name='subtree')
        # # return subtree
        # skeleton_viewer.skeleton.graph.remove_nodes_from(subtree)
        # skeleton_viewer.refresh()
    

    def render_part_of_tree(self, depth:int, refresh:bool = True):
        #only render edges up to a certain depth
        if refresh:
            self.update()
        new_features = self.edges_layer.features[self.edges_layer.features['level'] <depth]
        self.edges_layer.data = [self.edges_layer.data[i] for i in new_features.index]
        self.edges_layer.features = new_features
        self.edges_layer.edge_width = 10






class QtImageSliceWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.viewer: napari.Viewer = None
        self.image_slices: Optional[np.ndarray] = None
        self.results_table: Optional[pd.DataFrame] = None
        self.pixel_size_um: float = 5.79
        self.stain_channeL_names: List[str] = []
        self.min_slice: int = 0
        self.max_slice: int = 0
        self.current_channel_index: int = 0

        # create the slider
        self.slice_slider = QLabeledSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setRange(0, 99)
        self.slice_slider.setSliderPosition(50)
        self.slice_slider.setSingleStep(1)
        self.slice_slider.setTickInterval(1)
        self.slice_slider.valueChanged.connect(self._on_slider_moved)

        #add a "load new edge" button
        self.load_button = QPushButton("Load New Edge")
        self.load_button.clicked.connect(self.load_new_image_slices)

        # create the stain channel selection box
        self.image_selector = QComboBox()
        self.image_selector.currentIndexChanged.connect(self._update_current_channel)



        # create the image
        pg.setConfigOptions(imageAxisOrder="row-major")
        self.image_widget = pg.ImageView(parent=self)

        # Add a ScatterPlotItem for the red dot in the middle
        self.red_dot = pg.ScatterPlotItem()
        self.red_dot.setData([0], [0], pen='r', symbol='+', size=15)
        self.image_widget.getView().addItem(self.red_dot)




        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.load_button)
        self.layout().addWidget(self.slice_slider)
        self.layout().addWidget(self.image_selector)
        self.layout().addWidget(self.image_widget)
        # self.layout().addWidget(self.plot_widget)

    def _on_slider_moved(self, event=None):
        self.draw_at_current_slice_index()

    def draw_at_current_slice_index(self):
        current_slice_index = int(self.slice_slider.value())
        self.draw_at_slice_index(current_slice_index)
        # self._update_plot_slice_line(current_slice_index)

    def draw_at_slice_index(self, slice_index: int):
        self._update_image(slice_index)
        self._update_red_dot()

    def _update_image(self, slice_index: int):
        # offset the slice index since we only have a subset of the slices
        if self.image_slices is None:
            # if images haven't been set yet, do nothing
            return

        offset_slice_index = slice_index - self.min_slice
        image_slice = self.image_slices[self.current_channel_index, offset_slice_index, ...]

        # update the image slice
        self.image_widget.setImage(image_slice)



    def _update_red_dot(self):
        # Update the position of the red dot in the middle
        image_shape = self.image_slices.shape[2:]  # Assuming shape is (slice, y, x)
        middle_y, middle_x = image_shape[0] // 2, image_shape[1] // 2
        self.red_dot.setData([middle_x], [middle_y])




    def set_data(
            self,
            stain_image: np.ndarray,
            results_table: pd.DataFrame,
            pixel_size_um: float,
            stain_channel_names: Optional[List[str]]=None,
    ):
        if stain_image.ndim == 3:
            # make sure the image is 4D
            # (channel, slice, y, x)
            stain_image = np.expand_dims(stain_image, axis=0)
        # set the range slider range
        self.min_slice = results_table["slice_index"].min()
        self.max_slice = results_table["slice_index"].max()
        self.slice_slider.setRange(self.min_slice, self.max_slice)

        self.pixel_size_um = pixel_size_um
        self.image_slices = stain_image
        self.results_table = results_table



        # add the image channels
        if stain_channel_names is not None:
            self.stain_channeL_names = stain_channel_names

        else:
            n_channels = stain_image.shape[0]
            self.stain_channeL_names = [
                f"channel {channel_index}" for channel_index in range(n_channels)
            ]


        self.image_selector.clear()
        self.image_selector.addItems(self.stain_channeL_names)

        # refresh the selected channel index and redraw
        self._update_current_channel()

        self.setVisible(True)
    
    def load_new_image_slices(self):
        highlighted_edge = self.viewer.highlighted_edges
        start_node = highlighted_edge['start_node'].values[0]
        end_node = highlighted_edge['end_node'].values[0]

        #eventually move up 
        slice_width = 300
        slice_spacing = 5
        interpolation_order = 1
        slice_pixel_size = self.viewer.image_voxel_size

        new_image_slices = self.viewer.skeleton.sample_slices_on_edge(
            image=self.viewer.image,
            image_voxel_size=self.viewer.image_voxel_size,
            start_node=start_node,
            end_node=end_node,
            slice_width=slice_width,
            slice_pixel_size=slice_pixel_size,
            slice_spacing=slice_spacing,
            interpolation_order=interpolation_order
        )

        results_table = pd.DataFrame({"slice_index": np.arange(new_image_slices.shape[0])})

        self.set_data(new_image_slices, results_table, 1)

        
    def _update_current_channel(self, event=None):
        if len(self.stain_channeL_names) == 0:
            # don't do anything if there aren't any channels
            return
        self.current_channel_index = self.image_selector.currentIndex()
        self.draw_at_current_slice_index()