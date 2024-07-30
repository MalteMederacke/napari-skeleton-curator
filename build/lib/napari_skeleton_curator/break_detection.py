from math import floor
from typing import Tuple

import h5py
import napari
import numpy as np
from numba import jit, prange
from numba.typed import List
from numba.types import bool_, float32

from skan import Skeleton
from scipy.spatial import KDTree


def _get_degree_1_nodes(skeleton_obj: Skeleton) -> np.ndarray:
    """Get the end points of the skeleton.

    End points are defined as nodes with degree 1.

    Parameters
    ----------
    skeleton_obj : Skeleton
        The skan skeleton object that you want to get the end points from.

    Returns
    -------
    np.ndarray
        (n x 3) array of the coordinates of the end points of the skeleton.
    """
    return skeleton_obj.coordinates[skeleton_obj.degrees == 1]


@jit(nopython=True, fastmath=True, cache=True, nogil=True)
def check_background_intersection(
        start_point: np.ndarray,
        end_point: np.ndarray,
        label_image: np.ndarray,
        skeleton_label_image: np.ndarray,
        step_size: float
) -> Tuple[bool, float]:
    line_length = np.linalg.norm(end_point - start_point)
    num_points = floor(line_length / step_size)

    # unit vector in the direction to check
    unit_vector = (end_point - start_point) / line_length
    step_vector = step_size * unit_vector

    # get the branch label values for the start and end point
    start_skeleton_value = skeleton_label_image[
        int(start_point[0]), int(start_point[1]), int(start_point[2])
    ]
    end_skeleton_value = skeleton_label_image[
        int(end_point[0]), int(end_point[1]), int(end_point[2])
    ]

    for step_index in range(num_points):
        query_point = start_point + step_index * step_vector
        label_value = label_image[
            int(query_point[0]), int(query_point[1]), int(query_point[2])
        ]
        skeleton_value = skeleton_label_image[
            int(query_point[0]), int(query_point[1]), int(query_point[2])
        ]
        if (label_value == 0):
            # line intersects background
            return False, line_length
        if not (
                (skeleton_value == 0) or (skeleton_value == start_skeleton_value) or (
                skeleton_value == end_skeleton_value)
        ):
            # line intersects a different branch
            return False, line_length

    return True, line_length


@jit(nopython=True, fastmath=True, cache=True)
def filter_points_from_same_branch(
    start_point: np.ndarray,
    query_points: np.ndarray,
    skeleton_label_image: np.ndarray
) -> np.ndarray:
    start_point_label_value = skeleton_label_image[
        int(start_point[0]),
        int(start_point[1]),
        int(start_point[2])
    ]

    # loop through the query points
    n_query_points = query_points.shape[0]
    point_in_different_branch = np.zeros((n_query_points,), bool_)
    for query_point_index in range(n_query_points):
        query_label_value = skeleton_label_image[
            int(query_points[query_point_index, 0]),
            int(query_points[query_point_index, 1]),
            int(query_points[query_point_index, 2])
        ]
        if query_label_value == start_point_label_value:
            point_in_different_branch[query_point_index] = False
        else:
            point_in_different_branch[query_point_index] = True

    return point_in_different_branch


@jit(nopython=True, fastmath=True, cache=True)
def find_nearest_point_in_segmentation(
        start_point: np.ndarray,
        points_to_check: np.ndarray,
        label_image: np.ndarray,
        skeleton_label_image: np.ndarray
) -> np.ndarray:
    # get the points to query
    points_to_query_mask = filter_points_from_same_branch(
        start_point,
        points_to_check,
        skeleton_label_image,
    )
    points_to_query = points_to_check[points_to_query_mask]

    # loop through array
    n_query_points = points_to_query.shape[0]
    all_intersects_background = np.zeros((n_query_points,), dtype=bool_)
    all_distances = np.zeros((n_query_points,), dtype=float32)
    for query_point_index in range(n_query_points):
        query_point = points_to_query[query_point_index]

        intersects_background, distance = check_background_intersection(
            start_point=start_point,
            end_point=query_point,
            label_image=label_image,
            skeleton_label_image=skeleton_label_image,
            step_size=0.5
        )

        # store the values
        all_intersects_background[query_point_index] = intersects_background
        all_distances[query_point_index] = distance

    # find the shortest distance that doesn't intersect background
    if all_intersects_background.sum() == 0:
        # return nans if no point passes the filters
        return np.array([np.nan, np.nan, np.nan])
    else:
        shortest_distance_index = all_distances[all_intersects_background].argmin()
        return points_to_query[all_intersects_background][shortest_distance_index]


@jit(nopython=True, fastmath=True, cache=True, parallel=True)
def find_missing_branches(
        start_points: np.ndarray,
        proximal_points: List[np.ndarray],
        skeleton_coordinates: np.ndarray,
        label_image: np.ndarray,
        skeleton_label_image: np.ndarray
) -> np.ndarray:
    n_proximal_points = len(proximal_points)

    # loop through the start points
    all_nearest_points = np.zeros((n_proximal_points, 3))
    for point_index in prange(n_proximal_points):
        points_in_radius = skeleton_coordinates[proximal_points[point_index]]

        # get the start point
        start_point = start_points[point_index]

        nearest_point = find_nearest_point_in_segmentation(
            start_point=start_point,
            points_to_check=points_in_radius,
            label_image=label_image,
            skeleton_label_image=skeleton_label_image
        )
        all_nearest_points[point_index, :] = nearest_point

    return all_nearest_points


def find_breaks_in_skeleton(
        skeleton_obj: Skeleton,
        end_point_radius: float,
        segmentation_label_image: np.ndarray,
        skeleton_label_image: np.ndarray,
        n_workers: int = -1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # build the tree for finding points in radius
    skeleton_coordinates = skeleton_obj.coordinates.astype(float)
    skeleton_tree = KDTree(skeleton_obj.coordinates)

    # get points proximal to end points
    degree_1_nodes = _get_degree_1_nodes(skeleton_obj)
    proximal_points = skeleton_tree.query_ball_point(degree_1_nodes, r=end_point_radius, workers=n_workers)

    points_numba = List(np.array(indices) for indices in proximal_points)
    nearest_points = find_missing_branches(
        start_points=degree_1_nodes,
        proximal_points=points_numba,
        skeleton_coordinates=skeleton_coordinates,
        label_image=segmentation_label_image,
        skeleton_label_image=skeleton_label_image
    )

    to_join_mask = np.logical_not(np.any(np.isnan(nearest_points), axis=1))
    source_coordinates = degree_1_nodes[to_join_mask]
    destination_coordinates = nearest_points[to_join_mask]

    # get the ids of the nodes of the source_coordinates
    degree_1_node_ids = np.squeeze(np.argwhere(skeleton_obj.degrees == 1))
    node_ids = degree_1_node_ids[to_join_mask]
    return node_ids, source_coordinates, destination_coordinates
