
import napari
import numpy as np
import warnings
from qtpy.QtWidgets import QVBoxLayout, QWidget, QPushButton, QLabel
import skimage as ski
import pandas as pd
from skimage import morphology as ski_morphology
from skimage import measure as ski_measure
from skimage.segmentation import find_boundaries

class SliceAnnotator:
    def __init__(self, viewer: napari.Viewer = None):
        if viewer is None:
            viewer = napari.Viewer()
        self.viewer = viewer

        # Initialize widgets
        self._initialize_widgets()
        # Set up keyboard shortcuts
        self._setup_shortcuts()

    def _initialize_widgets(self):
        # Add each widget to the viewer window
        self.viewer.window.add_dock_widget(CropWidget(self.viewer), area="right")
        self.viewer.window.add_dock_widget(ReduceSlicesWidget(self.viewer), area="right")
        self.viewer.window.add_dock_widget(DeleteSliceWidget(self.viewer), area="right")
        self.viewer.window.add_dock_widget(AddLabelWidget(self.viewer), area="right")
        self.viewer.window.add_dock_widget(MoveLayerWidget(self.viewer), area="right")

    def _setup_shortcuts(self):
        # Define keyboard shortcuts
        self.viewer.bind_key("q", self.previous_label)
        self.viewer.bind_key("e", self.next_label)

    def previous_label(self, viewer):
        active_layer = viewer.layers.selection.active
        if hasattr(active_layer, "selected_label"):
            active_layer.selected_label -= 1
        else:
            active_layer.selected_label = active_layer.data.max()

    def next_label(self, viewer):
        active_layer = viewer.layers.selection.active
        if hasattr(active_layer, "selected_label"):
            active_layer.selected_label += 1


class CropWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.start_slice_button = QPushButton("Get start slice")
        self.start_slice_button.clicked.connect(self._get_current_slice_start)
        self.end_slice_button = QPushButton("Get end slice")
        self.end_slice_button.clicked.connect(self._get_current_slice_end)
        self.crop_button = QPushButton("Crop")
        self.crop_button.clicked.connect(self.crop_image)
        self.start_slice_number_label = QLabel("Start slice number: ")
        self.end_slice_number_label = QLabel("End slice number: ")
        self.current_image_selection_label = QLabel("Current image: ")

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.current_image_selection_label)
        self.layout().addWidget(QLabel("Get start slice"))
        self.layout().addWidget(self.start_slice_button)
        self.layout().addWidget(self.start_slice_number_label)
        self.layout().addWidget(QLabel("Get end slice"))
        self.layout().addWidget(self.end_slice_button)
        self.layout().addWidget(self.end_slice_number_label)
        self.layout().addWidget(QLabel("Crop Image"))
        self.layout().addWidget(self.crop_button)

    def _get_current_slice_start(self):
        viewer = self.viewer
        viewer.layers.selection._current.metadata["start"] = viewer.dims.current_step[0]
        self.start_slice_number_label.setText("Start slice number: " + str(viewer.dims.current_step[0]))
        self.current_image_selection_label.setText("Current image: " + str(viewer.layers.selection._current.name))

    def _get_current_slice_end(self):
        viewer = self.viewer
        viewer.layers.selection._current.metadata["end"] = viewer.dims.current_step[0]
        self.end_slice_number_label.setText("End slice number: " + str(viewer.dims.current_step[0]))
        self.current_image_selection_label.setText("Current image: " + str(viewer.layers.selection._current.name))

    def crop_image(self):
        viewer = self.viewer
        start = viewer.layers.selection._current.metadata.get("start")
        end = viewer.layers.selection._current.metadata.get("end")

        if not viewer.layers.selection._current.visible:
            warnings.warn("Layer not visible")
            return

        if viewer.layers.selection._current.name.endswith("_label"):
            base_layer_name = viewer.layers.selection._current.name.split("_label")[0]
            viewer.layers.selection._current.data = viewer.layers.selection._current.data[start:end, :, :]

            if base_layer_name in viewer.layers:
                viewer.layers[base_layer_name].data = viewer.layers[base_layer_name].data[start:end, :, :]
            else:
                warnings.warn(f"Corresponding image layer '{base_layer_name}' not found.")
        else:
            viewer.layers.selection._current.data = viewer.layers.selection._current.data[start:end, :, :]
            label_layer_name = viewer.layers.selection._current.name + "_label"
            if label_layer_name in viewer.layers:
                viewer.layers[label_layer_name].data = viewer.layers[label_layer_name].data[start:end, :, :]
            else:
                warnings.warn(f"Corresponding label layer '{label_layer_name}' not found.")

        print(viewer.layers.selection._current.data.shape)


class ReduceSlicesWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.reduce_slices_button = QPushButton("Reduce slices by half")
        self.reduce_slices_button.clicked.connect(self.reduce_slices)
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.reduce_slices_button)

    def reduce_slices(self):
        viewer = self.viewer
        if not viewer.layers.selection._current.visible:
            warnings.warn("Layer not visible")
            return

        if viewer.layers.selection._current.name.endswith("_label"):
            base_layer_name = viewer.layers.selection._current.name.split("_label")[0]
            viewer.layers.selection._current.data = viewer.layers.selection._current.data[::2, :, :]
            if base_layer_name in viewer.layers:
                viewer.layers[base_layer_name].data = viewer.layers[base_layer_name].data[::2, :, :]
            else:
                warnings.warn(f"Corresponding image layer '{base_layer_name}' not found.")
        else:
            viewer.layers.selection._current.data = viewer.layers.selection._current.data[::2, :, :]
            label_layer_name = viewer.layers.selection._current.name + "_label"
            if label_layer_name in viewer.layers:
                viewer.layers[label_layer_name].data = viewer.layers[label_layer_name].data[::2, :, :]
            else:
                warnings.warn(f"Corresponding label layer '{label_layer_name}' not found.")

        print(viewer.layers.selection._current.data.shape)


class DeleteSliceWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.delete_slice_button = QPushButton("Delete slice")
        self.delete_slice_button.clicked.connect(self.delete_slice)
        self.delete_slice_number_label = QLabel("Slice number to delete: ")
        self.current_image_selection_label = QLabel("Current image: ")
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.current_image_selection_label)
        self.layout().addWidget(QLabel("Delete slice"))
        self.layout().addWidget(self.delete_slice_button)
        self.layout().addWidget(self.delete_slice_number_label)

    def delete_slice(self):
        viewer = self.viewer
        slice_number = viewer.dims.current_step[0]
        if viewer.layers.selection._current.name.endswith("_label"):
            base_layer_name = viewer.layers.selection._current.name.split("_label")[0]
            viewer.layers.selection._current.data = np.delete(viewer.layers.selection._current.data, slice_number, axis=0)
            if base_layer_name in viewer.layers:
                viewer.layers[base_layer_name].data = np.delete(viewer.layers[base_layer_name].data, slice_number, axis=0)
            else:
                warnings.warn(f"Corresponding image layer '{base_layer_name}' not found.")
        else:
            viewer.layers.selection._current.data = np.delete(viewer.layers.selection._current.data, slice_number, axis=0)
            label_layer_name = viewer.layers.selection._current.name + "_label"
            if label_layer_name in viewer.layers:
                viewer.layers[label_layer_name].data = np.delete(viewer.layers[label_layer_name].data, slice_number, axis=0)
            else:
                warnings.warn(f"Corresponding label layer '{label_layer_name}' not found.")

        self.delete_slice_number_label.setText("Slice number to delete: " + str(slice_number))
        self.current_image_selection_label.setText("Current image: " + str(viewer.layers.selection._current.name))


class AddLabelWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.add_label_button = QPushButton("Add label")
        self.add_label_button.clicked.connect(self.add_label)
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(QLabel("Add label"))
        self.layout().addWidget(self.add_label_button)

    def add_label(self):
        viewer = self.viewer
        image_layer = viewer.layers.selection._current.data
        label = np.zeros(image_layer.shape, dtype=np.uint8)
        image_name = viewer.layers.selection._current.name
        viewer.add_labels(label, name=image_name + "_label")
        label_index = viewer.layers.index(image_name + "_label")
        current_index = viewer.layers.index(image_name)
        viewer.layers.move(label_index, current_index + 1)


class MoveLayerWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.move_label_button = QPushButton('Move label')
        self.move_label_button.clicked.connect(self.move_label)
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(QLabel('Move label'))
        self.layout().addWidget(self.move_label_button)

    def move_label(self):
        viewer = self.viewer
        top_layer_index = viewer.layers.index(viewer.layers[-1].name)
        current_index = viewer.layers.index(viewer.layers.selection._current.name)
        viewer.layers.move(top_layer_index, current_index+1)


#Read out functions-----
def filter_largest_label_layers(viewer, tissue_value=2, lumen_value=1, min_size=10):
    """
    Filters layers in the viewer with the largest label components for tissue and lumen.
    
    Parameters:
    viewer (napari.Viewer): The napari viewer containing the layers to filter.
    tissue_value (int): The integer value representing tissue in the label image.
    lumen_value (int): The integer value representing lumen in the label image.
    min_size (int): Minimum size of objects to keep.

    """
    for layer in viewer.layers:
        if layer.visible and layer.name.endswith('_label'):
            print(f"Processing layer: {layer.name}")
            
            # Get label data
            label = layer.data
            tissue = label == tissue_value
            lumen = label == lumen_value
            
            # Remove small objects
            tissue = ski.morphology.remove_small_objects(tissue, min_size=min_size)
            lumen = ski.morphology.remove_small_objects(lumen, min_size=min_size)

            # Throw warnings if no tissue or lumen is found
            if np.sum(tissue) == 0:
                warnings.warn(f'No tissue found in layer {layer.name}')
            if np.sum(lumen) == 0:
                warnings.warn(f'No lumen found in layer {layer.name}')

            # Create the filtered label array
            label_filtered = np.zeros(label.shape, dtype=np.uint8)
            label_filtered[tissue] = tissue_value
            label_filtered[lumen] = lumen_value
            label_filtered[label_filtered == 0] = 3  # Set background or other regions to 3

            # Update the layer data with the filtered labels
            viewer.layers[layer.name].data = label_filtered


import h5py
import numpy as np
from skimage import measure as ski_measure

def add_measurements_to_dataframe(viewer, dataframe, file, label_layer, label_number, pixel_size, save_segmentation=True, save_path=None):
    """
    Adds measurements from a segmented image to a dataframe.

    Parameters:
    - dataframe (pd.DataFrame): DataFrame to store the computed measurements.
    - file (str): File name for the image data.
    - label_layer (str): Name of the layer in the viewer containing the label image data.
    - label_number (int): The label number identifying the region of interest within the label image.
    - pixel_size (float): The scaling factor to convert pixel measurements to physical units.
    - save_segmentation (bool, optional): Whether to save the segmentation to an HDF5 file. Default is True.
    - save_path (str, optional): Path to save the segmentation file, required if `save_segmentation` is True.

    Returns:
    - pd.DataFrame: The input dataframe updated with computed measurements for the specified file.
    """

    scaling_factor = pixel_size

    # Extract the label image from the viewer layer
    label_image = viewer.layers[label_layer].data
    print(f"Label image shape: {label_image.shape}")
    
    # Segment the specified label
    segmentation = ski_measure.label(label_image == label_number)
    print(f"Segmentation shape: {segmentation.shape}")
    
    # Save segmentation if requested
    if save_segmentation:
        if save_path is None:
            raise ValueError("save_path must be provided when save_segmentation is True.")
        if file.endswith('.tif'):
            file_name = file.split('.tif')[0]
        else:
            file_name = file
        with h5py.File(f"{save_path}/{file_name}_segmentation.h5", 'w') as f:
            f.create_dataset('segmentation', data=segmentation)
            f.create_dataset('image', data=viewer.layers[file].data)

    # Initialize lists for measurement properties
    perimeter = []
    major_axis_length = []
    minor_axis_length = []
    area = []
    radii = []
    diameters = []

    # Loop through each slice in the segmentation and calculate properties
    for slice in segmentation:
        if np.sum(slice) == 0:
            # If no region is found, append zero values for measurements
            perimeter.append(0)
            major_axis_length.append(0)
            minor_axis_length.append(0)
            area.append(0)
            radii.append(0)
            diameters.append(0)
            continue
        
        # Calculate region properties for the slice
        reg_props = ski_measure.regionprops(slice)[0]
        scaled_area = reg_props.area * scaling_factor**2
        area.append(scaled_area)
        
        # Calculate radius and diameter assuming area represents a circle
        radius = np.sqrt(scaled_area / np.pi)
        diameter = 2 * radius
        radii.append(radius)
        diameters.append(diameter)
        
        # Append scaled properties
        perimeter.append(reg_props.perimeter * scaling_factor)
        major_axis_length.append(reg_props.major_axis_length * scaling_factor)
        minor_axis_length.append(reg_props.minor_axis_length * scaling_factor)

    # Calculate means and standard deviations of each property
    dataframe.loc[file, 'perimeter'] = np.mean(perimeter)
    dataframe.loc[file, 'perimeter_sd'] = np.std(perimeter)
    dataframe.loc[file, 'major_axis_length'] = np.mean(major_axis_length)
    dataframe.loc[file, 'major_axis_length_sd'] = np.std(major_axis_length)
    dataframe.loc[file, 'minor_axis_length'] = np.mean(minor_axis_length)
    dataframe.loc[file, 'minor_axis_length_sd'] = np.std(minor_axis_length)
    dataframe.loc[file, 'area'] = np.mean(area)
    dataframe.loc[file, 'area_sd'] = np.std(area)
    dataframe.loc[file, 'radius'] = np.mean(radii)
    dataframe.loc[file, 'radius_sd'] = np.std(radii)
    dataframe.loc[file, 'diameter'] = np.mean(diameters)
    dataframe.loc[file, 'diameter_sd'] = np.std(diameters)

    return dataframe

def extract_parameters_h5(file):
    """
    Extracts parameters from a given filename formatted with specific attributes.

    Parameters:
    - file (str): The filename containing parameters, structured with fields like 
                  'level', 'start_node', 'end_node', 'number_tips', and 'edge_length'.

    Returns:
    - dict: A dictionary containing extracted parameters:
        - 'level' (int): The level value extracted from the filename.
        - 'start_node' (int): The start node value.
        - 'end_node' (int): The end node value.
        - 'number_tips' (int): The number of tips.
        - 'edge_length' (float): The edge length value.
    """
    
    # Extract parameters from the filename
    level = int(file.split('level_')[1].split('_sn')[0])
    start_node = int(file.split('sn_')[1].split('_en')[0])
    end_node = int(file.split('en_')[1].split('_number')[0])
    number_tips = int(file.split('number_tips_')[1].split('_edge')[0])
    edge_length = float(file.split('edge_length_')[1].split('.h5')[0])
    
    # Store extracted parameters in a dictionary
    parameters = {
        'level': level,
        'start_node': start_node,
        'end_node': end_node,
        'number_tips': number_tips,
        'edge_length': edge_length
    }
    
    return parameters


def find_parent_edges(metadata):
    """
    Finds the parent edge for each row in the metadata DataFrame and adds it as a new column.

    Parameters:
    - metadata (pd.DataFrame): DataFrame containing edge metadata with at least 'level', 'start_node', and 'end_node' columns.

    Returns:
    - pd.DataFrame: The updated DataFrame with a new column 'parent_edge', which contains the index of the parent edge for each row.
    """
    # Initialize the 'parent_edge' column
    metadata['parent_edge'] = None

    # Iterate over each row to find and assign the parent edge
    for i, row in metadata.iterrows():
        if row['level'] == 0:
            continue
        try:
            # Find the index of the row where 'end_node' matches the current row's 'start_node'
            parent_index = metadata.loc[metadata['end_node'] == row['start_node']].index[0]
            metadata.loc[i, 'parent_edge'] = parent_index
        except IndexError:
            # Print the row index if no parent edge is found
            print(f"No parent edge found for row {i}")

    return metadata





def process_layers_and_compute_thickness(viewer, metadata, scaling_factor=100, 
                                         lumen_label_value=1, tissue_label_value=2):
    """
    Processes visible layers in the viewer, calculates label images for tissue and lumen,
    computes tissue area and thickness, and updates metadata with the results.

    Parameters:
    - viewer (napari.Viewer): The napari viewer containing the layers to process.
    - metadata (pd.DataFrame): DataFrame to store computed measurements for each image.
    - scaling_factor (float, optional): Scaling factor to convert pixel measurements to physical units.

    Returns:
    - pd.DataFrame: Updated metadata DataFrame with tissue area, tissue area standard deviation,
      and tissue thickness.
    """

    label_image_stack = []

    for layer in viewer.layers:
        if not layer.visible or not layer.name.endswith('_label'):
            continue

        label = layer.data
        img_name = layer.name.split('_label')[0]
        img = viewer.layers[img_name].data
        
        # Identify tissue and lumen labels
        lumen = label == lumen_label_value
        tissue = label == tissue_label_value

        # Label connected tissue regions and find boundaries
        tissue_label = ski_measure.label(tissue)
        tissue_boundaries = find_boundaries(tissue_label)
        lumen_boundaries = find_boundaries(lumen)
        touching_lumen = np.logical_and(tissue_boundaries, lumen)

        # Dilate boundaries to find overlapping regions
        footprint = ski_morphology.ball(1)
        overlap = np.unique(tissue_label[ski_morphology.binary_dilation(touching_lumen, footprint=footprint)])
        overlap = overlap[overlap != 0]

        # If multiple overlapping regions, select the largest one
        if len(overlap) > 1:
            overlap = [max(overlap, key=lambda x: np.sum(tissue_label == x))]

        # Create binary label image for tissue and lumen
        tissue_label = tissue_label == overlap[0]
        label_image = np.zeros_like(tissue_label, dtype=np.uint8)
        label_image[tissue_label != 0] = 2
        label_image[lumen] = 1
        label_image_stack.append(label_image)

        # Calculate tissue area per slice
        tissue_area = []
        for slice in tissue_label:
            if np.sum(slice) == 0:
                tissue_area.append(0)
                continue
            area = np.sum(slice) * scaling_factor**2
            tissue_area.append(area)

        # Update metadata with tissue area statistics
        metadata.loc[img_name, 'tissue_area'] = np.mean(tissue_area)
        metadata.loc[img_name, 'tissue_area_sd'] = np.std(tissue_area)

    # Compute total area and thickness
    lumen_area = metadata.get('area', pd.Series(0, index=metadata.index))
    area_total = (lumen_area + metadata['tissue_area']).fillna(0).astype(np.float64)
    radius_total = np.sqrt(area_total / np.pi)
    tissue_thickness = radius_total - metadata.get('radius', pd.Series(0, index=metadata.index))

    # Update metadata with tissue thickness
    metadata['tissue_thickness'] = tissue_thickness.astype(np.float64)

    return metadata

    