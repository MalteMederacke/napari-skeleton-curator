# Skeleton curator for napari

This pakage provides tooling intended to inspect and curate image skeletons in napari.  
Additionally, slicing and exporting of images along edges is supported.

Tooling includes:
- inspect skeleton
- inspect subgraphs
- compare with original image
- remove nodes
- remove edges
- measure length of edges
- measure angles between edges
- quantify lobe dimensions
- export image along edges


In principle it is extandable to any networkx compatible graph structure. 



## Installation
It is recommended to install this package in a virtual environment.

    conda create -n napari-skeleton-curator python=3.9.9  
    conda activate napari-skeleton-curator

You can install `napari-skeleton-curator` via [pip]:

    pip install napari-skeleton-curator

Or the latest 

## Usage
`code/skeleton.py` contains the functionallity  
`examples/skeleton_merge_nodes.ipynb` is a notebook that demonstrates the data type and workflow.