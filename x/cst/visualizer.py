# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math

from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

"""
Node Abstract Class
Class OperatorNode(Node):
   name: str
   start_time: int
   end_time: int
   children: List[Node]
   runtimes: List[RuntimeNode]
   input_shape: List[List[int]]
   input_type: List[str]
   callstack: str
   self_host_duration: int
   self_device_duration: int
"""


def draw_tree(root_node, img_path="/tmp/output.png", verbose=False):
    """
    Draw the tree structure of the root node
    show the name, start_time, end_time
    """
    
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    # Calculate dimensions of the tree 
    # add the max_children_width for each node
    # get the node_count_per_layer for each layer
    # set the index of each node at each layer
    def calculate_tree_dimensions(node, depth=0, horizontal_pos=0):
        nonlocal max_depth, max_end_time
        
        # Initialize layer node count if not exists
        if depth not in node_count_per_layer:
            node_count_per_layer[depth] = 0
        
        # Set node index in its layer
        node.layer_index = node_count_per_layer[depth]
        node_count_per_layer[depth] += 1
        
        # Update tree dimensions
        max_depth = max(max_depth, depth)
        max_end_time = max(max_end_time, node.end_time) if hasattr(node, 'end_time') else max_end_time
        
        width = len(node.children)
        if width == 0:
            return 1
        
        child_width = 0
        for child in node.children:
            child_width += calculate_tree_dimensions(child, depth+1, horizontal_pos+child_width)
        
        node.max_children_width = max(width, child_width)
        return max(1, child_width)
    
    max_depth = 0
    max_end_time = 0
    node_count_per_layer = {}
    calculate_tree_dimensions(root_node)
    
    base_time = root_node.start_time
    
    # Determine position
    box_width = 90
    box_height = 60
    margin = 10
    
    max_layer_width = max(node_count_per_layer.values())
    
    # Create the image
    img_width = (2 + max_layer_width) * (box_width + margin) + margin
    img_height = max_depth * (box_height + margin)*2 + margin
    vertical_spacing = img_height / (max_depth + 2)
    
    img = Image.new(mode="RGB", size=(img_width, img_height), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    
    # Draw the nodes
    def draw_node(node, depth=0, horizontal_pos=0, parent_x=None, parent_y=None):
        nonlocal node_count
        
        node_index = node.layer_index if node_count_per_layer[depth] > 1 else 0.5
        horizontal_pos = (node_index / node_count_per_layer[depth]) * max_layer_width
        x = horizontal_pos * (box_width + margin) + margin
        
        y = depth * vertical_spacing + margin
        
        # Ensure x is within image bounds
        x = min(x, img_width - box_width - margin)
        
        # Draw connection to parent
        if parent_x is not None and parent_y is not None:
            draw.line((parent_x + box_width/2, parent_y + box_height, 
                       x + box_width/2, y), fill=(100, 100, 100), width=2)
        
        # Draw the node
        node_color = (230, 230, 255)  # Light blue
        if depth == 0:
            node_color = (200, 255, 200)  # Light green for root
            
        # Draw box with rounded corners
        draw.rounded_rectangle((x, y, x + box_width, y + box_height), radius=8, 
                              fill=node_color, outline=(0, 0, 0))
        
        # Draw node info
        if hasattr(node, 'name'):
            draw.text((x + 5, y + 5), node.name[:20], fill=(0, 0, 0))
        
        if hasattr(node, 'start_time') and hasattr(node, 'end_time'):
            time_text = f"S: {node.start_time - base_time:.2f}\nE: {node.end_time - base_time:.2f}"
            draw.text((x + 5, y + 25), time_text, fill=(0, 0, 0))
            
            # # Show duration
            # if hasattr(node, 'self_host_duration') or hasattr(node, 'self_device_duration'):
            #     host_dur = getattr(node, 'self_host_duration', 0)
            #     device_dur = getattr(node, 'self_device_duration', 0)
            #     dur_text = f"H: {host_dur}, D: {device_dur}"
            #     draw.text((x + 5, y + 40), dur_text[:18], fill=(0, 0, 0))
        
        node_count += 1
        
        # Draw children
        child_horizontal_pos = horizontal_pos
        for child in node.children:
            child_width = draw_node(child, depth+1, child_horizontal_pos, x, y)
            child_horizontal_pos += child_width
            
        return max(1, len(node.children))
    
    node_count = 0
    draw_node(root_node)
    
    logger.info(f"Drew tree with {node_count} nodes")
    logger.info(f"Saving image to {img_path}")
    img.save(img_path, "PNG")
    
    return img_path