"""
Visualization utilities for HRHUB.
Handles network graph creation using PyVis.
"""

from pyvis.network import Network
import streamlit as st
from typing import Dict, Any, List
import tempfile
import os


def create_network_graph(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    height: str = "600px",
    width: str = "100%"
) -> str:
    """
    Create interactive network graph using PyVis.
    
    Args:
        nodes: List of node dictionaries with id, label, color, etc.
        edges: List of edge dictionaries with from, to, value, etc.
        height: Graph height (CSS format)
        width: Graph width (CSS format)
    
    Returns:
        HTML string of the network graph
    """
    
    # Initialize network
    net = Network(
        height=height,
        width=width,
        bgcolor="#1E1E1E",  # Dark background
        font_color="#FFFFFF",
        notebook=False
    )
    
    # Configure physics for better layout
    net.set_options("""
    {
        "physics": {
            "enabled": true,
            "barnesHut": {
                "gravitationalConstant": -15000,
                "centralGravity": 0.3,
                "springLength": 200,
                "springConstant": 0.04,
                "damping": 0.09,
                "avoidOverlap": 0.5
            },
            "minVelocity": 0.75,
            "solver": "barnesHut"
        },
        "nodes": {
            "font": {
                "size": 14,
                "face": "Arial",
                "color": "#FFFFFF"
            },
            "borderWidth": 2,
            "borderWidthSelected": 4
        },
        "edges": {
            "color": {
                "color": "#FFFFFF",
                "highlight": "#00FF00"
            },
            "smooth": {
                "type": "continuous",
                "forceDirection": "none"
            },
            "width": 2
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "zoomView": true,
            "dragView": true
        }
    }
    """)
    
    # Add nodes
    for node in nodes:
        net.add_node(
            node['id'],
            label=node.get('label', ''),
            title=node.get('title', ''),
            color=node.get('color', '#FFFFFF'),
            shape=node.get('shape', 'dot'),
            size=node.get('size', 20)
        )
    
    # Add edges
    for edge in edges:
        # Calculate width based on score/value
        width = edge.get('value', 0.5) * 5
        
        net.add_edge(
            edge['from'],
            edge['to'],
            title=edge.get('title', ''),
            value=width,
            color=edge.get('color', {'opacity': 0.8})
        )
    
    # Generate HTML
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.html', encoding='utf-8') as f:
        net.save_graph(f.name)
        with open(f.name, 'r', encoding='utf-8') as html_file:
            html_content = html_file.read()
        os.unlink(f.name)
    
    return html_content


def display_network_in_streamlit(html_content: str, height: int = 600):
    """
    Display PyVis network graph in Streamlit using components.html.
    
    Args:
        html_content: HTML string from create_network_graph
        height: Height of the display area in pixels
    """
    
    st.components.v1.html(html_content, height=height, scrolling=False)
