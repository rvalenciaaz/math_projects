"""
separate_components.py

This script reads a GraphML file, extracts each connected component,
and writes each component to its own GraphML file in the specified output
directory.

Usage:
    python separate_components.py input_graphml_file.graphml --output_dir components/
"""

import networkx as nx
import os
import argparse


def separate_components(input_graphml: str, output_dir: str) -> None:
    # Read the graph
    G = nx.read_graphml(input_graphml)

    # Choose the correct component function
    if G.is_directed():
        components = nx.weakly_connected_components(G)
    else:
        components = nx.connected_components(G)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over components and write each to its own GraphML file
    for idx, comp_nodes in enumerate(components, start=1):
        H = G.subgraph(comp_nodes).copy()
        output_path = os.path.join(output_dir, f"component_{idx}.graphml")
        nx.write_graphml(H, output_path)
        print(f"Saved component {idx}: {H.number_of_nodes()} nodes, {H.number_of_edges()} edges -> {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Separate a GraphML file into its connected components."
    )
    parser.add_argument(
        "input_graphml",
        help="Path to the input GraphML file (.graphml)."
    )
    parser.add_argument(
        "-o", "--output_dir",
        default="components",
        help="Directory where component GraphMLs will be saved."
    )
    args = parser.parse_args()

    separate_components(args.input_graphml, args.output_dir)


if __name__ == "__main__":
    main()
