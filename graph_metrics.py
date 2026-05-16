import argparse
import json
import math
import os
import time
from collections import Counter
from datetime import datetime

import networkx as nx
from neo4j import GraphDatabase

from config import (
    edge_name,
    neo4j_dbname,
    neo4j_password,
    neo4j_url,
    neo4j_user,
    node_name,
)


def fetch_graph(label, relationship_type, undirected=True):
    driver = GraphDatabase.driver(
        neo4j_url,
        auth=(neo4j_user, neo4j_password),
        database=neo4j_dbname,
    )
    graph = nx.Graph() if undirected else nx.DiGraph()

    with driver.session() as session:
        node_result = session.run(
            f"""
            MATCH (n:{label})
            RETURN elementId(n) AS id
            """
        )
        node_ids = [record["id"] for record in node_result]
        graph.add_nodes_from(node_ids)

        edge_result = session.run(
            f"""
            MATCH (a:{label})-[r:{relationship_type}]->(b:{label})
            RETURN elementId(a) AS source, elementId(b) AS target
            """
        )
        edge_rows = [(record["source"], record["target"]) for record in edge_result]
        graph.add_edges_from(edge_rows)

    driver.close()
    graph.graph["neo4j_node_count"] = len(node_ids)
    graph.graph["neo4j_relationship_count"] = len(edge_rows)
    return graph


def _safe_average(values):
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _component_metrics(graph):
    if graph.number_of_nodes() == 0:
        return {
            "connected_component_count": 0,
            "largest_component_size": 0,
            "largest_component_ratio": 0.0,
            "isolated_node_count": 0,
            "component_sizes": [],
        }

    if graph.is_directed():
        components = list(nx.weakly_connected_components(graph))
    else:
        components = list(nx.connected_components(graph))

    component_sizes = sorted((len(c) for c in components), reverse=True)
    largest_size = component_sizes[0] if component_sizes else 0

    return {
        "connected_component_count": len(component_sizes),
        "largest_component_size": largest_size,
        "largest_component_ratio": largest_size / graph.number_of_nodes(),
        "isolated_node_count": nx.number_of_isolates(graph),
        "component_sizes": component_sizes,
    }


def _path_metrics(graph, sample_size=None, seed=13):
    if graph.number_of_nodes() == 0:
        return {
            "average_shortest_path_length_lcc": None,
            "diameter_lcc": None,
            "radius_lcc": None,
            "approx_average_shortest_path_length_lcc": None,
            "approx_eccentricity_sample_size": 0,
        }

    if graph.is_directed():
        components = list(nx.weakly_connected_components(graph))
        graph_for_paths = graph.to_undirected()
    else:
        components = list(nx.connected_components(graph))
        graph_for_paths = graph

    if not components:
        return {
            "average_shortest_path_length_lcc": None,
            "diameter_lcc": None,
            "radius_lcc": None,
            "approx_average_shortest_path_length_lcc": None,
            "approx_eccentricity_sample_size": 0,
        }

    largest_nodes = max(components, key=len)
    lcc = graph_for_paths.subgraph(largest_nodes).copy()

    if lcc.number_of_nodes() <= 1:
        return {
            "average_shortest_path_length_lcc": 0.0,
            "diameter_lcc": 0,
            "radius_lcc": 0,
            "approx_average_shortest_path_length_lcc": 0.0,
            "approx_eccentricity_sample_size": lcc.number_of_nodes(),
        }

    exact_threshold = 2000
    if sample_size is None and lcc.number_of_nodes() <= exact_threshold:
        return {
            "average_shortest_path_length_lcc": nx.average_shortest_path_length(lcc),
            "diameter_lcc": nx.diameter(lcc),
            "radius_lcc": nx.radius(lcc),
            "approx_average_shortest_path_length_lcc": None,
            "approx_eccentricity_sample_size": 0,
        }

    sample_count = sample_size or min(256, lcc.number_of_nodes())
    sample_nodes = list(lcc.nodes())
    rng = nx.utils.create_random_state(seed)
    sampled = list(rng.choice(sample_nodes, size=min(sample_count, len(sample_nodes)), replace=False))

    distances = []
    eccentricities = []
    for source in sampled:
        lengths = nx.single_source_shortest_path_length(lcc, source)
        distances.extend(length for target, length in lengths.items() if target != source)
        eccentricities.append(max(lengths.values()) if lengths else 0)

    return {
        "average_shortest_path_length_lcc": None,
        "diameter_lcc": max(eccentricities) if eccentricities else None,
        "radius_lcc": min(eccentricities) if eccentricities else None,
        "approx_average_shortest_path_length_lcc": _safe_average(distances),
        "approx_eccentricity_sample_size": len(sampled),
    }


def compute_graph_metrics(graph, label, relationship_type, sample_size=None):
    start = time.time()
    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    degrees = [degree for _, degree in graph.degree()]
    degree_counter = Counter(degrees)

    component_metrics = _component_metrics(graph)
    path_metrics = _path_metrics(graph, sample_size=sample_size)

    density = nx.density(graph) if node_count > 1 else 0.0
    average_degree = _safe_average(degrees)
    max_possible_edges = node_count * (node_count - 1)
    if not graph.is_directed():
        max_possible_edges = max_possible_edges / 2

    metrics = {
        "label": label,
        "relationship_type": relationship_type,
        "directed": graph.is_directed(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "node_count": node_count,
        "edge_count": edge_count,
        "neo4j_node_count": graph.graph.get("neo4j_node_count", node_count),
        "neo4j_relationship_count": graph.graph.get("neo4j_relationship_count", edge_count),
        "max_possible_edges": int(max_possible_edges),
        "density": density,
        "average_degree": average_degree,
        "min_degree": min(degrees) if degrees else 0,
        "max_degree": max(degrees) if degrees else 0,
        "median_degree": float(sorted(degrees)[len(degrees) // 2]) if degrees else 0.0,
        "degree_histogram": dict(sorted(degree_counter.items())),
        "self_loop_count": nx.number_of_selfloops(graph),
        "average_clustering": nx.average_clustering(graph.to_undirected()) if node_count else 0.0,
        "transitivity": nx.transitivity(graph.to_undirected()) if node_count else 0.0,
        "compute_seconds": None,
    }
    metrics.update(component_metrics)
    metrics.update(path_metrics)
    metrics["compute_seconds"] = time.time() - start
    return metrics


def print_metrics(metrics):
    print("=" * 80)
    print("Graph Metrics")
    print("=" * 80)
    print(f"Label: {metrics['label']}")
    print(f"Relationship: {metrics['relationship_type']}")
    print(f"Directed: {metrics['directed']}")
    print("-" * 80)
    print(f"Nodes: {metrics['node_count']}")
    print(f"Unique graph edges: {metrics['edge_count']}")
    print(f"Neo4j relationships: {metrics['neo4j_relationship_count']}")
    print(f"Density: {metrics['density']:.6f}")
    print(f"Average degree: {metrics['average_degree']:.4f}")
    print(f"Min degree: {metrics['min_degree']}")
    print(f"Median degree: {metrics['median_degree']}")
    print(f"Max degree: {metrics['max_degree']}")
    print(f"Self loops: {metrics['self_loop_count']}")
    print("-" * 80)
    print(f"Connected components: {metrics['connected_component_count']}")
    print(f"Largest component size: {metrics['largest_component_size']}")
    print(f"Largest component ratio: {metrics['largest_component_ratio']:.4f}")
    print(f"Isolated nodes: {metrics['isolated_node_count']}")
    print(f"Top component sizes: {metrics['component_sizes'][:10]}")
    print("-" * 80)
    print(f"Average clustering: {metrics['average_clustering']:.6f}")
    print(f"Transitivity: {metrics['transitivity']:.6f}")
    print("-" * 80)
    if metrics["average_shortest_path_length_lcc"] is not None:
        print(
            "Average shortest path length, LCC: "
            f"{metrics['average_shortest_path_length_lcc']:.6f}"
        )
    else:
        print(
            "Approx. average shortest path length, LCC: "
            f"{metrics['approx_average_shortest_path_length_lcc']:.6f}"
        )
        print(f"Path sample size: {metrics['approx_eccentricity_sample_size']}")
    print(f"Diameter, LCC: {metrics['diameter_lcc']}")
    print(f"Radius, LCC: {metrics['radius_lcc']}")
    print("-" * 80)
    print(f"Degree histogram: {metrics['degree_histogram']}")
    print(f"Compute seconds: {metrics['compute_seconds']:.4f}")
    print("=" * 80)


def save_metrics(metrics, output_path):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute graph metrics for a HopRAG Neo4j graph.")
    parser.add_argument("--label", default=node_name, help="Neo4j node label to analyze.")
    parser.add_argument("--relationship", default=edge_name, help="Neo4j relationship type to analyze.")
    parser.add_argument("--directed", action="store_true", help="Compute metrics on a directed graph.")
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Sample size for path metrics. Exact path metrics are used for small graphs by default.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path. Defaults to outputs/metrics/<label>_graph_metrics.json.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = args.output or f"outputs/metrics/{args.label}_graph_metrics.json"

    graph = fetch_graph(
        label=args.label,
        relationship_type=args.relationship,
        undirected=not args.directed,
    )
    metrics = compute_graph_metrics(
        graph,
        label=args.label,
        relationship_type=args.relationship,
        sample_size=args.sample_size,
    )
    print_metrics(metrics)
    save_metrics(metrics, output_path)
    print(f"Saved metrics to: {output_path}")


if __name__ == "__main__":
    main()
