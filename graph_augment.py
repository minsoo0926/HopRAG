import argparse
import itertools
import json
import os
import re
import sys
import time
from collections import Counter

import networkx as nx
import numpy as np
from neo4j import GraphDatabase

from config import (
    augmentation_bridge_question_prompt,
    create_edge_dense_index_template,
    create_edge_sparse_index_template,
    create_node_dense_index_template,
    create_node_sparse_index_template,
    edge_name,
    embed_model,
    embed_dim,
    local_model_name,
    neo4j_dbname,
    neo4j_notification_filter,
    neo4j_password,
    neo4j_url,
    neo4j_user,
    node_name,
)


SAFE_NEO4J_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_neo4j_name(name, kind):
    if not SAFE_NEO4J_NAME.match(name):
        raise ValueError(
            f"Invalid Neo4j {kind}: {name!r}. Use only letters, numbers, and underscores; "
            "the first character must be a letter or underscore."
        )
    return name


def count_graph(session, label, relationship_type):
    node_count = session.run(
        f"""
        MATCH (n:{label})
        RETURN count(n) AS count
        """
    ).single()["count"]
    relationship_count = session.run(
        f"""
        MATCH (a:{label})-[r:{relationship_type}]->(b:{label})
        RETURN count(r) AS count
        """
    ).single()["count"]
    return node_count, relationship_count


def count_augmented_relationships(session, label, relationship_type):
    result = session.run(
        f"""
        MATCH (a:{label})-[r:{relationship_type}]->(b:{label})
        WHERE coalesce(r.augmentation_edge, false) = true
        RETURN count(r) AS count
        """
    )
    return result.single()["count"]


def drop_graph(session, label):
    session.run(
        f"""
        MATCH (n:{label})
        DETACH DELETE n
        """
    )


def clone_nodes(session, source_label, target_label):
    result = session.run(
        f"""
        MATCH (source:{source_label})
        CREATE (clone:{target_label})
        SET clone = properties(source)
        SET clone.source_element_id = elementId(source),
            clone.source_label = $source_label,
            clone.graph_variant = $target_label
        RETURN count(clone) AS count
        """,
        source_label=source_label,
        target_label=target_label,
    )
    return result.single()["count"]


def clone_relationships(session, source_label, source_relationship, target_label, target_relationship):
    result = session.run(
        f"""
        MATCH (source_a:{source_label})-[source_r:{source_relationship}]->(source_b:{source_label})
        MATCH (clone_a:{target_label} {{source_element_id: elementId(source_a)}})
        MATCH (clone_b:{target_label} {{source_element_id: elementId(source_b)}})
        CREATE (clone_a)-[clone_r:{target_relationship}]->(clone_b)
        SET clone_r = properties(source_r)
        SET clone_r.source_element_id = elementId(source_r),
            clone_r.source_relationship_type = $source_relationship,
            clone_r.graph_variant = $target_label
        RETURN count(clone_r) AS count
        """,
        source_relationship=source_relationship,
        target_label=target_label,
    )
    return result.single()["count"]


def create_indexes(session, target_label, target_relationship):
    index_specs = [
        (
            f"{target_label}_node_dense_index",
            create_node_dense_index_template.format(
                name=f"{target_label}_node_dense_index",
                property="embed",
                dim=embed_dim,
                type=target_label,
            ),
        ),
        (
            f"{target_label}_edge_dense_index",
            create_edge_dense_index_template.format(
                name=f"{target_label}_edge_dense_index",
                property="embed",
                dim=embed_dim,
                type=target_relationship,
            ),
        ),
        (
            f"{target_label}_node_sparse_index",
            create_node_sparse_index_template.format(
                name=f"{target_label}_node_sparse_index",
                property="text",
                type=target_label,
            ),
        ),
        (
            f"{target_label}_edge_sparse_index",
            create_edge_sparse_index_template.format(
                name=f"{target_label}_edge_sparse_index",
                property="question",
                type=target_relationship,
            ),
        ),
    ]
    for index_name, cypher in index_specs:
        validate_neo4j_name(index_name, "index name")
        session.run(cypher)


def fetch_graph_for_augmentation(session, label, relationship_type):
    graph = nx.Graph()

    node_result = session.run(
        f"""
        MATCH (n:{label})
        RETURN elementId(n) AS id,
               n.text AS text,
               n.keywords AS keywords,
               n.embed AS embed
        """
    )
    for record in node_result:
        embed = record["embed"]
        if embed is None:
            continue
        graph.add_node(
            record["id"],
            text=record["text"] or "",
            keywords=list(record["keywords"] or []),
            embed=np.array(embed, dtype=np.float32),
        )

    edge_result = session.run(
        f"""
        MATCH (a:{label})-[r:{relationship_type}]->(b:{label})
        RETURN elementId(a) AS source, elementId(b) AS target
        """
    )
    for record in edge_result:
        source = record["source"]
        target = record["target"]
        if source in graph and target in graph:
            graph.add_edge(source, target)

    return graph


def r_neighborhood(graph, node, radius):
    return set(nx.single_source_shortest_path_length(graph, node, cutoff=radius).keys())


def greedy_r_dominating_set(graph, radius):
    if radius < 0:
        raise ValueError("radius must be >= 0")

    dominating_nodes = []
    for component_nodes in nx.connected_components(graph):
        component = graph.subgraph(component_nodes)
        uncovered = set(component.nodes())
        neighborhoods = {
            node: r_neighborhood(component, node, radius)
            for node in component.nodes()
        }

        while uncovered:
            best_node = max(
                component.nodes(),
                key=lambda node: (
                    len(neighborhoods[node] & uncovered),
                    component.degree(node),
                ),
            )
            dominating_nodes.append(best_node)
            uncovered -= neighborhoods[best_node]

    return dominating_nodes


def cosine_similarity(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def normalized_average_embedding(source_embed, target_embed):
    embed = (source_embed + target_embed) / 2.0
    norm = np.linalg.norm(embed)
    if norm > 0:
        embed = embed / norm
    return embed.astype(float).tolist()


def load_question_cache(cache_path):
    if not cache_path or not os.path.exists(cache_path):
        return {}
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_question_cache(cache_path, cache):
    if not cache_path:
        return
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def fallback_bridge_question(shared_keywords):
    if shared_keywords:
        topic = ", ".join(shared_keywords[:3])
        return f"What specific information is provided about {topic}?"
    return "What specific information is provided in the target passage?"


def generate_bridge_question(source_text, target_text, shared_keywords, question_model):
    from tool import get_chat_completion

    prompt = augmentation_bridge_question_prompt.format(
        source_text=str(source_text).replace("\n", " ")[:900],
        target_text=str(target_text).replace("\n", " ")[:900],
        shared_keywords=", ".join(shared_keywords[:12]) if shared_keywords else "None",
    )
    chat = [{"role": "user", "content": prompt}]
    question, _ = get_chat_completion(
        chat,
        keys=["Question"],
        model=question_model,
        max_tokens=160,
    )
    if question is None:
        return fallback_bridge_question(shared_keywords)

    if isinstance(question, list):
        question = question[0] if question else None

    question = str(question or "").strip()
    if not question:
        return fallback_bridge_question(shared_keywords)
    return question


def edge_key(source, target):
    return tuple(sorted((source, target)))


def build_backbone_edges(
    graph,
    backbone_nodes,
    radius,
    k,
    similarity_threshold,
    max_aug_degree,
):
    if len(backbone_nodes) <= 1:
        return []

    existing_edges = {edge_key(source, target) for source, target in graph.edges()}
    pair_scores = {}
    complete_graph = nx.Graph()
    complete_graph.add_nodes_from(backbone_nodes)

    for source, target in itertools.combinations(backbone_nodes, 2):
        similarity = cosine_similarity(graph.nodes[source]["embed"], graph.nodes[target]["embed"])
        pair_scores[edge_key(source, target)] = similarity
        complete_graph.add_edge(source, target, weight=1.0 - similarity)

    selected = {}
    aug_degree = Counter()
    mst = nx.minimum_spanning_tree(complete_graph, weight="weight")
    for source, target in mst.edges():
        key = edge_key(source, target)
        if key in existing_edges:
            continue
        similarity = pair_scores[key]
        selected[key] = {
            "source": source,
            "target": target,
            "similarity": similarity,
            "edge_kind": "mst",
        }
        aug_degree[source] += 1
        aug_degree[target] += 1

    if k <= 0:
        return list(selected.values())

    for source in backbone_nodes:
        candidates = []
        for target in backbone_nodes:
            if source == target:
                continue
            key = edge_key(source, target)
            if key in existing_edges or key in selected:
                continue
            similarity = pair_scores[key]
            if similarity < similarity_threshold:
                continue
            candidates.append((similarity, target))

        candidates.sort(reverse=True)
        added_for_source = 0
        for similarity, target in candidates:
            if added_for_source >= k:
                break
            if max_aug_degree is not None:
                if aug_degree[source] >= max_aug_degree or aug_degree[target] >= max_aug_degree:
                    continue

            key = edge_key(source, target)
            selected[key] = {
                "source": source,
                "target": target,
                "similarity": similarity,
                "edge_kind": "knn",
            }
            aug_degree[source] += 1
            aug_degree[target] += 1
            added_for_source += 1

    for row in selected.values():
        row["radius"] = radius
        row["k"] = k
    return list(selected.values())


def build_augmented_relationship_rows(
    graph,
    edges,
    method,
    generate_questions=False,
    question_cache_path=None,
    question_model=local_model_name,
    embedding_model_name=embed_model,
):
    question_cache = load_question_cache(question_cache_path)
    embedder = None
    if generate_questions:
        from tool import get_doc_embeds, get_ner_eng, load_embed_model

        embedder = load_embed_model(embedding_model_name)

    rows = []
    for idx, edge in enumerate(edges, start=1):
        source = edge["source"]
        target = edge["target"]
        source_keywords = list(graph.nodes[source].get("keywords") or [])
        target_keywords = list(graph.nodes[target].get("keywords") or [])
        shared_keywords = sorted(set(source_keywords) & set(target_keywords))
        keywords = list(dict.fromkeys(shared_keywords + source_keywords + target_keywords))[:24]

        if generate_questions:
            cache_key = f"{source}->{target}"
            if cache_key in question_cache:
                question = question_cache[cache_key]["question"]
            else:
                question = generate_bridge_question(
                    graph.nodes[source].get("text", ""),
                    graph.nodes[target].get("text", ""),
                    shared_keywords,
                    question_model,
                )
                question_cache[cache_key] = {
                    "source": source,
                    "target": target,
                    "question": question,
                }
                save_question_cache(question_cache_path, question_cache)
                print(f"generated bridge question {idx}/{len(edges)}: {question}")
            question_keywords = list(get_ner_eng(question))
            keywords = list(dict.fromkeys(question_keywords + keywords))[:24]
            edge_embed = get_doc_embeds(question, embedder)
        else:
            question = fallback_bridge_question(shared_keywords)
            edge_embed = normalized_average_embedding(
                graph.nodes[source]["embed"],
                graph.nodes[target]["embed"],
            )

        rows.append(
            {
                "source": source,
                "target": target,
                "question": question,
                "keywords": keywords,
                "embed": edge_embed,
                "similarity": edge["similarity"],
                "edge_kind": edge["edge_kind"],
                "method": method,
                "radius": edge["radius"],
                "k": edge["k"],
            }
        )
    return rows


def upload_augmented_relationships(session, label, relationship_type, rows):
    if not rows:
        return 0

    result = session.run(
        f"""
        UNWIND $rows AS row
        MATCH (source:{label})
        WHERE elementId(source) = row.source
        MATCH (target:{label})
        WHERE elementId(target) = row.target
        CREATE (source)-[r:{relationship_type}]->(target)
        SET r.question = row.question,
            r.keywords = row.keywords,
            r.embed = row.embed,
            r.augmentation_edge = true,
            r.augmentation_method = row.method,
            r.augmentation_edge_kind = row.edge_kind,
            r.augmentation_r = row.radius,
            r.augmentation_k = row.k,
            r.augmentation_similarity = row.similarity
        RETURN count(r) AS count
        """,
        rows=rows,
    )
    return result.single()["count"]


def augment_graph(
    label,
    relationship_type,
    radius,
    k,
    similarity_threshold=0.5,
    max_aug_degree=4,
    generate_questions=False,
    question_cache_path=None,
    question_model=local_model_name,
    embedding_model_name=embed_model,
):
    label = validate_neo4j_name(label, "label")
    relationship_type = validate_neo4j_name(relationship_type, "relationship type")
    method = "r_dominating_mst_knn"

    driver = GraphDatabase.driver(
        neo4j_url,
        auth=(neo4j_user, neo4j_password),
        database=neo4j_dbname,
        notifications_disabled_categories=neo4j_notification_filter,
    )

    start = time.time()
    try:
        with driver.session() as session:
            graph = fetch_graph_for_augmentation(session, label, relationship_type)
            if graph.number_of_nodes() == 0:
                raise RuntimeError(f"Graph has no nodes with usable embeddings: {label}")

            existing_aug_edges = count_augmented_relationships(session, label, relationship_type)
            if existing_aug_edges:
                raise RuntimeError(
                    f"{label} already has {existing_aug_edges} augmented relationships. "
                    "Use a fresh clone for a new experiment."
                )

            backbone_nodes = greedy_r_dominating_set(graph, radius)
            candidate_edges = build_backbone_edges(
                graph=graph,
                backbone_nodes=backbone_nodes,
                radius=radius,
                k=k,
                similarity_threshold=similarity_threshold,
                max_aug_degree=max_aug_degree,
            )
            rows = build_augmented_relationship_rows(
                graph,
                candidate_edges,
                method,
                generate_questions=generate_questions,
                question_cache_path=question_cache_path,
                question_model=question_model,
                embedding_model_name=embedding_model_name,
            )
            uploaded_edges = upload_augmented_relationships(session, label, relationship_type, rows)
            final_nodes, final_relationships = count_graph(session, label, relationship_type)
    finally:
        driver.close()

    edge_kinds = Counter(row["edge_kind"] for row in rows)
    similarities = [row["similarity"] for row in rows]
    return {
        "label": label,
        "relationship_type": relationship_type,
        "method": method,
        "radius": radius,
        "k": k,
        "similarity_threshold": similarity_threshold,
        "max_aug_degree": max_aug_degree,
        "generate_questions": generate_questions,
        "question_cache_path": question_cache_path,
        "question_model": question_model,
        "node_count": graph.number_of_nodes(),
        "original_unique_edges": graph.number_of_edges(),
        "component_count": nx.number_connected_components(graph),
        "backbone_node_count": len(backbone_nodes),
        "candidate_edge_count": len(candidate_edges),
        "uploaded_edge_count": uploaded_edges,
        "mst_edge_count": edge_kinds.get("mst", 0),
        "knn_edge_count": edge_kinds.get("knn", 0),
        "min_similarity": min(similarities) if similarities else None,
        "avg_similarity": float(sum(similarities) / len(similarities)) if similarities else None,
        "max_similarity": max(similarities) if similarities else None,
        "final_nodes": final_nodes,
        "final_relationships": final_relationships,
        "elapsed_seconds": time.time() - start,
    }


def clone_graph(
    source_label,
    source_relationship,
    target_label,
    target_relationship,
    replace=False,
    with_indexes=True,
):
    source_label = validate_neo4j_name(source_label, "source label")
    source_relationship = validate_neo4j_name(source_relationship, "source relationship type")
    target_label = validate_neo4j_name(target_label, "target label")
    target_relationship = validate_neo4j_name(target_relationship, "target relationship type")

    driver = GraphDatabase.driver(
        neo4j_url,
        auth=(neo4j_user, neo4j_password),
        database=neo4j_dbname,
        notifications_disabled_categories=neo4j_notification_filter,
    )

    start = time.time()
    try:
        with driver.session() as session:
            source_nodes, source_edges = count_graph(session, source_label, source_relationship)
            target_nodes, target_edges = count_graph(session, target_label, target_relationship)

            if source_nodes == 0:
                raise RuntimeError(f"Source graph has no nodes: {source_label}")

            if target_nodes or target_edges:
                if not replace:
                    raise RuntimeError(
                        f"Target graph already exists: {target_label} has {target_nodes} nodes "
                        f"and {target_edges} relationships. Use --replace to rebuild it."
                    )
                drop_graph(session, target_label)

            cloned_nodes = clone_nodes(session, source_label, target_label)
            cloned_edges = clone_relationships(
                session,
                source_label,
                source_relationship,
                target_label,
                target_relationship,
            )

            if with_indexes:
                create_indexes(session, target_label, target_relationship)

            final_nodes, final_edges = count_graph(session, target_label, target_relationship)
    finally:
        driver.close()

    return {
        "source_label": source_label,
        "source_relationship": source_relationship,
        "target_label": target_label,
        "target_relationship": target_relationship,
        "source_nodes": source_nodes,
        "source_relationships": source_edges,
        "cloned_nodes": cloned_nodes,
        "cloned_relationships": cloned_edges,
        "target_nodes": final_nodes,
        "target_relationships": final_edges,
        "indexes_created": with_indexes,
        "elapsed_seconds": time.time() - start,
    }


def print_clone_result(result):
    print("=" * 80)
    print("Graph Clone Result")
    print("=" * 80)
    print(f"Source label: {result['source_label']}")
    print(f"Source relationship: {result['source_relationship']}")
    print(f"Target label: {result['target_label']}")
    print(f"Target relationship: {result['target_relationship']}")
    print("-" * 80)
    print(f"Source nodes: {result['source_nodes']}")
    print(f"Source relationships: {result['source_relationships']}")
    print(f"Cloned nodes: {result['cloned_nodes']}")
    print(f"Cloned relationships: {result['cloned_relationships']}")
    print(f"Final target nodes: {result['target_nodes']}")
    print(f"Final target relationships: {result['target_relationships']}")
    print(f"Indexes created: {result['indexes_created']}")
    print(f"Elapsed seconds: {result['elapsed_seconds']:.2f}")
    print("=" * 80)


def print_augment_result(result):
    print("=" * 80)
    print("Graph Augmentation Result")
    print("=" * 80)
    print(f"Label: {result['label']}")
    print(f"Relationship: {result['relationship_type']}")
    print(f"Method: {result['method']}")
    print(f"r: {result['radius']}")
    print(f"k: {result['k']}")
    print(f"Similarity threshold: {result['similarity_threshold']}")
    print(f"Max augmented degree: {result['max_aug_degree']}")
    print(f"Generated bridge questions: {result['generate_questions']}")
    if result["generate_questions"]:
        print(f"Question model: {result['question_model']}")
        print(f"Question cache: {result['question_cache_path']}")
    print("-" * 80)
    print(f"Nodes: {result['node_count']}")
    print(f"Original unique graph edges: {result['original_unique_edges']}")
    print(f"Connected components before augmentation: {result['component_count']}")
    print(f"Backbone nodes: {result['backbone_node_count']}")
    print(f"Candidate augmented edges: {result['candidate_edge_count']}")
    print(f"Uploaded augmented edges: {result['uploaded_edge_count']}")
    print(f"MST edges: {result['mst_edge_count']}")
    print(f"kNN shortcut edges: {result['knn_edge_count']}")
    print("-" * 80)
    if result["uploaded_edge_count"]:
        print(f"Min similarity: {result['min_similarity']:.4f}")
        print(f"Avg similarity: {result['avg_similarity']:.4f}")
        print(f"Max similarity: {result['max_similarity']:.4f}")
    print(f"Final nodes: {result['final_nodes']}")
    print(f"Final relationships: {result['final_relationships']}")
    print(f"Elapsed seconds: {result['elapsed_seconds']:.2f}")
    print("=" * 80)


def add_clone_args(parser):
    parser.add_argument("--source-label", default=node_name)
    parser.add_argument("--source-relationship", default=edge_name)
    parser.add_argument("--target-label", default=None)
    parser.add_argument("--target-relationship", default=None)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Delete only the target graph before cloning if it already exists.",
    )
    parser.add_argument(
        "--no-indexes",
        action="store_true",
        help="Skip vector/fulltext index creation on the cloned graph.",
    )


def add_augment_args(parser):
    parser.add_argument("--label", default=node_name)
    parser.add_argument("--relationship", default=edge_name)
    parser.add_argument("--r", type=int, default=1, help="Radius for the r-dominating set.")
    parser.add_argument("--k", type=int, default=2, help="Top-k shortcut edges per backbone node.")
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.5,
        help="Minimum cosine similarity for kNN shortcut edges. MST edges are always kept.",
    )
    parser.add_argument(
        "--max-aug-degree",
        type=int,
        default=4,
        help="Maximum augmented degree per backbone node. Use -1 for no cap.",
    )
    parser.add_argument(
        "--generate-questions",
        action="store_true",
        help="Generate bridge questions with the configured local LLM and use question embeddings.",
    )
    parser.add_argument(
        "--question-cache",
        default=None,
        help="JSON cache path for generated bridge questions.",
    )
    parser.add_argument(
        "--question-model",
        default=local_model_name,
        help="LLM used to generate augmentation bridge questions.",
    )
    parser.add_argument(
        "--embedding-model",
        default=embed_model,
        help="Embedding model used for generated bridge question embeddings.",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clone and augment HopRAG Neo4j graphs."
    )
    subparsers = parser.add_subparsers(dest="command")

    clone_parser = subparsers.add_parser(
        "clone",
        help="Clone a graph into a separate label/relationship pair.",
    )
    add_clone_args(clone_parser)

    augment_parser = subparsers.add_parser(
        "augment",
        help="Add r-dominating-set MST+kNN augmentation edges to a cloned graph.",
    )
    add_augment_args(augment_parser)

    if len(sys.argv) == 1 or (
        len(sys.argv) > 1 and sys.argv[1] not in {"clone", "augment", "-h", "--help"}
    ):
        legacy_parser = argparse.ArgumentParser(
            description="Clone a HopRAG Neo4j graph into a separate label/relationship pair."
        )
        add_clone_args(legacy_parser)
        args = legacy_parser.parse_args()
        args.command = "clone"
        return args

    return parser.parse_args()


def main():
    args = parse_args()
    command = args.command or "clone"

    if command == "clone":
        target_label = args.target_label or f"{args.source_label}_aug_clone"
        target_relationship = args.target_relationship or f"{args.source_relationship}_aug_clone"
        result = clone_graph(
            source_label=args.source_label,
            source_relationship=args.source_relationship,
            target_label=target_label,
            target_relationship=target_relationship,
            replace=args.replace,
            with_indexes=not args.no_indexes,
        )
        print_clone_result(result)
        return

    if command == "augment":
        max_aug_degree = None if args.max_aug_degree < 0 else args.max_aug_degree
        result = augment_graph(
            label=args.label,
            relationship_type=args.relationship,
            radius=args.r,
            k=args.k,
            similarity_threshold=args.similarity_threshold,
            max_aug_degree=max_aug_degree,
            generate_questions=args.generate_questions,
            question_cache_path=args.question_cache,
            question_model=args.question_model,
            embedding_model_name=args.embedding_model,
        )
        print_augment_result(result)
        return

    raise RuntimeError(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
