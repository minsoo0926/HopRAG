import heapq
import numpy as np
from loguru import logger
from typing import Any, List, Tuple, Dict, Set
from config import expand_logic_query
from tool import sparse_similarity


class HopQMixin:
    # Stubs for attributes/methods provided by HopRetriever
    topk: int
    max_hop: int
    epsilon: float
    driver: Any

    def process_query(self, query: str) -> Tuple[List, str]:
        raise NotImplementedError

    def search_docs_mock(self, query_embedding: List, query_keywords: str, topk: int) -> Tuple:
        raise NotImplementedError

    def topk_filter(self, sim_dict: Dict[str, float]) -> Tuple[List[str], List[float]]:
        raise NotImplementedError

    def _hopq_prune(self, C_score: Dict[str, float], node2emb: Dict[str, list],
                    node2kw: Dict[str, set], q_emb: np.ndarray,
                    query_keywords: str) -> Tuple[List[str], List[float]]:
        texts = list(C_score.keys())
        embeds = np.array([node2emb[t] for t in texts])
        q_norm = np.linalg.norm(q_emb)
        embed_norms = np.linalg.norm(embeds, axis=1)
        dense_sims = np.dot(embeds, q_emb) / (embed_norms * q_norm + 1e-9)

        hybrid: Dict[str, float] = {}
        for i, text in enumerate(texts):
            sparse_sim = sparse_similarity(node2kw[text], query_keywords)
            hybrid[text] = 0.5 * float(dense_sims[i]) + 0.5 * sparse_sim

        return self.topk_filter(hybrid)

    def search_docs_hopq(self, query: str) -> Tuple[List[str], List[float]]:
        """Priority-queue graph traversal with explore-exploit scoring (no LLM calls)."""
        query_embedding, query_keywords = self.process_query(query)
        mock_result = self.search_docs_mock(query_embedding, query_keywords, self.topk)
        if mock_result[0] is not None:
            return mock_result
        start_nodes = mock_result[1]  # List[Tuple[Dict, float]]

        q_emb = np.array(query_embedding)
        q_norm = np.linalg.norm(q_emb)

        # Max-heap via negated scores; counter breaks ties to avoid dict comparison
        H = []
        counter = 0
        C_score: Dict[str, float] = {}
        node2emb: Dict[str, list] = {}
        node2kw: Dict[str, set] = {}

        for node, score in start_nodes[:self.topk]:
            text = node['text']
            heapq.heappush(H, (-score, counter, node))
            counter += 1
            C_score[text] = max(score, C_score.get(text, float('-inf')))
            node2emb[text] = node['embed']
            node2kw[text] = set(node['keywords'])

        expanded: Set[str] = set()

        with self.driver.session() as session:
            for _ in range(self.max_hop):
                for _ in range(self.topk):
                    v = None
                    while H:
                        _, _, candidate = heapq.heappop(H)
                        if candidate['text'] not in expanded:
                            v = candidate
                            break
                    if v is None:
                        break

                    expanded.add(v['text'])
                    v_emb = np.array(v['embed'])
                    v_norm = np.linalg.norm(v_emb)
                    if v_norm < 1e-9:
                        continue

                    v_best = None
                    s_best = float('-inf')

                    result = session.run(expand_logic_query, {'text': v['text']})
                    for record in result:
                        vp = record['logic_node']
                        vp_emb = np.array(vp['embed'])
                        vp_norm = np.linalg.norm(vp_emb)

                        if vp_norm < 1e-9:
                            continue

                        exploration = float(np.dot(v_emb, vp_emb) / (v_norm * vp_norm))
                        exploitation = float(np.dot(q_emb, vp_emb) / (q_norm * vp_norm))
                        score = self.epsilon * exploration + (1 - self.epsilon) * exploitation

                        if score > s_best:
                            s_best = score
                            v_best = vp

                    if v_best is None:
                        continue

                    text = v_best['text']
                    heapq.heappush(H, (-s_best, counter, v_best))
                    counter += 1
                    C_score[text] = max(s_best, C_score.get(text, float('-inf')))
                    node2emb[text] = v_best['embed']
                    node2kw[text] = set(v_best['keywords'])

        logger.info(f"hopq: visited {len(expanded)} nodes, C_score size {len(C_score)}")
        return self._hopq_prune(C_score, node2emb, node2kw, q_emb, query_keywords)
