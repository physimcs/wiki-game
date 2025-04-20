import torch 
from sentence_transformers import SentenceTransformer, util

class WikiNavigator:
    def __init__(self, current_link, target_link, possible_links):
        self.current_link = current_link
        self.target_link = target_link
        self.links = possible_links
        self._embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def get_embeddings(self, s):
        return self._embedder.encode(s, convert_to_tensor=True)

#hits = util.semantic_search(query_embeddings, corpus_embeddings, top_k=3)