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
    
    def similarity(self, target, comparison):
        target_embedding = self.get_embeddings(target)
        
        if not isinstance(comparison, list):
            comparison = [comparison]

        comparison_embeddings = [self.get_embeddings(item) for item in comparison]
        results = util.semantic_search(target_embedding, comparison_embeddings, top_k=1)
        return results

    def best_link(self):
        results = self.similarity(self.target_link, self.links)
        index = results[0][0].get('corpus_id')
        link = self.links[index]
        similarity = results[0][0].get('score')
        return link, similarity

link = 'Hypertension'
target = 'Eggs'
links = ['Diet', 'Heart attack', 'Doctor']

wiki = WikiNavigator(link, target, links)
print(wiki.best_link())

#hits = semantic_search(query_embeddings, corpus_embeddings, top_k=3)