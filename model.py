from sentence_transformers import SentenceTransformer
model = SentenceTransformer("multi-qa-mpnet-base-cos-v1")

class WikiNavigator:
    def __init__(self, current_link, target_link, possible_links):
        self.current_link = current_link
        self.target_link = target_link
        self.possible_links = possible_links

# current_link, target_link, possible_links