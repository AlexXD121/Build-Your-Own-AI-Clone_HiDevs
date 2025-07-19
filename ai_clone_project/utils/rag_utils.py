from sentence_transformers import SentenceTransformer
import faiss
import os
import pickle

class RAGPipeline:
    def __init__(self, embedding_model='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(embedding_model)
        self.index = None
        self.texts = []

    def create_vector_db(self, texts, save_path="vectorstore"):
        os.makedirs(save_path, exist_ok=True)
        self.texts = texts
        embeddings = self.model.encode(texts)

        # Save index
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        faiss.write_index(self.index, f"{save_path}/index.faiss")

        # Save original text chunks
        with open(f"{save_path}/chunks.pkl", "wb") as f:
            pickle.dump(self.texts, f)

    def load_vector_db(self, path="vectorstore"):
        self.index = faiss.read_index(f"{path}/index.faiss")
        with open(f"{path}/chunks.pkl", "rb") as f:
            self.texts = pickle.load(f)

    def retrieve(self, query, k=3):
        query_vec = self.model.encode([query])
        distances, indices = self.index.search(query_vec, k)
        return [self.texts[i] for i in indices[0]]
