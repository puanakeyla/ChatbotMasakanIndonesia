"""
Modul untuk text embedding menggunakan model sentence-transformers
Mengubah teks resep menjadi representasi vektor numerik
"""

from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np


class RecipeEmbedding:
    """
    Kelas untuk menghasilkan embedding dari teks resep
    """
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        """
        Inisialisasi model embedding
        
        Args:
            model_name: Nama model sentence-transformers yang digunakan
        """
        print(f"Memuat model embedding: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        print(f"Model dimuat. Dimensi embedding: {self.embedding_dimension}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Menghasilkan embedding untuk satu teks
        
        Args:
            text: Teks yang akan di-embed
            
        Returns:
            Array numpy berisi vektor embedding
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def embed_batch(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """
        Menghasilkan embedding untuk batch teks
        
        Args:
            texts: List teks yang akan di-embed
            batch_size: Ukuran batch untuk processing
            show_progress: Tampilkan progress bar
            
        Returns:
            Array numpy berisi vektor embedding untuk semua teks
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        return embeddings
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Menghitung cosine similarity antara dua embedding
        
        Args:
            embedding1: Vektor embedding pertama
            embedding2: Vektor embedding kedua
            
        Returns:
            Nilai similarity (0-1)
        """
        # Normalisasi vektor
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)
    
    def find_most_similar(self, query_embedding: np.ndarray, 
                          document_embeddings: np.ndarray, 
                          top_k: int = 5) -> List[tuple]:
        """
        Mencari dokumen paling mirip dengan query
        
        Args:
            query_embedding: Vektor embedding query
            document_embeddings: Array vektor embedding dokumen
            top_k: Jumlah dokumen teratas yang dikembalikan
            
        Returns:
            List tuple (index, similarity_score)
        """
        similarities = []
        
        for idx, doc_embedding in enumerate(document_embeddings):
            similarity = self.compute_similarity(query_embedding, doc_embedding)
            similarities.append((idx, similarity))
        
        # Sort berdasarkan similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]


if __name__ == "__main__":
    # Test embedding
    embedder = RecipeEmbedding()
    
    # Test single text
    sample_text = "Cara membuat nasi goreng yang enak"
    embedding = embedder.embed_text(sample_text)
    print(f"\nDimensi embedding: {embedding.shape}")
    print(f"Sample embedding values: {embedding[:5]}")
    
    # Test batch
    texts = [
        "Nasi Goreng dengan telur dan kecap",
        "Rendang daging sapi pedas",
        "Soto ayam kuning dengan kuah kaldu"
    ]
    
    embeddings = embedder.embed_batch(texts, show_progress=False)
    print(f"\nBatch embeddings shape: {embeddings.shape}")
    
    # Test similarity
    query = "Resep nasi goreng"
    query_emb = embedder.embed_text(query)
    
    results = embedder.find_most_similar(query_emb, embeddings, top_k=3)
    print(f"\nHasil pencarian untuk: '{query}'")
    for idx, score in results:
        print(f"  {idx}. {texts[idx]} (score: {score:.4f})")
