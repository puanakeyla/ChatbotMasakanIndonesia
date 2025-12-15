"""
Modul Retriever untuk mengambil informasi relevan dari vector store
"""

from typing import List, Dict, Optional
from src.vector_store import RecipeVectorStore


class RecipeRetriever:
    """
    Kelas untuk melakukan retrieval dokumen resep dari vector store
    """
    
    def __init__(self, vector_store: RecipeVectorStore, top_k: int = 3):
        """
        Inisialisasi retriever
        
        Args:
            vector_store: Instance RecipeVectorStore
            top_k: Jumlah dokumen yang diambil
        """
        self.vector_store = vector_store
        self.top_k = top_k
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        Mengambil dokumen relevan berdasarkan query
        
        Args:
            query: Query pencarian
            top_k: Override jumlah dokumen (opsional)
            
        Returns:
            List dokumen relevan
        """
        k = top_k if top_k is not None else self.top_k
        
        # Search di vector store
        search_results = self.vector_store.search(query, top_k=k)
        
        return search_results['results']
    
    def retrieve_with_scores(self, query: str, top_k: Optional[int] = None, 
                            min_score: float = 0.0) -> List[Dict]:
        """
        Mengambil dokumen dengan filtering berdasarkan score
        
        Args:
            query: Query pencarian
            top_k: Override jumlah dokumen
            min_score: Minimum similarity score (threshold)
            
        Returns:
            List dokumen yang memenuhi threshold
        """
        results = self.retrieve(query, top_k)
        
        # Filter berdasarkan score (distance yang lebih kecil = similarity lebih tinggi)
        # Untuk Chroma, distance adalah L2 distance, jadi kita filter yang distance-nya kecil
        filtered_results = []
        for result in results:
            if result['distance'] is not None:
                # Convert distance to similarity score (0-1 range)
                # Semakin kecil distance, semakin tinggi similarity
                similarity = 1 / (1 + result['distance'])
                
                if similarity >= min_score:
                    result['similarity_score'] = similarity
                    filtered_results.append(result)
        
        return filtered_results
    
    def retrieve_by_category(self, query: str, category: str, 
                            top_k: Optional[int] = None) -> List[Dict]:
        """
        Mengambil dokumen dengan filter kategori
        
        Args:
            query: Query pencarian
            category: Kategori resep
            top_k: Override jumlah dokumen
            
        Returns:
            List dokumen dari kategori tertentu
        """
        k = top_k if top_k is not None else self.top_k
        
        search_results = self.vector_store.search_by_category(query, category, top_k=k)
        
        return search_results['results']
    
    def format_context(self, retrieved_docs: List[Dict], 
                       include_metadata: bool = True) -> str:
        """
        Memformat dokumen yang diambil menjadi context string
        untuk diberikan ke LLM
        
        Args:
            retrieved_docs: List dokumen hasil retrieval
            include_metadata: Include metadata dalam context
            
        Returns:
            String context yang terformat
        """
        if not retrieved_docs:
            return "Tidak ada resep yang relevan ditemukan."
        
        context_parts = []
        context_parts.append("Berikut adalah resep-resep yang relevan:\n")
        
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(f"\n=== Resep {i}: {doc['metadata']['nama']} ===")
            
            if include_metadata:
                metadata = doc['metadata']
                if metadata.get('kategori'):
                    context_parts.append(f"Kategori: {metadata['kategori']}")
                if metadata.get('porsi'):
                    context_parts.append(f"Porsi: {metadata['porsi']}")
                if metadata.get('waktu_masak'):
                    context_parts.append(f"Waktu Memasak: {metadata['waktu_masak']}")
                if metadata.get('tingkat_kesulitan'):
                    context_parts.append(f"Tingkat Kesulitan: {metadata['tingkat_kesulitan']}")
            
            context_parts.append(f"\n{doc['document']}")
        
        return "\n".join(context_parts)
    
    def get_retrieval_summary(self, retrieved_docs: List[Dict]) -> Dict:
        """
        Membuat summary dari hasil retrieval
        
        Args:
            retrieved_docs: List dokumen hasil retrieval
            
        Returns:
            Dictionary berisi summary
        """
        if not retrieved_docs:
            return {
                "total_retrieved": 0,
                "recipes": [],
                "categories": []
            }
        
        recipes = [doc['metadata']['nama'] for doc in retrieved_docs]
        categories = list(set([doc['metadata'].get('kategori', '') 
                              for doc in retrieved_docs if doc['metadata'].get('kategori')]))
        
        return {
            "total_retrieved": len(retrieved_docs),
            "recipes": recipes,
            "categories": categories
        }


if __name__ == "__main__":
    # Test retriever
    from src.vector_store import RecipeVectorStore
    
    vector_store = RecipeVectorStore()
    retriever = RecipeRetriever(vector_store, top_k=3)
    
    # Test retrieval
    query = "bagaimana cara membuat nasi goreng yang enak?"
    
    print(f"Query: {query}\n")
    
    # Retrieve documents
    docs = retriever.retrieve(query)
    print(f"Retrieved {len(docs)} documents")
    
    # Get summary
    summary = retriever.get_retrieval_summary(docs)
    print(f"\nSummary:")
    print(f"Total: {summary['total_retrieved']}")
    print(f"Recipes: {summary['recipes']}")
    print(f"Categories: {summary['categories']}")
    
    # Format context
    context = retriever.format_context(docs)
    print(f"\n{context}")
