"""
Modul Vector Store untuk penyimpanan dan pencarian embedding resep
Menggunakan ChromaDB sebagai basis data vektor
"""

import os
import json
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import numpy as np


class RecipeVectorStore:
    """
    Kelas untuk mengelola penyimpanan vektor resep dalam ChromaDB
    """
    
    def __init__(self, 
                 persist_directory: str = "./chroma_db",
                 collection_name: str = "indonesian_recipes",
                 embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        """
        Inisialisasi vector store
        
        Args:
            persist_directory: Direktori untuk menyimpan database
            collection_name: Nama collection
            embedding_model: Model untuk embedding
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Setup ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Setup embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        
        # Buat atau ambil collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"description": "Indonesian cooking recipes"}
        )
        
        print(f"Vector store initialized: {collection_name}")
        print(f"Total documents: {self.collection.count()}")
    
    def add_recipes(self, recipes: List[Dict], recipe_texts: List[str]):
        """
        Menambahkan resep ke vector store
        
        Args:
            recipes: List dictionary resep (metadata)
            recipe_texts: List teks resep yang sudah diformat
        """
        if len(recipes) != len(recipe_texts):
            raise ValueError("Jumlah recipes dan recipe_texts harus sama")
        
        # Generate IDs
        ids = [f"recipe_{i}" for i in range(len(recipes))]
        
        # Prepare metadata
        metadatas = []
        for recipe in recipes:
            metadata = {
                "nama": recipe.get("nama", ""),
                "kategori": recipe.get("kategori", ""),
                "porsi": recipe.get("porsi", ""),
                "waktu_masak": recipe.get("waktu_masak", ""),
                "tingkat_kesulitan": recipe.get("tingkat_kesulitan", "")
            }
            metadatas.append(metadata)
        
        # Add to collection
        self.collection.add(
            documents=recipe_texts,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Added {len(recipes)} recipes to vector store")
    
    def search(self, query: str, top_k: int = 3) -> Dict:
        """
        Mencari resep berdasarkan query
        
        Args:
            query: Pertanyaan atau query pencarian
            top_k: Jumlah hasil teratas
            
        Returns:
            Dictionary berisi hasil pencarian
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        # Format results
        formatted_results = {
            "query": query,
            "results": []
        }
        
        if results and results['documents'] and len(results['documents'][0]) > 0:
            for i in range(len(results['documents'][0])):
                result_item = {
                    "id": results['ids'][0][i],
                    "document": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i] if 'distances' in results else None
                }
                formatted_results["results"].append(result_item)
        
        return formatted_results
    
    def search_by_category(self, query: str, category: str, top_k: int = 3) -> Dict:
        """
        Mencari resep berdasarkan query dan filter kategori
        
        Args:
            query: Pertanyaan atau query pencarian
            category: Kategori resep
            top_k: Jumlah hasil teratas
            
        Returns:
            Dictionary berisi hasil pencarian
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where={"kategori": category}
        )
        
        # Format results (sama seperti search)
        formatted_results = {
            "query": query,
            "category": category,
            "results": []
        }
        
        if results and results['documents'] and len(results['documents'][0]) > 0:
            for i in range(len(results['documents'][0])):
                result_item = {
                    "id": results['ids'][0][i],
                    "document": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i] if 'distances' in results else None
                }
                formatted_results["results"].append(result_item)
        
        return formatted_results
    
    def get_all_categories(self) -> List[str]:
        """
        Mendapatkan semua kategori yang tersedia
        
        Returns:
            List kategori unik
        """
        # Get all documents
        all_docs = self.collection.get()
        
        if all_docs and 'metadatas' in all_docs:
            categories = set()
            for metadata in all_docs['metadatas']:
                if 'kategori' in metadata:
                    categories.add(metadata['kategori'])
            return sorted(list(categories))
        
        return []
    
    def delete_all(self):
        """
        Menghapus semua dokumen dari collection
        """
        # Delete collection
        self.client.delete_collection(name=self.collection_name)
        
        # Recreate collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function
        )
        
        print("All documents deleted from vector store")
    
    def get_stats(self) -> Dict:
        """
        Mendapatkan statistik vector store
        
        Returns:
            Dictionary berisi statistik
        """
        count = self.collection.count()
        categories = self.get_all_categories()
        
        return {
            "total_recipes": count,
            "categories": categories,
            "num_categories": len(categories)
        }


if __name__ == "__main__":
    # Test vector store
    vector_store = RecipeVectorStore()
    
    # Sample data
    sample_recipes = [
        {
            "nama": "Nasi Goreng",
            "kategori": "Makanan Utama",
            "porsi": "2 porsi",
            "waktu_masak": "20 menit",
            "tingkat_kesulitan": "Mudah"
        },
        {
            "nama": "Rendang Sapi",
            "kategori": "Makanan Utama",
            "porsi": "4 porsi",
            "waktu_masak": "3 jam",
            "tingkat_kesulitan": "Sulit"
        }
    ]
    
    sample_texts = [
        "Nasi Goreng adalah makanan khas Indonesia yang terbuat dari nasi putih yang digoreng dengan kecap, telur, dan bumbu.",
        "Rendang Sapi adalah masakan khas Minangkabau yang terkenal dengan cita rasa pedas dan santan yang kental."
    ]
    
    # Add recipes
    # vector_store.add_recipes(sample_recipes, sample_texts)
    
    # Search
    query = "cara membuat nasi goreng"
    results = vector_store.search(query, top_k=2)
    
    print(f"\nHasil pencarian untuk: '{query}'")
    for result in results['results']:
        print(f"\n- {result['metadata']['nama']}")
        print(f"  Kategori: {result['metadata']['kategori']}")
        print(f"  Distance: {result['distance']:.4f}")
    
    # Stats
    stats = vector_store.get_stats()
    print(f"\nStatistik Vector Store:")
    print(f"Total resep: {stats['total_recipes']}")
    print(f"Kategori: {stats['categories']}")
