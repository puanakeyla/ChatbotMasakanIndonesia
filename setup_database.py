"""
Script untuk setup awal: load data resep ke vector store
"""

import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_processor import RecipePreprocessor
from src.vector_store import RecipeVectorStore


def setup_vector_store(data_path: str = "./data/resep_indonesia.json"):
    """
    Load data resep dan simpan ke vector store
    
    Args:
        data_path: Path ke file JSON resep
    """
    print("=" * 60)
    print("SETUP VECTOR STORE - CHATBOT ASISTEN MEMASAK")
    print("=" * 60)
    
    # 1. Initialize preprocessor
    print("\n1. Inisialisasi Data Preprocessor...")
    preprocessor = RecipePreprocessor()
    
    # 2. Load dan process recipes
    print(f"\n2. Memuat data resep dari {data_path}...")
    try:
        recipes = preprocessor.load_from_json(data_path)
        print(f"   ✓ Berhasil memuat {len(recipes)} resep")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return
    
    # 3. Format recipes untuk embedding
    print("\n3. Memformat resep untuk embedding...")
    recipe_texts = []
    for recipe in recipes:
        text = preprocessor.format_recipe_for_embedding(recipe)
        recipe_texts.append(text)
    print(f"   ✓ {len(recipe_texts)} resep siap untuk embedding")
    
    # 4. Initialize vector store
    print("\n4. Inisialisasi Vector Store...")
    vector_store = RecipeVectorStore(
        persist_directory="./chroma_db",
        collection_name="indonesian_recipes"
    )
    
    # 5. Check if already has data
    current_count = vector_store.collection.count()
    if current_count > 0:
        print(f"\n   ⚠ Vector store sudah berisi {current_count} dokumen")
        response = input("   Hapus data lama dan load ulang? (y/n): ")
        if response.lower() == 'y':
            print("   Menghapus data lama...")
            vector_store.delete_all()
        else:
            print("   Setup dibatalkan")
            return
    
    # 6. Add recipes to vector store
    print("\n5. Menambahkan resep ke Vector Store...")
    print("   (Proses embedding membutuhkan waktu...)")
    try:
        vector_store.add_recipes(recipes, recipe_texts)
        print(f"   ✓ Berhasil menambahkan {len(recipes)} resep ke vector store")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return
    
    # 7. Verify
    print("\n6. Verifikasi...")
    stats = vector_store.get_stats()
    print(f"   Total resep dalam database: {stats['total_recipes']}")
    print(f"   Jumlah kategori: {stats['num_categories']}")
    print(f"   Kategori tersedia: {', '.join(stats['categories'])}")
    
    # 8. Test search
    print("\n7. Test Pencarian...")
    test_query = "cara membuat nasi goreng"
    print(f"   Query test: '{test_query}'")
    
    results = vector_store.search(test_query, top_k=3)
    print(f"   Hasil pencarian:")
    for i, result in enumerate(results['results'], 1):
        print(f"   {i}. {result['metadata']['nama']} (distance: {result['distance']:.4f})")
    
    print("\n" + "=" * 60)
    print("SETUP SELESAI!")
    print("=" * 60)
    print("\nVector store siap digunakan.")
    print("Anda dapat menjalankan chatbot dengan: streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    # Check if data file exists
    data_file = "./data/resep_indonesia.json"
    
    if not os.path.exists(data_file):
        print(f"Error: File {data_file} tidak ditemukan!")
        print("Pastikan file data resep sudah tersedia.")
        sys.exit(1)
    
    # Run setup
    setup_vector_store(data_file)
