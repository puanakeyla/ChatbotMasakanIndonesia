"""
Modul untuk pengolahan data resep masakan Indonesia
Melakukan preprocessing, cleaning, dan normalisasi data resep
"""

import re
import json
from typing import List, Dict
import pandas as pd


class RecipePreprocessor:
    """
    Kelas untuk preprocessing data resep masakan
    """
    
    def __init__(self):
        self.unwanted_patterns = [
            r'<[^>]+>',  # HTML tags
            r'http[s]?://\S+',  # URLs
            r'\[.*?\]',  # Brackets dengan konten
        ]
    
    def clean_text(self, text: str) -> str:
        """
        Membersihkan teks dari karakter tidak diinginkan
        
        Args:
            text: Teks yang akan dibersihkan
            
        Returns:
            Teks yang sudah dibersihkan
        """
        if not text:
            return ""
        
        # Hapus HTML tags dan URLs
        for pattern in self.unwanted_patterns:
            text = re.sub(pattern, '', text)
        
        # Hapus multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Hapus whitespace di awal dan akhir
        text = text.strip()
        
        return text
    
    def normalize_ingredients(self, ingredients: List[str]) -> List[str]:
        """
        Normalisasi format bahan-bahan masakan
        
        Args:
            ingredients: List bahan-bahan
            
        Returns:
            List bahan-bahan yang sudah dinormalisasi
        """
        normalized = []
        for ingredient in ingredients:
            # Clean dan lowercase
            cleaned = self.clean_text(ingredient)
            if cleaned:
                normalized.append(cleaned)
        
        return normalized
    
    def normalize_steps(self, steps: List[str]) -> List[str]:
        """
        Normalisasi langkah-langkah memasak
        
        Args:
            steps: List langkah memasak
            
        Returns:
            List langkah yang sudah dinormalisasi
        """
        normalized = []
        for i, step in enumerate(steps, 1):
            cleaned = self.clean_text(step)
            if cleaned:
                # Tambahkan nomor jika belum ada
                if not re.match(r'^\d+\.', cleaned):
                    cleaned = f"{i}. {cleaned}"
                normalized.append(cleaned)
        
        return normalized
    
    def process_recipe(self, recipe: Dict) -> Dict:
        """
        Memproses satu resep lengkap
        
        Args:
            recipe: Dictionary berisi data resep
            
        Returns:
            Dictionary resep yang sudah diproses
        """
        processed = {
            'nama': self.clean_text(recipe.get('nama', '')),
            'kategori': self.clean_text(recipe.get('kategori', '')),
            'porsi': self.clean_text(recipe.get('porsi', '')),
            'waktu_masak': self.clean_text(recipe.get('waktu_masak', '')),
            'tingkat_kesulitan': self.clean_text(recipe.get('tingkat_kesulitan', '')),
            'bahan': self.normalize_ingredients(recipe.get('bahan', [])),
            'langkah': self.normalize_steps(recipe.get('langkah', [])),
            'tips': self.clean_text(recipe.get('tips', ''))
        }
        
        return processed
    
    def format_recipe_for_embedding(self, recipe: Dict) -> str:
        """
        Mengubah resep menjadi format teks terstruktur untuk embedding
        
        Args:
            recipe: Dictionary resep yang sudah diproses
            
        Returns:
            String teks terstruktur
        """
        text_parts = []
        
        # Judul dan metadata
        text_parts.append(f"Nama Masakan: {recipe['nama']}")
        
        if recipe.get('kategori'):
            text_parts.append(f"Kategori: {recipe['kategori']}")
        
        if recipe.get('porsi'):
            text_parts.append(f"Porsi: {recipe['porsi']}")
        
        if recipe.get('waktu_masak'):
            text_parts.append(f"Waktu Memasak: {recipe['waktu_masak']}")
        
        if recipe.get('tingkat_kesulitan'):
            text_parts.append(f"Tingkat Kesulitan: {recipe['tingkat_kesulitan']}")
        
        # Bahan-bahan
        if recipe.get('bahan'):
            text_parts.append("\nBahan-bahan:")
            for bahan in recipe['bahan']:
                text_parts.append(f"- {bahan}")
        
        # Langkah-langkah
        if recipe.get('langkah'):
            text_parts.append("\nCara Membuat:")
            for langkah in recipe['langkah']:
                text_parts.append(langkah)
        
        # Tips
        if recipe.get('tips'):
            text_parts.append(f"\nTips: {recipe['tips']}")
        
        return "\n".join(text_parts)
    
    def load_from_json(self, filepath: str) -> List[Dict]:
        """
        Memuat data resep dari file JSON
        
        Args:
            filepath: Path ke file JSON
            
        Returns:
            List resep yang sudah diproses
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            recipes = json.load(f)
        
        processed_recipes = []
        for recipe in recipes:
            processed = self.process_recipe(recipe)
            processed_recipes.append(processed)
        
        return processed_recipes
    
    def save_to_json(self, recipes: List[Dict], filepath: str):
        """
        Menyimpan resep yang sudah diproses ke file JSON
        
        Args:
            recipes: List resep
            filepath: Path file output
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(recipes, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # Test preprocessing
    preprocessor = RecipePreprocessor()
    
    # Sample recipe
    sample_recipe = {
        'nama': 'Nasi Goreng',
        'kategori': 'Makanan Utama',
        'porsi': '2 porsi',
        'waktu_masak': '20 menit',
        'tingkat_kesulitan': 'Mudah',
        'bahan': [
            '2 piring nasi putih',
            '2 butir telur',
            '3 siung bawang putih',
            'Kecap manis secukupnya'
        ],
        'langkah': [
            'Panaskan minyak',
            'Tumis bawang putih hingga harum',
            'Masukkan telur, orak-arik',
            'Tambahkan nasi dan kecap'
        ],
        'tips': 'Gunakan nasi dingin untuk hasil yang lebih pulen'
    }
    
    processed = preprocessor.process_recipe(sample_recipe)
    formatted_text = preprocessor.format_recipe_for_embedding(processed)
    
    print("Resep yang sudah diformat:")
    print(formatted_text)
