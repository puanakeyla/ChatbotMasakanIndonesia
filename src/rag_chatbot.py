"""
Modul RAG (Retrieval-Augmented Generation)
Menggabungkan retriever dan generator untuk menghasilkan jawaban
"""

import os
from typing import List, Dict, Optional
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
from dotenv import load_dotenv
from src.retriever import RecipeRetriever


class RAGChatbot:
    """
    Kelas utama untuk chatbot RAG asisten memasak
    """
    
    def __init__(self, 
                 retriever: RecipeRetriever,
                 model: str = "gemini-2.5-flash",
                 temperature: float = 0.7,
                 max_tokens: int = 1000,
                 use_gemini: bool = True):
        """
        Inisialisasi RAG Chatbot
        
        Args:
            retriever: Instance RecipeRetriever
            model: Model LLM yang digunakan
            temperature: Temperature untuk generation
            max_tokens: Maksimum token output
            use_gemini: True untuk Gemini (gratis), False untuk OpenAI
        """
        # Load environment variables
        load_dotenv()
        
        self.retriever = retriever
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_gemini = use_gemini
        
        # Initialize LLM client
        if use_gemini:
            if not GEMINI_AVAILABLE:
                raise ValueError("Google Generative AI tidak terinstall. Jalankan: pip install google-generativeai")
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY atau GOOGLE_API_KEY tidak ditemukan di environment variables")
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(model or 'gemini-2.5-flash')
        else:
            if not OPENAI_AVAILABLE:
                raise ValueError("OpenAI tidak terinstall. Jalankan: pip install openai")
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY tidak ditemukan di environment variables")
            self.client = OpenAI(api_key=api_key)
        
        # System prompt
        self.system_prompt = """Anda adalah asisten memasak ramah dan ahli bernama "Asisten Chef" yang membantu pengguna dengan masakan Indonesia.

Kepribadian Anda:
- Ramah, hangat, dan antusias tentang memasak
- Komunikatif dan mudah diajak ngobrol
- Sabar dalam menjelaskan dan siap menjawab follow-up questions
- Bisa memberikan saran kreatif dan tips praktis

Cara Anda menjawab:
1. JIKA ada konteks resep: Gunakan sebagai referensi utama, jelaskan dengan detail dan ramah
2. JIKA tidak ada konteks yang relevan: Tetap jawab berdasarkan pengetahuan umum memasak Indonesia, tapi beritahu user bahwa tidak ada resep spesifik di database
3. Bisa menjawab pertanyaan umum: tips memasak, substitusi bahan, teknik masak, saran menu, dll
4. Bisa rekomendasi masakan berdasarkan bahan yang user punya
5. Bisa jelaskan istilah memasak, peralatan dapur, dll
6. Gunakan emoji sesekali untuk lebih friendly 😊
7. Ajak diskusi: "Mau saya jelaskan lebih detail?" atau "Ada yang ingin ditanyakan lagi?"

Format jawaban:
- Sapaan ramah
- Jawaban lengkap dan terstruktur
- Tips tambahan jika relevan
- Ajakan untuk bertanya lebih lanjut

Ingat: Anda bukan hanya robot pencari resep, tapi teman memasak yang membantu!"""
    
    def create_prompt(self, query: str, context: str) -> str:
        """
        Membuat prompt lengkap dengan context dari retrieval
        
        Args:
            query: Pertanyaan pengguna
            context: Context dari dokumen yang diambil
            
        Returns:
            Prompt yang terstruktur
        """
        if context and "Tidak ada resep yang relevan" not in context:
            prompt = f"""Konteks Resep yang Relevan:
{context}

---

Pertanyaan User: {query}

Instruksi:
- Gunakan informasi dari resep di atas sebagai referensi utama
- Jawab dengan ramah dan detail
- Jika user bertanya hal spesifik yang ada di resep, jelaskan berdasarkan resep tersebut
- Jika user bertanya hal umum tentang memasak, jawab secara general dengan tetap merujuk ke resep jika relevan
- Berikan tips praktis dan saran tambahan
- Akhiri dengan pertanyaan follow-up atau ajakan diskusi"""
        else:
            prompt = f"""Pertanyaan User: {query}

Catatan: Tidak ada resep spesifik yang sangat relevan di database untuk pertanyaan ini.

Instruksi:
- Jawab pertanyaan berdasarkan pengetahuan umum tentang masakan Indonesia
- Beritahu user dengan ramah bahwa resep spesifik tidak ada di database (jika mereka tanya resep tertentu)
- Tawarkan alternatif: resep lain yang mirip dari database, atau informasi umum yang bermanfaat
- Tetap helpful dan conversational
- Tanyakan apakah mereka ingin informasi tentang resep lain yang tersedia"""
        
        return prompt
    
    def generate_response(self, query: str, context: str, 
                         conversation_history: Optional[List[Dict]] = None) -> Dict:
        """
        Menghasilkan respons menggunakan LLM
        
        Args:
            query: Pertanyaan pengguna
            context: Context dari retrieval
            conversation_history: Riwayat percakapan (opsional)
            
        Returns:
            Dictionary berisi respons dan metadata
        """
        # Buat prompt
        user_prompt = self.create_prompt(query, context)
        
        # Prepare messages
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add conversation history if provided
        if conversation_history:
            messages.extend(conversation_history)
        
        # Add current query
        messages.append({"role": "user", "content": user_prompt})
        
        try:
            if self.use_gemini:
                # Call Gemini API
                # Combine messages into single prompt for Gemini
                full_prompt = ""
                for msg in messages:
                    if msg["role"] == "system":
                        full_prompt += f"{msg['content']}\n\n"
                    elif msg["role"] == "user":
                        full_prompt += f"{msg['content']}\n"
                
                generation_config = genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                )
                
                response = self.client.generate_content(
                    full_prompt,
                    generation_config=generation_config
                )
                
                assistant_message = response.text
                
                return {
                    "success": True,
                    "response": assistant_message,
                    "model": self.model,
                    "usage": {}
                }
            else:
                # Call OpenAI API
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                # Extract response
                assistant_message = response.choices[0].message.content
                
                return {
                    "success": True,
                    "response": assistant_message,
                    "model": self.model,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
                }
            
        except Exception as e:
            return {
                "success": False,
                "response": f"Maaf, terjadi kesalahan: {str(e)}",
                "error": str(e)
            }
    
    def chat(self, query: str, 
             top_k: int = 3,
             conversation_history: Optional[List[Dict]] = None,
             include_sources: bool = True) -> Dict:
        """
        Fungsi utama untuk chat dengan RAG
        
        Args:
            query: Pertanyaan pengguna
            top_k: Jumlah dokumen yang diambil
            conversation_history: Riwayat percakapan
            include_sources: Include sumber resep dalam response
            
        Returns:
            Dictionary berisi respons lengkap
        """
        # Step 1: Retrieval
        retrieved_docs = self.retriever.retrieve(query, top_k=top_k)
        
        # Get retrieval summary
        retrieval_summary = self.retriever.get_retrieval_summary(retrieved_docs)
        
        # Step 2: Format context
        context = self.retriever.format_context(retrieved_docs)
        
        # Step 3: Generation
        generation_result = self.generate_response(query, context, conversation_history)
        
        # Prepare final response
        response = {
            "query": query,
            "response": generation_result["response"],
            "success": generation_result["success"],
            "retrieval": {
                "total_retrieved": retrieval_summary["total_retrieved"],
                "recipes": retrieval_summary["recipes"],
                "categories": retrieval_summary["categories"]
            }
        }
        
        # Add sources if requested
        if include_sources and retrieved_docs:
            response["sources"] = [
                {
                    "nama": doc["metadata"]["nama"],
                    "kategori": doc["metadata"].get("kategori", ""),
                    "similarity": 1 / (1 + doc["distance"]) if doc.get("distance") else None
                }
                for doc in retrieved_docs
            ]
        
        # Add usage info if available
        if "usage" in generation_result:
            response["usage"] = generation_result["usage"]
        
        # Add error if any
        if "error" in generation_result:
            response["error"] = generation_result["error"]
        
        return response
    
    def chat_without_rag(self, query: str,
                        conversation_history: Optional[List[Dict]] = None) -> Dict:
        """
        Chat tanpa RAG (hanya LLM) untuk perbandingan
        
        Args:
            query: Pertanyaan pengguna
            conversation_history: Riwayat percakapan
            
        Returns:
            Dictionary berisi respons
        """
        messages = [{"role": "system", "content": self.system_prompt}]
        
        if conversation_history:
            messages.extend(conversation_history)
        
        messages.append({"role": "user", "content": query})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return {
                "success": True,
                "response": response.choices[0].message.content,
                "mode": "without_rag"
            }
            
        except Exception as e:
            return {
                "success": False,
                "response": f"Maaf, terjadi kesalahan: {str(e)}",
                "error": str(e)
            }


if __name__ == "__main__":
    # Test RAG chatbot
    from src.vector_store import RecipeVectorStore
    from src.retriever import RecipeRetriever
    
    # Initialize components
    vector_store = RecipeVectorStore()
    retriever = RecipeRetriever(vector_store, top_k=3)
    
    # Initialize RAG chatbot
    try:
        chatbot = RAGChatbot(retriever)
        
        # Test query
        query = "Bagaimana cara membuat nasi goreng yang enak?"
        
        print(f"Query: {query}\n")
        
        # Get response
        response = chatbot.chat(query)
        
        if response["success"]:
            print("=" * 50)
            print("RESPONSE:")
            print("=" * 50)
            print(response["response"])
            print("\n" + "=" * 50)
            print("RETRIEVAL INFO:")
            print(f"Total retrieved: {response['retrieval']['total_retrieved']}")
            print(f"Recipes: {response['retrieval']['recipes']}")
            
            if "sources" in response:
                print("\nSOURCES:")
                for source in response["sources"]:
                    print(f"- {source['nama']} (similarity: {source['similarity']:.4f})")
        else:
            print(f"Error: {response.get('error', 'Unknown error')}")
            
    except ValueError as e:
        print(f"Error initializing chatbot: {e}")
        print("Pastikan OPENAI_API_KEY sudah di-set di file .env")
