"""
Aplikasi Chatbot Asisten Memasak - Interface Streamlit
"""

import streamlit as st
import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.vector_store import RecipeVectorStore
from src.retriever import RecipeRetriever
from src.rag_chatbot import RAGChatbot


# Page config
st.set_page_config(
    page_title="Asisten Chef Indonesia",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a1a;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    .sub-header {
        font-size: 1rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Sidebar Stats */
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    
    .stat-label {
        font-size: 0.875rem;
        opacity: 0.9;
        font-weight: 500;
    }
    
    /* Category Pills */
    .category-pill {
        display: inline-block;
        background: #f3f4f6;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.25rem;
        font-size: 0.875rem;
        color: #374151;
        font-weight: 500;
    }
    
    /* Chat Messages */
    .stChatMessage {
        background: #ffffff;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-weight: 500;
        border: 1px solid #e5e7eb;
        background: white;
        color: #374151;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background: #f9fafb;
        border-color: #667eea;
        color: #667eea;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(102, 126, 234, 0.15);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #f9fafb;
        border-radius: 8px;
        font-weight: 500;
    }
    
    /* Success/Info Messages */
    .stSuccess {
        background: #ecfdf5;
        color: #065f46;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #10b981;
    }
    
    /* Input Box */
    .stChatInputContainer {
        border-top: 1px solid #e5e7eb;
        padding-top: 1rem;
    }
    
    /* Icons */
    .icon {
        display: inline-block;
        width: 20px;
        height: 20px;
        margin-right: 8px;
        vertical-align: middle;
    }
    
    /* Section Title */
    .section-title {
        font-size: 1.125rem;
        font-weight: 600;
        color: #1f2937;
        margin: 1.5rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_chatbot():
    """
    Initialize chatbot components (cached)
    """
    try:
        # Load environment variables
        load_dotenv()
        
        # Check if vector store exists
        if not os.path.exists("./chroma_db"):
            st.error("Error: Vector store belum disetup. Jalankan setup_database.py terlebih dahulu!")
            st.stop()
        
        # Initialize components
        vector_store = RecipeVectorStore(
            persist_directory="./chroma_db",
            collection_name="indonesian_recipes"
        )
        
        retriever = RecipeRetriever(vector_store, top_k=3)
        
        chatbot = RAGChatbot(
            retriever=retriever,
            model=os.getenv("LLM_MODEL", "gemini-2.5-flash"),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            use_gemini=os.getenv("USE_GEMINI", "true").lower() == "true"
        )
        
        return chatbot, vector_store
        
    except ValueError as e:
        st.error(f"Error: {str(e)}")
        st.info("Info: Pastikan OPENAI_API_KEY sudah diset di file .env")
        st.stop()
    except Exception as e:
        st.error(f"Error initializing chatbot: {str(e)}")
        st.stop()


def main():
    """
    Main application
    """
    # Header
    st.markdown('<div class="main-header">Asisten Chef Indonesia</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Platform AI untuk Resep & Panduan Memasak Nusantara</div>', unsafe_allow_html=True)
    
    # Initialize chatbot
    with st.spinner("Memuat chatbot..."):
        chatbot, vector_store = initialize_chatbot()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Pengaturan Sistem")
        
        # Database stats
        stats = vector_store.get_stats()
        st.markdown("### Database Statistics")
        
        # Stat cards
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{stats['total_recipes']}</div>
            <div class="stat-label">Total Resep Tersedia</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{stats['num_categories']}</div>
            <div class="stat-label">Kategori Masakan</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Categories
        st.markdown('<div class="section-title">Kategori Masakan</div>', unsafe_allow_html=True)
        
        # Initialize selected category
        if 'selected_category' not in st.session_state:
            st.session_state.selected_category = None
        
        # Create clickable category buttons
        if st.button("Semua Kategori", use_container_width=True, type="primary" if st.session_state.selected_category is None else "secondary"):
            st.session_state.selected_category = None
            st.rerun()
        
        for category in stats['categories']:
            is_selected = st.session_state.selected_category == category
            if st.button(category, use_container_width=True, type="primary" if is_selected else "secondary"):
                st.session_state.selected_category = category
                # Add prompt about this category
                category_prompt = f"Tolong rekomendasikan resep dari kategori {category}"
                st.session_state.messages.append({"role": "user", "content": category_prompt})
                response = chatbot.chat(query=category_prompt, top_k=top_k, include_sources=show_sources)
                if response["success"]:
                    message_data = {"role": "assistant", "content": response["response"]}
                    if "sources" in response:
                        message_data["sources"] = response["sources"]
                    st.session_state.messages.append(message_data)
                st.rerun()
        
        st.markdown("---")
        
        # RAG Settings
        st.markdown('<div class="section-title">Konfigurasi RAG</div>', unsafe_allow_html=True)
        top_k = st.slider("Jumlah resep yang diambil", 1, 5, 3, help="Semakin banyak, semakin lengkap konteksnya")
        show_sources = st.checkbox("Tampilkan sumber resep", value=True, help="Lihat resep mana yang digunakan AI")
        
        st.markdown("---")
        
        # Clear chat button
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.selected_category = None
            st.rerun()
        
        st.markdown("---")
        st.markdown('<div class="section-title">About</div>', unsafe_allow_html=True)
        st.markdown("""
        <small style="color: #6b7280;">
        <strong>Technology Stack:</strong><br>
        • RAG (Retrieval-Augmented Generation)<br>
        • ChromaDB Vector Database<br>
        • Google Gemini AI<br>
        • Sentence Transformers<br><br>
        <strong>Data:</strong> 100 Resep Masakan Indonesia
        </small>
        """, unsafe_allow_html=True)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(content)
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(content)
                
                # Show sources if available
                if show_sources and "sources" in message:
                    with st.expander("Referensi Resep"):
                        for idx, source in enumerate(message["sources"], 1):
                            similarity_pct = source['similarity'] * 100 if source['similarity'] else 0
                            st.markdown(f"""
                            <div style="background: #f9fafb; padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem; border-left: 3px solid #667eea;">
                                <strong>{idx}. {source['nama']}</strong><br>
                                <small style="color: #6b7280;">
                                    Kategori: {source['kategori']} | 
                                    Relevansi: <strong>{similarity_pct:.0f}%</strong>
                                </small>
                            </div>
                            """, unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ketik pertanyaan Anda di sini..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        
        # Get bot response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Memproses pertanyaan Anda..."):
                # Get response from chatbot
                response = chatbot.chat(
                    query=prompt,
                    top_k=top_k,
                    include_sources=show_sources
                )
                
                if response["success"]:
                    # Display response
                    st.markdown(response["response"])
                    
                    # Display retrieval info in a nicer way
                    if response["retrieval"]["total_retrieved"] > 0:
                        recipes_list = ", ".join(response['retrieval']['recipes'])
                        st.markdown(f"""
                        <div style="background: #ecfdf5; padding: 0.75rem; border-radius: 8px; border-left: 3px solid #10b981; margin-top: 1rem;">
                            <small style="color: #065f46;">
                                <strong>Ditemukan {response['retrieval']['total_retrieved']} resep relevan:</strong> {recipes_list}
                            </small>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Save message with sources
                    message_data = {
                        "role": "assistant",
                        "content": response["response"]
                    }
                    if "sources" in response:
                        message_data["sources"] = response["sources"]
                    
                    st.session_state.messages.append(message_data)
                else:
                    error_msg = response.get("error", "Terjadi kesalahan")
                    st.error(f"Error: {error_msg}")
    
    # Example questions
    st.markdown("---")
    st.markdown('<div class="section-title">Contoh Pertanyaan</div>', unsafe_allow_html=True)
    st.markdown('<small style="color: #6b7280;">Klik tombol atau ketik pertanyaan sendiri</small>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Resep Nasi Goreng", use_container_width=True):
            st.session_state.example_query = "Bagaimana cara membuat nasi goreng yang enak dan pulen?"
            st.rerun()
        if st.button("Cara Masak Rendang", use_container_width=True):
            st.session_state.example_query = "Bagaimana cara membuat rendang sapi yang empuk dan bumbu meresap?"
            st.rerun()
    
    with col2:
        if st.button("Resep Soto Ayam", use_container_width=True):
            st.session_state.example_query = "Apa bahan-bahan dan cara membuat soto ayam kuning?"
            st.rerun()
        if st.button("Tips Tumis Sayur", use_container_width=True):
            st.session_state.example_query = "Bagaimana tips menumis sayuran agar tetap renyah?"
            st.rerun()
    
    with col3:
        if st.button("Substitusi Bahan", use_container_width=True):
            st.session_state.example_query = "Kalau tidak ada kecap manis, bisa diganti dengan apa?"
            st.rerun()
        if st.button("Rekomendasi Menu", use_container_width=True):
            st.session_state.example_query = "Bisa rekomendasikan menu masakan Indonesia untuk makan siang keluarga?"
            st.rerun()
    
    # Handle example query
    if "example_query" in st.session_state:
        query = st.session_state.example_query
        del st.session_state.example_query
        
        # Add to messages and get response
        st.session_state.messages.append({"role": "user", "content": query})
        
        response = chatbot.chat(
            query=query,
            top_k=top_k,
            include_sources=show_sources
        )
        
        if response["success"]:
            message_data = {
                "role": "assistant",
                "content": response["response"]
            }
            if "sources" in response:
                message_data["sources"] = response["sources"]
            
            st.session_state.messages.append(message_data)
        
        st.rerun()


if __name__ == "__main__":
    main()
