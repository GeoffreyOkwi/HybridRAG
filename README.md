HybridRAG – Configurable Hybrid Retrieval-Augmented Generation 
        Overview 
HybridRAG is a Streamlit-based Retrieval-Augmented Generation (RAG) system that 
combines vector similarity search with keyword-based scoring to improve document 
retrieval relevance. 
Instead of relying purely on embeddings, this system implements a hybrid scoring 
mechanism that blends: 
• Vector similarity score 
• Keyword match score 
This allows more accurate and context-aware retrieval. 
 
    Architecture 
User Query 
↓ 
Embedding Generation 
↓ 
Vector Search (FAISS) 
↓ 
Keyword Matching 
↓ 
Hybrid Re-ranking 
↓ 
Top-k Context Injection 
↓ 
LLM Response Generation 
 
      Hybrid Scoring Formula 
The final hybrid score is computed as: 
FinalScore = (1 / (1 + VectorDistance)) + (β × KeywordMatches) 
Where: 
• VectorDistance = FAISS L2 distance (lower is better) 
• KeywordMatches = number of overlapping query terms 
• β = keyword weighting factor (currently 0.1) 
Documents are then re-ranked by HybridScore in descending order. 
 
     Tech Stack 
• Python 
• Streamlit 
• FAISS 
• LangChain 
• OpenAI API 
 
       Installation 
git clone https://github.com/GeoffreyOkwi/HybridRAG.git 
cd HybridRAG 
pip install -r requirements.txt 
streamlit run app.py 
Features 
• Configurable hybrid scoring 
• Top-k re-ranking 
• Transparent retrieval logic 
• Streamlit interactive interface 
• Chat history support 
Future Improvements 
• Add adjustable α parameter via UI 
• Add visualization of retrieval scores 
• Deploy on Streamlit Cloud 
• Add document upload interface 

Author 
Geoffrey Okwi 
AI Engineer | Retrieval Systems | Applied LLM Systems