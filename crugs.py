import streamlit as st
import PyPDF2
import re
import nltk
import arxiv
import io
import requests
from typing import Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


st.set_page_config(page_title="Research Similarity Tool", layout="wide")

@st.cache_resource
def initialize_assets():

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model, set(nltk.corpus.stopwords.words('english'))

embedding_model, STOP_WORDS = initialize_assets()

class PDFProcessor:
    @staticmethod
    def extract_text(uploaded_file) -> str:
        text = ""
        try:
            reader = PyPDF2.PdfReader(uploaded_file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
        return text

class ArxivProcessor:
    @staticmethod
    def fetch_by_id(arxiv_id: str):
        """Fetches paper text and metadata from arXiv."""
        try:
            search = arxiv.Search(id_list=[arxiv_id.strip().replace(" ", "")])
            paper = next(search.results())
            
            # Stream the PDF into memory
            response = requests.get(paper.pdf_url)
            pdf_file = io.BytesIO(response.content)
            
            text = PDFProcessor.extract_text(pdf_file)
            return text, paper.title
        except Exception as e:
            st.error(f"ArXiv Fetch Error: {e}")
            return None, None

class ResearchSimilarityEngine:
    def preprocess(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = text.split()
        tokens = [w for w in tokens if w not in STOP_WORDS and len(w) > 2]
        return " ".join(tokens)

    def semantic_similarity(self, text1: str, text2: str) -> float:

        chunk1 = text1[:10000] 
        chunk2 = text2[:10000]
        
        embeddings = embedding_model.encode([chunk1, chunk2])
        score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return round(float(score) * 100, 2)

    def keyword_analysis(self, text1: str, text2: str) -> Dict:
        clean1 = self.preprocess(text1)
        clean2 = self.preprocess(text2)

        if not clean1 or not clean2:
            return {"keyword_score": 0, "matched_keywords": [], "missing_keywords": []}

        vectorizer = TfidfVectorizer(max_features=50, ngram_range=(1, 2))
        vectors = vectorizer.fit_transform([clean1, clean2])
        feature_names = vectorizer.get_feature_names_out()

        tfidf_1 = vectors[0].toarray()[0]
        tfidf_2 = vectors[1].toarray()[0]

        keywords1 = {feature_names[i] for i in range(len(tfidf_1)) if tfidf_1[i] > 0}
        keywords2 = {feature_names[i] for i in range(len(tfidf_2)) if tfidf_2[i] > 0}

        matched = list(keywords1 & keywords2)
        missing = list(keywords2 - keywords1)
        
        keyword_score = (len(matched) / len(keywords2)) * 100 if keywords2 else 0

        return {
            "keyword_score": round(keyword_score, 2),
            "matched_keywords": matched[:15],
            "missing_keywords": missing[:15]
        }

    def calculate_overall(self, semantic: float, keyword: float) -> float:
        return round((semantic * 0.7 + keyword * 0.3), 2)

st.title("Research Paper Similarity Finder")
st.markdown("Compare your work against local files or **arXiv** publications.")


st.sidebar.header("Settings")
input_mode = st.sidebar.radio("Compare against:", ["Local PDF Upload", "arXiv ID"])
st.sidebar.divider()
st.sidebar.info("Tip: arXiv IDs look like `2305.16300` or `cs/0502051`")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Your Paper")
    file1 = st.file_uploader("Upload Base PDF", type=["pdf"], key="file1")

with col2:
    st.subheader("Comparator Paper")
    if input_mode == "Local PDF Upload":
        file2 = st.file_uploader("Upload Comparison PDF", type=["pdf"], key="file2")
        arxiv_id = None
    else:
        arxiv_id = st.text_input("Enter arXiv ID", placeholder="e.g. 1706.03762")
        file2 = None

st.divider()

if st.button(" Run Analysis", use_container_width=True):
    if file1 and (file2 or arxiv_id):
        try:
            with st.spinner("Analyzing papers... this may take a moment."):
                engine = ResearchSimilarityEngine()

                text1 = PDFProcessor.extract_text(file1)

                paper2_title = "Comparator Paper"
                if input_mode == "Local PDF Upload":
                    text2 = PDFProcessor.extract_text(file2)
                else:
                    text2, paper2_title = ArxivProcessor.fetch_by_id(arxiv_id)
                
                if not text1.strip() or not text2:
                    st.error(" Failed to extract text. Please check your files or arXiv ID.")
                    st.stop()

                sem_score = engine.semantic_similarity(text1, text2)
                kw_data = engine.keyword_analysis(text1, text2)
                final_score = engine.calculate_overall(sem_score, kw_data["keyword_score"])

                st.success(f" Analysis Complete: Comparing with '{paper2_title}'")
                st.balloons()

                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric(" Overall Score", f"{final_score}%")
                m_col2.metric(" Semantic Similarity", f"{sem_score}%")
                m_col3.metric(" Keyword Match", f"{kw_data['keyword_score']}%")

                st.divider()

                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.subheader("Shared Concepts")
                    if kw_data["matched_keywords"]:
                        for kw in kw_data["matched_keywords"]:
                            st.info(kw)
                    else:
                        st.write("No major overlapping keywords found.")

                with res_col2:
                    st.subheader(" Missing in Your Paper")
                    if kw_data["missing_keywords"]:
                        for kw in kw_data["missing_keywords"]:
                            st.warning(kw)
                    else:
                        st.write("No unique keywords found in the comparator.")

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please provide both papers to proceed.")

st.caption("Built for Researchers")
