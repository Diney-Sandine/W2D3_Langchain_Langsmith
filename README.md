# W2D3_Langchain_Langsmith
import os
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 0. Load Environment Variables & Enforce LangSmith Tracing Configuration
# ---------------------------------------------------------------------------
load_dotenv(override=True)

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "default"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"

# If your key is not picking up from .env, uncomment and paste it below:
# os.environ["LANGSMITH_API_KEY"] = "lsv2_pt_..."

from langsmith import traceable
from langsmith.wrappers import wrap_openai
from openai import OpenAI

# Wrap native OpenAI client for traceable direct calls
wrapped_openai_client = wrap_openai(OpenAI())

# ---------------------------------------------------------------------------
# 1. Write the CA course details to 'Sample.txt'
# ---------------------------------------------------------------------------
ca_text = (
    "The Chartered Accountancy (CA) course, governed by The Institute of Chartered "
    "Accountants of India (ICAI), is a globally recognized professional program designed "
    "to create experts in financial reporting, auditing, taxation, and business advisory, "
    "typically spanning 4 to 4.5 years. The educational journey begins with CA Foundation, "
    "an entry-level exam for Class 12 graduates covering Accounting, Business Laws, Quantitative "
    "Aptitude, and Business Economics over a 4-month preparation period. Next is CA Intermediate, "
    "spanning 8 months of study and divided into Group I (Advanced Accounting, Corporate and Other Laws, "
    "and Taxation) and Group II (Cost and Management Accounting, Auditing and Ethics, and Financial "
    "Management and Strategic Management). Following intermediate exams, students undergo 2 years of "
    "mandatory practical training (articleship) under a practicing Chartered Accountant for hands-on "
    "professional experience. Finally, the CA Final level tests advanced strategic and multidisciplinary "
    "decision-making across Group I (Financial Reporting, Advanced Financial Management, Advanced Auditing, "
    "Assurance and Professional Ethics) and Group II (Direct Tax Laws and International Taxation, "
    "Indirect Tax Laws, and Integrated Business Solutions)."
)

try:
    with open("Sample.txt", "w", encoding="utf-8") as file:
        file.write(ca_text)
    print("Successfully created and wrote text to 'Sample.txt'!")
except Exception as e:
    print(f"An error occurred while creating Sample.txt: {e}")


# ---------------------------------------------------------------------------
# 2. Load the PDF using LangChain's PyPDFLoader
# ---------------------------------------------------------------------------
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("ICAI-Prospectus.pdf")
pages = loader.load()
print(f"Successfully loaded {len(pages)} page(s) from the PDF.")


# ---------------------------------------------------------------------------
# 3. Split the documents into smaller chunks
# ---------------------------------------------------------------------------
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,     # Maximum size of each chunk (in characters)
    chunk_overlap=50    # Overlap to maintain context between chunks
)

docs = text_splitter.split_documents(pages)
print(f"Successfully split the document into {len(docs)} chunks!")


# ---------------------------------------------------------------------------
# 4. Initialize OpenAI Embeddings Model
# ---------------------------------------------------------------------------
from langchain_openai import OpenAIEmbeddings

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
print("Successfully initialized OpenAI embeddings model!")


# ---------------------------------------------------------------------------
# 5. Create or Load the FAISS Vector Store Locally & Configure Retriever (k=3)
# ---------------------------------------------------------------------------
from langchain_community.vectorstores import FAISS

VECTORSTORE_PATH = "faiss_index"

if os.path.exists(VECTORSTORE_PATH):
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH, 
        embeddings_model, 
        allow_dangerous_deserialization=True
    )
    print("Successfully loaded FAISS vector store from local folder!")
else:
    vectorstore = FAISS.from_documents(docs, embeddings_model)
    vectorstore.save_local(VECTORSTORE_PATH)
    print("Successfully created and saved FAISS vector store to local folder!")

# Configure retriever to fetch top 3 matching chunks (k=3)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("Successfully created a retriever with k=3 from the FAISS vector store!")


# ---------------------------------------------------------------------------
# 6. Initialize ChatOpenAI Model (gpt-4o-mini)
# ---------------------------------------------------------------------------
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
print("Successfully initialized ChatOpenAI (gpt-4o-mini) model!")


# ---------------------------------------------------------------------------
# 7. Define Chained Workflows with @traceable
# ---------------------------------------------------------------------------

@traceable(name="Format Retrieved Documents")
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- Chain 1: RAG Extraction Chain ---
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert academic advisor for the ICAI Chartered Accountancy course. Answer accurately based on the context provided."),
    ("human", "Context:\n{context}\n\nQuestion:\n{question}")
])

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

# --- Chain 2: Formatting/Refinement Chain ---
formatting_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a professional editor. Clean up and structure the provided text into clear, neat markdown bullet points grouped clearly by groups and subjects."),
    ("human", "Raw Answer to format:\n{raw_answer}")
])

formatting_chain = (
    formatting_prompt
    | llm
    | StrOutputParser()
)

# --- Full Pipeline (Chaining Chain 1 and Chain 2 together) ---
full_pipeline = {"raw_answer": rag_chain} | formatting_chain


# ---------------------------------------------------------------------------
# 8. Execute the Pipeline and Print Results in Terminal
# ---------------------------------------------------------------------------
stage_prompts = [
    "What are the specific groups and respective subjects included in the CA Intermediate stage?",
    "What are the specific groups and respective subjects included in the CA Final stage?"
]

for question in stage_prompts:
    print(f"\n" + "="*50)
    print(f"QUESTION: {question}")
    print("="*50)
    
    # Invoke the full chained pipeline and print output
    final_output = full_pipeline.invoke(question)
    print(final_output)
    print("\n" + "-"*50)
    
