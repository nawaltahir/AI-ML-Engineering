from web_scraper import scrape_and_save_text

# 📌 Step 1: Scrape website and write to file
scrape_and_save_text("https://en.wikipedia.org/wiki/LangChain", "data/langchain.txt")

# 📌 Step 2: Now load it for embeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# Load and split
loader = TextLoader("data/langchain.txt")
docs = loader.load()
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embedding)

# Load Zephyr via Hugging Face Inference API
pipe = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-beta")
llm = HuggingFacePipeline(pipeline=pipe)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory
)
