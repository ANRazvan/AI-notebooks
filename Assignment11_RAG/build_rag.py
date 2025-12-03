import os
import lancedb
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import LanceDB
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load API Keys
load_dotenv()


urls = [
    "https://www.almanac.com/plant/aloe-vera",
    "https://www.almanac.com/plant/cucumbers",
    "https://www.almanac.com/plant/tomatoes",
    "https://www.almanac.com/plant/spinach",
    "https://www.almanac.com/plant/mangoes",
    "https://www.gardendesign.com/succulents/aloe-vera.html",
    "https://www.gardendesign.com/vegetables/corn.html",
    "https://www.gardendesign.com/vegetables/eggplant.html",
    "https://www.gardendesign.com/fruit/watermelon.html",
    "https://www.gardendesign.com/fruit/cantaloupe.html"
]

def build_knowledge_base():
    print(f"📡 Scrapping {len(urls)} websites...")
    
    # WebBaseLoader uses BeautifulSoup to strip HTML tags and get just the text
    loader = WebBaseLoader(urls)
    docs = loader.load()
    
    # We split text into chunks of 1000 characters with 200 overlap.
    # This ensures the context isn't cut off in the middle of a sentence.
    print("✂️ Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)
    print(f"🧩 Created {len(chunks)} text chunks.")

    # We create a local folder 'lancedb_data' to store the vectors
    db = lancedb.connect("./lancedb_data")
    
    # This sends text to OpenAI to get numbers (embeddings) and saves them to LanceDB
    print("💾 Embedding and saving to LanceDB...")
    table_name = "plant_care_guides"
    
    # If table exists, we overwrite it for this demo to keep it clean
    try:
        db.drop_table(table_name)
    except:
        pass

    vector_store = LanceDB.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(),
        connection=db,
        table_name=table_name
    )
    
    print("✅ Success! Knowledge base built at './lancedb_data'")

if __name__ == "__main__":
    build_knowledge_base()