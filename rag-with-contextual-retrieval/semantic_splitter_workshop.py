import argparse
import os
import shutil
import pandas as pd
from langchain_community.document_loaders import PyPDFDirectoryLoader
#from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

CHROMA_PATH = "chroma"
DATA_PATH = "data"

CHUNKING_PROMPT_TEMPLATE = """
Here is the whole document: {document}.
Here is the chunk we want to situate within the whole document: {chunk}.
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else.
"""

#CHUNKING_PROMPT_TEMPLATE = """
#Here is the whole document: {document}.
#Here is the chunk we want to situate within the whole document: {chunk}.
#Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
#Answer only with the succinct context and nothing else. Start the context text with a game name.
#"""

def main():

    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    #documents = load_documents
    df_paths = get_paths_to_files(DATA_PATH)
    for path in df_paths.path:
        documents = load_document_from_path(path)
        chunks = split_documents_semantically(documents)
        ok_chunks = filter_out_too_short_chunks(chunks)
        chunks_with_rephrasing = add_rephrasing_to_chunks(ok_chunks, documents)
        add_to_chroma(chunks_with_rephrasing)


def get_paths_to_files(src, filter_if_contains=None, keep=True):
    df = []
    for directory, subs, files in os.walk(src, topdown=False):
        if files != []:
            for file in files:
                dict_to_df = {'file': file, 'folder': directory, 'path': os.path.join(directory, file)}
                if (filter_if_contains==None) or (keep==True and filter_if_contains in file) or \
                        (keep==False and not filter_if_contains in file):
                        df.append(dict_to_df)
    df = pd.DataFrame(df)
    return df


def load_document_from_path(file_path):
    loader = PyPDFLoader(file_path)
    pages = []
    for page in loader.lazy_load():
        pages.append(page)
    return pages


def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def split_documents_semantically(documents): #documents
    # There is a bug in SemanticChunker. For some reason metadata gets cleared and later on is shared among chunks
    splitter = SemanticChunker(get_embedding_function(), breakpoint_threshold_type="percentile")
    doc_metadata = documents[0].metadata
    docs = splitter.create_documents([document.page_content for document in documents])
    docs_with_metadata = []
    for doc in docs:
        doc.metadata = doc_metadata
        docs_with_metadata.append(doc)
    return docs_with_metadata
    

def filter_out_too_short_chunks(chunks, min_len=40):
    ok_chunks = []
    for chunk in chunks:
        if len(chunk.page_content) > min_len:
            ok_chunks.append(chunk)
    return ok_chunks


def add_rephrasing_to_chunks(chunks, documents):
    rephrase_template = ChatPromptTemplate.from_template(CHUNKING_PROMPT_TEMPLATE)
    model = Ollama(model="llama3.2", mirostat_tau=0) #mistral
    count = 0
    for chunk in chunks:
        rephrasing_prompt = rephrase_template.format(chunk=chunk, document=[document.page_content for document in documents])
        rephrased_text = model.invoke(rephrasing_prompt)
        #rephrased_text = "placeholder" + str(count)
        count += 1
        text_with_context = rephrased_text + "\nfull text:\n" + chunk.page_content
        chunk.page_content = text_with_context
        print("context")
        print(rephrased_text)

    return chunks


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )

    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function(), collection_metadata={"hnsw:space": "cosine"}
    )

    # Calculate Page IDs.
    chunk_ids = calculate_chunk_ids(chunks)
    
    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    new_ids = []
    for chunk_num in range(len(chunks)):
        if chunk_ids[chunk_num] not in existing_ids:
            new_chunks.append(chunks[chunk_num])
            new_ids.append(chunk_ids[chunk_num])
    #print(new_ids)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        db.add_documents(new_chunks, ids=new_ids)
        #new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        #db.add_documents(new_chunks, ids=new_chunk_ids)
        #db.persist()
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0
    chunks_ids = []
    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunks_ids.append(chunk_id)
        # Add it to the page meta-data.

    return chunks_ids


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()
