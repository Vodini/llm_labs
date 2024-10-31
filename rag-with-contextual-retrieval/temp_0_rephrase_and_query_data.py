import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

REPHRASE_TEMPLATE = """
Rephrase a question to a following format:
- state the name of the game
- after a colon state the question about the game rule. Be as strict as possible. Do not repeat the game name.
Example of correctly rephrased question for question "how do I build a hotel in monopoly?":
Monopoly: how to build a hotel?
Rephrase the following question according to abovementioned rules: {question} 
"""

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):

    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    #Rephrase the question
    rephrase_template = ChatPromptTemplate.from_template(REPHRASE_TEMPLATE)
    rephrasing_prompt = rephrase_template.format(question=query_text)
    #print(rephrasing_prompt)
    model = Ollama(model="mistral", mirostat_tau=0)
    rephrased_text = model.invoke(rephrasing_prompt)
    print(rephrased_text)
    
    # Search the DB.
    results = db.similarity_search_with_score(rephrased_text, k=5)
    print(results)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=rephrased_text)
    # print(prompt)

    model = Ollama(model="mistral", mirostat_tau=0)
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()
