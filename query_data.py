import argparse
from xml.parsers.expat import model

from click import prompt
# from dataclasses import dataclass
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

CHROMA_PATH = 'chroma'

PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
Answer the question based on the above context: {query}"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type='str', help='This is the query text')
    args = parser.parse_args()
    query_text = args.query_text
    print(query_text)

    # prepare the db
    embedding_function = OpenAIEmbeddings()
# vector = embedding_function.embed_query('apple')
# print(vector)
# print(len(vector))

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    #search the db
    results = db.similarity_search_with_relevance_scores(query_text, k=3)

#return type of the search
# List[tuple[Document, float]]

    if len(results) == 0 or results[0][1] < 0.7:
        print('Unable to find matching results')
        return


    context_text = '\n\n---\n\n'.join([doc.page_content for doc,_score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, query=query_text)

    model = ChatOpenAI()
    response = model.predict(prompt)

    sources = [doc.metadata.get('source', None) for doc, _score in results]
    formatted_response = f'Response: {response}\n Sources: {sources}'
    print(formatted_response)


if __name__ == '__main__':
    main()