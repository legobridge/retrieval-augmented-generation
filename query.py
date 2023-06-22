# A major problem right now is Google blocking the queries for some safety reasons.
import sys
import time

import chromadb
import google.generativeai as palm
from chromadb import Settings
from google.generativeai.types import SafetySettingDict, HarmCategory, HarmBlockThreshold

from generate_embeddings import generate_embedding

SAFETY_SETTINGS = [
    SafetySettingDict(category=hc, threshold=HarmBlockThreshold.BLOCK_NONE) for hc in HarmCategory
]


def retrieve_documents_from_vector_db(collection, model, query, num_candidates=5):
    query_embedding = generate_embedding(model, query)
    results = collection.query(query_embeddings=[query_embedding], n_results=num_candidates)
    retrieved_documents = results['documents'][0]
    return retrieved_documents


def main(palm_key):
    palm.configure(api_key=palm_key)

    models = [m for m in palm.list_models() if 'embedText' in m.supported_generation_methods]
    embedding_model = models[0]

    models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
    text_model = models[0].name

    chroma_client = chromadb.Client(Settings(chroma_db_impl='duckdb+parquet', persist_directory='chroma_db'))
    collection = chroma_client.get_collection(name="email_vector_db")

    # query = 'Which emails have instances of hiding knowledge of a crashing game engine?'
    query = input('Enter your query in the format of the question you want answered.\n')
    retrieved_emails = retrieve_documents_from_vector_db(collection, embedding_model, query)

    with open('prompt_templates/final_prompt.txt') as f:
        final_prompt_template = f.read()
        for retrieved_email in retrieved_emails:
            final_prompt = final_prompt_template.format(retrieved_email, query)
            response = palm.generate_text(
                model=text_model,
                prompt=final_prompt,
                temperature=0,
                safety_settings=SAFETY_SETTINGS
            )
            time.sleep(2)
            print(final_prompt)
            print('=======================================================================')
            if len(response.candidates) > 0:
                print(response.candidates[0]['output'])
            else:
                print('Prompt blocked by PaLM safety settings')
            print('=======================================================================\n')


if __name__ == '__main__':
    palm_key = sys.argv[1]
    main(palm_key)
