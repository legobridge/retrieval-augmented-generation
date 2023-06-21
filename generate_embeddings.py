import os
import sys
import time

import chromadb
import google.generativeai as palm
from chromadb import Settings
from tqdm import tqdm


def generate_embedding(model, document):
    try:
        return palm.generate_embeddings(model=model, text=document)['embedding']
    except:
        time.sleep(30)
        return generate_embedding(model, document)


def main(palm_key):
    palm.configure(api_key=palm_key)

    models = [m for m in palm.list_models() if 'embedText' in m.supported_generation_methods]
    model = models[0]

    chroma_client = chromadb.Client(Settings(chroma_db_impl='duckdb+parquet', persist_directory='chroma_db'))
    db_name = "email_vector_db"
    collection = chroma_client.create_collection(name=db_name, get_or_create=True)

    emails = []
    embeddings = []

    for email_file in tqdm(os.listdir('generated_emails')):
        if email_file[-4:] == '.txt':
            with open(f'generated_emails/{email_file}') as f:
                email = f.read()
                embedding = generate_embedding(model, email)
                emails.append(email)
                embeddings.append(embedding)
                time.sleep(0.2)

    collection.add(embeddings=embeddings,
                   documents=emails,
                   ids=[str(i) for i in range(1, len(emails) + 1)])

    chroma_client.persist()


if __name__ == '__main__':
    palm_key = sys.argv[1]
    main(palm_key)
