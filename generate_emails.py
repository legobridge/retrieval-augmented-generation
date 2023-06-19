# The script generates very vanilla emails as of now,
# and does not attempt to maintain coherence between subsequent emails.
import sys
import time

import google.generativeai as palm
import pandas as pd
import numpy as np
from tqdm import tqdm


def generate_email(model, employee_list, sender, recipient, include_scandal):
    prompt = f"""\
Spintendo is a game development company. The following people work at Spintendo:

{employee_list}

Write an email from {sender} to {recipient}.
Make sure that the email is relevant to their position at the company, \
and the tone is appropriate for their relationship and seniority."""

    if include_scandal:
        prompt += f" Add a subtle but legally incriminating secret in the email, \
        which {sender} and {recipient} are trying to cover up."
    try:
        email_text = palm.generate_text(
            model=model,
            prompt=prompt,
            temperature=0.6
        )
        return email_text.result
    except:
        time.sleep(30)
        return generate_email(model, employee_list, sender, recipient, include_scandal)


def main(palm_key):
    palm.configure(api_key=palm_key)

    models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
    model = models[0].name

    employee_df = pd.read_csv('employees.csv')
    employee_list = '\n'.join([f"{row[0]} [{row[1]}]" for _, row in employee_df.iterrows()])
    employee_df = employee_df.set_index('Name')
    EMAILS_TO_GENERATE = 50
    HOT_DOC_RATE = 0.01
    for i in tqdm(range(EMAILS_TO_GENERATE)):
        sender, recipient = np.random.choice(employee_df.index.values, size=2, replace=False)
        include_scandal = False
        if np.random.rand() < HOT_DOC_RATE:
            include_scandal = True

        generated_email = generate_email(model, employee_list, sender, recipient, include_scandal)
        time.sleep(2)
        hot = ''
        if include_scandal:
            hot = 'hot_'
        filename = f'{hot}email_{i}_{sender}_{recipient}'
        with open(f'generated_emails/{filename}.txt', 'w') as f:
            f.write(generated_email)


if __name__ == '__main__':
    palm_key = sys.argv[1]
    main(palm_key)
