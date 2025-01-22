import os
import pandas as pd
import openai
from openai import OpenAI
from tqdm import tqdm

# For example, replace with an environment variable:
openai_api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key = openai_api_key)
# Function to get response from OpenAI
def get_openai_response(question):
    try:
        completion = client.chat.completions.create(
            model="ft:gpt-4o-2024-08-06:carnegie-mellon-university::ADM1aI5p",
            messages=[
                {"role": "system", "content": "WiLL is a FCC regulations expert chatbot."},
                {"role": "user", "content": question}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error occurred: {e}")
        return None

# Read questions from CSV file
# df = pd.read_csv('./../csv_files/rag_questions.csv', low_memory=False)
df = pd.read_csv('/home/wairimu/Downloads/finetuning_test_dataset.csv', low_memory=False)


results = []


# Process each question
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing questions"):
    question = row['Question']
    answer = get_openai_response(question)
    results.append({
        'Question': question,
        'Answer': answer
    })

results = pd.DataFrame(results)
# Save results to a new CSV file
output_file = '/home/wairimu/Downloads/finetuning_test_results.csv'
results.to_csv(output_file, index=False)
print(f"Responses saved to {output_file}")