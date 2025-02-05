import ast  # for converting embeddings saved as strings back to arrays
import openai
import pandas as pd  # for storing text and embeddings data
import tiktoken  # for counting tokens
import os
from scipy import spatial
from datasets import load_dataset
import re  # for cutting <ref> links out of Wikipedia articles
from tqdm.notebook import tqdm

# Chunking Text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument

# Reranking
from ragatouille import RAGPretrainedModel

from typing import Optional

# Asynchronous requests
import aiohttp
import asyncio
from tqdm.asyncio import tqdm as atqdm

# Embedded chuck 
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from collections import defaultdict

# Load environment variables
load_dotenv()

# Global variables
client = None  # Initialize global client variable
GPT_MODEL = "gpt-4-turbo"  # Reader model
EMBEDDING_MODEL = "text-embedding-ada-002"  # Embedding model
EMBEDDING_VECTOR_FILE = "./csv_files/embeddings/part15_embeddingvector_ada.csv"

def setup_openai():
    """Initialize OpenAI client with API key."""
    global client  # Declare we're using the global client
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    return client

def ask(query: str, client, model: str = GPT_MODEL) -> str:
    """Sends a query to the OpenAI API and returns the response."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": query},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )

    return response.choices[0].message.content

def load_embeddings():
    """Load embeddings from CSV file"""
    df = pd.read_csv(EMBEDDING_VECTOR_FILE, sep=',')
    sections = df['section'].tolist()
    embeddings = df['embedding'].apply(ast.literal_eval).tolist()
    return pd.DataFrame({"text": sections, "embedding": embeddings})

def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 10
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    
    query_embedding_response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response.data[0].embedding

    max_sections = 15
    # Group chunks by headers
    header_to_chunks = defaultdict(list)
    for i, row in df.iterrows():
        header = row["text"].split("\n")[0]  # Assuming the header is the first line
        header_to_chunks[header].append((row["text"], row["embedding"]))

    # Calculate relatedness for each header
    header_relatednesses = []
    for header, chunks in header_to_chunks.items():
        max_relatedness = max(relatedness_fn(query_embedding, chunk[1]) for chunk in chunks)
        header_relatednesses.append((header, max_relatedness))

    # Sort headers by relatedness
    header_relatednesses.sort(key=lambda x: x[1], reverse=True)

    # Collect all chunks for top headers, limiting to max_sections
    strings = []
    relatednesses = []
    for header, relatedness in header_relatednesses[:min(top_n, max_sections)]:
        for chunk, _ in header_to_chunks[header]:
            strings.append(chunk)
            relatednesses.append(relatedness)
        if len(strings) >= max_sections:
            break

    return strings[:max_sections], relatednesses[:max_sections]

def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def query_message(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int,
    top_n: int = 10,
) -> tuple[str, list[str], list[str]]:
    strings, relatednesses = strings_ranked_by_relatedness(query=query, df=df, top_n=top_n)
    introduction = 'Use the below passages on the FCC regulations to answer the subsequent question. Ensure your answer includes a yes/no/"I could not find an answer." response, the relevant section number(s), and a detailed explanation.'
    question = f"\n\nQuestion: {query}"
    message = introduction
    sources = []
    embedding_strings = strings

    for string in strings:
        next_article = f'\n\nFCC excerpt:\n"""\n{string}\n"""'
        if num_tokens(message + next_article + question, model=model) > token_budget:
            break
        else:
            message += next_article
            sources.append(next_article)

    return message + question, sources, embedding_strings

def ask(
    query: str,
    df: pd.DataFrame,
    model: str = GPT_MODEL,
    token_budget: int = 4096 - 500,
    top_n: int = 10,
) -> tuple[str, list[str]]:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    message, sources, embedding_strings = query_message(
        query, df, model=model, token_budget=token_budget, top_n=top_n
    )

    # Use only the top N most relevant sections
    top_sections = embedding_strings[:top_n]
    
    focused_prompt = f"""Based on the following FCC regulation excerpts, please answer the question. 
    Be sure to cite the specific section number(s) in your answer.

    Relevant FCC Regulations:
    {"".join(top_sections)}

    Question: {query}

    Your answer must strictly follow this format:
    Answer: [Yes/No/I could not find an answer]
    Section: [Relevant FCC regulation section number(s) if the answer is Yes or No. "N/A" if the answer is I could not find an answer]
    Calculation: [Detailed calculation based on the provided regulations]

    If the provided regulations do not contain enough information to answer the question, state "I could not find an answer".
    """

    messages = [
        {"role": "system", "content": "You are an FCC regulations expert. Provide concise and accurate responses based solely on the given information."},
        {"role": "user", "content": focused_prompt},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )

    return response.choices[0].message.content, embedding_strings

def main():
    """Main function to handle user interaction."""
    # Setup OpenAI client
    setup_openai()  # This will set the global client
    
    print("\nWelcome to WiLL - Your personal FCC regulations assistant")
    print("================================================")
    print("\nLoading FCC regulations database...")
    
    try:
        df = load_embeddings()
        print("Database loaded successfully!")
    except Exception as e:
        print(f"Error loading database: {e}")
        return

    while True:
        print("\nPlease enter your question about FCC regulations (or 'quit' to exit):")
        query = input("> ")
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("\nThank you for using WiLL. Goodbye!")
            break

        print("\nAnalyzing FCC regulations...\n")
        try:
            answer, _ = ask(query, df, top_n=9)
            print("=" * 50)
            print(answer)
            print("=" * 50)
        except Exception as e:
            print(f"Error processing your question: {e}")
            print("Please try again or contact support if the problem persists.")

if __name__ == "__main__":
    main() 