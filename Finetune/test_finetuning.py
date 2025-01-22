import json
import tiktoken # for token counting
import numpy as np
from collections import defaultdict

data_path = "./finetuning_data.jsonl"

# Load the dataset
with open(data_path, 'r', encoding='utf-8') as f:
    dataset = [json.loads(line) for line in f]

# Initial dataset stats
print("Num examples:", len(dataset))
print("First example:")
for message in dataset[0]["messages"]:
    print(message)




'''
done with help from https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset and https://cookbook.openai.com/examples/chat_finetuning_data_prep 
'''

# examples:
# {"messages": [{"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."}, \
#               {"role": "user", "content": "What's the capital of France?"}, \
#                 {"role": "assistant", "content": "Paris, as if everyone doesn't know that already."}]}
# {"messages": [{"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."}, \
#               {"role": "user", "content": "Who wrote 'Romeo and Juliet'?"}, \
#                 {"role": "assistant", "content": "Oh, just some guy named William Shakespeare. Ever heard of him?"}]}
# {"messages": [{"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."}, \
#               {"role": "user", "content": "How far is the Moon from Earth?"}, \
#                 {"role": "assistant", "content": "Around 384,400 kilometers. Give or take a few, like that really matters."}]}
