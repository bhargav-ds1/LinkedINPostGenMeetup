import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import snapshot_download
from dotenv import load_dotenv

"""
This script downloads specific language model and embedding model from Hugging Face Hub, and it tests their 
initial functionality. 
This includes setting up environmental variables, downloading models, and verifying basic operation with example text.

Note: Ensure that all required dependencies are installed before running this script.
"""

# loading env to get the huggingface token
if not load_dotenv('../.envfile'):
    raise ValueError('Could not load the specified env file')

# Downloading an embedding model from hugginface repository
embedding_model_id = 'sentence-transformers/all-MiniLM-L12-v2'

downloaded_model_path = snapshot_download(
    repo_id=embedding_model_id,
    token=os.environ['hugging_face_token'],
    local_dir='./Models/' + str(embedding_model_id)
)

print('Embedding model downloaded to:' + str(downloaded_model_path))

# Downloading an LLM model from huggingface repository
LLM_model_id = 'meta-llama/Llama-2-7b-chat-hf'

downloaded_model_path = snapshot_download(
    repo_id=LLM_model_id,
    token=os.environ['hugging_face_token'],
    local_dir='./Models/' + str(LLM_model_id)
)

print('LLM model downloaded to:' + str(downloaded_model_path))

""" 
    The following commented section can be used to verify the functionality of the embedding model.
    Uncomment this to check embedding outputs for example sentences using the SentenceTransformer class.
"""
# from sentence_transformers import SentenceTransformer
# sentences = ["This is an example sentence", "Each sentence is converted"]

# model = SentenceTransformer(embedding_model_id)
# embeddings = model.encode(sentences)
# print(embeddings)

# Initialize and verify the functionality of the LLM
tokenizer = AutoTokenizer.from_pretrained(LLM_model_id)

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path='./Models/' + LLM_model_id,
    device_map="cpu",
    torch_dtype=torch.float16,
)
messages = [
    {"role": "user", "content": "What is your favourite condiment?"},
    {"role": "assistant",
     "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    {"role": "user", "content": "Do you have mayonnaise recipes?"}
]

# Prepare messages for model input using the tokenizer's chat-specific functionality
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cpu")
# Generate responses using the loaded model
outputs = model.generate(inputs, max_new_tokens=20, pad_token_id=tokenizer.eos_token_id)
# Decode and print the generated response
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
