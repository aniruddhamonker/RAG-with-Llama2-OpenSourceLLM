# LLM Document Retrieval
This project is a simple implementation of a document retrieval system using an Open Source Large Language Model (LLM) "LLAMA2" from Meta. 

The LLM Document Retrieval system is designed to retrieve documents based on a user’s query. 

The project implemenation is done using Langchain framework and an open sourece vector database "Chroma".
For vector embeddings, the project uses an open source instructor embedding model from HuggingFace model library.

The advantage of using an open source LLM model for implementation is that organizations can use these models without having to pay for expensive licensing fees.
Also Organizations can see how they work and what data they are using. This can help organizations ensure that the models are working as intended and that they are not biased, 
not to mention the data remains protected inside an organizations domain.


# Installation

1. clone the repository from GitHub:

git clone https://github.com/aniruddhamonker/llm-document-retrieval.git

2. Install requirements

cd llm-document-retrieval
pip install -r requirements.txt


Usage
To use the LLM Document Retrieval system, you need to first download the pre-trained LLM model from Hugging Face:

pip install transformers
Copy
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
Copy
Then you can use the retrieve_documents function to retrieve documents based on a user’s query:

from llm_document_retrieval import retrieve_documents

query = "What is the capital of France?"
documents = retrieve_documents(query, model, tokenizer)
Copy
Contributing
Contributions are welcome! If you find any bugs or have any suggestions for improvement, please open an issue or submit a pull request.