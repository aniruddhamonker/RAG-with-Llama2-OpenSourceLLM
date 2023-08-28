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

    * git clone https://github.com/aniruddhamonker/llm-document-retrieval.git

2. Install requirements

    * cd llm-document-retrieval

    * pip install -r requirements.txt

3. download Llama2-chat 7B parameter open source model and instructor-large embedding model from hugging face library

    * mkdir models && cd models

    * wget https://huggingface.co/localmodels/Llama-2-7B-Chat-ggml/resolve/main/llama-2-7b-chat.ggmlv3.q4_0.bin

    * wget https://huggingface.co/hkunlp/instructor-large/resolve/main/pytorch_model.bin

4. If you are going to use OpenAI GPT models , then update .env file with your Open AI api key

    * cd ..
    * mv .env.bak .env

    * "update .env file with your API key"

5. create a virtual environment and activate

    * virtualenv .
    * source bin/activate

6. Run the app with Streamlit

    * streamlit run chat_app.py

# Demo

![image](https://github.com/aniruddhamonker/llm-document-retrieval/assets/17957255/53b90e0d-62a3-4ab9-83af-ed2099944086)

