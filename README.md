# PDF-Based AI Chatbot with RAG

This project is an AI-powered chatbot designed to interact with PDF documents. It employs Retrieval-Augmented Generation (RAG) and language models to provide meaningful insights and personalized training plans based on the content of the PDF and user interactions.

## Features

### 1. Querying the PDF (Level 1)
- Users can ask questions directly related to the PDF's content.
- The chatbot retrieves relevant information using vector embeddings and provides concise, context-aware answers.

### 2. Generating Questions (Level 2)
- The chatbot generates open-ended questions based on the PDF's main topics.
- These questions aim to test the user's understanding of the document.

### 3. Training Plan Evaluation (Level 3)
- Based on user responses to the generated questions, the chatbot evaluates knowledge gaps.
- It suggests a structured training plan tailored to improve understanding.

## Tech Stack

- **Programming Language**: Python
- **Key Libraries**:
  - [LangChain](https://www.langchain.com): For chaining AI models and retrieval mechanisms.
  - [Streamlit](https://streamlit.io): For building the interactive web application.
  - [HuggingFace Transformers](https://huggingface.co): For state-of-the-art embeddings and models.
  - [Chroma](https://docs.trychroma.com/): For vector storage and similarity search.
  - [PyPDF2](https://pypdf2.readthedocs.io/): For loading and parsing PDF documents.

## Installation

### Prerequisites
- Python 3.8 or later
- A Hugging Face account and an API token ([Sign up here](https://huggingface.co/join)).

### Steps
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/pdf-ai-chatbot.git
    cd pdf-ai-chatbot
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up environment variables:
   - Create a `.env` file in the project root.
   - Add your Hugging Face API token:
     ```
     HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
     ```

4. Add your PDF file:
   - Place the document(s) you want to analyze in the `data/` folder.

5. Initialize the vector database:
    ```bash
    python utils.py
    ```

6. Run the application:
    ```bash
    streamlit run app.py
    ```

## Usage

1. Open the app in your browser (default: http://localhost:8501).
2. **Level 1**: Ask questions about the PDF and get precise answers.
3. **Level 2**: Generate open-ended questions about the document's key topics.
4. **Level 3**: Provide responses to the questions and receive a customized training plan.
