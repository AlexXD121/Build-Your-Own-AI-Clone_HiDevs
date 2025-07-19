# AI Clone Project

This project is a simple implementation of a Retrieval-Augmented Generation (RAG) based AI assistant that can respond to queries using documents stored locally.

## Features

- Ingests PDFs into a vector store using embeddings
- Uses LangChain for RAG implementation
- Streamlit-based user interface
- Environment variable support via `.env`

## Requirements

- Python 3.10+
- All dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/ai_clone_project.git
   cd ai_clone_project
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ```

## Running the Project

### Step 1: Create RAG Index
This will convert all PDFs in the `./docs` folder into a FAISS vector store:
```
python rag_indexer.py
```

### Step 2: Run the App
Start the Streamlit app:
```
streamlit run app.py
```

The app will open in your browser.

## Troubleshooting

- If you see `ImportError: cannot import name 'OpenInferenceTracer'`, remove the line from `app.py` that says:
  ```python
  from phoenix.trace import OpenInferenceTracer
  ```

## License

This project is licensed under the MIT License.

Demo link : https://drive.google.com/file/d/1A_J4ukzkecA0yUAZzCh_UkIivdnwPUom/view?usp=drive_link
