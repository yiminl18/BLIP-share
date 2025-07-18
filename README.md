### Load OpenAI API 

Store your OpenAI API in by using 'export OPENAI_API_KEY="your-api-key-here"' or set the OpenAI API key as an environment variable in your system log. Alternatively, you can store the API keys locally and copy the path to the key in the codes for specific models under models/ folder. 

### Install Package

```bash
pip install -e . 
```

### Infer a single minimal provenance

Run BLIP/infer_single_provenance.py, where you can choose any strategy combination, any dataset, and any LLM models. 

### Infer k minimal provenance

Run BLIP/infer_k_provenance.py, where you can choose any dataset, and any LLM models.

### Infer minimal provenance in tableQA

Run BLIP/infer_provenance_tableqa.py, where you can choose any strategy combination, any dataset, and any LLM models. 