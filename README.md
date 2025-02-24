# ccrag
Simple implementation of RAG (Retrial-Augmented Generation), that works with local text files and Ollama for embedding generation and LLM query.

This tool can generate vector embedding for set of text files. Then it can use those embedding to find a closes document to a user query. Then it can query and LLM with that document prepended to the context.

# Making embeddings from a file library

```bash
find /Users/kif/roam -name "*.org" | ccrag -e
```

# Making query

```bash
ccrag -q "What do Icelandic pop stars do with television?"

# Only run similarity caparison without feeding result to LLM. This will output best matched files paths
ccrag -s -q "What do Icelandic pop stars do with television?"
```
# Configuration

You can configure this tool by setting the following environmental variables:

```
# The values in this example are default values that the tool is using
export CCRAG_OLLAMA_ADDRESS="http://localhost:11434"
export CCRAG_EMBED_MODEL="mxbai-embed-large"
export CCRAG_LLM_MODEL="mistral:latest"
export CCRAG_MAX_RESULTS=3
```

# How it works

## Preprocessing 
1. Take a collection if text files (other file types should be also possible with extra work)
2. Feed each file or its chunk into an embedding model
3. Store generated embedding vectors for each chunk

## Query
1. Take user query
2. Generate embedding for user query
3. Go over all stored embeddings and compare it to user query embedding using Cosine Similarity or Euclidean distance.
   Basically find how close user query is to particular chunk in higher dimensional space. That usually corresponds to semantic closeness.
4. Sort all result base in the distance from the previous step
5. Select best N result (usually 3-5, depending on the chunk size)
6. Load actual text chunks that correspond to that N embeddings
7. Prepend it to a prompt that looks rough the following "I have an information: {text from N chunks}, Please answer this user question {user_query} using that information"

