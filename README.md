# ccrag

Simple implementation of RAG (Retrieval-Augmented Generation) that works with local text files and Ollama for embedding generation and LLM querying.

This tool can generate vector embeddings for a set of text files. Then it can use those embeddings to find the closest document to a user query. Finally, it can query an LLM with that document prepended to the context.

# Key Features
- Generate vector embeddings for local text documents
- Semantic search across document corpus
- Fully local processing with Ollama integration
- No other dependencies other then Ollama

# Prerequisites

You will need [Ollama](https://ollama.com/) running on your local or remote machine.

# Build and install

```bash
mkdir bin
go build -o bin/ccrag main.go
cp bin/ccrag /usr/local/bin/ # Or any other location in your path, alternatively you can also use a symlink
```

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

# How It Works

## Preprocessing 
1. Take a collection of text files (other file types should also be possible with additional work)
2. Feed each file or its chunk into an embedding model
3. Store generated embedding vectors for each chunk

## Query
1. Take user query
2. Generate embedding for user query
3. Go over all stored embeddings and compare it to the user query embedding using Cosine Similarity or Euclidean distance. 
   Basically, find how close the user query is to a particular chunk in higher-dimensional space. This usually corresponds to semantic closeness.
4. Sort all results based on the distance from the previous step
5. Select the best N results (usually 3-5, depending on the chunk size)
6. Load the actual text chunks that correspond to those N embeddings
7. Prepend it to a prompt that looks roughly like the following: "I have information: {text from N chunks}. Please answer this user question {user_query} using that information"

