# Making embedings from a file library

```bash
find /Users/kif/roam -name "*.org" | go run main.go -e
```

# Making query

```bash
go run main.go -q "What do Icelandic pop stars do with television?"

# Only run similarity coparison without feeding result to LLM. This will output best matched files.
go run main.go -s -q "What do Icelandic pop stars do with television?"
```