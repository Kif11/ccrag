package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"time"

	cc "github.com/kif11/cclib"
)

type EmbeddingResponse struct {
	Model           string      `json:"model"`
	Embeddings      [][]float64 `json:"embeddings"`
	TotalDuration   int64       `json:"total_duration"`
	LoadDuration    int64       `json:"load_duration"`
	PromptEvalCount int         `json:"prompt_eval_count"`
	Source          string      `json:"source"`
}

type ScoredResult struct {
	Score float64
	Path  string
}

type OllamaResponse struct {
	Model              string `json:"model"`
	CreatedAt          string `json:"created_at"`
	Response           string `json:"response"`
	Done               bool   `json:"done"`
	DoneReason         string `json:"done_reason"`
	Context            []int  `json:"context"`
	TotalDuration      int64  `json:"total_duration"`
	LoadDuration       int64  `json:"load_duration"`
	PromptEvalCount    int    `json:"prompt_eval_count"`
	PromptEvalDuration int64  `json:"prompt_eval_duration"`
	EvalCount          int    `json:"eval_count"`
	EvalDuration       int64  `json:"eval_duration"`
}

var ollamaAddress = cc.GetEnv("CCRAG_OLLAMA_ADDRESS", "http://localhost:11434")
var embedModel = cc.GetEnv("CCRAG_EMBED_MODEL", "mxbai-embed-large")
var llmModel = cc.GetEnv("CCRAG_LLM_MODEL", "mistral:latest")
var maxResults = cc.GetEnvInt("CCRAG_MAX_RESULTS", 3)
var embedDirName = "embed"
var embedFormat = "json"

var client = &http.Client{
	Timeout: 3 * time.Minute,
}

// cosineSimilarity calculates cosine similarity (magnitude-adjusted dot
// product) between two vectors that must be of the same size.
func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
		panic("different lengths")
	}

	var aMag, bMag, dotProduct float64
	for i := 0; i < len(a); i++ {
		aMag += a[i] * a[i]
		bMag += b[i] * b[i]
		dotProduct += a[i] * b[i]
	}
	return dotProduct / (math.Sqrt(aMag) * math.Sqrt(bMag))
}

func embed(data string) (EmbeddingResponse, error) {
	payload := map[string]string{
		"model": embedModel,
		"input": data,
	}

	jsonData, err := json.Marshal(payload)
	if err != nil {
		return EmbeddingResponse{}, err
	}

	resp, err := client.Post(ollamaAddress+"/api/embed", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return EmbeddingResponse{}, err
	}
	defer resp.Body.Close()

	var result EmbeddingResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return EmbeddingResponse{}, err
	}

	return result, nil
}

func main() {
	embedMode := flag.Bool("e", false, "Embedding mode. Process list of text file provided over stdin.")
	query := flag.String("q", "", "Query mode. Search for the given query. And generate LLM response with context from similarity search.")
	similarityOnly := flag.Bool("s", false, "Run similarity search only. Output found file list.")
	verbose := flag.Bool("v", false, "Verbose mode.")
	flag.Parse()

	homeDir, err := os.UserHomeDir()
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	embedDir := filepath.Join(homeDir, ".ccrag", embedDirName)

	if *verbose {
		fmt.Printf("[D] Embedding storage directory: %s\n", embedDir)
		fmt.Printf("[D] CCRAG_OLLAMA_ADDRESS: %s\n", ollamaAddress)
		fmt.Printf("[D] CCRAG_EMBED_MODEL: %s\n", embedModel)
		fmt.Printf("[D] CCRAG_LLM_MODEL: %s\n", llmModel)
	}

	if _, err := os.Stat(embedDir); os.IsNotExist(err) {
		err := os.MkdirAll(embedDir, 0755)
		if err != nil {
			fmt.Println(err)
			os.Exit(1)
		}
	}

	if *embedMode {
		// Accept list of paths from stdin
		scanner := bufio.NewScanner(os.Stdin)

		var paths []string
		for scanner.Scan() {
			paths = append(paths, scanner.Text())
		}

		if err := scanner.Err(); err != nil {
			fmt.Fprintf(os.Stderr, "Error reading input: %v\n", err)
			os.Exit(1)
		}

		for _, p := range paths {
			data, err := os.ReadFile(p)
			if err != nil {
				fmt.Printf("[!] Can not read source file %s\n", p)
				continue
			}

			name := cc.FileName(p)
			embedFileName := name + "." + embedFormat
			embedFilePath := filepath.Join(embedDir, embedFileName)

			if _, err := os.Stat(embedFilePath); err == nil {
				if *verbose {
					fmt.Printf("[-] Skipping existing file: %s\n", embedFilePath)
				}
				continue
			}

			if *verbose {
				fmt.Printf("[D] Embedding: %s\n", p)
			}

			res, err := embed(string(data))
			if err != nil {
				fmt.Printf("[!] Failed to generate embedding for source file %s\n", p)
				continue

			}

			res.Source = p

			embedData, err := json.Marshal(res)
			if err != nil {
				fmt.Printf("[!] Failed to marshal source file %s\n", p)
				continue

			}

			if err := os.WriteFile(embedFilePath, embedData, 0644); err != nil {
				fmt.Printf("[!] Error writing file: %s\n", embedFilePath)
			}
		}
	} else if *query != "" {
		embUserQuery, err := embed(*query)
		if err != nil {
			log.Fatal(err)
		}

		if len(embUserQuery.Embeddings) == 0 {
			fmt.Printf("[!] Failed to create embedding for user query. %v\n", embUserQuery.Embeddings)
			os.Exit(1)
		}

		embedFiles, err := filepath.Glob(filepath.Join(embedDir, "*."+embedFormat))
		if err != nil {
			log.Fatal(err)
		}

		scores := []ScoredResult{}

		for _, file := range embedFiles {
			data, err := os.ReadFile(file)
			if err != nil {
				log.Fatal(err)
			}

			var embNote EmbeddingResponse
			if err := json.Unmarshal(data, &embNote); err != nil {
				log.Fatal(err)
			}

			if len(embNote.Embeddings) == 0 {
				fmt.Printf("[!] Stored note embedding is empty. %s\n", file)
				continue
			}

			score := cosineSimilarity(embUserQuery.Embeddings[0], embNote.Embeddings[0])

			if *verbose {
				fmt.Printf("[D] Scoring file: %s, %f\n", file, score)
			}

			scores = append(scores, ScoredResult{
				Score: score,
				Path:  embNote.Source,
			})

			// fmt.Printf("[D] Computed score for %s %f\n", file, score)
		}

		// fmt.Printf("[D] Total scored files: %d\n", len(scores))
		slices.SortFunc(scores, func(a, b ScoredResult) int {
			// The scores are in the range [0, 1], so scale them to get non-zero
			// integers for comparison.
			return int((100.0*a.Score - 100.0*b.Score))
		})

		// Take the N best-scoring chunks
		selectedScores := []ScoredResult{}
		for i := len(scores) - 1; i > len(scores)-(maxResults+1); i-- {
			selectedScores = append(selectedScores, scores[i])
		}

		// Print best matches and exit
		if *similarityOnly {
			for _, v := range selectedScores {
				fmt.Println(v.Path)
			}
			os.Exit(0)
		}

		// Concat selected chunks into context to prepend to the LLM prompt
		context := ""
		for _, v := range selectedScores {
			if *verbose {
				fmt.Printf("[D] Selected file: %s %f\n", v.Path, v.Score)
			}

			data, err := os.ReadFile(v.Path)
			if err != nil {
				log.Fatal(err)
			}
			context += string(data) + "\n"
		}

		// Make a request to an LLM with context of the note appended to the prompt
		prompt := fmt.Sprintf(`Use the below information provided in org-mode markdown to answer the subsequent question. Do not offer any helpful advice! If can not be derived from provided Information use your best take to answer the question. 
Information:
%v

Question: %v`, context, *query)

		// fmt.Printf("[D] Prompt: %s\n", prompt)

		url := ollamaAddress + "/api/generate"
		payload := map[string]interface{}{
			"model":  llmModel,
			"prompt": prompt,
			"stream": false,
		}
		jsonPayload, err := json.Marshal(payload)
		if err != nil {
			log.Fatal(err)
		}

		resp, err := client.Post(url, "application/json", bytes.NewBuffer(jsonPayload))
		if err != nil {
			log.Fatal(err)
		}
		defer resp.Body.Close()

		ollamaResp := OllamaResponse{}

		if err := json.NewDecoder(resp.Body).Decode(&ollamaResp); err != nil {
			log.Fatal(err)
		}

		fmt.Println(ollamaResp.Response)
	} else {
		flag.PrintDefaults()
		os.Exit(1)
	}

}
