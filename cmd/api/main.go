package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"github.com/joho/godotenv"
	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/llms/openai"
	"github.com/tmc/langchaingo/vectorstores/pinecone"
	"io"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"text/template"
)

const (
	model                = "gpt-4o-mini"
	chatCompletionUrl    = "https://api.openai.com/v1/chat/completions"
	ragMsgTemplate       = "Kontext: {{.RagContext}}\nNachricht: {{.Message}}"
	similarDocumentCount = 7
)

var systemMessage = message{
	Role:    "system",
	Content: "Answer the questions in the language they were asked and help the user find answers to their developer-specific questions. Refer all questions to the DORA context and try to provide as concrete advice as possible.",
}

type message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}
type chatCompletionRequest struct {
	Model         string        `json:"model"`
	Messages      []message     `json:"messages"`
	MaxTokens     int           `json:"max_tokens,omitempty"`
	Temperature   float64       `json:"temperature,omitempty"`
	Stream        bool          `json:"stream,omitempty"`
	StreamOptions streamOptions `json:"stream_options"`
}

type chatCompletionResponse struct {
	ID                string `json:"id"`
	Object            string `json:"object"`
	Created           int64  `json:"created"`
	Model             string `json:"model"`
	ServiceTier       string `json:"service_tier"`
	SystemFingerprint string `json:"system_fingerprint"`
	Choices           []struct{}
	Usage             struct {
		PromptTokens        int `json:"prompt_tokens"`
		CompletionTokens    int `json:"completion_tokens"`
		TotalTokens         int `json:"total_tokens"`
		PromptTokensDetails struct {
			CachedTokens int `json:"cached_tokens"`
			AudioTokens  int `json:"audio_tokens"`
		} `json:"prompt_tokens_details"`
		CompletionTokensDetails struct {
			ReasoningTokens          int `json:"reasoning_tokens"`
			AudioTokens              int `json:"audio_tokens"`
			AcceptedPredictionTokens int `json:"accepted_prediction_tokens"`
			RejectedPredictionTokens int `json:"rejected_prediction_tokens"`
		} `json:"completion_tokens_details"`
	} `json:"usage"`
}

func (s *server) proxyHandler(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()

	slog.InfoContext(ctx, "received requestData")

	reqData, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "failed to read requestData body", http.StatusBadRequest)
		return
	}

	var requestData chatCompletionRequest
	err = json.Unmarshal(reqData, &requestData)
	if err != nil {
		http.Error(w, "failed to read requestData body", http.StatusBadRequest)
		return
	}
	defer r.Body.Close()

	if len(requestData.Messages) == 0 {
		http.Error(w, "request data must contain at least one message", http.StatusBadRequest)
	}

	requestData.MaxTokens = 0
	requestData.Model = model
	requestData.StreamOptions = streamOptions{IncludeUsage: true}

	requestData.Messages[0] = systemMessage

	lastMessage := requestData.Messages[len(requestData.Messages)-1]

	// get documents from pinecone store
	docs, err := s.pineconeStore.SimilaritySearch(ctx, lastMessage.Content, similarDocumentCount)

	// merge documents into a single string
	var ragContext string
	for _, doc := range docs {
		ragContext += fmt.Sprintf("%s\n%s", ragContext, doc.PageContent)
	}

	if ragContext == "" {
		ragContext = "Leer, Nachricht kann also ohne weiteren Kontext verarbeitet werden."
	}

	// render template
	templateData := struct {
		RagContext string
		Message    string
	}{
		RagContext: ragContext,
		Message:    lastMessage.Content,
	}

	var renderedTemplateBuffer bytes.Buffer
	err = s.ragTemplate.Execute(&renderedTemplateBuffer, templateData)
	if err != nil {
		http.Error(w, "Failed to execute template", http.StatusInternalServerError)
		return
	}

	// store rendered template in message store
	s.messageStore[lastMessage.Content] = renderedTemplateBuffer.String()

	// check all messages for stored modified messages with rag context and use it if available
	for i, msg := range requestData.Messages {
		if content, ok := s.messageStore[msg.Content]; ok {
			requestData.Messages[i].Content = content
		}
	}

	modifiedReqData := &bytes.Buffer{}
	err = json.NewEncoder(modifiedReqData).Encode(requestData)
	if err != nil {
		http.Error(w, "failed to marshal modified request data", http.StatusInternalServerError)
		return
	}

	req, err := http.NewRequest("POST", chatCompletionUrl, modifiedReqData)
	if err != nil {
		http.Error(w, "Failed to create request data", http.StatusInternalServerError)
		return
	}

	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", s.openaiAPIKey))
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		http.Error(w, "Failed to contact OpenAI API", http.StatusInternalServerError)
		return
	}
	defer resp.Body.Close()

	slog.InfoContext(ctx, "response status code", "statusCode", resp.StatusCode)

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(resp.StatusCode)

	// use a tee reader to send it to another buffer while writing it to the client
	// this way we can print the used tokens for the chat completion request
	var buf bytes.Buffer
	tee := io.TeeReader(resp.Body, &buf)

	_, err = io.Copy(w, tee)
	if err != nil {
		slog.ErrorContext(ctx, "failed to write response body", "err", err)
		return
	}
	slog.InfoContext(ctx, "forwarded response to client")

	chunks := strings.Split(buf.String(), "data: ")
	var lastJSON string

	// find the last valid json before "data: [DONE]"
	for _, chunk := range chunks {
		if strings.TrimSpace(chunk) == "[DONE]" {
			break
		}
		if len(strings.TrimSpace(chunk)) > 0 {
			lastJSON = chunk
		}
	}

	// Now lastJSON should contain the last valid JSON string
	var response chatCompletionResponse
	err = json.Unmarshal([]byte(lastJSON), &response)
	if err != nil {
		fmt.Println("Error parsing JSON:", err)
		return
	}

	slog.InfoContext(ctx, fmt.Sprintf("used tokens: %v (prompt: %v, completion: %v)", response.Usage.TotalTokens, response.Usage.PromptTokens, response.Usage.CompletionTokens))
}

func main() {
	ctx := context.Background()

	err := godotenv.Load()
	if err != nil {
		slog.ErrorContext(ctx, fmt.Sprintf("error loading .env file: %v", err))
	}

	tmpl, err := template.New("").Parse(ragMsgTemplate)
	if err != nil {
		slog.ErrorContext(ctx, fmt.Sprintf("error parsing RAG message template: %v", err))
		os.Exit(1)
	}

	llm, err := openai.New(
		openai.WithEmbeddingModel(os.Getenv("OPENAI_EMBEDDING_MODEL")),
		openai.WithToken(os.Getenv("OPENAI_API_KEY")),
	)
	if err != nil {
		slog.ErrorContext(ctx, fmt.Sprintf("error creating openai client: %v", err))
		os.Exit(1)
	}

	embedder, err := embeddings.NewEmbedder(llm)
	if err != nil {
		slog.ErrorContext(ctx, fmt.Sprintf("error creating embeddings embedder: %v", err))
		os.Exit(1)
	}

	store, err := pinecone.New(
		pinecone.WithHost(os.Getenv("PINECONE_HOST")),
		pinecone.WithAPIKey(os.Getenv("PINECONE_API_KEY")),
		pinecone.WithNameSpace(os.Getenv("PINECONE_NAMESPACE")),
		pinecone.WithEmbedder(embedder),
	)
	if err != nil {
		slog.ErrorContext(ctx, fmt.Sprintf("error creating pinecone store: %v", err))
		os.Exit(1)
	}

	s := &server{
		pineconeStore: &store,
		messageStore:  make(map[string]string),
		openaiAPIKey:  os.Getenv("OPENAI_API_KEY"),
		ragTemplate:   tmpl,
	}

	http.HandleFunc("/v1/chat/completions", s.proxyHandler)

	err = http.ListenAndServe(":9000", nil)
	if err != nil {
		slog.ErrorContext(ctx, fmt.Sprintf("error listening on port 9000: %v", err))
		os.Exit(1)
	}
}

type server struct {
	pineconeStore *pinecone.Store
	messageStore  map[string]string
	openaiAPIKey  string
	ragTemplate   *template.Template
}
