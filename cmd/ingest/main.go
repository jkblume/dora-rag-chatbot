package main

import (
	"context"
	"fmt"
	"github.com/joho/godotenv"
	"github.com/tmc/langchaingo/documentloaders"
	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/llms/openai"
	"github.com/tmc/langchaingo/textsplitter"
	"github.com/tmc/langchaingo/vectorstores/pinecone"
	"log/slog"
	"os"
)

func main() {
	ctx := context.Background()

	err := godotenv.Load()
	if err != nil {
		slog.WarnContext(ctx, fmt.Sprintf("error loading .env file: %v", err))
	}

	file, err := os.Open("res/dora.txt")
	if err != nil {
		slog.ErrorContext(ctx, fmt.Sprintf("error opening dora.txt file: %v", err))
		os.Exit(1)
	}
	defer file.Close()

	documentLoader := documentloaders.NewText(file)

	docs, err := documentLoader.LoadAndSplit(ctx, textsplitter.NewRecursiveCharacter(
		textsplitter.WithChunkSize(1000),
		textsplitter.WithChunkOverlap(20),
	))
	if err != nil {
		slog.ErrorContext(ctx, fmt.Sprintf("error loading documents: %v", err))
		os.Exit(1)
	}

	llm, err := openai.New(
		openai.WithEmbeddingModel(os.Getenv("OPENAI_EMBEDDING_MODEL")),
		openai.WithToken(os.Getenv("OPENAI_API_KEY")),
	)
	if err != nil {
		slog.ErrorContext(ctx, fmt.Sprintf("error loading openai llm: %v", err))
		os.Exit(1)
	}

	embedder, err := embeddings.NewEmbedder(llm)
	if err != nil {
		slog.ErrorContext(ctx, fmt.Sprintf("error creating new embedder: %v", err))
		os.Exit(1)
	}

	// Create a new Pinecone vector store.
	store, err := pinecone.New(
		pinecone.WithHost(os.Getenv("PINECONE_HOST")),
		pinecone.WithAPIKey(os.Getenv("PINECONE_API_KEY")),
		pinecone.WithNameSpace(os.Getenv("PINECONE_NAMESPACE")),
		pinecone.WithEmbedder(embedder),
	)
	if err != nil {
		slog.ErrorContext(ctx, fmt.Sprintf("error creating new pinecone store: %v", err))
		os.Exit(1)
	}

	// add documents in batches
	batchSize := 100
	for i := 0; i < len(docs); i += batchSize {
		end := i + batchSize
		if end > len(docs) {
			end = len(docs)
		}
		batch := docs[i:end]

		res, err := store.AddDocuments(ctx, batch)
		if err != nil {
			slog.ErrorContext(ctx, fmt.Sprintf("error adding documents to store: %v", err))
			os.Exit(1)
		}

		slog.InfoContext(ctx, "Added %d documents to Pinecone", len(res))
	}
}
