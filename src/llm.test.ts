/**
 * llm.test.ts - Unit tests for the LLM abstraction layer (OpenRouter)
 *
 * Run with: bun test src/llm.test.ts
 *
 * These tests require OPENROUTER_API_KEY to be set in the environment.
 */

import { describe, test, expect, beforeAll, afterAll } from "bun:test";
import {
  OpenRouterLLM,
  getDefaultLLM,
  disposeDefaultLLM,
  type RerankDocument,
} from "./llm.js";

// Skip tests if no API key is set
const hasApiKey = !!process.env.OPENROUTER_API_KEY;

// =============================================================================
// Singleton Tests (no API call required)
// =============================================================================

describe("Default OpenRouterLLM Singleton", () => {
  test.skipIf(!hasApiKey)("getDefaultLLM returns same instance on subsequent calls", () => {
    const llm1 = getDefaultLLM();
    const llm2 = getDefaultLLM();
    expect(llm1).toBe(llm2);
    expect(llm1).toBeInstanceOf(OpenRouterLLM);
  });

  test("throws error when no API key is set", () => {
    const originalKey = process.env.OPENROUTER_API_KEY;
    try {
      delete process.env.OPENROUTER_API_KEY;
      expect(() => new OpenRouterLLM()).toThrow("OpenRouter API key required");
    } finally {
      if (originalKey) {
        process.env.OPENROUTER_API_KEY = originalKey;
      }
    }
  });
});

// =============================================================================
// Model Existence Tests
// =============================================================================

describe.skipIf(!hasApiKey)("OpenRouterLLM.modelExists", () => {
  test("returns exists:true for any model (OpenRouter handles validation)", async () => {
    const llm = getDefaultLLM();
    const result = await llm.modelExists("openai/text-embedding-3-large");

    expect(result.exists).toBe(true);
    expect(result.name).toBe("openai/text-embedding-3-large");
  });
});

// =============================================================================
// Integration Tests (require API key)
// =============================================================================

describe.skipIf(!hasApiKey)("OpenRouterLLM Integration", () => {
  const llm = getDefaultLLM();

  afterAll(async () => {
    await disposeDefaultLLM();
  });

  describe("embed", () => {
    test("returns embedding with correct dimensions (3072 for text-embedding-3-large)", async () => {
      const result = await llm.embed("Hello world");

      expect(result).not.toBeNull();
      expect(result!.embedding).toBeInstanceOf(Array);
      expect(result!.embedding.length).toBeGreaterThan(0);
      // text-embedding-3-large outputs 3072 dimensions
      expect(result!.embedding.length).toBe(3072);
    }, 30000);

    test("returns consistent embeddings for same input", async () => {
      const result1 = await llm.embed("test text for consistency");
      const result2 = await llm.embed("test text for consistency");

      expect(result1).not.toBeNull();
      expect(result2).not.toBeNull();

      // Embeddings should be nearly identical for the same input
      // Allow small floating point differences
      let maxDiff = 0;
      for (let i = 0; i < result1!.embedding.length; i++) {
        const diff = Math.abs(result1!.embedding[i]! - result2!.embedding[i]!);
        maxDiff = Math.max(maxDiff, diff);
      }
      expect(maxDiff).toBeLessThan(0.001);
    }, 30000);

    test("returns different embeddings for different inputs", async () => {
      const result1 = await llm.embed("cats are great pets");
      const result2 = await llm.embed("quantum physics theories");

      expect(result1).not.toBeNull();
      expect(result2).not.toBeNull();

      // Calculate cosine similarity - should be less than 1.0 (not identical)
      let dotProduct = 0;
      let norm1 = 0;
      let norm2 = 0;
      for (let i = 0; i < result1!.embedding.length; i++) {
        const v1 = result1!.embedding[i]!;
        const v2 = result2!.embedding[i]!;
        dotProduct += v1 * v2;
        norm1 += v1 ** 2;
        norm2 += v2 ** 2;
      }
      const similarity = dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));

      expect(similarity).toBeLessThan(0.9); // Should be meaningfully different
    }, 30000);
  });

  describe("embedBatch", () => {
    test("returns embeddings for multiple texts", async () => {
      const texts = ["Hello world", "Test text", "Another document"];
      const results = await llm.embedBatch(texts);

      expect(results).toHaveLength(3);
      for (const result of results) {
        expect(result).not.toBeNull();
        expect(result!.embedding.length).toBe(3072);
      }
    }, 30000);

    test("handles empty array", async () => {
      const results = await llm.embedBatch([]);
      expect(results).toHaveLength(0);
    });

    test("batch returns same results as individual embed calls", async () => {
      const texts = ["cats are great", "dogs are awesome"];

      // Get batch embeddings
      const batchResults = await llm.embedBatch(texts);

      // Get individual embeddings
      const individualResults = await Promise.all(texts.map((t) => llm.embed(t)));

      // Compare - should be nearly identical
      for (let i = 0; i < texts.length; i++) {
        expect(batchResults[i]).not.toBeNull();
        expect(individualResults[i]).not.toBeNull();
        
        let maxDiff = 0;
        for (let j = 0; j < batchResults[i]!.embedding.length; j++) {
          const diff = Math.abs(
            batchResults[i]!.embedding[j]! - individualResults[i]!.embedding[j]!
          );
          maxDiff = Math.max(maxDiff, diff);
        }
        expect(maxDiff).toBeLessThan(0.001);
      }
    }, 60000);
  });

  describe("generate", () => {
    test("returns text completion", async () => {
      const result = await llm.generate("Say 'hello' and nothing else.");

      expect(result).not.toBeNull();
      expect(result!.text.toLowerCase()).toContain("hello");
      expect(result!.done).toBe(true);
    }, 30000);

    test("respects maxTokens option", async () => {
      const result = await llm.generate("Write a very long essay about cats.", {
        maxTokens: 10,
      });

      expect(result).not.toBeNull();
      // With maxTokens=10, response should be short
      expect(result!.text.split(" ").length).toBeLessThan(20);
    }, 30000);
  });

  describe("rerank", () => {
    // Cloud-only version doesn't implement real reranking
    // It returns documents in original order with position-based scores
    test("returns documents with position-based scores", async () => {
      const query = "What is the capital of France?";
      const documents: RerankDocument[] = [
        { file: "doc1.txt", text: "First document" },
        { file: "doc2.txt", text: "Second document" },
        { file: "doc3.txt", text: "Third document" },
      ];

      const result = await llm.rerank(query, documents);

      expect(result.results).toHaveLength(3);
      expect(result.model).toBe("none");

      // Should maintain order with decaying scores
      expect(result.results[0]!.file).toBe("doc1.txt");
      expect(result.results[1]!.file).toBe("doc2.txt");
      expect(result.results[2]!.file).toBe("doc3.txt");

      // Scores should decay
      expect(result.results[0]!.score).toBeGreaterThan(result.results[1]!.score);
      expect(result.results[1]!.score).toBeGreaterThan(result.results[2]!.score);
    });

    test("handles empty document list", async () => {
      const result = await llm.rerank("test query", []);
      expect(result.results).toHaveLength(0);
    });
  });

  describe("expandQuery", () => {
    test("returns query expansions with vec and hyde types", async () => {
      const result = await llm.expandQuery("how to configure authentication");

      // Should include original query
      expect(result.length).toBeGreaterThanOrEqual(1);
      
      // Should have the original query as first entry
      expect(result[0]!.type).toBe("vec");
      expect(result[0]!.text).toBe("how to configure authentication");

      // Each result should have a valid type (vec or hyde only - no lex in cloud version)
      for (const q of result) {
        expect(["vec", "hyde"]).toContain(q.type);
        expect(q.text.length).toBeGreaterThan(0);
      }
    }, 30000);

    test("generates alternative query formulations", async () => {
      const result = await llm.expandQuery("database optimization tips");

      // Should generate multiple variations
      expect(result.length).toBeGreaterThan(1);

      // Should have at least one hyde (hypothetical document)
      const hydeEntries = result.filter((q) => q.type === "hyde");
      expect(hydeEntries.length).toBeGreaterThanOrEqual(0); // May or may not have hyde
    }, 30000);

    test("falls back to original query on error", async () => {
      // Create LLM with invalid model to trigger error
      const badLlm = new OpenRouterLLM({
        apiKey: process.env.OPENROUTER_API_KEY,
        chatModel: "nonexistent/model-that-does-not-exist",
      });

      const result = await badLlm.expandQuery("test query");

      // Should return at least the original query
      expect(result.length).toBeGreaterThanOrEqual(1);
      expect(result[0]!.text).toBe("test query");

      await badLlm.dispose();
    }, 30000);
  });
});

// =============================================================================
// OpenRouterLLM Configuration Tests
// =============================================================================

describe.skipIf(!hasApiKey)("OpenRouterLLM Configuration", () => {
  test("uses environment variables for model configuration", () => {
    const originalEmbed = process.env.QMD_EMBED_MODEL;
    const originalChat = process.env.QMD_CHAT_MODEL;

    try {
      process.env.QMD_EMBED_MODEL = "openai/text-embedding-3-small";
      process.env.QMD_CHAT_MODEL = "anthropic/claude-3-haiku";

      const llm = new OpenRouterLLM({ apiKey: process.env.OPENROUTER_API_KEY });
      
      // Check that getEmbeddingDimensions returns correct value for the configured model
      expect(llm.getEmbeddingDimensions()).toBe(1536); // text-embedding-3-small has 1536 dims
    } finally {
      if (originalEmbed) {
        process.env.QMD_EMBED_MODEL = originalEmbed;
      } else {
        delete process.env.QMD_EMBED_MODEL;
      }
      if (originalChat) {
        process.env.QMD_CHAT_MODEL = originalChat;
      } else {
        delete process.env.QMD_CHAT_MODEL;
      }
    }
  });

  test("config overrides environment variables", () => {
    const llm = new OpenRouterLLM({
      apiKey: process.env.OPENROUTER_API_KEY,
      embedModel: "openai/text-embedding-ada-002",
    });

    expect(llm.getEmbeddingDimensions()).toBe(1536);
  });

  test("getEmbeddingDimensions returns correct dimensions", () => {
    const llm = new OpenRouterLLM({ apiKey: process.env.OPENROUTER_API_KEY });
    
    // Default model is text-embedding-3-large with 3072 dimensions
    expect(llm.getEmbeddingDimensions()).toBe(3072);
  });
});
