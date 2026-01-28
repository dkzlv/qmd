/**
 * OpenRouter API client for qmd
 *
 * Handles embeddings and chat completions via OpenRouter's unified API.
 * OpenRouter provides access to OpenAI, Anthropic, and other models through
 * a single API endpoint.
 */

import type {
  LLM,
  EmbeddingResult,
  GenerateResult,
  EmbedOptions,
  GenerateOptions,
  RerankOptions,
  RerankResult,
  RerankDocument,
  ModelInfo,
  Queryable,
} from "./llm.js";

// OpenRouter API base URL
const OPENROUTER_API_BASE = "https://openrouter.ai/api/v1";

// Default models
const DEFAULT_EMBED_MODEL = "openai/text-embedding-3-large";
const DEFAULT_CHAT_MODEL = "openai/gpt-4o-mini";

// Embedding dimensions by model
const EMBEDDING_DIMENSIONS: Record<string, number> = {
  "openai/text-embedding-3-large": 3072,
  "openai/text-embedding-3-small": 1536,
  "openai/text-embedding-ada-002": 1536,
};

export type OpenRouterConfig = {
  apiKey?: string; // Falls back to OPENROUTER_API_KEY env var
  embedModel?: string;
  chatModel?: string;
  baseUrl?: string;
};

export class OpenRouterLLM implements LLM {
  private apiKey: string;
  private embedModel: string;
  private chatModel: string;
  private baseUrl: string;

  constructor(config: OpenRouterConfig = {}) {
    this.apiKey = config.apiKey || process.env.OPENROUTER_API_KEY || "";
    if (!this.apiKey) {
      throw new Error(
        "OpenRouter API key required. Set OPENROUTER_API_KEY environment variable."
      );
    }
    this.embedModel = config.embedModel || process.env.QMD_EMBED_MODEL || DEFAULT_EMBED_MODEL;
    this.chatModel = config.chatModel || process.env.QMD_CHAT_MODEL || DEFAULT_CHAT_MODEL;
    this.baseUrl = config.baseUrl || OPENROUTER_API_BASE;
  }

  /**
   * Get embedding dimensions for the configured model
   */
  getEmbeddingDimensions(): number {
    return EMBEDDING_DIMENSIONS[this.embedModel] || 3072;
  }

  /**
   * Make authenticated request to OpenRouter API
   */
  private async request<T>(
    endpoint: string,
    body: Record<string, unknown>
  ): Promise<T> {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${this.apiKey}`,
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/tobi/qmd", // Required by OpenRouter
        "X-Title": "qmd", // Optional, shows in OpenRouter dashboard
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`OpenRouter API error (${response.status}): ${error}`);
    }

    return response.json() as Promise<T>;
  }

  // ===========================================================================
  // LLM Interface Implementation
  // ===========================================================================

  async embed(
    text: string,
    options: EmbedOptions = {}
  ): Promise<EmbeddingResult | null> {
    try {
      const response = await this.request<{
        data: Array<{ embedding: number[] }>;
        model: string;
      }>("/embeddings", {
        model: this.embedModel,
        input: text,
      });

      return {
        embedding: response.data[0]?.embedding || [],
        model: response.model,
      };
    } catch (error) {
      console.error("Embedding error:", error);
      return null;
    }
  }

  /**
   * Batch embed multiple texts in a single API call
   */
  async embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]> {
    if (texts.length === 0) return [];

    try {
      // OpenAI embeddings API supports batch input
      const response = await this.request<{
        data: Array<{ embedding: number[]; index: number }>;
        model: string;
      }>("/embeddings", {
        model: this.embedModel,
        input: texts,
      });

      // Sort by index to maintain order
      const sorted = response.data.sort((a, b) => a.index - b.index);

      return sorted.map((item) => ({
        embedding: item.embedding,
        model: response.model,
      }));
    } catch (error) {
      console.error("Batch embedding error:", error);
      return texts.map(() => null);
    }
  }

  async generate(
    prompt: string,
    options: GenerateOptions = {}
  ): Promise<GenerateResult | null> {
    try {
      const response = await this.request<{
        choices: Array<{ message: { content: string } }>;
        model: string;
      }>("/chat/completions", {
        model: this.chatModel,
        messages: [{ role: "user", content: prompt }],
        max_tokens: options.maxTokens || 500,
        temperature: options.temperature || 0,
      });

      return {
        text: response.choices[0]?.message?.content || "",
        model: response.model,
        done: true,
      };
    } catch (error) {
      console.error("Generation error:", error);
      return null;
    }
  }

  async modelExists(model: string): Promise<ModelInfo> {
    // OpenRouter models are always "available" if the API key is valid
    // We could call /models endpoint to verify, but it's not worth the latency
    return { name: model, exists: true };
  }

  async expandQuery(
    query: string,
    options: { context?: string; includeLexical?: boolean } = {}
  ): Promise<Queryable[]> {
    const context = options.context;

    // Simplified prompt for cloud model - they follow instructions better
    const prompt = `You are a search query optimizer. Generate alternative queries for semantic search.

Original Query: ${query}
${context ? `\nContext: ${context}` : ""}

Generate exactly 3 outputs:
1. A rephrased version of the query using different words
2. A more specific version with additional relevant terms
3. A hypothetical document snippet (1-2 sentences) that would answer this query

Format your response EXACTLY as:
vec: [rephrased query]
vec: [specific query]
hyde: [hypothetical document passage]

Do not include any other text.`;

    try {
      const result = await this.generate(prompt, {
        maxTokens: 300,
        temperature: 0.7,
      });

      if (!result) {
        return [{ type: "vec", text: query }];
      }

      const lines = result.text.trim().split("\n");
      const queryables: Queryable[] = [];

      for (const line of lines) {
        const match = line.match(/^(vec|hyde):\s*(.+)$/i);
        if (match) {
          const type = match[1]!.toLowerCase() as "vec" | "hyde";
          const text = match[2]!.trim();
          if (text && text !== query) {
            queryables.push({ type, text });
          }
        }
      }

      // Always include original query
      return [{ type: "vec", text: query }, ...queryables];
    } catch (error) {
      console.error("Query expansion error:", error);
      return [{ type: "vec", text: query }];
    }
  }

  async rerank(
    query: string,
    documents: RerankDocument[],
    options: RerankOptions = {}
  ): Promise<RerankResult> {
    // Not implemented for cloud-only version
    // Return documents in original order with uniform scores
    return {
      results: documents.map((doc, index) => ({
        file: doc.file,
        score: 1 - index * 0.01, // Slight decay to maintain order
        index,
      })),
      model: "none",
    };
  }

  async dispose(): Promise<void> {
    // Nothing to dispose - no persistent connections
  }
}
