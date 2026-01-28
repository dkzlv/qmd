/**
 * llm.ts - LLM abstraction layer for QMD using OpenRouter
 *
 * Provides embeddings, text generation via OpenRouter cloud API.
 * This replaces the previous node-llama-cpp local implementation.
 */

import { OpenRouterLLM, type OpenRouterConfig } from "./openrouter.js";

// =============================================================================
// Types
// =============================================================================

/**
 * Token with log probability
 */
export type TokenLogProb = {
  token: string;
  logprob: number;
};

/**
 * Embedding result
 */
export type EmbeddingResult = {
  embedding: number[];
  model: string;
};

/**
 * Generation result with optional logprobs
 */
export type GenerateResult = {
  text: string;
  model: string;
  logprobs?: TokenLogProb[];
  done: boolean;
};

/**
 * Rerank result for a single document
 */
export type RerankDocumentResult = {
  file: string;
  score: number;
  index: number;
};

/**
 * Batch rerank result
 */
export type RerankResult = {
  results: RerankDocumentResult[];
  model: string;
};

/**
 * Model info
 */
export type ModelInfo = {
  name: string;
  exists: boolean;
  path?: string;
};

/**
 * Options for embedding
 */
export type EmbedOptions = {
  model?: string;
  isQuery?: boolean;
  title?: string;
};

/**
 * Options for text generation
 */
export type GenerateOptions = {
  model?: string;
  maxTokens?: number;
  temperature?: number;
};

/**
 * Options for reranking
 */
export type RerankOptions = {
  model?: string;
};

/**
 * Supported query types for different search backends
 */
export type QueryType = "lex" | "vec" | "hyde";

/**
 * A single query and its target backend type
 */
export type Queryable = {
  type: QueryType;
  text: string;
};

/**
 * Document to rerank
 */
export type RerankDocument = {
  file: string;
  text: string;
  title?: string;
};

// =============================================================================
// LLM Interface
// =============================================================================

/**
 * Abstract LLM interface - implement this for different backends
 */
export interface LLM {
  /**
   * Get embeddings for text
   */
  embed(text: string, options?: EmbedOptions): Promise<EmbeddingResult | null>;

  /**
   * Generate text completion
   */
  generate(
    prompt: string,
    options?: GenerateOptions
  ): Promise<GenerateResult | null>;

  /**
   * Check if a model exists/is available
   */
  modelExists(model: string): Promise<ModelInfo>;

  /**
   * Expand a search query into multiple variations for different backends.
   * Returns a list of Queryable objects.
   */
  expandQuery(
    query: string,
    options?: { context?: string; includeLexical?: boolean }
  ): Promise<Queryable[]>;

  /**
   * Rerank documents by relevance to a query
   * Returns list of documents with relevance scores (higher = more relevant)
   */
  rerank(
    query: string,
    documents: RerankDocument[],
    options?: RerankOptions
  ): Promise<RerankResult>;

  /**
   * Dispose of resources
   */
  dispose(): Promise<void>;
}

// =============================================================================
// Re-export OpenRouter implementation
// =============================================================================

export { OpenRouterLLM } from "./openrouter.js";

// =============================================================================
// Singleton for default OpenRouter instance
// =============================================================================

let defaultLLM: OpenRouterLLM | null = null;

/**
 * Get the default OpenRouter LLM instance (creates one if needed)
 */
export function getDefaultLLM(): OpenRouterLLM {
  if (!defaultLLM) {
    defaultLLM = new OpenRouterLLM();
  }
  return defaultLLM;
}

/**
 * Alias for backward compatibility with existing code
 */
export function getDefaultLlamaCpp(): OpenRouterLLM {
  return getDefaultLLM();
}

/**
 * Set a custom default LLM instance (useful for testing)
 */
export function setDefaultLLM(llm: OpenRouterLLM | null): void {
  defaultLLM = llm;
}

/**
 * Dispose the default LLM instance if it exists.
 */
export async function disposeDefaultLLM(): Promise<void> {
  if (defaultLLM) {
    await defaultLLM.dispose();
    defaultLLM = null;
  }
}

/**
 * Alias for backward compatibility with existing code
 */
export async function disposeDefaultLlamaCpp(): Promise<void> {
  return disposeDefaultLLM();
}
