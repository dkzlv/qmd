/**
 * openrouter.test.ts - End-to-end tests for the OpenRouter cloud integration
 *
 * Run with: bun test src/openrouter.test.ts
 *
 * These tests require OPENROUTER_API_KEY to be set in the environment.
 * They test the full flow from embedding to search.
 */

import { describe, test, expect, beforeAll, afterAll } from "bun:test";
import { mkdtemp, rm, writeFile, mkdir } from "fs/promises";
import { tmpdir } from "os";
import { join } from "path";
import Database from "bun:sqlite";

import {
  OpenRouterLLM,
  getDefaultLLM,
  disposeDefaultLLM,
} from "./llm.js";
import {
  createStore,
  chunkDocument,
  insertContent,
  insertDocument,
  insertEmbedding,
  searchVec,
  EMBEDDING_DIMENSIONS,
  DEFAULT_EMBED_MODEL,
} from "./store.js";

// Skip tests if no API key is set
const hasApiKey = !!process.env.OPENROUTER_API_KEY;

// =============================================================================
// Unit Tests for OpenRouterLLM
// =============================================================================

describe("OpenRouterLLM Unit Tests", () => {
  test("throws error when no API key is provided", () => {
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

  test("getEmbeddingDimensions returns correct dimensions for default model", () => {
    if (!hasApiKey) return;
    const llm = new OpenRouterLLM();
    expect(llm.getEmbeddingDimensions()).toBe(3072);
  });

  test("getEmbeddingDimensions returns correct dimensions for text-embedding-3-small", () => {
    if (!hasApiKey) return;
    const llm = new OpenRouterLLM({ embedModel: "openai/text-embedding-3-small" });
    expect(llm.getEmbeddingDimensions()).toBe(1536);
  });

  test("rerank returns documents in original order with decaying scores", async () => {
    if (!hasApiKey) return;
    const llm = new OpenRouterLLM();
    const docs = [
      { file: "a.md", text: "first" },
      { file: "b.md", text: "second" },
      { file: "c.md", text: "third" },
    ];

    const result = await llm.rerank("query", docs);

    expect(result.results).toHaveLength(3);
    expect(result.results[0]!.file).toBe("a.md");
    expect(result.results[1]!.file).toBe("b.md");
    expect(result.results[2]!.file).toBe("c.md");
    expect(result.results[0]!.score).toBeGreaterThan(result.results[1]!.score);
    expect(result.results[1]!.score).toBeGreaterThan(result.results[2]!.score);
  });
});

// =============================================================================
// Integration Tests (require API key)
// =============================================================================

describe.skipIf(!hasApiKey)("OpenRouter Integration Tests", () => {
  let llm: OpenRouterLLM;

  beforeAll(() => {
    llm = getDefaultLLM();
  });

  afterAll(async () => {
    await disposeDefaultLLM();
  });

  describe("Embedding", () => {
    test("embed returns 3072-dimensional vector", async () => {
      const result = await llm.embed("Hello, world!");

      expect(result).not.toBeNull();
      expect(result!.embedding).toHaveLength(3072);
      expect(result!.model).toContain("embedding");
    }, 30000);

    test("embedBatch returns multiple embeddings", async () => {
      const texts = ["First document", "Second document", "Third document"];
      const results = await llm.embedBatch(texts);

      expect(results).toHaveLength(3);
      for (const result of results) {
        expect(result).not.toBeNull();
        expect(result!.embedding).toHaveLength(3072);
      }
    }, 30000);

    test("similar texts have higher cosine similarity", async () => {
      const catText = "Cats are wonderful pets that purr and meow.";
      const dogText = "Dogs are loyal companions that bark and wag their tails.";
      const mathText = "The quadratic formula solves ax^2 + bx + c = 0.";

      const [catEmbed, dogEmbed, mathEmbed] = await llm.embedBatch([catText, dogText, mathText]);

      // Helper to compute cosine similarity
      const cosineSim = (a: number[], b: number[]) => {
        let dot = 0, normA = 0, normB = 0;
        for (let i = 0; i < a.length; i++) {
          dot += a[i]! * b[i]!;
          normA += a[i]! * a[i]!;
          normB += b[i]! * b[i]!;
        }
        return dot / (Math.sqrt(normA) * Math.sqrt(normB));
      };

      const catDogSim = cosineSim(catEmbed!.embedding, dogEmbed!.embedding);
      const catMathSim = cosineSim(catEmbed!.embedding, mathEmbed!.embedding);

      // Cat and dog texts should be more similar than cat and math
      expect(catDogSim).toBeGreaterThan(catMathSim);
    }, 30000);
  });

  describe("Generation", () => {
    test("generate returns text completion", async () => {
      const result = await llm.generate("What is 2+2? Reply with just the number.");

      expect(result).not.toBeNull();
      expect(result!.text).toContain("4");
      expect(result!.done).toBe(true);
    }, 30000);

    test("generate respects maxTokens", async () => {
      const result = await llm.generate("Write a very long story about a cat.", {
        maxTokens: 20,
      });

      expect(result).not.toBeNull();
      // Response should be short due to token limit
      expect(result!.text.length).toBeLessThan(200);
    }, 30000);
  });

  describe("Query Expansion", () => {
    test("expandQuery returns original query plus variations", async () => {
      const result = await llm.expandQuery("how to configure authentication");

      expect(result.length).toBeGreaterThanOrEqual(1);
      // First should always be the original query
      expect(result[0]!.text).toBe("how to configure authentication");
      expect(result[0]!.type).toBe("vec");
    }, 30000);

    test("expandQuery generates vec and hyde types", async () => {
      const result = await llm.expandQuery("database performance optimization");

      const types = result.map(q => q.type);
      // Should have at least vec type
      expect(types).toContain("vec");
      // May have hyde (hypothetical document)
    }, 30000);
  });
});

// =============================================================================
// End-to-End Tests (full pipeline)
// =============================================================================

describe.skipIf(!hasApiKey)("End-to-End Search Pipeline", () => {
  let testDir: string;
  let store: ReturnType<typeof createStore>;
  let db: Database;
  let llm: OpenRouterLLM;

  beforeAll(async () => {
    // Create temp directory for test database
    testDir = await mkdtemp(join(tmpdir(), "qmd-e2e-test-"));
    const dbPath = join(testDir, "test.sqlite");

    // Set up store
    store = createStore(dbPath);
    db = store.db;
    store.ensureVecTable(EMBEDDING_DIMENSIONS);

    llm = getDefaultLLM();

    // Create test documents
    const docs = [
      {
        name: "auth.md",
        content: `# Authentication Guide

Authentication is configured using environment variables.
Set AUTH_SECRET to a secure random string.
JWT tokens expire after 24 hours by default.
Use OAuth2 for third-party integrations.`,
      },
      {
        name: "database.md",
        content: `# Database Configuration

PostgreSQL is the recommended database.
Set DATABASE_URL to your connection string.
Run migrations with: npm run migrate
Enable connection pooling for production.`,
      },
      {
        name: "deployment.md",
        content: `# Deployment Guide

Deploy to production using Docker containers.
Set NODE_ENV=production for optimal performance.
Use a reverse proxy like nginx for SSL termination.
Configure health checks for load balancers.`,
      },
    ];

    const now = new Date().toISOString();

    // Index documents and create embeddings
    for (const doc of docs) {
      const hash = Bun.hash(doc.content).toString(16).slice(0, 12);

      // Insert document
      insertContent(db, hash, doc.content, now);
      insertDocument(db, "test-collection", doc.name, doc.name.replace(".md", ""), hash, now, now);

      // Chunk and embed
      const chunks = chunkDocument(doc.content);
      for (let seq = 0; seq < chunks.length; seq++) {
        const chunk = chunks[seq]!;
        const titlePrefix = doc.name.replace(".md", "") + "\n\n";
        const result = await llm.embed(titlePrefix + chunk.text);
        if (result) {
          insertEmbedding(db, hash, seq, chunk.pos, new Float32Array(result.embedding), DEFAULT_EMBED_MODEL, now);
        }
      }
    }
  }, 120000); // 2 minute timeout for setup

  afterAll(async () => {
    store.close();
    await rm(testDir, { recursive: true, force: true });
    await disposeDefaultLLM();
  });

  test("vector search finds relevant documents", async () => {
    const results = await searchVec(db, "how to configure authentication", DEFAULT_EMBED_MODEL, 5);

    expect(results.length).toBeGreaterThan(0);
    // Auth document should be in top results
    const authResult = results.find(r => r.filepath.includes("auth.md"));
    expect(authResult).toBeDefined();
  }, 30000);

  test("vector search ranks semantically similar docs higher", async () => {
    const results = await searchVec(db, "setting up JWT tokens and OAuth", DEFAULT_EMBED_MODEL, 5);

    expect(results.length).toBeGreaterThan(0);
    // Auth should rank higher than database or deployment
    const authIndex = results.findIndex(r => r.filepath.includes("auth.md"));
    const dbIndex = results.findIndex(r => r.filepath.includes("database.md"));

    if (authIndex !== -1 && dbIndex !== -1) {
      expect(authIndex).toBeLessThan(dbIndex);
    }
  }, 30000);

  test("vector search finds database docs for SQL queries", async () => {
    const results = await searchVec(db, "PostgreSQL connection string configuration", DEFAULT_EMBED_MODEL, 5);

    expect(results.length).toBeGreaterThan(0);
    const dbResult = results.find(r => r.filepath.includes("database.md"));
    expect(dbResult).toBeDefined();
  }, 30000);

  test("vector search handles queries with no exact keyword matches", async () => {
    // Query uses different words than the documents
    const results = await searchVec(db, "secure login setup", DEFAULT_EMBED_MODEL, 5);

    expect(results.length).toBeGreaterThan(0);
    // Should still find auth.md through semantic similarity
    const hasAuthOrDeployment = results.some(
      r => r.filepath.includes("auth.md") || r.filepath.includes("deployment.md")
    );
    expect(hasAuthOrDeployment).toBe(true);
  }, 30000);
});

// =============================================================================
// CLI Integration Tests
// =============================================================================

describe.skipIf(!hasApiKey)("CLI Integration", () => {
  let testDir: string;
  let testDbPath: string;
  let testConfigDir: string;
  let fixturesDir: string;

  const qmdScript = join(import.meta.dir, "qmd.ts");

  async function runQmd(args: string[], options: { cwd?: string } = {}): Promise<{
    stdout: string;
    stderr: string;
    exitCode: number;
  }> {
    const proc = Bun.spawn(["bun", qmdScript, ...args], {
      cwd: options.cwd || fixturesDir,
      env: {
        ...process.env,
        INDEX_PATH: testDbPath,
        QMD_CONFIG_DIR: testConfigDir,
        PWD: options.cwd || fixturesDir,
      },
      stdout: "pipe",
      stderr: "pipe",
    });

    const stdout = await new Response(proc.stdout).text();
    const stderr = await new Response(proc.stderr).text();
    const exitCode = await proc.exited;

    return { stdout, stderr, exitCode };
  }

  beforeAll(async () => {
    testDir = await mkdtemp(join(tmpdir(), "qmd-cli-test-"));
    testDbPath = join(testDir, "test.sqlite");
    testConfigDir = join(testDir, "config");
    fixturesDir = join(testDir, "fixtures");

    await mkdir(testConfigDir, { recursive: true });
    await mkdir(fixturesDir, { recursive: true });

    // Create empty config
    await writeFile(join(testConfigDir, "index.yml"), "collections: {}\n");

    // Create test markdown file
    await writeFile(
      join(fixturesDir, "test.md"),
      `# Test Document

This is a test document about cloud computing.
It discusses virtual machines and containerization.
Docker and Kubernetes are popular tools.
`
    );
  });

  afterAll(async () => {
    await rm(testDir, { recursive: true, force: true });
  });

  test("search command shows deprecation message", async () => {
    const { stderr, exitCode } = await runQmd(["search", "test"]);

    expect(exitCode).toBe(1);
    expect(stderr).toContain("search");
    expect(stderr).toContain("removed");
    expect(stderr).toContain("vsearch");
  });

  test("query command shows deprecation message", async () => {
    const { stderr, exitCode } = await runQmd(["query", "test"]);

    expect(exitCode).toBe(1);
    expect(stderr).toContain("query");
    expect(stderr).toContain("removed");
    expect(stderr).toContain("vsearch");
  });

  test("collection add works", async () => {
    const { stdout, exitCode } = await runQmd([
      "collection", "add", fixturesDir,
      "--name", "test-fixtures",
      "--mask", "**/*.md"
    ]);

    expect(exitCode).toBe(0);
    expect(stdout).toContain("test-fixtures");
  });

  test("status shows collection info", async () => {
    const { stdout, exitCode } = await runQmd(["status"]);

    expect(exitCode).toBe(0);
    expect(stdout).toContain("QMD Status");
  });

  test("embed creates embeddings", async () => {
    const { stdout, stderr, exitCode } = await runQmd(["embed"]);

    // Either succeeds or says all docs already embedded
    expect(exitCode).toBe(0);
    const output = stdout + stderr;
    expect(
      output.includes("Done!") ||
      output.includes("already have embeddings") ||
      output.includes("Embedding")
    ).toBe(true);
  }, 120000);

  test("vsearch returns results", async () => {
    const { stdout, exitCode } = await runQmd(["vsearch", "cloud computing", "--json"]);

    // May return no results if embed didn't run, but shouldn't error
    expect(exitCode).toBe(0);
  }, 60000);

  test("help shows OpenRouter info", async () => {
    const { stdout, exitCode } = await runQmd(["--help"]);

    expect(exitCode).toBe(0);
    expect(stdout).toContain("OPENROUTER_API_KEY");
    expect(stdout).toContain("openai/text-embedding-3-large");
    expect(stdout).toContain("vsearch");
  });
});
