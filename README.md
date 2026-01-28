# QMD-Cloud - Quick Markdown Search (Cloud Version)

A cloud-based fork of [tobi/qmd](https://github.com/tobi/qmd) that uses the OpenRouter API for embeddings instead of local GGUF models. This makes it lightweight and easy to deploy on servers without GPU requirements.

## Key Differences from Original QMD

| Feature | Original (tobi/qmd) | This Fork (qmd-cloud) |
|---------|---------------------|----------------------|
| Embeddings | Local GGUF models via node-llama-cpp | OpenRouter API (text-embedding-3-large) |
| Query Expansion | Local Qwen3-1.7B | OpenRouter API (gpt-4o-mini) |
| Re-ranking | Local qwen3-reranker | Not available (simplified pipeline) |
| Search Commands | `search`, `vsearch`, `query` | `vsearch` only |
| Dependencies | ~3GB of model downloads | None (API-based) |
| Requirements | GPU recommended | Just an API key |

## Installation

```sh
# Install globally via bun from GitHub
bun install -g github:dkzlv/qmd

# Or clone and link for development
git clone https://github.com/dkzlv/qmd.git
cd qmd
bun install
bun link
```

## Setup

Set your OpenRouter API key:

```sh
export OPENROUTER_API_KEY="sk-or-..."
```

Get an API key at [openrouter.ai](https://openrouter.ai/).

## Quick Start

```sh
# Create collections for your notes
qmd-cloud collection add ~/notes --name notes
qmd-cloud collection add ~/Documents/meetings --name meetings

# Add context to help with search results
qmd-cloud context add qmd://notes "Personal notes and ideas"
qmd-cloud context add qmd://meetings "Meeting transcripts"

# Generate embeddings (requires OPENROUTER_API_KEY)
qmd-cloud embed

# Search across everything
qmd-cloud vsearch "how to deploy"

# Get a specific document
qmd-cloud get "meetings/2024-01-15.md"

# Get a document by docid (shown in search results)
qmd-cloud get "#abc123"
```

## Commands

```sh
qmd-cloud collection add [path] --name <name>  # Create/index collection
qmd-cloud collection list                      # List all collections
qmd-cloud collection remove <name>             # Remove a collection
qmd-cloud collection rename <old> <new>        # Rename a collection
qmd-cloud ls [collection[/path]]               # List files in a collection
qmd-cloud context add [path] "text"            # Add context for path
qmd-cloud context list                         # List all contexts
qmd-cloud context rm <path>                    # Remove context
qmd-cloud get <file>                           # Get document by path or docid
qmd-cloud multi-get <pattern>                  # Get multiple docs by glob
qmd-cloud status                               # Show index status
qmd-cloud update [--pull]                      # Re-index all collections
qmd-cloud embed [-f]                           # Generate vector embeddings
qmd-cloud vsearch <query>                      # Vector similarity search
qmd-cloud mcp                                  # Start MCP server
```

## Search Options

```sh
-n <num>                 # Number of results (default: 5)
-c, --collection <name>  # Filter to a specific collection
--all                    # Return all matches
--min-score <num>        # Minimum similarity score
--full                   # Show full document content
--line-numbers           # Add line numbers to output
--json                   # JSON output
--csv                    # CSV output
--md                     # Markdown output
--xml                    # XML output
--files                  # Output: docid,score,filepath,context
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENROUTER_API_KEY` | Yes | - | OpenRouter API key |
| `QMD_EMBED_MODEL` | No | `openai/text-embedding-3-large` | Embedding model |
| `QMD_CHAT_MODEL` | No | `openai/gpt-4o-mini` | Query expansion model |

## MCP Server

For AI agent integration, qmd-cloud exposes an MCP server:

**Claude Desktop** (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "qmd": {
      "command": "qmd-cloud",
      "args": ["mcp"],
      "env": {
        "OPENROUTER_API_KEY": "sk-or-..."
      }
    }
  }
}
```

**Tools exposed:**
- `qmd_vsearch` - Semantic vector search
- `qmd_get` - Retrieve document by path or docid
- `qmd_multi_get` - Retrieve multiple documents
- `qmd_status` - Index health and collection info

## Architecture

```
Document ──► Chunk (800 tokens) ──► OpenRouter API ──► Store Vectors
                                    text-embedding-3-large
                                    (3072 dimensions)

Query ──► OpenRouter API ──► Vector Search ──► Results
          Embed query         sqlite-vec
```

## Data Storage

Index stored in: `~/.cache/qmd/index.sqlite`

## Credits

This is a fork of [tobi/qmd](https://github.com/tobi/qmd) by Tobias Lutke. The original project uses local GGUF models for fully offline operation. This fork trades that for simpler deployment by using cloud APIs.

## License

MIT
