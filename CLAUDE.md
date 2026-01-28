# QMD-Cloud - Quick Markdown Search (Cloud Version)

A fork of [tobi/qmd](https://github.com/tobi/qmd) using OpenRouter API instead of local models.

**Note**: This project uses [bd (beads)](https://github.com/steveyegge/beads) for issue tracking. Use `bd` commands instead of markdown TODOs. See AGENTS.md for workflow details.

Use Bun instead of Node.js (`bun` not `node`, `bun install` not `npm install`).

## Commands

```sh
qmd-cloud collection add . --name <n>   # Create/index collection
qmd-cloud collection list               # List all collections with details
qmd-cloud collection remove <name>      # Remove a collection by name
qmd-cloud collection rename <old> <new> # Rename a collection
qmd-cloud ls [collection[/path]]        # List collections or files in a collection
qmd-cloud context add [path] "text"     # Add context for path (defaults to current dir)
qmd-cloud context list                  # List all contexts
qmd-cloud context check                 # Check for collections/paths missing context
qmd-cloud context rm <path>             # Remove context
qmd-cloud get <file>                    # Get document by path or docid (#abc123)
qmd-cloud multi-get <pattern>           # Get multiple docs by glob or comma-separated list
qmd-cloud status                        # Show index status and collections
qmd-cloud update [--pull]               # Re-index all collections (--pull: git pull first)
qmd-cloud embed                         # Generate vector embeddings (uses OpenRouter API)
qmd-cloud vsearch <query>               # Vector similarity search (primary search command)
```

## Environment Variables

Required:
- `OPENROUTER_API_KEY` - Your OpenRouter API key

Optional:
- `QMD_EMBED_MODEL` - Embedding model (default: `openai/text-embedding-3-large`)
- `QMD_CHAT_MODEL` - Chat model for query expansion (default: `openai/gpt-4o-mini`)

## Collection Management

```sh
# List all collections
qmd-cloud collection list

# Create a collection with explicit name
qmd-cloud collection add ~/Documents/notes --name mynotes --mask '**/*.md'

# Remove a collection
qmd-cloud collection remove mynotes

# Rename a collection
qmd-cloud collection rename mynotes my-notes

# List all files in a collection
qmd-cloud ls mynotes

# List files with a path prefix
qmd-cloud ls journals/2025
qmd-cloud ls qmd://journals/2025
```

## Context Management

```sh
# Add context to current directory (auto-detects collection)
qmd-cloud context add "Description of these files"

# Add context to a specific path
qmd-cloud context add /subfolder "Description for subfolder"

# Add global context to all collections (system message)
qmd-cloud context add / "Always include this context"

# Add context using virtual paths
qmd-cloud context add qmd://journals/ "Context for entire journals collection"
qmd-cloud context add qmd://journals/2024 "Journal entries from 2024"

# List all contexts
qmd-cloud context list

# Check for collections or paths without context
qmd-cloud context check

# Remove context
qmd-cloud context rm qmd://journals/2024
qmd-cloud context rm /  # Remove global context
```

## Document IDs (docid)

Each document has a unique short ID (docid) - the first 6 characters of its content hash.
Docids are shown in search results as `#abc123` and can be used with `get` and `multi-get`:

```sh
# Search returns docid in results
qmd-cloud vsearch "query" --json
# Output: [{"docid": "#abc123", "score": 0.85, "file": "docs/readme.md", ...}]

# Get document by docid
qmd-cloud get "#abc123"
qmd-cloud get abc123              # Leading # is optional

# Docids also work in multi-get comma-separated lists
qmd-cloud multi-get "#abc123, #def456"
```

## Options

```sh
# Search & retrieval
-c, --collection <name>  # Restrict search to a collection (matches pwd suffix)
-n <num>                 # Number of results
--all                    # Return all matches
--min-score <num>        # Minimum score threshold
--full                   # Show full document content
--line-numbers           # Add line numbers to output

# Multi-get specific
-l <num>                 # Maximum lines per file
--max-bytes <num>        # Skip files larger than this (default 10KB)

# Output formats (search and multi-get)
--json, --csv, --md, --xml, --files
```

## Development

```sh
bun src/qmd.ts <command>   # Run from source
bun link                   # Install globally as 'qmd-cloud'
```

## Architecture

- SQLite FTS5 for full-text search (not exposed in CLI, used internally)
- sqlite-vec for vector similarity search
- OpenRouter API for embeddings (text-embedding-3-large, 3072 dimensions)
- OpenRouter API for query expansion (gpt-4o-mini)
- Token-based chunking: 800 tokens/chunk with 15% overlap

## Important: Do NOT run automatically

- Never run `qmd-cloud collection add`, `qmd-cloud embed`, or `qmd-cloud update` automatically
- Never modify the SQLite database directly
- Write out example commands for the user to run manually
- Index is stored at `~/.cache/qmd/index.sqlite`

## Do NOT compile

- Never run `bun build --compile` - it overwrites the shell wrapper and breaks sqlite-vec
- The `qmd-cloud` file is a shell script that runs `bun src/qmd.ts` - do not replace it
