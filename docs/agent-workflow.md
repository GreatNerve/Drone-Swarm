# Agent workflow

## Graphify

Use the graph only when you do not already know where to look (architecture / "where is X"). Files already open in this chat, or `no-push/runs/` after a demo, come first.

```
graphify query "<question>"
graphify path "A" "B"
graphify explain "Node"
```

Do **not** query, detect, extract, or `--update` for a bugfix in a known file, a viewer tweak, or a "the sim looked wrong" report.

Rebuild only when the user runs `/graphify`, or when a local query misses because the graph is stale:

```
graphify --update          # local AST only
graphify --cluster-only    # reclustering, no re-extract
```

**Gemini:** do not use it. Local extract only. If that local extract fails, then Gemini is allowed.

Put expensive one-off extracts (PDF text, OCR, API dumps) in `no-push/extracts/` and reuse them.

## no-push/

Already ignored by global git. Holds:

| Path | For |
| --- | --- |
| `no-push/extracts/` | Cached text/md from PDFs, pages, datasets |
| `no-push/local/` | Scratch code not part of the product |
| `no-push/runs/` | Last 3 view/eval run logs (positions, slots, issues) |

Never stage this folder. Never treat it as the module docs — those live in `docs/`.

## Rules

`.cursor/rules/` plus this file and `AGENTS.md`. If they conflict, `AGENTS.md` wins, then update the rule.
