# Conventions

## Code

- Smallest diff that solves the request. Reuse before add.
- No unused abstractions, config, or files "for later".
- Comments only for non-obvious intent or a known ceiling (`// ponytail: global lock; per-account if throughput matters`).
- Loading and error paths at IO, network, and UI boundaries. Do not swallow errors.

## Git

- Commit only when asked.
- Never `Co-authored-by` / AI authorship on commits, files, or comments.
- Do not change git config.
- Do not force-push `main`/`master`.

## Files

- Group by kind. One module, one folder under `src/swarm/`.
- Local-only work goes in `no-push/`, not the repo root.

## Names (code)

- Package: `swarm`. Repo name is not the import.
- snake_case files and functions; PascalCase classes.
- Use glossary identifiers: `drone`, `shared_target`, `slot`, `geometry`, `neighbor_set`, `pass_gate`.
- Do not name things `agent`, `goal`, `quads_`, or `formation`.
