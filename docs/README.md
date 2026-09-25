# Docs

Modular docs. One concern per file. Update the matching file when that concern changes; do not grow a single dump.

| Doc | Use when |
| --- | --- |
| [COMMAND.md](../COMMAND.md) | Install, make/cli truth, local run order |
| [prd.md](./prd.md) | Product requirements (tests, gates, out of scope) |
| [trd.md](./trd.md) | Stack, folders, names, numbers |
| [architecture.md](./architecture.md) | How the system is shaped |
| [conventions.md](./conventions.md) | How we write code, git, and comments |
| [agent-workflow.md](./agent-workflow.md) | Graphify, `no-push/`, agent rules |
| [modules/](./modules/README.md) | One file per code module |
| [adr/](./adr/README.md) | Decisions that are expensive to reverse |

New module in code → add `docs/modules/<name>.md` in the same change.
