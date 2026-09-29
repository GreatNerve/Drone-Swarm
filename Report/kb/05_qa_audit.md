# 05 — QA Audit and LaTeX Build (Agent 5)

Build: `tectonic -X compile main.tex` in `Report/latex/` → `main.pdf` (version 1, with lists, **34 pages**) and `main_without_lists.tex` → version 2 (**32 pages**)
(title, certificate, declaration, acknowledgement, abstract, contents, list of figures, list of tables on separate
pages; 6 chapters; bibliography; appendix A). 12 pt, 1.1 line spacing, no errors, no overfull boxes, no
undefined references; every page keeps at least 17.9 pt between content and the page number (checked with
`figures/scripts/check_footer.py`). `.tex`/`.bib` are ASCII (pdfLaTeX-safe). Overleaf zip compiles standalone.

## Bibliography (revised on user request: only works actually used by the project) — 9 entries

From `docs/research-plan.md` (design sources): Batra 2021, Huang 2024, QuadSwarm 2023, Yu 2022 (MAPPO),
Panerati 2021, van den Berg (ORCA). Implemented algorithms (`train.py`): PPO, GAE, DAgger.
Removed 20 entries that came from the team context list but are not directly used; their in-text citations were
removed and sentences rewritten.

## Checks

| Check | Result |
|---|---|
| Names / roll numbers / supervisor ("Prof.", designation "Professor") / institution | exact |
| Numbers vs source files | from run3 showcase reports, stage configs/logs, checkpoint, recorded traces |
| Unit tests | 40 passed |
| Figures / tables | 13 / 8, all captioned and referenced |
| Forbidden terms (baseline, hand-coded, shield, V1, V2) | 0 occurrences in PDF |
| V4 | trained on separate machine; errors during training; V3 = final model |
| Restored after page-cut review | imitation-convergence figure, Batra figure, full Sec. 5.7 text, full Appendix A (software, structure, commands, scenarios, unit tests, figure generation) |
| Page-number bug | arabic numbering started on the List-of-Tables page → fixed with `\clearpage` |

## Remaining items for the team

- Sign certificate / declaration / acknowledgement.
- Plagiarism page removed on request; add back if required at submission.
