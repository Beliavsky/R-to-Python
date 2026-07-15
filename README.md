# R-to-Python

`xr2p.py` is an experimental source-to-source transpiler from a practical subset
of R to Python. It is aimed at numerical scripts, simulation examples, small
data-analysis programs, and regression fixtures rather than full general R
compatibility.

The generated Python is designed to run directly. It primarily targets NumPy,
with pandas, SciPy, statsmodels, scikit-learn, and optional numba used where
those libraries are a closer match to R behavior or performance needs.

## Status

This project is actively evolving. It supports many common R constructs, but it
is not a complete R interpreter or a drop-in replacement for R.

The transpiler works best on small, self-contained `.r` scripts using base R and
straightforward numerical code. Package-heavy code, advanced non-standard
evaluation, and complex R object systems may require additional translator work.

For a quick feel for the generated code, see [small translation examples](docs/examples.md).

## Quick start

Translate an R file:

```bat
python xr2p.py analysis.r
```

This writes `analysis.py`.

Translate and run the generated Python:

```bat
python xr2p.py analysis.r --run
```

Run both the original R script and generated Python:

```bat
python xr2p.py analysis.r --run-both
```

Use the most recently modified `.r` or `.R` file in the current directory:

```bat
python xr2p.py @last --run
```

Print the generated Python while writing it:

```bat
python xr2p.py analysis.r --tee
```

Print both the original R source and generated Python:

```bat
python xr2p.py analysis.r --tee-both
```

Skip Python syntax checking:

```bat
python xr2p.py analysis.r --no-py-compile
```

Generated files start with a banner comment recording the source file and
translation time, for example<br>
`# Translated from analysis.r by xr2p.py on 2026-07-12 10:00:56.`
Suppress it with

```bat
python xr2p.py analysis.r --no-banner
```

## Useful command-line options

Show nonblank, noncomment lines of code for R and Python:

```bat
python xr2p.py analysis.r --loc
```

Run Python with transpilation and run time displayed:

```bat
python xr2p.py analysis.r --time
```

Run both R and Python with timing:

```bat
python xr2p.py analysis.r --time-both
```

Try to remove unused generated helpers from the output:

```bat
python xr2p.py analysis.r --lean
```

Move generated runtime helpers into `xr2p_runtime.py`:

```bat
python xr2p.py analysis.r --runtime-module
```

Keep only needed runtime helpers in that module:

```bat
python xr2p.py analysis.r --runtime-module --prune-runtime
```

Refresh an existing runtime module:

```bat
python xr2p.py analysis.r --runtime-module --update-runtime
```

Disable generated numba fast paths:

```bat
python xr2p.py analysis.r --no-numba
```

`--lean` and `--prune-runtime` check that the slimmed output still compiles. If
not, `xr2p.py` falls back to the safer unslimmed output and prints a warning.

## Output comparison options

Run both R and Python and show a unified output diff:

```bat
python xr2p.py analysis.r --run-diff
```

Normalize common R/Python formatting differences in R output:

```bat
python xr2p.py analysis.r --run-both --pretty
```

`--pretty` removes R display markers like `[1]`, maps `TRUE`/`FALSE` to
`True`/`False`, maps `NA` to `nan`, and removes quotes around simple printed
strings.

Compare numeric output summaries instead of full text:

```bat
python xr2p.py analysis.r --stats
```

`--stats` scans output for numeric values and reports count, minimum, maximum,
and sum for R and Python output.

Round numeric output before comparison:

```bat
python xr2p.py analysis.r --run-diff --round-both 6
```

Make displayed output easier to compare visually:

```bat
python xr2p.py analysis.r --run-both --flush-left --squeeze
```

`--flush-left` strips leading whitespace from displayed lines. `--squeeze`
collapses runs of two or more spaces to one space.

## Optional GUI

A small Tkinter GUI is available:

```bat
python scripts\xr2p_ide.py
```

Open an R file directly:

```bat
python scripts\xr2p_ide.py analysis.r
```

The GUI is a thin wrapper around `xr2p.py`. It provides:

- R source and generated Python panes.
- Separate R and Python output panes for Run Both.
- Syntax highlighting for R and Python.
- Optional autocomplete for R editing.
- Adjustable font size.
- Clear buttons for source and generated code panes.
- Timing displays for R, Python, and transpilation.
- Line-of-code displays for R and generated Python.
- Checkboxes for `--lean`, `--runtime-module`, `--prune-runtime`, `--pretty`,
  `--no-numba`, and related run options.
- Buttons for translate, Run R, Run Python, Run Both, and diff.

## Supported feature areas

Support is partial, but the current translator covers a broad base-R subset:

- Assignments with `<-`, `=`, and selected `<<-` cases.
- Scalar and vector arithmetic with R-style recycling where needed, while using
  ordinary Python operators for simple numeric expressions.
- Vectors, named vectors, and name-based indexing/assignment.
- R 1-based indexing, logical indexing, negative indexing, and matrix
  coordinate indexing.
- `for`, `while`, `repeat`, `break`, and `next`.
- `if`/`else` as expressions, including one-line forms with assignments in both
  branches.
- Functions, defaults, partial argument matching, `...`, and `do.call`.
- Inline anonymous functions (`function(x) ...` and the R 4.1 `\(x)`
  shorthand) in call arguments such as `sapply`, `apply`, and `outer`,
  including `FUN =` keyword form and the backtick `` `[` `` extractor.
- Native `|>` and magrittr-style `%>%` pipes.
- `%%`, `%/%`, and `L`-suffixed integer literals.
- Lists, `$`, `[[ ]]`, `lapply`, `sapply`, `mapply`, `split`, `unsplit`, and
  `Reduce`.
- Matrices and arrays with R column-major ordering.
- `cbind`, `rbind`, row/column sums and means, `apply`, `sweep`, `outer`,
  `diag`, `lower.tri`, `upper.tri`, `crossprod`, and `tcrossprod`.
- pandas-backed `data.frame`, tibble-style construction for common cases,
  subsetting, filtering, modification, `merge`, `aggregate`, `stack`, and
  `unstack`.
- Factors, `table`, `tapply`, ordering, sorting, ranking, `unique`,
  `duplicated`, `match`, `setdiff`, and `%in%`.
- Missing/infinite values: `NA`, typed `NA_*` constants, `NaN`, `Inf`,
  `is.na`, `is.nan`, `is.finite`, `is.infinite`, and `complete.cases`.
- Strings and regex helpers including `paste`, `paste0`, `sprintf`, `substr`,
  `substr<-`, `strsplit`, `format`, `grep`, `grepl`, `sub`, `gsub`, and
  `regexpr`.
- Dates, date sequences, date formatting, and simple time series helpers.
- Random generators and distribution `d`, `p`, and `q` functions using
  NumPy/SciPy where appropriate.
- Mathematical special functions including `gamma`, `lgamma`, `beta`,
  `lbeta`, `digamma`, `trigamma`, `psigamma`, `choose`, `lchoose`,
  `factorial`, `lfactorial`, and selected Bessel functions.
- Basic modeling/statistics helpers including `lm`, `glm` binomial, `aov`,
  `model.matrix`, `prcomp`, `kmeans`, `arima`, `cor`, `cov`, `eigen`, `svd`,
  `cov2cor`, `qr`, `polyroot`, `uniroot`, `integrate`, `kruskal.test`, and
  `wilcox.test`.
- File I/O helpers for CSV, tables, lines, simple text connections, and
  pickle-backed RDS-like save/load.
- Minimal S3-style class, attributes, `UseMethod`, `try`, `tryCatch`, `stop`,
  `warning`, `message`, and `capture.output`.

## Dependencies

Core translation uses Python's standard library. Generated Python may require:

- NumPy
- pandas
- SciPy
- statsmodels
- scikit-learn
- numba, optional, for selected generated fast paths

R is only needed for `--run-both`, `--time-both`, `--run-diff`, `--stats`, or
tests that compare against `rscript`.

## Tests

Run the Python test suite:

```bat
pytest -q
```

The current baseline is 139 passing tests. The suite is self-contained and
does not require a separate checkout of the R-to-Fortran project.

The tests include:

- Unit-style checks for generated Python snippets.
- Regression checks for recently added R features.
- Compile checks for selected fixture scripts.
- Runtime smoke tests for deterministic fixture scripts.
- Translation checks for 20 small examples vendored in
  `fixtures/xr2f_corpus/`.

Fixture scripts live in `fixtures/`. Generated `x*.py` outputs are ignored by
Git; regenerate them locally when needed.

Some fixtures are exploratory examples. Not every local `x*.r` file is part of
the hard pytest contract.

To test exactly what a commit contains before pushing it, clone the local Git
repository into a new directory and run pytest there:

```bat
git clone . ..\R-to-Python-clean-test
cd ..\R-to-Python-clean-test
python -m pytest -q
```

Only committed files appear in this clone. Staged and unstaged working-tree
changes are intentionally excluded, making this a useful final pre-push check.

## Batch translation checks

`xr2p_batch.py` can run translation sweeps over user-provided R files, glob
patterns, `@` lists, or directories:

```bat
python xr2p_batch.py *.r --quiet
```

Add syntax checking:

```bat
python xr2p_batch.py *.r --quiet --check-syntax
```

Run generated Python as well:

```bat
python xr2p_batch.py *.r --run --quiet
```

Run recursively over a downloaded corpus and save generated Python plus a CSV
summary:

```bat
python xr2p_batch.py path\to\r_corpus --recursive --check-syntax --out-dir translated_output --summary-csv translated_output\summary.csv --quiet
```

Useful corpus options include:

- `--limit N` to process only the first N expanded inputs.
- `--skip N` to skip the first N expanded inputs after filtering.
- `--max-fail N` to stop after N failures.
- `--only-r-pass CSV` to process only scripts that passed an `xrbatch.py` run.
- `--only-r-output CSV` to process only scripts that passed R and produced
  output. This implies the practical intent of `--only-r-pass`.

Failure output includes the generated Python line, nearby generated context,
and source file path. Batch runs also print the finish time and elapsed seconds.
The summary CSV includes status and elapsed-time information so unusually slow
Python translations can be identified.

## Running original R corpora

`xrbatch.py` runs original R scripts without translating them. This is useful
for deciding which programs are valid R examples before asking `xr2p_batch.py`
to translate and run them.

Run a recursive R sweep:

```bat
python xrbatch.py path\to\r_corpus --recursive --summary-csv r_runs.csv
```

Save stdout and stderr for each R script:

```bat
python xrbatch.py path\to\r_corpus --recursive --log-dir r_logs --summary-csv r_runs.csv
```

The `xrbatch.py` CSV records pass/fail status, elapsed time, and stdout size.
Scripts with no stdout are often library/function files rather than standalone
programs; `xr2p_batch.py --only-r-output burkardt_r_runs.csv` can use that CSV
to focus on runnable examples.

## Burkardt-style source paths

Some public R corpora contain absolute `source()` calls that point to the
author's machine. The repository includes a helper script for rewriting those
to relative paths in a local corpus copy. Use it on a disposable downloaded
copy, not on upstream source files you want to keep pristine.

## Performance notes

Generated Python is designed first for understandable, runnable translations.
Fast Python sometimes requires additional specialization.

`optim()` is a notable case. R code often passes an objective function that uses
scalar loops and flexible R indexing. Translating that literally can be slow in
Python because SciPy calls the objective many times, and each call may pass
through R-compatibility helpers. The transpiler includes fast paths for some
recognized numerical kernels, such as ARMA residuals, VARMA residuals, and
NAGARCH variance recursions, but arbitrary `optim()` objectives may still need
manual vectorization, a custom SciPy formulation, or a new generated fast path.

SciPy optimizer convergence codes also do not map exactly to R `optim()` codes.
The compatibility wrapper handles common finite precision-loss exits, but
numerical results and convergence flags can still differ from R.

Use these options when investigating performance or output differences:

```bat
python xr2p.py script.r --time
python xr2p.py script.r --time-both --round-both 6
python xr2p.py script.r --loc --lean --runtime-module --prune-runtime
```

## Development notes

The translator is intentionally pragmatic. Many R constructs are lowered to
small runtime helper functions emitted into the generated Python file. This
keeps individual translations simple and makes R-specific behavior explicit in
the output.

When generated files contain too much boilerplate, use `--lean` or
`--runtime-module --prune-runtime` to separate or reduce the runtime helpers.

When adding support for a new R construct, prefer adding:

- A small fixture script, usually named `xfeature_name.r`.
- A compile or runtime regression test in `tests/`.
- A focused helper function when Python/NumPy semantics differ from R.

## Limitations

Known limitations include:

- No full parser for all R syntax.
- No general package/import translation.
- Incomplete support for non-standard evaluation.
- Partial S3/S4 behavior only.
- Formatting may differ from R even when numeric results match.
- Statistical routines may use different algorithms from R and can differ
  numerically.
- Automatically translated optimization code may be correct but slower than R
  until inner numerical kernels are specialized.

Use `--pretty`, `--round-both`, `--flush-left`, `--squeeze`, and `--stats` when
comparing outputs from R and Python.
