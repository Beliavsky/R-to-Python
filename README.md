# R-to-Python

`xr2p.py` is an experimental source-to-source transpiler from a practical
subset of R to Python.  It is aimed at numerical scripts, simulation examples,
small data-analysis programs, and regression fixtures rather than full general
R compatibility.

The generated Python is designed to run directly.  It primarily targets NumPy,
with pandas, SciPy, statsmodels, and scikit-learn used for features where those
libraries are a closer match to R behavior.

## Status

This project is actively evolving.  It supports many common R constructs, but
it is not a complete R interpreter or a drop-in replacement for R.

The transpiler works best on small, self-contained `.r` scripts that use base R
features.  Package-heavy code, advanced non-standard evaluation, and complex R
object systems may require additional translator work.

## Quick start

Translate an R file:

```bat
python xr2p.py fixtures\xseq.r
```

This writes `fixtures\xseq.py`.

Translate and run the generated Python:

```bat
python xr2p.py fixtures\xseq.r --run
```

Run both the original R script and generated Python:

```bat
python xr2p.py fixtures\xseq.r --run-both
```

Use the most recently modified `.r` or `.R` file in the current directory:

```bat
python xr2p.py @last --run
```

Print the generated Python while writing it:

```bat
python xr2p.py fixtures\xseq.r --tee
```

Skip Python syntax checking:

```bat
python xr2p.py fixtures\xseq.r --no-py-compile
```

## Output comparison options

Run both R and Python and show a unified output diff:

```bat
python xr2p.py fixtures\xseq.r --run-diff
```

Normalize common R/Python formatting differences in R output:

```bat
python xr2p.py fixtures\xseq.r --run-both --pretty
```

`--pretty` removes R display markers like `[1]`, maps `TRUE`/`FALSE` to
`True`/`False`, maps `NA` to `nan`, and removes quotes around simple printed
strings.

Compare numeric output summaries instead of full text:

```bat
python xr2p.py fixtures\xseq.r --stats
```

`--stats` scans output for numeric values and reports count, minimum, maximum,
and sum for R and Python output.

Round numeric output before comparison:

```bat
python xr2p.py fixtures\xseq.r --run-diff --round-both 6
```

## Supported feature areas

Support is partial, but the current translator covers a broad base-R subset:

- Assignments with `<-`, `=`, and selected `<<-` cases.
- Scalar and vector arithmetic with R-style recycling.
- Vectors, named vectors, and name-based indexing/assignment.
- R 1-based indexing, logical indexing, negative indexing, and matrix
  coordinate indexing.
- `for`, `while`, `repeat`, `break`, and `next`.
- Functions, defaults, partial argument matching, `...`, and `do.call`.
- Lists, `$`, `[[ ]]`, `lapply`, `sapply`, `mapply`, `split`, `unsplit`, and
  `Reduce`.
- Matrices and arrays with R column-major ordering.
- `cbind`, `rbind`, row/column sums and means, `apply`, `sweep`, `outer`,
  `diag`, `lower.tri`, `upper.tri`, `crossprod`, and `tcrossprod`.
- pandas-backed `data.frame`, subsetting, filtering, modification, `merge`,
  `aggregate`, `stack`, and `unstack`.
- Factors, `table`, `tapply`, ordering, sorting, ranking, `unique`,
  `duplicated`, `match`, and `%in%`.
- Missing/infinite values: `NA`, `NaN`, `Inf`, `is.na`, `is.nan`,
  `is.finite`, and `is.infinite`.
- Strings and regex helpers including `paste`, `paste0`, `sprintf`, `substr`,
  `grep`, `grepl`, `sub`, `gsub`, and `regexpr`.
- Dates and simple time series helpers.
- Random generators and distribution functions using NumPy/SciPy where
  appropriate.
- Basic modeling/statistics helpers including `lm`, `glm` binomial, `aov`,
  `model.matrix`, `prcomp`, `kmeans`, `arima`, `cor`, `cov`, `eigen`, `svd`,
  and `qr`.
- File I/O helpers for CSV, lines, simple text connections, and RDS-like
  pickle-backed save/load.
- Minimal S3-style class, attributes, `UseMethod`, `try`, `tryCatch`,
  `stop`, `warning`, `message`, and `capture.output`.

## Dependencies

Core translation uses Python's standard library.  Generated Python may require:

- NumPy
- pandas
- SciPy
- statsmodels
- scikit-learn

R is only needed for `--run-both`, `--run-diff`, `--stats`, or tests that
compare against `rscript`.

## Tests

Run the Python test suite:

```bat
pytest -q
```

The tests include:

- Unit-style checks for generated Python snippets.
- Regression checks for recently added R features.
- Compile checks for selected fixture scripts.
- Runtime smoke tests for deterministic fixture scripts.

Some fixtures are exploratory examples.  Not every local `x*.r` file is part of
the hard pytest contract.

Fixture scripts live in `fixtures/`.  Generated `x*.py` outputs are ignored by
Git; regenerate them locally when needed.

## Batch corpus check

If `c:\python\R-to-Fortran` is present, run a sweep over the reusable examples
from that project:

```bat
python xr2p_batch.py --xr2f-pytest-corpus --quiet
```

Add syntax checking:

```bat
python xr2p_batch.py --xr2f-pytest-corpus --quiet --check-syntax
```

## Development notes

The translator is intentionally pragmatic.  Many R constructs are lowered to
small runtime helper functions emitted into the generated Python file.  This
keeps individual translations simple and makes R-specific behavior explicit in
the output.

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

Use `--pretty`, `--round-both`, and `--stats` when comparing outputs from R and
Python.
