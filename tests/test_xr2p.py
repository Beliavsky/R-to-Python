from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from xr2p import translate_source


def test_vector_loop_translation():
    out = translate_source(
        "x <- 3\n"
        "y <- c(1.0, 4.0, 9.0)\n"
        "for (v in y) {\n"
        "  print(v)\n"
        "}\n"
    )
    assert "x = 3" in out
    assert "y = r_c(1.0, 4.0, 9.0)" in out
    assert "for v in y:" in out
    assert "r_print(v" in out


def test_matrix_translation_uses_column_major_reshape():
    out = translate_source("x = 1:4\nxmat = matrix(x, 2, 2)\n")
    assert "x = r_seq(1, 4)" in out
    assert "np.resize(r_matrix_data(x), r_mul(2, 2)).reshape((2, 2), order='F')" in out


def test_matrix_size_expression_preserves_precedence():
    out = translate_source("xlag <- matrix(NA_real_, nrow = n - p, ncol = p)\n")
    assert "np.resize(r_matrix_data(np.nan), r_mul(r_sub(n, p), p)).reshape((r_sub(n, p), p), order='F')" in out


def test_one_line_for_translation():
    out = translate_source("for (i in 1:3) print(sqrt(i))\n")
    assert "for i in range(1, 4):" in out
    assert "r_s3_print(np.sqrt(i))" in out


def test_simple_cat_newline_loop_omits_numpy_import_and_end_argument():
    out = translate_source('for (i in 1:3) {\n  cat(i, i^2, "\\n")\n}\n')
    assert not out.startswith("import numpy as np")
    assert "for i in range(1, 4):" in out
    assert "print(i, i ** 2)" in out
    assert 'end=""' not in out


def test_unused_for_counter_uses_zero_based_arange():
    out = translate_source(
        "for (i in 1:3) {\n"
        "  x = rnorm(10)\n"
        "  print(mean(x))\n"
        "}\n"
    )
    assert "for i in range(1, 4):" in out
    assert "r_seq(1, 3)" not in out


def test_cat_with_nested_calls_and_string_literal():
    out = translate_source('x = rnorm(10)\ncat("\\n", mean(x), sd(x))\n')
    assert 'x = np.random.normal(0, 1, size=10)' in out
    assert 'print("\\n", np.mean(x), np.std(x, ddof=1), end="")' in out


def test_normal_distribution_functions_use_scipy():
    out = translate_source("print(dnorm(0), pnorm(0), qnorm(0.5))\n")
    assert "from scipy import stats" in out
    assert "stats.norm.pdf(0, loc=0, scale=1)" in out
    assert "stats.norm.cdf(0, loc=0, scale=1)" in out
    assert "stats.norm.ppf(0.5, loc=0, scale=1)" in out


def test_run_both_option_runs_r_and_python(tmp_path: Path):
    source = tmp_path / "xrnorm.r"
    source.write_text('n = 10\nx = rnorm(n)\ncat("\\n", mean(x), sd(x))\n', encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(ROOT / "xr2p.py"), str(source), "--run-both"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Run (R): PASS" in proc.stdout
    assert "Run (Python): PASS" in proc.stdout


def test_tee_option_prints_generated_python(tmp_path: Path):
    source = tmp_path / "x.r"
    out = tmp_path / "x.py"
    source.write_text("x <- 3\nprint(x)\n", encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(ROOT / "xr2p.py"), str(source), "-o", str(out), "--tee"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "wrote " in proc.stdout
    assert "import numpy as np" in proc.stdout
    assert "x = 3" in proc.stdout
    assert "print(x)" in proc.stdout


def test_tee_both_option_prints_source_and_generated_python(tmp_path: Path):
    source = tmp_path / "x.r"
    out = tmp_path / "x.py"
    source.write_text("x <- 3\nprint(x)\n", encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(ROOT / "xr2p.py"), str(source), "-o", str(out), "--tee-both"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "R source:" in proc.stdout
    assert "x <- 3" in proc.stdout
    assert "Python translation:" in proc.stdout
    assert "x = 3" in proc.stdout


def test_cli_checks_generated_python_by_default(tmp_path: Path):
    source = tmp_path / "bad.r"
    out = tmp_path / "bad.py"
    source.write_text("x <- 1 +\n", encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(ROOT / "xr2p.py"), str(source), "-o", str(out)],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode != 0
    assert "Python syntax check failed" in proc.stderr


def test_no_py_compile_skips_generated_python_check(tmp_path: Path):
    source = tmp_path / "bad.r"
    out = tmp_path / "bad.py"
    source.write_text("x <- 1 +\n", encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(ROOT / "xr2p.py"), str(source), "-o", str(out), "--no-py-compile"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert out.exists()


def test_function_definition_and_member_access_translate():
    out = translate_source(
        "f <- function(x) {\n"
        "  list(y = x)\n"
        "}\n"
        "z <- f(3)\n"
        "print(z$y)\n"
    )

    assert "def f(x):" in out
    assert "return RList(y=x, _r_names=['y'])" in out
    assert "r_print(z.y, colnames=getattr(z, 'y_colnames', None))" in out


def test_member_access_survives_nested_sprintf():
    out = translate_source('cat(sprintf("n = %d\\n", sim$true$n))\n')
    assert "sim.true.n" in out
    assert "sim_true_n" not in out


def test_r_subscripts_are_one_based():
    out = translate_source("print(pi[1])\nprint(mu[z])\nprint(x[is.finite(x)])\n")
    assert "r_matrix_index_get(pi, 1)" in out
    assert "r_matrix_index_get(mu, z)" in out
    assert "r_matrix_index_get(x, np.isfinite(x))" in out


def test_parenthesized_range_subscript():
    out = translate_source(
        "n <- length(x)\n"
        "lagx <- x[1:(n - 1)]\n"
    )
    assert "lagx = r_matrix_index_get(x, r_seq(1, r_sub(n, 1)))" in out
    assert "1:(n - 1)" not in out


def test_negative_integer_subscript_drops_element():
    out = translate_source("phi <- coef_vec[-1]\n")
    assert "phi = np.delete(coef_vec, r_sub(1, 1))" in out


def test_parenthesized_range_in_for_loop():
    out = translate_source("for (t in (p + 1):n) print(t)\n")
    assert "for t in r_range(r_add(p, 1), n):" in out
    assert "(p + 1):n" not in out


def test_descending_range_subscript_uses_r_seq_helper():
    out = translate_source("lag_values <- x[(t - 1):(t - p)]\n")
    assert "def r_seq(start, stop):" in out
    assert "lag_values = r_matrix_index_get(x, r_seq(r_sub(t, 1), r_sub(t, p)))" in out


def test_range_endpoint_can_be_function_call():
    out = translate_source("for (j in 1:length(a)) print(j)\n")
    assert "for j in r_range(1, r_length(a)):" in out
    assert "r_seq(1, length)(a)" not in out


def test_range_endpoint_can_be_member_access():
    out = translate_source("for (i in 1:fit$p) print(i)\n")
    assert "for i in r_range(1, fit.p):" in out


def test_simple_symbol_stop_for_loop_uses_python_range():
    out = translate_source("for (i in 1:n) print(i)\n")
    assert "for i in range(1, n + 1):" in out


def test_floor_maps_to_numpy():
    out = translate_source("for (i in 2:floor(sqrt(n))) print(i)\n")
    assert "np.floor(np.sqrt(n))" in out
    assert " floor(" not in out


def test_unknown_direction_for_loop_keeps_r_range():
    out = translate_source("for (i in a:b) print(i)\n")
    assert "for i in r_range(a, b):" in out


def test_data_frame_with_mixed_named_and_unnamed_args():
    out = translate_source("df <- data.frame(y = y, xlag)\n")
    assert "import pandas as pd" in out
    assert "df = r_data_frame(y=y, xlag=xlag)" in out


def test_lm_dot_formula_with_data_frame():
    out = translate_source("df <- data.frame(y = y, xlag)\nfit <- lm(y ~ ., data = df)\n")
    assert "fit = lm_py(df.y, df.xlag)" in out


def test_named_matrix_column_lookup_and_which_min():
    out = translate_source(
        'colnames(table) <- c("order", "aic")\n'
        'aic_order <- table[which.min(table[, "aic"]), "order"]\n'
    )
    assert 'table_colnames = list(r_c("order", "aic"))' in out
    assert "r_subset(table" in out
    assert "r_which_min(" in out


def test_named_matrix_column_lookup_with_variable_row():
    out = translate_source(
        'colnames(tab) <- c("p", "q", "aic")\n'
        'aic_idx <- which.min(tab[, "aic"])\n'
        'aic_p <- tab[aic_idx, "p"]\n'
    )
    assert "aic_idx = r_which_min(" in out
    assert 'aic_p = r_subset(tab, r_sub(aic_idx, 1), r_col_key(tab, "p", globals().get(\'tab_colnames\')))' in out


def test_named_matrix_column_lookup_inside_list_result():
    out = translate_source(
        'colnames(tab) <- c("p", "q", "aic")\n'
        'aic_idx <- which.min(tab[, "aic"])\n'
        'out <- list(aic_p = tab[aic_idx, "p"], aic_q = tab[aic_idx, "q"])\n'
    )
    assert "aic_p=r_subset(tab" in out
    assert "aic_q=r_subset(tab" in out


def test_names_assignment_is_omitted_as_metadata():
    out = translate_source('names(phi) <- paste0("phi", 1:p)\n')
    assert "phi = RNamedVector(phi, list(" in out
    assert "names(phi)" not in out


def test_cbind_named_columns_translates_to_column_stack():
    out = translate_source("print(cbind(r1 = r1, r2 = r2))\n")
    assert "cbind_py(r_data_frame(r1=r1), r_data_frame(r2=r2))" in out
    assert "def cbind_py(*cols):" in out


def test_print_round_matrix_uses_r_style_formatter():
    out = translate_source("print(round(tab, 4))\n")
    assert "r_print(np.round(tab, 4), digits=4, colnames=(tab_colnames if 'tab_colnames' in locals() else None))" in out
    assert "def r_print(*args, digits=None, colnames=None):" in out


def test_list_result_carries_column_names_for_printing():
    out = translate_source(
        'colnames(tab) <- c("p", "q", "aic")\n'
        'out <- list(table = tab)\n'
        'print(round(out$table, 4))\n'
    )
    assert "table_colnames=(tab_colnames if 'tab_colnames' in locals() else None)" in out
    assert "colnames=getattr(out, 'table_colnames', None)" in out


def test_tail_one_returns_last_element():
    out = translate_source("print(tail(x, 1))\n")
    assert "x[-1]" in out


def test_blank_line_after_function_block():
    out = translate_source(
        "f <- function(x) {\n"
        "  list(y = x)\n"
        "}\n"
        "z <- f(3)\n"
    )
    assert "return RList(y=x, _r_names=['y'])\n\nz = f(3)" in out


def test_else_if_translates_to_elif():
    out = translate_source(
        "if (is.null(a)) {\n"
        "  x <- 1\n"
        "} else if (is.null(b)) {\n"
        "  x <- 2\n"
        "} else {\n"
        "  x <- 3\n"
        "}\n"
    )
    assert "elif (b is None):" in out
    assert "else if" not in out


def test_matrix_subscripts_and_apply_helpers():
    out = translate_source(
        "log_dens[, j] <- x\n"
        "row_max <- apply(log_dens, 1, max)\n"
        "nk <- colSums(resp)\n"
        "cluster <- max.col(resp, ties.method = \"first\")\n"
    )
    assert "r_set_subset(log_dens, x, slice(None), (j) - 1)" in out
    assert "row_max = r_apply(log_dens, 1, 'max')" in out
    assert "nk = np.sum(resp, axis=0)" in out
    assert "cluster = r_add(np.argmax(resp, axis=1), 1)" in out


def test_drop_false_is_ignored_in_matrix_subscript():
    out = translate_source("y <- x[2:n, , drop = FALSE]\n")
    assert "y = r_subset(x, r_sub(r_seq(2, n), 1), slice(None))" in out
    assert "drop" not in out


def test_negative_matrix_subscript_drops_axis_element():
    out = translate_source("a <- t(coef_mat[-1, , drop = FALSE])\n")
    assert "a = (np.delete(coef_mat, r_sub(1, 1), axis=0)).T" in out
    assert "drop" not in out


def test_tail_expression_and_paste_formatting():
    out = translate_source(
        "fmt_vec <- function(x) {\n"
        "  paste0(\"(\", paste(sprintf(\"%.6f\", x), collapse = \", \"), \")\")\n"
        "}\n"
    )
    assert "return " in out
    assert "np.char.mod(\"%.6f\", x)" in out
    assert "collapse=\", \"" in out


def test_explicit_return_call_is_not_double_returned():
    out = translate_source(
        "f <- function(y) {\n"
        "  return(y)\n"
        "}\n"
    )
    assert "return y" in out
    assert "return return" not in out


def test_default_argument_can_reference_prior_argument():
    out = translate_source(
        "f <- function(mu, x0 = mu) {\n"
        "  return(x0)\n"
        "}\n"
    )
    assert "def f(mu, x0=None):" in out
    assert "if x0 is None:" in out
    assert "x0 = mu" in out


def test_default_argument_expression_can_reference_prior_arguments():
    out = translate_source(
        "f <- function(x, p, q, start_order = max(p, q)) {\n"
        "  return(start_order)\n"
        "}\n"
    )
    assert "def f(x, p, q, start_order=None):" in out
    assert "if start_order is None:" in out
    assert "start_order = np.maximum(p, q)" in out


def test_c_concatenates_existing_vector_arguments():
    out = translate_source("par <- c(par, as.numeric(t(a[[i]])))\n")
    assert "par = r_c(par, np.asarray((r_matrix_index_get(a, r_sub(i, 1))).T, dtype=float))" in out


def test_try_error_translation_for_cholesky():
    out = translate_source(
        "r <- try(chol(sigma), silent = TRUE)\n"
        "if (inherits(r, \"try-error\")) return(1.0e100)\n"
    )
    assert "class TryError:" in out
    assert "r = try_(lambda: np.linalg.cholesky(sigma).T, silent=True)" in out
    assert "if isinstance(r, TryError):" in out


def test_unnamed_list_and_double_bracket_indexing():
    out = translate_source(
        "xs <- list(c(1, 2), c(3, 4))\n"
        "print(xs[[2]])\n"
        "ys <- vector(\"list\", 2)\n"
        "ys[[1]] <- xs[[2]]\n"
    )
    assert "xs = [r_c(1, 2), r_c(3, 4)]" in out
    assert "r_s3_print(r_matrix_index_get(xs, 1))" in out
    assert "ys = ([None] * (2))" in out
    assert "ys[(1) - 1] = r_matrix_index_get(xs, 1)" in out


def test_common_matrix_vector_recycling_patterns():
    out = translate_source(
        "dens_shifted <- exp(log_dens - row_max)\n"
        "resp <- dens_shifted / rowSums(dens_shifted)\n"
        "mu <- colSums(resp * x) / nk\n"
    )
    assert "r_sub(log_dens, row_max[:, None])" in out
    assert "r_div(dens_shifted, np.sum(dens_shifted, axis=1)[:, None])" in out
    assert "r_mul(resp, x[:, None])" in out


def test_row_and_column_means_translate():
    out = translate_source("a <- colMeans(x)\nb <- rowMeans(x)\n")
    assert "a = np.mean(x, axis=0)" in out
    assert "b = np.mean(x, axis=1)" in out


def test_sweep_and_matrix_var_helpers_are_emitted():
    out = translate_source(
        "sigma <- var(x) + ridge * diag(p)\n"
        "xc <- sweep(x, 2, mu, \"-\")\n"
    )
    assert "def var_r(x):" in out
    assert "sigma = r_add(var_r(x), r_mul(ridge," in out
    assert "def sweep_py(x, margin, stats, op):" in out
    assert 'xc = sweep_py(x, 2, mu, "-")' in out


def test_coef_result_supports_r_one_based_subscript():
    out = translate_source(
        "fit <- lm(y ~ x)\n"
        "b <- coef(fit)[2]\n"
        "print(summary(fit))\n"
    )
    assert "fit = lm_py(y, x)" in out
    assert "b = RNamedVector(fit.coef" in out
    assert "r_s3_print(summary_py(fit))" in out
    assert "def summary_lm_py(fit):" in out


def test_rnorm_one_returns_scalar():
    out = translate_source("x <- rnorm(1)\ny <- rnorm(3)\n")
    assert "x = np.random.normal(0, 1)" in out
    assert "y = np.random.normal(0, 1, size=3)" in out
