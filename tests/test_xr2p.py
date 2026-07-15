from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from xr2p import flush_left_output, normalize_output, round_numeric_tokens, squeeze_output_spaces, translate_source


def test_round_numeric_tokens_leaves_integers_unchanged():
    out = round_numeric_tokens("iterations = 64\ncounts\n461 292 -247\nx = 1.23456 1e-3", 4)
    assert "iterations = 64" in out
    assert "461 292 -247" in out
    assert "1.2346 0.0010" in out


def test_flush_left_output_strips_leading_spaces_per_line():
    assert flush_left_output("  a\n b\n") == "a\nb\n"


def test_normalize_output_can_flush_left_after_rounding():
    assert normalize_output("  1.23456\n", 4, flush_left=True) == ["1.2346"]


def test_squeeze_output_spaces_collapses_runs_of_spaces():
    assert squeeze_output_spaces("a   b\n  c    d\n") == "a b\n c d\n"


def test_normalize_output_can_squeeze_after_rounding():
    assert normalize_output("  1.23456    2.0\n", 4, squeeze=True) == ["1.2346 2.0000"]


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


def test_character_vector_loop_variable_selects_matrix_column_by_name():
    out = translate_source(
        "prices <- matrix(c(100, 101, 102), ncol = 1)\n"
        'colnames(prices) <- c("SPY")\n'
        'response.assets <- c("SPY")\n'
        "for (response.asset in response.assets) {\n"
        "  y <- diff(log(prices[, response.asset]))\n"
        "}\n"
    )
    assert "for response_asset in response_assets:" in out
    assert "r_subset(prices, slice(None), r_col_key(prices, response_asset, globals().get('prices_colnames')))" in out
    assert "response_asset - 1" not in out


def test_full_line_r_comments_are_preserved():
    out = translate_source("# estimate pi\nx <- 1\n# print result\nprint(x)\n")
    assert "# estimate pi" in out
    assert "# print result" in out
    assert "x = 1" in out


def test_simple_numeric_arithmetic_keeps_python_operators():
    out = translate_source(
        "inside <- x^2 + y^2 <= 1.0\n"
        "pi_hat <- 4.0 * mean(inside)\n"
        "cat(\"error =\", pi_hat - pi, \"\\n\")\n"
    )
    assert "inside = x ** 2 + y ** 2 <= 1.0" in out
    assert "pi_hat = 4.0 * np.mean(inside)" in out
    assert 'print("error =", pi_hat - pi)' in out
    assert "def r_recycle_binary" not in out


def test_as_character_translates_to_string_array():
    out = translate_source('dates <- as.Date(as.character(prices$Date), format = "%Y%m%d")\n')
    assert 'dates = r_as_date(np.asarray(prices.Date, dtype=str), format="%Y%m%d")' in out
    assert "as_character" not in out


def test_setdiff_names_can_select_dataframe_columns():
    out = translate_source('x <- as.matrix(prices[, setdiff(names(prices), "Date")])\n')
    assert 'r_subset(prices, slice(None), r_setdiff(r_names(prices), "Date"))' in out
    assert "r_setdiff" in out
    assert 'r_setdiff(r_names(prices), "Date") - 1' not in out


def test_setdiff_names_variable_can_select_dataframe_columns():
    out = translate_source('price_names <- setdiff(names(prices), "Date")\nx <- as.matrix(prices[, price_names])\n')
    assert 'price_names = r_setdiff(r_names(prices), "Date")' in out
    assert "r_subset(prices, slice(None), r_col_key(prices, price_names, globals().get('prices_colnames')))" in out
    assert "price_names - 1" not in out


def test_as_matrix_dataframe_column_subset_preserves_colnames():
    out = translate_source('price_mat <- as.matrix(prices[, setdiff(names(prices), "Date")])\n')
    assert 'price_mat_colnames = list(r_setdiff(r_names(prices), "Date"))' in out
    assert "asset_names = r_colnames(price_mat, globals().get('price_mat_colnames', []))" in translate_source(
        "asset_names <- colnames(price_mat)\n"
    )


def test_as_matrix_dataframe_column_subset_ignores_drop_option_for_colnames():
    out = translate_source("price_mat <- as.matrix(prices[, price_names, drop = FALSE])\n")
    assert "price_mat_colnames = list(price_names)" in out
    assert "drop = False" not in out


def test_matrix_diff_uses_row_axis():
    out = translate_source("ret <- diff(log_price)\n")
    assert "def r_diff(x):" in out
    assert "np.diff(arr, axis=0) if arr.ndim >= 2 else np.diff(arr)" in out
    assert "ret = r_diff(log_price)" in out


def test_cor_use_option_translates_to_helper():
    out = translate_source('print(cor(ret_mat, use = "pairwise.complete.obs"))\n')
    assert 'cor_py(ret_mat, use="pairwise.complete.obs")' in out
    assert "np.corrcoef(ret_mat, use" not in out


def test_toeplitz_maps_to_scipy_linalg():
    out = translate_source("rmat <- toeplitz(acov[1:p])\n")
    assert "from scipy import linalg" in out
    assert "rmat = linalg.toeplitz(" in out


def test_gamma_besselk_and_hyperbolic_functions_map_to_scipy_and_numpy():
    out = translate_source(
        "a <- gamma(-Y)\n"
        "b <- besselK(x, nu)\n"
        "c <- besselK(x, nu, expon.scaled = TRUE)\n"
        "d <- cosh(z)\n"
    )
    assert "from scipy import special" in out
    assert "a = special.gamma(-Y)" in out
    assert "b = special.kv(nu, x)" in out
    assert "c = special.kve(nu, x)" in out
    assert "d = np.cosh(z)" in out


def test_quantile_names_false_returns_plain_array():
    out = translate_source("mu <- as.numeric(quantile(x, probs = q, names = FALSE))\n")
    assert "mu = r_as_numeric(np.quantile(x, q))" in out
    assert "lambda_" not in out
    assert "RNamedVector(np.quantile" not in out


def test_nagarch_var_uses_fast_recursion_helper():
    out = translate_source(
        "nagarch_var <- function(eps, omega, alpha, beta, theta) {\n"
        "  n <- length(eps)\n"
        "  h <- numeric(n)\n"
        "  h[1] <- var(eps)\n"
        "  for (i in 2:n) {\n"
        "    zlag <- eps[i - 1] / sqrt(h[i - 1])\n"
        "    h[i] <- omega + alpha * h[i - 1] * (zlag - theta)^2 + beta * h[i - 1]\n"
        "  }\n"
        "  return(h)\n"
        "}\n"
    )
    assert "def _nagarch_var_fast_impl(" in out
    assert "return nagarch_var_fast(eps, omega, alpha, beta, theta)" in out


def test_no_numba_keeps_nagarch_fast_path_without_numba_import():
    out = translate_source(
        "nagarch_var <- function(eps, omega, alpha, beta, theta) {\n"
        "  n <- length(eps)\n"
        "  h <- numeric(n)\n"
        "  h[1] <- var(eps)\n"
        "  for (i in 2:n) {\n"
        "    zlag <- eps[i - 1] / sqrt(h[i - 1])\n"
        "    h[i] <- omega + alpha * h[i - 1] * (zlag - theta)^2 + beta * h[i - 1]\n"
        "  }\n"
        "  return(h)\n"
        "}\n",
        use_numba=False,
    )
    assert "numba" not in out
    assert "nagarch_var_fast = _nagarch_var_fast_impl" in out
    assert "return nagarch_var_fast(eps, omega, alpha, beta, theta)" in out


def test_matrix_translation_uses_column_major_reshape():
    out = translate_source("x = 1:4\nxmat = matrix(x, 2, 2)\n")
    assert "x = r_seq(1, 4)" in out
    assert "np.resize(r_matrix_data(x), int(2 * 2)).reshape((int(2), int(2)), order='F')" in out


def test_matrix_size_expression_preserves_precedence():
    out = translate_source("xlag <- matrix(NA_real_, nrow = n - p, ncol = p)\n")
    assert "np.resize(r_matrix_data(np.nan), int(r_mul(r_sub(n, p), p))).reshape((int(r_sub(n, p)), int(p)), order='F')" in out


def test_nrow_ncol_inside_matrix_do_not_rewrite_shape_subscripts():
    out = translate_source("qprime <- t(matrix(rep(q, nrow(p)), ncol(p)))\n")
    assert "np.tile(np.repeat(q, 1), p.shape[0])" in out
    assert "reshape((int(p.shape[1]), -1), order='F').T" in out
    assert "r_matrix_index_get(p.shape" not in out


def test_attr_super_assignment_translates_to_set_attr():
    out = translate_source('attr(abc, "a_default") <<- a_default\n')
    assert 'r_set_attr(abc, "a_default", a_default)' in out
    assert "<=" not in out


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
    assert "phi = r_drop_index(coef_vec, 1)" in out


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
    assert 'table_colnames = list(np.atleast_1d(np.asarray(r_c("order", "aic"))))' in out
    assert "r_subset(table" in out
    assert "r_which_min(" in out


def test_named_matrix_column_lookup_with_variable_row():
    out = translate_source(
        'colnames(tab) <- c("p", "q", "aic")\n'
        'aic_idx <- which.min(tab[, "aic"])\n'
        'aic_p <- tab[aic_idx, "p"]\n'
    )
    assert "aic_idx = r_which_min(" in out
    assert 'aic_p = r_subset(tab, r_axis_index(aic_idx), r_col_key(tab, "p", globals().get(\'tab_colnames\')))' in out


def test_colnames_assignment_accepts_character_vector_variable():
    out = translate_source('coef_names <- c("order", paste0("phi", 1:2), "aic")\ncolnames(coef_mat) <- coef_names\n')
    assert "coef_mat_colnames = list(np.atleast_1d(np.asarray(coef_names)))" in out
    assert "pass  # R metadata assignment omitted" not in out


def test_nested_named_column_lookup_uses_r_col_key():
    out = translate_source(
        'colnames(coef_mat) <- coef_names\n'
        'aic_order <- coef_mat[which.min(coef_mat[, "aic"]), "order"]\n'
    )
    assert "r_col_key(coef_mat, 'aic', globals().get('coef_mat_colnames'))" in out
    assert 'r_col_key(coef_mat, "order", globals().get(\'coef_mat_colnames\'))' in out
    assert "coef_mat_colnames_index" not in out


def test_order_selection_from_named_column_is_cast_to_int():
    out = translate_source('aic_order <- coef_mat[which.min(coef_mat[, "aic"]), "order"]\n')
    assert "aic_order = int(" in out


def test_named_vector_character_assignment_uses_runtime_helper():
    out = translate_source('names(y) <- c("phi1", "phi2")\ny[paste0("phi", 1:2)] <- z\n')
    assert "def r_matrix_index_set(x, idx, value):" in out
    assert "isinstance(x, RNamedVector)" in out
    assert "x[str(name)] = val" in out


def test_order_decreasing_argument_is_preserved():
    out = translate_source("ord <- order(fit$prob, decreasing = TRUE)\n")
    assert "def r_order(x, decreasing=False):" in out
    assert "ord = r_order(fit.prob, decreasing=True)" in out


def test_member_assignment_uses_setattr_not_dotted_assignment():
    out = translate_source("fit$prob <- fit$prob[ord]\n")
    assert "setattr(fit, 'prob', r_matrix_index_get(r_member(fit, 'prob'), ord))" in out
    assert "fit_prob" not in out


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
    assert "phi = r_set_names(phi, " in out
    assert "names(phi)" not in out


def test_names_null_assignment_clears_named_vector_metadata():
    out = translate_source("names(beta) <- NULL\n")
    assert "beta = (beta.values if isinstance(beta, RNamedVector) else beta)" in out
    assert "list(None)" not in out


def test_cbind_named_columns_translates_to_column_stack():
    out = translate_source("print(cbind(r1 = r1, r2 = r2))\n")
    assert "cbind_py(r_data_frame(r1=r1), r_data_frame(r2=r2))" in out
    assert "def cbind_py(*cols, **named_cols):" in out


def test_print_round_matrix_uses_r_style_formatter():
    out = translate_source("print(round(tab, 4))\n")
    assert "r_print(np.round(tab, 4), digits=4, colnames=(tab_colnames if 'tab_colnames' in locals() else None))" in out
    assert "def r_print(*args, digits=None, colnames=None, row_names=True):" in out


def test_list_result_carries_column_names_for_printing():
    out = translate_source(
        'colnames(tab) <- c("p", "q", "aic")\n'
        'out <- list(table = tab)\n'
        'print(round(out$table, 4))\n'
    )
    assert "table_colnames=(tab_colnames if 'tab_colnames' in locals() else None)" in out
    assert "colnames=getattr(out, 'table_colnames', None)" in out


def test_print_row_names_option_passes_through():
    out = translate_source("print(out, row.names = FALSE)\n")
    assert "r_print(out, row_names=False)" in out


def test_tail_one_returns_last_element():
    out = translate_source("print(tail(x, 1))\n")
    assert "tail_py(x, 1)" in out


def test_tail_one_does_not_become_negative_r_subscript():
    out = translate_source('cat("last =", as.character(tail(ret_dates, 1)), "\\n")\n')
    assert "tail_py(ret_dates, 1)" in out
    assert "np.delete(ret_dates" not in out


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


def test_parenthesized_inline_if_translates_inside_arithmetic_expression():
    out = translate_source(
        "estim <- TRUE\n"
        "npar <- (if (estim) 1L else 0L) + 1L\n"
        "print(npar)\n"
    )
    assert "npar = r_add(1 if estim else 0, 1)" in out


def test_filter_negate_is_null_translates_bare_predicate_reference():
    out = translate_source(
        "fits <- list(1, NULL, 3)\n"
        "fits <- Filter(Negate(is.null), fits)\n"
    )
    assert "fits = r_filter(r_negate(r_is_null), fits)" in out
    assert "def r_filter(func, values):" in out
    assert "def r_negate(func):" in out
    assert "def r_is_null(value):" in out
    assert "is.null" not in out


def test_matrix_subscripts_and_apply_helpers():
    out = translate_source(
        "log_dens[, j] <- x\n"
        "row_max <- apply(log_dens, 1, max)\n"
        "nk <- colSums(resp)\n"
        "cluster <- max.col(resp, ties.method = \"first\")\n"
    )
    assert "r_set_subset(log_dens, x, slice(None), r_axis_index(j))" in out
    assert "row_max = r_apply(log_dens, 1, 'max')" in out
    assert "nk = np.sum(resp, axis=0)" in out
    assert "cluster = r_add(np.argmax(resp, axis=1), 1)" in out


def test_matrix_assignment_translates_nested_column_subscripts_in_logical_mask():
    out = translate_source(
        "h <- matrix(c(1, NA, -1, 2), nrow = 2)\n"
        "k <- 1L\n"
        "min.var <- 1.0e-10\n"
        "h[!is.finite(h[, k]) | h[, k] <= min.var, k] <- min.var\n"
    )
    assert "r_set_subset(h, min_var," in out
    assert "np.logical_not(np.isfinite(r_subset(h, slice(None), r_axis_index(k))))" in out
    assert "h[, k]" not in out


def test_drop_false_is_ignored_in_matrix_subscript():
    out = translate_source("y <- x[2:n, , drop = FALSE]\n")
    assert "y = r_subset(x, r_seq(2, n) - 1, slice(None))" in out
    assert "drop = False" not in out
    assert "drop=False" not in out


def test_negative_matrix_subscript_drops_axis_element():
    out = translate_source("a <- t(coef_mat[-1, , drop = FALSE])\n")
    assert "(r_drop_axis(coef_mat, 1, 0)).T" in out
    assert "drop = False" not in out
    assert "drop=False" not in out


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
    assert "par = r_c(par, r_as_numeric((r_list_get(a, i)).T))" in out


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
    assert "r_s3_print(r_list_get(xs, 2))" in out
    assert "ys = ([None] * (2))" in out
    assert "ys[(1) - 1] = r_list_get(xs, 2)" in out


def test_variable_double_bracket_index_uses_one_based_list_get():
    out = translate_source("print(mu[[k]])\n")
    assert "def r_list_get(x, idx):" in out
    assert "isinstance(x, RNamedVector)" in out
    assert "r_s3_print(r_list_get(mu, k))" in out


def test_common_matrix_vector_recycling_patterns():
    out = translate_source(
        "dens_shifted <- exp(log_dens - row_max)\n"
        "resp <- dens_shifted / rowSums(dens_shifted)\n"
        "mu <- colSums(resp * x) / nk\n"
    )
    assert "r_sub(log_dens, row_max[:, None])" in out
    assert "r_div(dens_shifted, np.sum(dens_shifted, axis=1)[:, None])" in out
    assert "r_mul(resp, x[:, None])" in out


def test_mixture_weight_vector_recycling_patterns():
    out = translate_source(
        "mu[[k]] <- as.numeric(colSums(x * wk) / nk[k])\n"
        "resp <- resp / row_sum\n"
    )
    assert "np.asarray(wk).reshape(-1, 1)" in out
    assert "np.asarray(row_sum).reshape(-1, 1)" in out


def test_storage_mode_double_assignment_translates_to_float_array():
    out = translate_source('storage.mode(x) <- "double"\n')
    assert "x = np.asarray(x, dtype=float)" in out


def test_read_table_default_separator_uses_normal_string_literal():
    out = translate_source("x <- read.table(in_file, header = FALSE)\n")
    assert "sep='\\\\s+'" in out
    assert "r__R_STR" not in out


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
    assert "b = r_matrix_index_get(RNamedVector(fit.coef" in out
    assert "r_s3_print(summary_py(fit))" in out
    assert "def summary_lm_py(fit):" in out


def test_rnorm_one_returns_scalar():
    out = translate_source("x <- rnorm(1)\ny <- rnorm(3)\n")
    assert "x = np.random.normal(0, 1)" in out
    assert "y = np.random.normal(0, 1, size=3)" in out

def test_replicate_rewrites_to_sapply_over_lambda():
    out = translate_source("x <- replicate(5, rnorm(1))\n")
    assert "x = r_sapply(np.arange(1, r_add(5, 1)), (lambda r_rep_i: np.random.normal(0, 1)))" in out
    assert "replicate(" not in out
    out_list = translate_source("x <- replicate(3, rnorm(2), simplify = FALSE)\n")
    assert "r_lapply(np.arange(1, r_add(3, 1))" in out_list
