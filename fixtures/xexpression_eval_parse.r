# test95_expression_eval_parse.R
expr <- expression(1 + 2 * 3)
print(eval(expr))

txt <- "x <- 10; y <- x^2; y + 1"
ans <- eval(parse(text = txt))
print(ans)
