# test66_formula_lm_with_interaction.R
set.seed(123)

n <- 100
x <- rnorm(n)
g <- factor(sample(c("a", "b"), n, replace = TRUE))
y <- 1 + 2 * x + ifelse(g == "b", 3, 0) + ifelse(g == "b", 2 * x, 0) + rnorm(n)

df <- data.frame(y = y, x = x, g = g)

fit <- lm(y ~ x * g, data = df)

print(coef(fit))
print(summary(fit))
