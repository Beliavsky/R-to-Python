# test79_glm_binomial.R
set.seed(123)

n <- 100
x <- rnorm(n)
eta <- -0.5 + 2 * x
p <- 1 / (1 + exp(-eta))
y <- rbinom(n, size = 1, prob = p)

fit <- glm(y ~ x, family = binomial())

print(coef(fit))
print(summary(fit))
print(head(fitted(fit)))
