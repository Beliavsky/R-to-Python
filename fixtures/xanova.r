# test78_anova.R
set.seed(123)

group <- factor(rep(c("a", "b", "c"), each = 20))
y <- c(rnorm(20, 0), rnorm(20, 1), rnorm(20, 2))

fit <- aov(y ~ group)

print(summary(fit))
