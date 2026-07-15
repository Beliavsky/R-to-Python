# xreg_fit.r
# Read fixed regression data and fit lm(y ~ x1 + x2).

path <- "r_examples/xreg_data.txt"

tab <- read.table(path, header = FALSE)

y  <- tab[, 1]
x1 <- tab[, 2]
x2 <- tab[, 3]

df <- data.frame(y = y, x1 = x1, x2 = x2)
fit <- lm(y ~ x1 + x2, data = df)

cat("coefficients:\n")
print(coef(fit))

cat("sigma:", summary(fit)$sigma, "\n")
cat("r.squared:", summary(fit)$r.squared, "\n")
