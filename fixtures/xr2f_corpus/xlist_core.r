fit <- list(a = 1.0, b = 2.0)
print(fit$a)
best <- fit
print(best$b)
fit2 <- list(meta = list(n = 3), ll = 1.25)
print(fit2$meta$n)
print(fit2$ll)
fit$a <- fit$a + 10.0
fit$b <- fit$b * 2.0
print(fit$a)
print(fit$b)
flag <- TRUE
if (flag) {
  fit$tag <- 1.0
} else {
  fit$tag <- 0.0
}
print(fit$tag)
