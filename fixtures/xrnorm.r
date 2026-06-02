n = 10^3
for (i in 1:3) {
	x = rnorm(n)
	cat("\n", min(x), max(x), median(x), mean(x), sd(x))
}