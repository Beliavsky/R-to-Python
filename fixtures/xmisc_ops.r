# Integer literals, integer division, pipes, format, and replacement functions.

# Integer division and modulo with L-suffixed literals.
print(7L %/% 2L)
print(7 %% 3)

# L-suffixed literals inside matrix subscript assignments.
temp <- matrix(0, 4, 4)
nx <- 4L
ny <- 4L
temp[nx %/% 2L, ny %/% 2L] <- 100
print(temp[2, 2])

# Zero-length numeric and integer vectors.
x <- numeric()
print(length(x))
i <- integer()
print(length(i))

# format() on numeric vectors.
nums <- c(1.5, 2, 3.25)
print(format(nums))
print(format(3.14159, nsmall = 2))

# Native pipe and magrittr-style pipe.
double_it <- function(x) x * 2
add_one <- function(x) x + 1
piped <- c(1, 2, 3) |>
  double_it() |>
  add_one()
print(piped)

# substr replacement assignment.
s2 <- "abcdef"
substr(s2, 2, 4) <- "XYZ"
print(s2)

# Dotted variable names with matrix subscript assignment.
strat.ret <- matrix(0, nrow = 3, ncol = 2)
sr <- c(1, 2, 3)
for (j in 1:2) {
  strat.ret[, j] <- sr * j
}
print(strat.ret)

# Chained assignment still expands.
a <- b <- 3
print(a + b)

# quantile with scalar probability.
loss <- c(5, 3, 9, 1, 7, 2, 8, 4, 6, 10)
print(as.numeric(quantile(loss, probs = 0.9, type = 7)))

# Range with nested call in parenthesized endpoint.
n <- 4
print(0:(length(1:n) - 1))
