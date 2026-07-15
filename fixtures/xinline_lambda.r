# Inline anonymous functions, if/else expressions, and lambda shorthand.

# Inline function literal in sapply.
print(sapply(1:4, function(i) i^2))

# Braced single-expression body.
print(sapply(c(1, 4, 9), function(v) { sqrt(v) }))

# Braced multi-statement body with intermediate assignment.
print(sapply(c(2, 4), function(v) { m <- v * 10; m + 1 }))

# R 4.1 backslash lambda shorthand.
print(sapply(1:3, \(x) x + 100))

# FUN as a keyword argument.
print(outer(1:2, 1:3, FUN = function(i, j) 10 * i + j))
print(sapply(X = 1:3, FUN = function(v) v - 1))

# Function defined with an expression body using if/else.
sign_word <- function(x) if (x > 0) "positive" else "nonpositive"
print(sign_word(3))
print(sign_word(-2))

# if/else expression on assignment right-hand side.
z <- if (length(1:5) > 4) "long" else "short"
print(z)

# One-line if/else with subscript assignments in both branches.
pred <- numeric(3)
vals <- c(2, 0, 5)
for (i in 1:3) {
  if (vals[i] > 1) pred[i] <- 1 else pred[i] <- 0
}
print(pred)

# Brace followed by one-line else.
runs <- c(1, 2)
if (length(runs) > 1) {
  runs <- runs * 2
} else runs <- NULL
print(runs)

# sapply with the `[` extractor and strsplit.
full_names <- c("John Smith", "Mary Jones")
first_names <- sapply(strsplit(full_names, " "), `[`, 1)
print(first_names)
