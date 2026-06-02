# test102_simple_gradient_descent.R
f <- function(x) {
  return((x - 3)^2)
}

grad <- function(x) {
  return(2 * (x - 3))
}

x <- 0
alpha <- 0.1

for (i in 1:50) {
  x <- x - alpha * grad(x)
}

print(x)
print(f(x))
