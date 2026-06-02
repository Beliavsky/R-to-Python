# test75_newton_method.R
newton <- function(f, fp, x0, niter) {
  x <- x0

  for (i in 1:niter) {
    x <- x - f(x) / fp(x)
  }

  return(x)
}

f <- function(x) {
  return(x^2 - 2)
}

fp <- function(x) {
  return(2 * x)
}

print(newton(f, fp, 1, 10))
