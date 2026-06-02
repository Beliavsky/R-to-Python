# test81_partial_argument_matching.R
f <- function(alpha, beta, gamma) {
  return(alpha + 10 * beta + 100 * gamma)
}

print(f(al = 1, be = 2, ga = 3))
