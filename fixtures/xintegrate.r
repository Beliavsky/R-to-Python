# test48_integrate.R
f <- function(x) {
  return(exp(-x^2))
}

ans <- integrate(f, lower = 0, upper = 1)

print(ans$value)
print(ans$abs.error)
