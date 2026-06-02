# test46_optimization_optim.R
objective <- function(par) {
  x <- par[1]
  y <- par[2]
  z <- (x - 2)^2 + (y + 3)^2
  return(z)
}

ans <- optim(c(0, 0), objective)

print(ans$par)
print(ans$value)
