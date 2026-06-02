# test49_density_distribution_functions.R
x <- seq(-3, 3, length.out = 7)

print(dnorm(x))
print(pnorm(x))
print(qnorm(c(0.025, 0.5, 0.975)))

print(dt(x, df = 5))
print(pt(x, df = 5))
print(qt(c(0.025, 0.5, 0.975), df = 5))
