# test35_random_numbers.R
set.seed(123)

x <- rnorm(10, mean = 0, sd = 1)
u <- runif(10, min = -1, max = 1)
z <- rbinom(10, size = 5, prob = 0.4)

print(x)
print(u)
print(z)
print(mean(x))
print(sd(x))
