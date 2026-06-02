# test59_sample.R
set.seed(123)

x <- 1:10

print(sample(x))
print(sample(x, size = 5))
print(sample(x, size = 20, replace = TRUE))
print(sample(c("a", "b", "c"), size = 10, replace = TRUE, prob = c(0.1, 0.2, 0.7)))
