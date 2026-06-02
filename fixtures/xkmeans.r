# test76_kmeans.R
set.seed(123)

x <- rbind(
  matrix(rnorm(50, mean = 0), ncol = 2),
  matrix(rnorm(50, mean = 5), ncol = 2)
)

fit <- kmeans(x, centers = 2)

print(fit$centers)
print(table(fit$cluster))
print(fit$withinss)
