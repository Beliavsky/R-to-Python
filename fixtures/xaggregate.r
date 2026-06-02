# test33_aggregate.R
df <- data.frame(
  group = c("a", "a", "b", "b", "b", "c"),
  x = c(1, 2, 10, 20, 30, 100),
  y = c(5, 6, 7, 8, 9, 10)
)

print(aggregate(x ~ group, data = df, FUN = sum))
print(aggregate(cbind(x, y) ~ group, data = df, FUN = mean))
