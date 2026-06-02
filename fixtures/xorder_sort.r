# test24_order_sort.R
x <- c(4, 1, 3, 2, 5)

print(sort(x))
print(sort(x, decreasing = TRUE))
print(order(x))
print(rank(x))

df <- data.frame(name = c("a", "b", "c", "d"), x = c(4, 1, 4, 2))
print(df[order(df$x), ])
print(df[order(-df$x), ])
