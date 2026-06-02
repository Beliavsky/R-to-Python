# test57_cbind_rbind.R
x <- 1:3
y <- 10:12

print(cbind(x, y))
print(rbind(x, y))

df1 <- data.frame(a = 1:2, b = 3:4)
df2 <- data.frame(a = 5:6, b = 7:8)

print(rbind(df1, df2))
print(cbind(df1, c = c(9, 10)))
