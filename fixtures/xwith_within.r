# test84_with_within.R
df <- data.frame(x = 1:5, y = 6:10)

print(with(df, x + y))

df2 <- within(df, {
  z <- x + y
  w <- z^2
})

print(df2)
