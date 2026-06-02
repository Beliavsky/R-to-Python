# test22_data_frame_modify.R
df <- data.frame(
  x = c(1, 2, 3, 4),
  y = c(10, 20, 30, 40)
)

df$z <- df$x + df$y
df$x2 <- df$x^2
df$flag <- df$z > 25

print(df)

df$y[df$flag] <- 999
print(df)
