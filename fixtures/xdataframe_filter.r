# test23_data_frame_filter.R
df <- data.frame(
  name = c("a", "b", "c", "d"),
  x = c(1, 5, 2, 8),
  y = c(10, 20, 30, 40)
)

print(df[df$x > 2, ])
print(df[df$y <= 30, c("name", "y")])
print(subset(df, x > 2))
print(subset(df, x > 2, select = c(name, x)))
