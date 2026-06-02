# test21_data_frame_basic.R
df <- data.frame(
  name = c("a", "b", "c"),
  x = c(1.0, 2.0, 3.0),
  y = c(10, 20, 30)
)

print(df)
print(names(df))
print(nrow(df))
print(ncol(df))
print(df$x)
print(df[["y"]])
print(df[1, ])
print(df[, "x"])
