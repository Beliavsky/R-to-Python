# test107_reorder_columns_rows.R
df <- data.frame(
  id = c("c", "a", "b"),
  x = c(30, 10, 20),
  y = c(300, 100, 200)
)

df <- df[order(df$id), c("id", "y", "x")]

print(df)
