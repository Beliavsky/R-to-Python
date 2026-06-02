# test39_model_matrix.R
df <- data.frame(
  y = c(1, 2, 3, 4, 5, 6),
  x = c(10, 20, 30, 40, 50, 60),
  g = factor(c("a", "a", "b", "b", "c", "c"))
)

mm <- model.matrix(y ~ x + g, data = df)
print(mm)
