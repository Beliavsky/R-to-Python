# test85_reshape_stack_unstack.R
df <- data.frame(
  id = 1:3,
  a = c(10, 20, 30),
  b = c(100, 200, 300)
)

s <- stack(df[, c("a", "b")])
print(s)

u <- unstack(s)
print(u)
