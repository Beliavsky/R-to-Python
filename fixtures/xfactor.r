# test27_factor.R
x <- factor(c("low", "high", "medium", "low", "high"))

print(x)
print(levels(x))
print(as.integer(x))
print(table(x))

y <- factor(
  c("low", "high", "medium", "low"),
  levels = c("low", "medium", "high"),
  ordered = TRUE
)

print(y)
print(y > "low")
