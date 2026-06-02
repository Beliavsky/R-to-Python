# test04_named_vectors.R
v <- c(a = 10, b = 20, c = 30)

print(v)
print(names(v))
print(v["a"])
print(v[c("c", "a")])
v["b"] <- 99
print(v)

w <- c(d = 4, e = 5)
print(c(v, w))
