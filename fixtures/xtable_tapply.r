# test28_table_tapply.R
group <- c("a", "a", "b", "b", "b", "c")
x <- c(1, 2, 10, 20, 30, 100)

print(table(group))
print(tapply(x, group, sum))
print(tapply(x, group, mean))
print(tapply(x, group, length))
