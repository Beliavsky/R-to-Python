# test69_unique_duplicated_match.R
x <- c("a", "b", "a", "c", "b", "d")

print(unique(x))
print(duplicated(x))
print(match(c("b", "d", "z"), x))
print("%in%")
print(c("a", "z") %in% x)
