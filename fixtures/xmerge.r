# test32_merge.R
df1 <- data.frame(id = c(1, 2, 3), x = c(10, 20, 30))
df2 <- data.frame(id = c(2, 3, 4), y = c(200, 300, 400))

print(merge(df1, df2, by = "id"))
print(merge(df1, df2, by = "id", all = TRUE))
print(merge(df1, df2, by = "id", all.x = TRUE))
print(merge(df1, df2, by = "id", all.y = TRUE))
