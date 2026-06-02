# test60_hist_cut_findInterval.R
x <- c(1, 2, 3, 4, 5, 10, 20)

print(cut(x, breaks = c(0, 3, 10, 100)))
print(table(cut(x, breaks = c(0, 3, 10, 100))))
print(findInterval(x, vec = c(0, 3, 10, 100)))
