# test71_time_series_ts.R
x <- ts(1:24, start = c(2020, 1), frequency = 12)

print(x)
print(start(x))
print(end(x))
print(frequency(x))
print(window(x, start = c(2020, 6), end = c(2021, 3)))
print(lag(x, k = 1))
