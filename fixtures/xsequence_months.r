# test70_date_sequence_months.R
d <- seq(as.Date("2024-01-01"), as.Date("2024-06-01"), by = "month")

print(d)
print(format(d, "%Y-%m"))
