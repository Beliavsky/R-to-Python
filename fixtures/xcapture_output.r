# test91_cat_sink_capture_output.R
x <- 1:5

cat("x =", x, "\n")
print(capture.output(summary(x)))
