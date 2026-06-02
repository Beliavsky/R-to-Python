# test72_autoregressive_fit_arima.R
set.seed(123)

n <- 200
x <- arima.sim(model = list(ar = c(0.7, -0.2)), n = n)

fit <- arima(x, order = c(2, 0, 0), include.mean = FALSE)

print(fit$coef)
print(fit$sigma2)
print(fit$aic)
