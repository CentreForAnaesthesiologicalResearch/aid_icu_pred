# setup -------------------------------------------------------------------

# read packages
library(tidymodels)


# helper function ---------------------------------------------------------
get_event_level <- function(truth, event_level = "first") {
  if (!is.factor(truth))
    stop("`truth` must be a factor")
  if (event_level == "first") {
    levels(truth)[1L]
  } else {
    levels(truth)[2L]
  }
}

# cox-snell pseudo r2 -----------------------------------------------

### Cox–Snell implementation
csr2_impl <- function(truth, estimate, case_weights = NULL) {
  eps <- 1e-14
  # Clamp probabilities
  estimate <- pmin(pmax(estimate, eps), 1 - eps)
  
  # Total weight / sample size
  N <- if (!is.null(case_weights)) sum(case_weights) else length(truth)
  
  # Log‐likelihood of model
  ll_mod <- if (!is.null(case_weights)) {
    sum(case_weights * (truth * log(estimate) +
                          (1 - truth) * log(1 - estimate)))
  } else {
    sum(truth * log(estimate) +
          (1 - truth) * log(1 - estimate))
  }
  
  # Null log‐likelihood (constant p0)
  p0 <- if (!is.null(case_weights)) {
    sum(case_weights * truth) / sum(case_weights)
  } else {
    mean(truth)
  }
  ll_null <- if (!is.null(case_weights)) {
    sum(case_weights * (truth * log(p0) +
                          (1 - truth) * log(1 - p0)))
  } else {
    sum(truth * log(p0) +
          (1 - truth) * log(1 - p0))
  }
  
  # cox–snell R2
  1 - exp((2 / N) * (ll_null - ll_mod))
}

csr2_vec <- function(truth, estimate,
                     estimator = NULL,
                     na_rm = TRUE,
                     case_weights = NULL,
                     event_level = "first", ...) {
  estimator <- finalize_estimator(truth, estimator, metric_class = "csr2")
  yardstick::check_prob_metric(truth, estimate, case_weights, estimator)
  
  if (na_rm) {
    res <- yardstick_remove_missing(truth, estimate, case_weights)
    truth <- res$truth; estimate <- res$estimate; case_weights <- res$case_weights
  } else if (yardstick_any_missing(truth, estimate, case_weights)) {
    return(NA_real_)
  }
  
  event <- get_event_level(truth, event_level)
  truth <- if_else(truth == event, 1, 0)
  
  csr2_impl(truth, estimate, case_weights)
}

# Generic and yardstick registration
csr2 <- function(data, ...) UseMethod("csr2")
csr2 <- new_prob_metric(csr2, direction = "maximize")

csr2.data.frame <- function(data,
                            truth,
                            ...,
                            na_rm       = TRUE,
                            case_weights= NULL,
                            event_level = "first") {
  prob_metric_summarizer(
    name          = "csr2",
    fn            = csr2_vec,
    data          = data,
    truth         = !!enquo(truth),
    na_rm         = na_rm,
    case_weights  = !!enquo(case_weights),
    event_level   = event_level,
    ...
  )
}
# observed / expected ratio -----------------------------------------------

oe_impl <- function(truth, estimate, case_weights = NULL) {
  if (!is.null(case_weights)) {
    sum(truth * case_weights) / sum(estimate * case_weights)
  } else {
    sum(truth) / sum(estimate)
  }
}

oe_vec <- function(truth,
                   estimate,
                   estimator = NULL,
                   na_rm = TRUE,
                   case_weights = NULL,
                   event_level = "first",
                   ...) {
  
  estimator <- finalize_estimator(truth, estimator, metric_class = "oe")
  
  yardstick::check_prob_metric(truth, estimate, case_weights, estimator)
  
  if (na_rm) {
    result <- yardstick::yardstick_remove_missing(truth, estimate, case_weights)
    truth <- result$truth
    estimate <- result$estimate
    case_weights <- result$case_weights
  } else if (yardstick::yardstick_any_missing(truth, estimate, case_weights)) {
    return(NA_real_)
  }
  
  event <- get_event_level(truth, event_level)
  
  truth <- if_else(truth == event, 1, 0)
  
  ratio <- oe_impl(truth, estimate, case_weights)
  
  ratio - 1
}

oe <- function(data, ...) {
  UseMethod("oe")
}

oe <- yardstick::new_prob_metric(oe, direction = "zero")

oe.data.frame <- function(data, truth, ..., na_rm = TRUE, case_weights = NULL, event_level = "first") {
  prob_metric_summarizer(
    name = "oe",
    fn = oe_vec,
    data = data,
    truth = !!enquo(truth),
    ...,
    na_rm = na_rm,
    case_weights = !!enquo(case_weights),
    event_level = event_level
  )
}


# calibration slope -------------------------------------------------------

cals_impl <- function(truth, estimate, case_weights = NULL) {
  eps <- 1e-14
  estimate <- pmin(pmax(estimate, eps), 1 - eps)
  
  logit_estimate <- log(estimate / (1 - estimate))
  
  if (!is.null(case_weights)) {
    data <- data.frame(truth = truth, logit_estimate = logit_estimate, weights = case_weights)
    fit <- glm(truth ~ logit_estimate, data = data, weights = weights, family = binomial)
  } else {
    data <- data.frame(truth = truth, logit_estimate = logit_estimate)
    fit <- glm(truth ~ logit_estimate, data = data, family = binomial)
  }
  
  coef(fit)["logit_estimate"]
}

cals_vec <- function(truth,
                     estimate,
                     estimator = NULL,
                     na_rm = TRUE,
                     case_weights = NULL,
                     event_level = "first",
                     ...) {
  
  estimator <- finalize_estimator(truth, estimator, metric_class = "cals")
  
  yardstick::check_prob_metric(truth, estimate, case_weights, estimator)
  
  if (na_rm) {
    result <- yardstick::yardstick_remove_missing(truth, estimate, case_weights)
    truth <- result$truth
    estimate <- result$estimate
    case_weights <- result$case_weights
  } else if (yardstick::yardstick_any_missing(truth, estimate, case_weights)) {
    return(NA_real_)
  }
  
  event <- get_event_level(truth, event_level)
  
  truth <- if_else(truth == event, 1, 0)
  
  cals_impl(truth, estimate, case_weights) - 1
}

cals <- function(data, ...) {
  UseMethod("cals")
}

cals <- yardstick::new_prob_metric(cals, direction = "zero")

cals.data.frame <- function(data, truth, ..., na_rm = TRUE, case_weights = NULL, event_level = "first") {
  yardstick::prob_metric_summarizer(
    name = "cals",
    fn = cals_vec,
    data = data,
    truth = !!enquo(truth),
    ...,
    na_rm = na_rm,
    case_weights = !!enquo(case_weights),
    event_level = event_level
  )
}

# calibration intercept ---------------------------------------------------
cali_impl <- function(truth, estimate, case_weights = NULL) {
  eps <- 1e-4
  estimate <- pmin(pmax(estimate, eps), 1 - eps)
  logit_est <- log(estimate / (1 - estimate))
  
  # Fit logistic with offset: logit(p) fixed, estimate only intercept
  if (!is.null(case_weights)) {
    df <- data.frame(truth = truth, logit_est = logit_est, w = case_weights)
    fit <- glm(truth ~ offset(logit_est),
               data   = df,
               weights= w,
               family = binomial)
  } else {
    df <- data.frame(truth = truth, logit_est = logit_est)
    fit <- glm(truth ~ offset(logit_est),
               data   = df,
               family = binomial)
  }
  
  coef(fit)[["(Intercept)"]]
}

cali_vec <- function(truth, estimate,
                     estimator = NULL,
                     na_rm = TRUE,
                     case_weights = NULL,
                     event_level = "first", ...) {
  estimator <- finalize_estimator(truth, estimator, metric_class = "cali")
  yardstick::check_prob_metric(truth, estimate, case_weights, estimator)
  
  if (na_rm) {
    res <- yardstick_remove_missing(truth, estimate, case_weights)
    truth <- res$truth; estimate <- res$estimate; case_weights <- res$case_weights
  } else if (yardstick_any_missing(truth, estimate, case_weights)) {
    return(NA_real_)
  }
  
  event <- get_event_level(truth, event_level)
  truth <- if_else(truth == event, 1, 0)
  
  cali_impl(truth, estimate, case_weights)
}

cali <- function(data, ...) UseMethod("cali")
cali <- new_prob_metric(cali, direction = "zero")

cali.data.frame <- function(data,
                            truth,
                            ...,
                            na_rm       = TRUE,
                            case_weights= NULL,
                            event_level = "first") {
  prob_metric_summarizer(
    name          = "cali",
    fn            = cali_vec,
    data          = data,
    truth         = !!enquo(truth),
    na_rm         = na_rm,
    case_weights  = !!enquo(case_weights),
    event_level   = event_level,
    ...
  )
}

# manual test -------------------------------------------------------------

## csR2
df_csr2 <- two_class_example %>% 
  mutate(
    truth01    = if_else(truth == "Class1", 1L, 0L),
    logit_pred = qlogis(Class1)
  )


csr2_val <- csr2_vec(
  truth = df_csr2$truth,
  estimate = df_csr2$Class1
)

#  Null log-likelihood
p0  <- mean(df_csr2$truth01)
ll0 <- sum(df_csr2$truth01*log(p0) + (1-df_csr2$truth01)*log(1-p0))

# 2) Model log-likelihood (original preds)
eps <- 1e-14
p1  <- pmin(pmax(df_csr2$Class1, eps), 1 - eps)
ll1 <- sum(df_csr2$truth01*log(p1) + (1-df_csr2$truth01)*log(1-p1))

N   <- nrow(df_csr2)
csr2_manual <- 1 - exp((2/N)*(ll0 - ll1))

near(csr2_val, csr2_manual, tol = 1e-14) # True to the 1e-14 tolerence limit

### OE
data(two_class_example)

two_class_example <- two_class_example |>
  mutate(new_truth = ifelse(truth == 'Class1', 1, 0)
         )

oe_manual_test <- two_class_example |>
  mutate(
    o = new_truth,
    e = Class1
  ) |>
  summarise(
    oe = sum(o)/sum(e)
  ) |>
  pull(oe)

oe_val <- oe_vec(
  truth = two_class_example$truth,
  estimate = two_class_example$Class1
) +1

near(oe_manual_test, oe_val, tol = 1e-14) # True to the 1e-14 tolerence limit

## CALS
cals_manual_test <- two_class_example |>
  mutate(
    logit_class = log(Class1/(1-Class1))
  ) %>%
  glm(new_truth ~ logit_class, data = ., family = binomial) |>
  tidy() |>
  filter(term == 'logit_class') |>
  pull(estimate)

cals_val <- cals_vec(
  truth = two_class_example$truth,
  estimate = two_class_example$Class1
) |>
 unname() %>%
  `+`(1)


cals_pmcal <- pmcalibration::logistic_cal(
    y = two_class_example$new_truth,
    p = two_class_example$Class1
)

near(cals_val, cals_manual_test, tol = 1e-14) # True to the 1e-14 tolerence limit
near(cals_val, cals_pmcal$calibration_slope$coefficients[[2]], tol = 1e-14) # True to the 1e-14 tolerence limit

## cali
cali_val <- cali_vec(
  truth = two_class_example$truth,
  estimate = two_class_example$Class1
)

cali_pmcal <- pmcalibration::logistic_cal(
  y = two_class_example$new_truth,
  p = two_class_example$Class1
)

near(cali_val, cali_pmcal$calibration_intercept[[1]], tol = 1e-5) # only equal to a tolerance level of 1e-5, which seems sufficient
