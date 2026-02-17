# load libraries ----------------------------------------------------------
library(tidyverse)
library(broom)
library(tidymodels)
library(probably)
library(stacks)
library(janitor)
library(probably)
library(discrim)
library(pak)
library(here)
library(shapviz)
library(kernelshap)
library(mgcv)
source('r/custom_performance_metrics.R')


# read data ---------------------------------------------------------------
dfRaw <- read_csv2(file = 'data/aid_icu.csv')


# preproccess data --------------------------------------------------------
df <- dfRaw |>
  clean_names() |>
  mutate(limitation = case_when(
    limitations %in% c(0,3,4) ~ 'noLimitations',
    limitations %in% c(2,3) ~ 'limitations',
    T ~ NA,
    .ptype = factor(levels = c('noLimitations', 'limitations'))
  ),
  pre_barthel = as.numeric(pre_barthel),
  pre_cfs = as.numeric(pre_cfs),
  pre_cps = as.numeric(pre_cps),
  strata_site = factor(strata_site),
  strata_delir = factor(strata_delir),
  intervention = factor(intervention, levels = c('Placebo', 'Haloperidol')),
  sex = case_when(
    subgroup_female == T ~ 'female',
    T ~ 'male',
    .ptype = factor(levels = c('male', 'female'))
  ),
  smoking = case_when(
    smoking == T ~ 'smoker',
    T ~ 'non-smoker',
    .ptype = factor(levels = c('non-smoker', 'smoker'))
  ),
  alcohol_abuse = case_when(
    alcohol_abuse == T ~ 'yes',
    T ~ 'no',
    .ptype = factor(levels = c('no', 'yes'))
  ),
  advanced_cancer = case_when(
    hematologic_cancer == T | metastatic_cancer == T ~ 'yes',
    T ~ 'no',
    .ptype = factor(levels = c('no', 'yes'))
  ),
  surgery = case_when(
    elective_surg == T | emergency_surg == T ~ 'yes',
    T ~ 'no',
    .ptype = factor(levels = c('no', 'yes'))
  ),
  mort_1y = case_when(
    fu2_dead1year == T ~ 1,
    fu2_dead1year == F ~ 0,
    T ~ NA
  ),
  mort_1y = factor(mort_1y, levels = c(0,1)),
  mort_90d = case_when(
    prim_90mort == T ~ 1,
    prim_90mort == F ~ 0,
    T ~ NA),
  mort_90d = factor(mort_90d, levels = c(0,1)),
  sms = as.numeric(sms),
  icu_type = case_when(
    strata_site %in% c('DK14') ~ 'primary',
    strata_site %in% c('DK01', 'DK04') ~ 'tertiary',
    T ~ NA,
    .ptype = factor(levels = c('primary', 'tertiary'))
  ),
  barthel_category = case_when(
    pre_barthel >= 17 ~ 'BarthelHigh',
    pre_barthel <= 16 ~ 'BarthelLow',
    T ~ NA,
    .ptype = factor(levels = c('BarthelHigh','BarthelLow'))
  ),
  cfs_category = case_when(
    pre_cfs <= 4 ~ 'CFSLow',
    pre_cfs >= 5 ~ 'CFSHigh',
    T ~ NA,
    .ptype = factor(levels = c('CFSLow', 'CFSHigh'))
  ),
  cps_category = case_when(
    pre_cps <= 13 ~ 'CPSLow',
    pre_cps >= 14 ~ 'CPSHigh',
    T ~ NA,
    .ptype = factor(levels = c('CPSLow', 'CPSHigh'))
  ),
  sms_category = case_when(
    sms <= 25 ~ 'SMSlow',
    sms >= 26 ~ 'SMShigh',
    T ~ NA,
    .ptype = factor(levels = c('SMSlow', 'SMShigh'))
  )) |>
  # select columns used in the study
  filter(!is.na(mort_90d)) |>
  dplyr::select(mort_90d, sex, age, alcohol_abuse, strata_delir, advanced_cancer, surgery, sms, icu_type, pre_barthel, pre_cps, pre_cfs, limitation, intervention)


# Model workflow setup ----------------------------------------------------
### Set a seed
set.seed(sum(utf8ToInt('ThisIsASplitSeed')))

### Create folds
folds <- vfold_cv(df, v = 5)

### detect core for parallelization
cores <- parallel::detectCores()

## Set recpie --------------------------------------------------------------
basic_formula <- mort_90d ~ .

pred_90d_recpie <- recipe(basic_formula, data = df) |>
  # Update levels in the outcome
  step_relevel(mort_90d, ref_level = "1", skip = T) |>
  # Create rules to use multiple imputation
  step_impute_knn(all_predictors()) |>
  # normalize predictors
  step_range(all_numeric_predictors(), min = 0, max = 1) |>
  # Create dummy variables for factors
  step_dummy(all_factor_predictors(), one_hot = F) |>
  # remove highly correlated features
  step_corr(all_predictors(), threshold = 0.9) |>
  # Remove predictors with zero variance
  step_zv(all_predictors())


## Add to workflow
pred_90d_wflow <-
  workflow() |>
  add_recipe(pred_90d_recpie)

## bake df dev -------------------------------------------------------------
prep_90d_recpie <- prep(pred_90d_recpie)
baked_df <- bake(prep_90d_recpie, df)

write_rds(baked_df, 'results/90d/model_training/preprocessed_data/baked_df.rds')

## set model specifications ------------------------------------------------

### gam formula
gam_formula <- mort_90d ~ 
  s(age) +
  strata_delir_Hypo +
  s(sms) +
  icu_type_tertiary +
  s(pre_barthel) +
  s(pre_cps) +
  s(pre_cfs) +
  intervention_Haloperidol +
  limitation_limitations +
  alcohol_abuse_yes +
  advanced_cancer_yes +
  surgery_yes

### set specs
dt_spec <- 
  decision_tree(
    cost_complexity = tune(),
    tree_depth = tune(),
    min_n = tune()
  ) |>
  set_engine("rpart") |> 
  set_mode("classification")

log_spec <- 
  logistic_reg(
  ) |>
  set_engine("glm") |> 
  set_mode("classification")

en_spec <- 
  logistic_reg(
    penalty = tune(),
    mixture = tune()
  ) |>
  set_engine("glmnet") |> 
  set_mode("classification")

l1_spec <- 
  logistic_reg(
    penalty = tune(),
    mixture = 1
  ) |>
  set_engine("glmnet") |> 
  set_mode("classification")

rf_spec <- 
  rand_forest(
    mtry = tune(),
    trees = tune(),
    min_n = tune()
  ) |>
  set_engine("ranger",
             num.threads = cores) |> 
  set_mode("classification")

mlp_spec <- 
  mlp(
    hidden_units = tune(),
    penalty = tune(),
    epochs = tune(),
  ) |>
  set_engine("nnet") |> 
  set_mode("classification")

xgb_spec <-
  boost_tree(
    mtry = tune(),
    trees = tune(),
    min_n = tune(),
    tree_depth = tune(),
    learn_rate = tune(),
    loss_reduction = tune(),
    sample_size = tune(),
    stop_iter = tune()
  ) |>
  set_engine("xgboost") |> 
  set_mode("classification")

svm_spec <-
  svm_linear(
    cost = tune(),
    margin = NULL
  ) |>
  set_engine("kernlab") |> 
  set_mode("classification")

nb_spec <-
  naive_Bayes(
    smoothness = tune(),
    Laplace = tune()
  ) |>
  set_engine("klaR") |> 
  set_mode("classification")

gam_spec <-
  gen_additive_mod() |>
  set_engine('mgcv') |>
  set_mode('classification') |>
  set_args(adjust_deg_free = tune())

### set hyperparameter grid
spec_names <- ls(pattern = "_spec$")
spec_names <- spec_names[spec_names != 'log_spec']

create_grid <- function(spec_name, df) {
  prefix <- sub("_spec$", "", spec_name)
  spec <- get(spec_name, envir = .GlobalEnv)
  params <- extract_parameter_set_dials(spec)
  params <- finalize(params, df)
  grid <- grid_space_filling(params, size = 25, type = 'max_entropy')
  assign(paste0(prefix, "_grid"), grid, envir = .GlobalEnv)
  invisible(NULL)
}

walk(spec_names, create_grid, df = df)


## finalize model workflow setup -------------------------------------------

### set metrics
metrics <- metric_set(accuracy,
                      bal_accuracy,
                      pr_auc,
                      roc_auc,
                      brier_class,
                      csr2,
                      oe,
                      cals,
                      cali
)

### add workflows to workflow sets
all_workflows <- 
  workflow_set(
    preproc = list(standard = pred_90d_recpie),
    models = list(en = en_spec,
                  dt = dt_spec,
                  l1 = l1_spec,
                  log = log_spec,
                  svm = svm_spec,
                  rf = rf_spec,
                  mlp = mlp_spec,
                  xgb = xgb_spec,
                  nb = nb_spec,
                  gam = gam_spec
    )
  ) |>
  update_workflow_model(
    id = 'standard_gam',
    spec = gam_spec,
    formula = gam_formula
  ) |>
  option_add(grid = dt_grid,
             id = "standard_dt") |>
  option_add(grid = en_grid,
             id = "standard_en") |>
  option_add(grid = l1_grid,
             id = "standard_l1") |> 
  option_add(grid = svm_grid,
             id = "standard_svm") |>  
  option_add(grid = rf_grid,
             id = "standard_rf") |>
  option_add(grid = mlp_grid,
             id = "standard_mlp") |>
  option_add(grid = xgb_grid,
             id = "standard_xgb") |>
  option_add(grid = nb_grid,
             id = "standard_nb") |>
  option_add(grid = gam_grid,
             id = "standard_gam")

# fit workflowsets --------------------------------------------------------
ctrl_grid <-
  control_grid(
    save_pred = TRUE,
    parallel_over = 'everything',
    save_workflow = TRUE,
    event_level = 'first'
  )

tictoc::tic()

all_workflows <- 
  all_workflows %>% 
  workflow_map(
    seed = sum(utf8ToInt('ThisIsAModelTrainingSeed')),
    fn = "tune_grid",
    metrics = metrics,
    resamples = folds,
    verbose = TRUE,
    control = ctrl_grid
  )

tictoc::toc()

saveRDS(all_workflows, file = ('results/90d/model_training/model_wfs/all_workflows.rds'))

# Select final model ------------------------------------------------------
lookup <- c(
  standard_dt = 'Decision tree',
  standard_en = 'Elastic net',
  standard_gam = 'Generalized additive model',
  standard_l1 = 'LASSO regression',
  standard_log = 'Logistic regression',
  standard_mlp = 'Multilayer perceptron',
  standard_nb = 'Naive Bayes',
  standard_rf = 'Random forest',
  standard_svm = 'Support vector machine',
  standard_xgb = 'Extreme gradient boosting'
)


###  accuraccy rank
accuracy_rank <- all_workflows |>
  collect_metrics() |>
  filter(.metric == 'accuracy') |>
  arrange(desc(mean)) |>
  mutate(rank = row_number()) |>
  ggplot(
    aes(x = rank,
        y = mean,
        color = wflow_id,
        ymin = mean - 1.96 * std_err,
        ymax = mean + 1.96 * std_err)) +
  geom_pointrange(alpha = 0.5) +
  theme_light() +
  labs(
    title = "Acuraccy of models",
    x = "Rank",
    y = "Acuraccy",
    color = 'Model type'
  ) +
  scale_color_discrete(labels = function(lvls) lookup[lvls]) +
  theme(
    plot.title = element_text(hjust = 0.5)
  )

###  balanced accuraccy rank
bal_accuracy_rank <- all_workflows |>
  collect_metrics() |>
  filter(.metric == 'bal_accuracy') |>
  arrange(desc(mean)) |>
  mutate(rank = row_number()) |>
  ggplot(
    aes(x = rank,
        y = mean,
        color = wflow_id,
        ymin = mean - 1.96 * std_err,
        ymax = mean + 1.96 * std_err)) +
  geom_pointrange(alpha = 0.5) +
  theme_light()+
  labs(
    title = "Balanced accuracy of models",
    x = "Rank",
    y = "Balanced accuracy",
    color = 'Model type'
  ) +
  scale_color_discrete(labels = function(lvls) lookup[lvls]) +
  theme(
    plot.title = element_text(hjust = 0.5)
  )

###  roc rank
roc_rank <- all_workflows |>
  collect_metrics() |>
  filter(.metric == 'roc_auc') |>
  arrange(desc(mean)) |>
  mutate(rank = row_number()) |>
  ggplot(
    aes(x = rank,
        y = mean,
        color = wflow_id,
        ymin = mean - 1.96 * std_err,
        ymax = mean + 1.96 * std_err)) +
  geom_pointrange(alpha = 0.5) +
  theme_light() +
  labs(
    title = "AUROC of models",
    x = "Rank",
    y = "AUROC",
    color = 'Model type'
  ) +
  scale_color_discrete(labels = function(lvls) lookup[lvls]) +
  theme(
    plot.title = element_text(hjust = 0.5)
  )

###  prc rank
pr_rank <- all_workflows |>
  collect_metrics() |>
  filter(.metric == 'pr_auc') |>
  arrange(desc(mean)) |>
  mutate(rank = row_number()) |>
  ggplot(
    aes(x = rank,
        y = mean,
        color = wflow_id,
        ymin = mean - 1.96 * std_err,
        ymax = mean + 1.96 * std_err)) +
  geom_pointrange(alpha = 0.5) +
  theme_light() +
  labs(
    title = "AUPRC of models",
    x = "Rank",
    y = "AUPRC",
    color = 'Model type'
  ) +
  scale_color_discrete(labels = function(lvls) lookup[lvls]) +
  theme(
    plot.title = element_text(hjust = 0.5)
  )

### brier rank
brier_rank <- all_workflows |>
  collect_metrics() |>
  filter(.metric == 'brier_class') |>
  arrange(mean) |>
  mutate(rank = row_number()) |>
  ggplot(
    aes(x = rank,
        y = mean,
        color = wflow_id,
        ymin = mean - 1.96 * std_err,
        ymax = mean + 1.96 * std_err)) +
  geom_pointrange(alpha = 0.5) +
  theme_light() +
  labs(
    title = "Brier score of models",
    x = "Rank",
    y = "Brier score",
    color = 'Model type'
  ) +
  scale_color_discrete(labels = function(lvls) lookup[lvls]) +
  theme(
    plot.title = element_text(hjust = 0.5)
  )

### cs R2 rank
csr2_rank <- all_workflows |>
  collect_metrics() |>
  filter(.metric == 'csr2') |>
  filter(model == "logistic_reg") |>
  arrange(abs(1-mean)) |>
  mutate(rank = row_number()) |>
  ggplot(
    aes(x = rank,
        y = mean,
        color = wflow_id,
        ymin = mean - 1.96 * std_err,
        ymax = mean + 1.96 * std_err)) +
  geom_pointrange(alpha = 0.5) +
  theme_light() +
  labs(
    title = "Cox-Snell R2 of models",
    x = "Rank",
    y = "Cox-Snell R2",
    color = 'Model type'
  ) +
  scale_color_discrete(labels = function(lvls) lookup[lvls]) +
  theme(
    plot.title = element_text(hjust = 0.5)
  )

### oe rank
oe_rank <- all_workflows |>
  collect_metrics() |>
  filter(.metric == 'oe') |>
  mutate(oe = mean + 1) |>
  arrange(abs(oe-1)) |>
  mutate(rank = row_number()) |>
  ggplot(
    aes(x = rank,
        y = oe,
        color = wflow_id)
  ) +
  geom_point(alpha = 0.5) +
  theme_light() +
  labs(
    title = "O/E ratio of models",
    x = "Rank",
    y = "O/E ratio",
    color = 'Model type'
  ) +
  scale_color_discrete(labels = function(lvls) lookup[lvls]) +
  theme(
    plot.title = element_text(hjust = 0.5)
  )

### calibration slope rank
cals_rank <- all_workflows |>
  collect_metrics() |>
  filter(.metric == 'cals') |>
  mutate(cals = mean + 1) |>
  arrange(abs(1-cals)) |>
  mutate(rank = row_number()) |>
  ggplot(
    aes(x = rank,
        y = cals,
        color = wflow_id,
        ymin = cals - 1.96 * std_err,
        ymax = cals + 1.96 * std_err)) +
  geom_pointrange(alpha = 0.5) +
  theme_light()+
  labs(
    title = "Calibration slope of models",
    x = "Rank",
    y = "Calibration slope",
    color = 'Model type'
  ) +
  scale_color_discrete(labels = function(lvls) lookup[lvls]) +
  theme(
    plot.title = element_text(hjust = 0.5)
  )

cali_rank <- all_workflows |>
  collect_metrics() |>
  filter(.metric == 'cali') |>
  arrange((abs(0-mean))) |>
  mutate(rank = row_number()) |>
  ggplot(
    aes(x = rank,
        y = mean,
        color = wflow_id,
        ymin = mean - 1.96 * std_err,
        ymax = mean + 1.96 * std_err)) +
  geom_pointrange(alpha = 0.5) +
  theme_light()+
  labs(
    title = "Calibration intercept of models",
    x = "Rank",
    y = "Calibration intercept",
    color = 'Model type'
  ) +
  scale_color_discrete(labels = function(lvls) lookup[lvls]) +
  theme(
    plot.title = element_text(hjust = 0.5)
  )

model_selection_plots <- list(
  accuracy_rank = accuracy_rank,
  bal_accuracy_rank = bal_accuracy_rank,
  roc_rank = roc_rank,
  pr_rank = pr_rank,
  brier_rank = brier_rank,
  csr2_rank = csr2_rank,
  cals_rank = cals_rank,
  cali_rank = cali_rank,
  oe_rank = oe_rank
)

for (name in names(model_selection_plots)) {
  ggsave(
    filename = paste0("results/90d/model_training/model_selection_plots/", name, ".png"),
    plot = model_selection_plots[[name]],
    width = 6,
    height = 4,
    dpi = 600
  )
}

# bootstrap validation of selected model ----------------------------------

### Find best elastic net hyperparameters
best_en_parameters <- 
  all_workflows %>% 
  extract_workflow_set_result("standard_en") |>
  select_best(metric = "csr2")

### Set up bootstrapped resamples and retrained model hyperparameters
set.seed(sum(utf8ToInt('ThisIsASeedForValidation')))
boots <- bootstraps(df, times = 500, apparent = TRUE)

bootstrap_spec_en <- en_spec |>
  update(
    penalty = best_en_parameters$penalty,
    mixture = best_en_parameters$mixture
  )

bootstrap_en_wflow <- pred_90d_wflow |>
  add_model(bootstrap_spec_en)


## train bootstrapped ------------------------------------------------------
tictoc::tic()
bootstrap_metrics <-
  bootstrap_en_wflow |>
  fit_resamples(resamples = boots,
                metrics = metrics,
                control = control_resamples(save_pred = TRUE,
                                            event_level = 'first',
                                            extract = function (x) extract_fit_parsnip(x),
                                            parallel_over = 'resamples',
                                            save_workflow = T)
  )
tictoc::toc()

### extract final model
final_model <- bootstrap_metrics$.extracts[[1]]$.extracts[[1]]
final_wf_fit <- fit(bootstrap_en_wflow, data = df)
final_model_tidy <- bootstrap_metrics$.extracts[[1]]$.extracts[[1]] |> tidy()

saveRDS(final_model, file = 'results/90d/model_training/final_model/final_model.rds')
saveRDS(final_wf_fit, file = 'results/90d/model_training/final_model/final_wf_fit.rds')
saveRDS(final_model_tidy, file = 'results/90d/model_training/final_model/final_model_tidy.rds')

### extract performance metrics on models trained on bootstrapped resamples
my_metrics <- metrics  


bootstrapValMetrics <- boots |>
  transmute(
    id,
    analysis_data = map(splits, analysis),
    metrics = map(analysis_data, function(d) {
      
      pred_prob  <- predict(final_wf_fit, new_data = d, type = "prob")
      pred_class <- predict(final_wf_fit, new_data = d, type = "class")
      
      truth_fix <- factor(d$mort_90d, levels = levels(pred_class$.pred_class))
      
      event_class <- levels(truth_fix)[1]
      prob_col <- paste0(".pred_", event_class)
      
      res <- bind_cols(pred_prob, pred_class, truth = truth_fix)
      
      res$.pred_event <- res[[prob_col]]
      
      my_metrics(
        res,
        truth = truth,
        estimate = .pred_class,
        .pred_event,
        event_level = "first"
      )
    })
  ) |>
  select(id, metrics) |>
  unnest(metrics) |>
  dplyr::select(-c(.estimator)) |>
  pivot_wider(names_from = .metric, values_from = .estimate) |>
  # Re-add the 1 substracted during deffination for optimizations towards 0
  mutate(oe = oe + 1,
         cals = cals + 1)


### extract bootstrapped models
extracted_boots_models <- bootstrap_metrics |>
  filter(id != 'Apparent') |>
  dplyr::select(id, .extracts) |>
  unnest(cols = .extracts) |>
  mutate(model = map(.extracts, ~ .x)) |>
  dplyr::select(id, model)

saveRDS(extracted_boots_models, file = 'results/90d/model_training/bootstrapped_results/bootstrapped_models.rds')

### calculate performance of bootstrapped models on original data
bootstrap_predictions_original_df <- extracted_boots_models |>
  mutate(
    pred_prob = map(model, ~ predict(.x, new_data = baked_df, type = 'prob')),
    pred_class = map(model, ~ predict(.x, new_data = baked_df, type = 'class')),
    truth =list(baked_df$mort_90d)
  ) |>
  mutate(results = pmap(list(pred_prob, pred_class, truth), 
                        ~ bind_cols(..1, .pred_class = ..2, truth = ..3))) |>
  dplyr::select(id, model, results) |>
  mutate(metrics = map(results, ~ metrics(.x, truth = truth, event_level = 'first', estimate = .pred_class, .pred_1))) |>
  unnest(cols = metrics) |>
  dplyr::select(-c(model,results, .estimator)) |>
  pivot_wider(names_from = .metric, values_from = .estimate) |>
  filter(id != 'Apparent') |>
  # Re-add the 1 substracted during deffination for optimizations towards 0
  mutate(oe = oe + 1,
         cals = cals + 1)

saveRDS(bootstrap_predictions_original_df, file = 'results/90d/model_training/bootstrapped_results/bootstrap_predictions_original_df.rds')

### apparent performance
apparent_performance_tmp <- bootstrapValMetrics |>
  filter(id == 'Apparent')
alpha <- 0.05

apparent_ci <- bootstrapValMetrics |>
  dplyr::filter(id != "Apparent") |>
  dplyr::summarise(dplyr::across(
    where(is.numeric),
    list(
      lo = ~ quantile(.x, probs = alpha/2, na.rm = TRUE),
      hi = ~ quantile(.x, probs = 1-alpha/2,, na.rm = TRUE)
    ),
    .names = "{.col}_{.fn}"
  ))

reorder_by_suffix <- function(.data, codes = c("lo", "hi")) {
  nm <- names(.data)
  rx <- paste0("(_", paste(codes, collapse = "|_"), ")$")
  
  key <- tibble(name = nm) |>
    mutate(
      base = str_replace(name, rx, ""),
      suf  = str_extract(name, rx),
      suf  = replace_na(suf, "")
    ) |>
    mutate(suf = factor(suf, levels = c("", paste0("_", codes)))) |>
    arrange(base, suf) |>
    pull(name)
  
  select(.data, all_of(key))
}

combined_apparent <- bind_cols(apparent_performance_tmp, apparent_ci)

apparent_performance <- reorder_by_suffix(combined_apparent, codes = c("lo","hi")) |>
  select(id, accuracy, accuracy_lo, accuracy_hi, bal_accuracy, bal_accuracy_lo, bal_accuracy_hi, brier_class, brier_class_lo, brier_class_hi, cali, cali_lo, cali_hi, cals, cals_lo, cals_hi, csr2, csr2_lo, csr2_hi, oe, oe_lo, oe_hi, pr_auc, pr_auc_lo, pr_auc_hi, roc_auc, roc_auc_lo, roc_auc_hi)

saveRDS(apparent_performance, file = 'results/90d/model_training/final_performance/apparent_performance.rds')

### optimism adjusted performance
optimism_df <- bootstrapValMetrics |>
  filter(id != 'Apparent') |>
  inner_join(bootstrap_predictions_original_df, by = 'id', suffix = c('_boot', '_original')) |>
  mutate(
    accuracy_optimism = accuracy_boot - accuracy_original,
    roc_auc_optimism = roc_auc_boot - roc_auc_original,
    brier_class_optimism = brier_class_boot - brier_class_original,
    bal_accuracy_optimism = bal_accuracy_boot - bal_accuracy_original,
    pr_auc_optimism = pr_auc_boot - pr_auc_original,
    csr2_optimism = csr2_boot - csr2_original,
    oe_optimism = oe_boot - oe_original,
    cals_optimism = cals_boot - cals_original,
    cali_optimism = cali_boot - cali_original
  ) |>
  dplyr::select(id, accuracy_optimism,
                roc_auc_optimism,
                brier_class_optimism,
                bal_accuracy_optimism,
                pr_auc_optimism,
                csr2_optimism,
                oe_optimism,
                cals_optimism,
                cali_optimism)

optimism_adjused_performance <- apparent_performance |>
  mutate(
    id = 'Optimism adjusted',
    accuracy = accuracy - mean(optimism_df$accuracy_optimism),
    accuracy_lo = accuracy_lo - mean(optimism_df$accuracy_optimism),
    accuracy_hi = accuracy_hi - mean(optimism_df$accuracy_optimism), 
    roc_auc = roc_auc - mean(optimism_df$roc_auc_optimism),
    roc_auc_lo = roc_auc_lo - mean(optimism_df$roc_auc_optimism),
    roc_auc_hi = roc_auc_hi - mean(optimism_df$roc_auc_optimism),
    brier_class = brier_class - mean(optimism_df$brier_class_optimism),
    brier_class_lo = brier_class_lo - mean(optimism_df$brier_class_optimism),
    brier_class_hi = brier_class_hi - mean(optimism_df$brier_class_optimism),
    bal_accuracy = bal_accuracy - mean(optimism_df$bal_accuracy_optimism),
    bal_accuracy_lo = bal_accuracy_lo - mean(optimism_df$bal_accuracy_optimism),
    bal_accuracy_hi = bal_accuracy_hi - mean(optimism_df$bal_accuracy_optimism),
    pr_auc = pr_auc - mean(optimism_df$pr_auc_optimism),
    pr_auc_lo = pr_auc_lo - mean(optimism_df$pr_auc_optimism),
    pr_auc_hi = pr_auc_hi - mean(optimism_df$pr_auc_optimism),
    csr2 = csr2 - mean(optimism_df$csr2_optimism),
    csr2_lo = csr2_lo - mean(optimism_df$csr2_optimism),
    csr2_hi = csr2_hi - mean(optimism_df$csr2_optimism),
    oe = oe - mean(optimism_df$oe_optimism),
    oe_lo = oe_lo - mean(optimism_df$oe_optimism),
    oe_hi = oe_hi - mean(optimism_df$oe_optimism),
    cals = cals - mean(optimism_df$cals_optimism),
    cals_lo = cals_lo - mean(optimism_df$cals_optimism),
    cals_hi = cals_hi - mean(optimism_df$cals_optimism),
    cali = cali - mean(optimism_df$cali_optimism),
    cali_lo = cali_lo - mean(optimism_df$cali_optimism),
    cali_hi = cali_hi - mean(optimism_df$cali_optimism)
  )

saveRDS(optimism_adjused_performance, file = 'results/90d/model_training/final_performance/optimism_adjusted_performance.rds')

### Write model performance csv
model_performance <- rbind(apparent_performance, optimism_adjused_performance)

format_ci_table <- function(df, id_col = "id", digits = 3, drop_lo_hi = TRUE) {
  
  metric_cols <- names(df) |>
    setdiff(id_col) |>
    keep(~ !str_detect(.x, "_lo$|_hi$")) |>
    keep(~ paste0(.x, "_lo") %in% names(df) && paste0(.x, "_hi") %in% names(df))
  
  fmt <- paste0("%.", digits, "f (%.", digits, "f-%.", digits, "f)")
  
  for (m in metric_cols) {
    df[[m]] <- sprintf(
      fmt,
      as.numeric(df[[m]]),
      as.numeric(df[[paste0(m, "_lo")]]),
      as.numeric(df[[paste0(m, "_hi")]])
    )
  }
  
  if (drop_lo_hi) {
    df <- df |> select(-ends_with("_lo"), -ends_with("_hi"))
  }
  
  df
}

model_performance_export <- format_ci_table(model_performance, id_col = "id", digits = 3, drop_lo_hi = TRUE)

write_csv2(model_performance, file = 'results/90d/model_training/final_performance/combined_performance.csv')
write_csv2(model_performance_export, file = 'results/90d/model_training/final_performance/combined_performance_export.csv')


### produce optimism plots
accuracy_optimism_plot <- optimism_df |>
  ggplot(aes(accuracy_optimism)) +
  geom_histogram(
    binwidth = 0.01
  ) +
  geom_vline(xintercept = 0, color = 'red') +
  geom_vline(xintercept = mean(optimism_df$accuracy_optimism), color = 'blue') + 
  theme_light() +
  labs(
    title = "Accuracy optimism plot",
    x = "Estimated optimism of accuracy",
    y = "Count"
  ) +
  theme(
    plot.title = element_text(hjust = 0.5)
  )

bal_accuracy_optimism_plot <- optimism_df |>
  ggplot(aes(bal_accuracy_optimism)) +
  geom_histogram(
    binwidth = 0.01
  ) +
  geom_vline(xintercept = 0, color = 'red') +
  geom_vline(xintercept = mean(optimism_df$bal_accuracy_optimism), color = 'blue') + 
  theme_light() +
  labs(
    title = "Balanced accuracy optimism plot",
    x = "Estimated optimism of balanced accuracy",
    y = "Count"
  ) +
  theme(
    plot.title = element_text(hjust = 0.5)
  )

roc_optimism_plot <- optimism_df |>
  ggplot(aes(roc_auc_optimism)) +
  geom_histogram(
    binwidth = 0.01
  ) +
  geom_vline(xintercept = 0, color = 'red') +
  geom_vline(xintercept = mean(optimism_df$roc_auc_optimism), color = 'blue') + 
  theme_light() +
  labs(
    title = "AUROC optimism plot",
    x = "Estimated optimism of AUROC",
    y = "Count"
  ) +
  theme(
    plot.title = element_text(hjust = 0.5)
  )

pr_optimism_plot <- optimism_df |>
  ggplot(aes(pr_auc_optimism)) +
  geom_histogram(
    binwidth = 0.01
  ) +
  geom_vline(xintercept = 0, color = 'red') +
  geom_vline(xintercept = mean(optimism_df$pr_auc_optimism), color = 'blue') + 
  theme_light() +
  labs(
    title = "AUPRC optimism plot",
    x = "Estimated optimism of AUPRC",
    y = "Count"
  ) +
  theme(
    plot.title = element_text(hjust = 0.5)
  )

brier_optimism_plot <- optimism_df |>
  ggplot(aes(brier_class_optimism)) +
  geom_histogram(
    binwidth = 0.005
  ) +
  geom_vline(xintercept = 0, color = 'red') +
  geom_vline(xintercept = mean(optimism_df$brier_class_optimism), color = 'blue') + 
  theme_light() +
  labs(
    title = "Brier score optimism plot",
    x = "Estimated optimism of Brier scrore",
    y = "Count"
  ) +
  theme(
    plot.title = element_text(hjust = 0.5)
  )

csr2_optimism_plot <- optimism_df |>
  ggplot(aes(csr2_optimism)) +
  geom_histogram(
    binwidth = 0.01
  ) +
  geom_vline(xintercept = 0, color = 'red') +
  geom_vline(xintercept = mean(optimism_df$csr2_optimism), color = 'blue') + 
  theme_light() +
  labs(
    title = "Cox-Snell r2 optimism plot",
    x = "Estimated optimism of Cox-Snell r2",
    y = "Count"
  ) +
  theme(
    plot.title = element_text(hjust = 0.5)
  )

oe_optimism_plot <- optimism_df |>
  ggplot(aes(oe_optimism)) +
  geom_histogram(
    binwidth = 0.02
  ) +
  geom_vline(xintercept = 0, color = 'red') +
  geom_vline(xintercept = mean(optimism_df$oe_optimism), color = 'blue') + 
  theme_light() +
  labs(
    title = "O/E ratio optimism plot",
    x = "Estimated optimism of O/E ratio",
    y = "Count"
  ) +
  theme(
    plot.title = element_text(hjust = 0.5)
  )

cals_optimism_plot <- optimism_df |>
  ggplot(aes(cals_optimism)) +
  geom_histogram(
    binwidth = 0.025
  ) +
  geom_vline(xintercept = 0, color = 'red') +
  geom_vline(xintercept = mean(optimism_df$cals_optimism), color = 'blue') + 
  theme_light() +
  labs(
    title = "Calibration slope optimism plot",
    x = "Estimated optimism of calibration slope",
    y = "Count"
  ) +
  theme(
    plot.title = element_text(hjust = 0.5)
  )

cali_optimism_plot <- optimism_df |>
  ggplot(aes(cali_optimism)) +
  geom_histogram(
    binwidth = 0.025
  ) +
  geom_vline(xintercept = 0, color = 'red') +
  geom_vline(xintercept = mean(optimism_df$cali_optimism), color = 'blue') + 
  theme_light() +
  labs(
    title = "Calibration intercept optimism plot",
    x = "Estimated optimism of calibration intercept",
    y = "Count"
  ) +
  theme(
    plot.title = element_text(hjust = 0.5)
  )

list_optimism_plots <- list(
  accuracy_optimism_plot = accuracy_optimism_plot,
  bal_accuracy_optimism_plot = bal_accuracy_optimism_plot,
  roc_optimism_plot = roc_optimism_plot,
  pr_optimism_plot = pr_optimism_plot,
  csr2_optimism_plot = csr2_optimism_plot,
  brier_optimism_plot = brier_optimism_plot,
  oe_optimism_plot = oe_optimism_plot,
  cals_optimism_plot = cals_optimism_plot,
  cali_optimism_plot = cali_optimism_plot
)

for (name in names(list_optimism_plots)) {
  ggsave(
    filename = paste0("results/90d/model_training/final_performance/optimism_plots/", name, ".png"),
    plot = list_optimism_plots[[name]],
    width = 6,
    height = 4,
    dpi = 600
  )
}


# create calibration curves -----------------------------------------------
bootstrap_resamples <-
  bootstrap_en_wflow |>
  fit_resamples(
    resamples = boots,
    metrics   = metrics,
    control   = control_resamples(
      save_pred = TRUE,
      extract   = function(x) x,
      parallel_over = "resamples",
      save_workflow = TRUE
    )
  )

boot_wflows <- bootstrap_resamples |>
  filter(id != "Apparent") |>
  select(id, .extracts) |>
  unnest(.extracts) |>
  rename(wflow_fit = .extracts) |>
  inner_join(boots |> select(id, splits), by = "id")

event_class <- "1"  

cal_bootstrapped <- boot_wflows |>
  transmute(
    id,
    analysis_data = map(splits, analysis),
    preds = map2(wflow_fit, analysis_data, function(wf, dat) {
      
      p_prob  <- predict(wf, new_data = dat, type = "prob")
      p_class <- predict(wf, new_data = dat, type = "class")
      
      truth_num <- as.integer(as.character(dat$mort_90d))
      
      prob_col <- paste0(".pred_", event_class)
      if (!prob_col %in% names(p_prob)) {
        stop("Probability column ", prob_col, " not found. Available: ",
             paste(names(p_prob), collapse = ", "))
      }
      
      out <- bind_cols(
        p_prob,
        p_class,
        mort_90d = truth_num
      )
      
      out$.pred_event <- out[[prob_col]]
      
      out
    })
  ) |>
  select(id, preds) |>
  unnest(preds) |>
  mutate(group = "bootstrapped_analysis") |>
  select(id, group, .pred_event, mort_90d)

cal_apparent <- bootstrap_metrics |>
  select(id, .predictions) |>
  unnest(.predictions) |>
  filter(id == "Apparent") |>
  transmute(
    id,
    group = "apparent",
    .pred_1,
    mort_90d = as.numeric(as.character(mort_90d))
  )

cal_plot <- ggplot() +
  geom_smooth(
    data   = cal_bootstrapped,
    aes(x = .pred_event, y = mort_90d, group = id),
    method      = "gam",
    formula     = y ~ s(x, bs = "cs", k = 5),
    method.args = list(family = gaussian),
    se     = FALSE,
    color  = "grey",
    size   = 0.1,
    alpha  = 0.3
  ) +
  geom_smooth(
    data        = cal_apparent,
    aes(x = .pred_1, y = mort_90d, group = 1),
    method      = "gam",
    formula     = y ~ s(x, bs = "cs", k = 5),
    method.args = list(family = gaussian),
    se          = TRUE,        
    level       = 0.95,        
    color       = "red",
    fill        = "red",
    size        = 1,
    alpha       = 0.3
  ) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "black") +
  labs(x = "Predicted Probability", 
       y = "Observed Proportion",
       title = "Calibration Curves: 'Apparent' vs. 'bootstrapped'") +
  theme_light() +
  theme(legend.position = "none")

ggsave(
  filename = paste0("results/90d/model_training/final_performance/calibration_plot.png"),
  plot = cal_plot,
  width = 6,
  height = 4,
  dpi = 600
)

# explainability ----------------------------------------------------------
apparent_fit <- bootstrap_metrics %>% 
  filter(id == "Apparent") %>%
  pull(.extracts) %>%
  .[[1]] 

en_model <- apparent_fit$.extracts[[1]]

X_pred_matrix <- baked_df |>
  dplyr::select(-c(mort_90d)) |>
  data.matrix()

xvars <- baked_df |>
  dplyr::select(-c(mort_90d)) |>
  names()

tictoc::tic()

ks <- kernelshap(
  en_model,
  type = 'prob',
  X = X_pred_matrix, 
  bg_X = X_pred_matrix
)

sv <- shapviz(ks) %>%
  .[[1]]

tictoc::toc()

write_rds(ks, 'results/90d/model_training/shap_values/kernelshap/ks.rds')
write_rds(sv, 'results/90d/model_training/shap_values/shap_values/sv.rds')


# create shap plots -------------------------------------------------------
shap_importance <- sv_importance(sv) +
  ggplot2::xlab("Mean absolute SHAP value") +
  scale_y_discrete(labels = c(
    'age' = 'Age',
    'pre_cfs' = 'Clinical Frailty Scale',
    'surgery_yes' = 'Surgical admission',
    'sms' = 'Simplified Mortality Score',
    'advanced_cancer_yes' = 'Advanced cancer',
    'limitation_limitations' = 'Limitation of escalation of care',
    'intervention_Haloperidol' = 'Allocation to haloperidol',
    'pre_cps' = 'Comorbidity–polypharmacy score',
    'alcohol_abuse_yes' = 'Alcohol abuse',
    'sex_female' = 'Female',
    'pre_barthel' = 'Barthel index',
    'icu_type_tertiary' = 'Teritary intensive care unit',
    'strata_delir_Hypo' = 'Hypoactive delirium'
  ))
shap_bee_importance_plot <- sv_importance(sv, kind = "bee", show_numbers = TRUE, alpha = 0.3)  +
  scale_y_discrete(labels = c(
    'age' = 'Age',
    'pre_cfs' = 'Clinical Frailty Scale',
    'surgery_yes' = 'Surgical admission',
    'sms' = 'Simplified Mortality Score',
    'advanced_cancer_yes' = 'Advanced cancer',
    'limitation_limitations' = 'Limitation of escalation of care',
    'intervention_Haloperidol' = 'Allocation to haloperidol',
    'pre_cps' = 'Comorbidity–polypharmacy score',
    'alcohol_abuse_yes' = 'Alcohol abuse',
    'sex_female' = 'Female',
    'pre_barthel' = 'Barthel index',
    'icu_type_tertiary' = 'Teritary intensive care unit',
    'strata_delir_Hypo' = 'Hypoactive delirium'
  ))
shap_dependence_plot <- sv_dependence(sv, v = xvars)

ggsave(
  filename = paste0("results/90d/model_training/shap_values/shap_plots/shap_importance.png"),
  plot = shap_importance,
  width = 6,
  height = 4,
  dpi = 600
)

ggsave(
  filename = paste0("results/90d/model_training/shap_values/shap_plots/shap_bee_importance_plot.png"),
  plot = shap_bee_importance_plot,
  width = 8,
  height = 4,
  dpi = 600
)

ggsave(
  filename = paste0("results/90d/model_training/shap_values/shap_plots/shap_dependence_plot.png"),
  plot = shap_dependence_plot,
  width = 15,
  height = 10,
  dpi = 600
)

# retrieve scaling values  -------------------------------------------

scaling_values <- tidy(prep_90d_recpie, number = 3)

write_rds(scaling_values, 'results/90d/model_training/final_model/scaling_values.rds')

# subgroup analyses -------------------------------------------------------
subgroup_df <- baked_df |>
  mutate(
    barthel_category = case_when(
      df$pre_barthel >= 17 ~ 'BarthelHigh',
      df$pre_barthel <= 16 ~ 'BarthelLow',
      T ~ NA,
      .ptype = factor(levels = c('BarthelHigh','BarthelLow'))
    ),
    cfs_category = case_when(
      df$pre_cfs <= 4 ~ 'CFSLow',
      df$pre_cfs >= 5 ~ 'CFSHigh',
      T ~ NA,
      .ptype = factor(levels = c('CFSLow', 'CFSHigh'))
    ),
    cps_category = case_when(
      df$pre_cps <= 13 ~ 'CPSLow',
      df$pre_cps >= 14 ~ 'CPSHigh',
      T ~ NA,
      .ptype = factor(levels = c('CPSLow', 'CPSHigh'))
    ),
    sms_category = case_when(
      df$sms <= 25 ~ 'SMSlow',
      df$sms >= 26 ~ 'SMShigh',
      T ~ NA,
      .ptype = factor(levels = c('SMSlow', 'SMShigh'))
    )
  )

subgroup_vars <- c("barthel_category", "cfs_category", 'cps_category', "sms_category")

subgroup_counts <- purrr::map_dfr(subgroup_vars, function(var) {
  subgroup_df |>
    mutate(subgroup_level = as.character(.data[[var]])) |>
    count(subgroup_level, name = "n") |>
    mutate(subgroup_var = var)
})

apparent_predictions_subgroups_df <- tibble(
  model = list(final_model),
  pred_prob = list(predict(final_model, new_data = subgroup_df, type = "prob")),
  pred_class = list(predict(final_model, new_data = subgroup_df, type = "class")),
  truth = list(subgroup_df$mort_90d),
  subgroups = list(dplyr::select(subgroup_df, all_of(subgroup_vars)))
) |>
  mutate(
    results = pmap(list(pred_prob, pred_class, truth, subgroups),
                   ~ bind_cols(..1, .pred_class = ..2, truth = ..3, ..4))
  )

apparent_subgroup_results_long <- purrr::map_dfr(subgroup_vars, function(var) {
  apparent_predictions_subgroups_df |>
    mutate(
      metrics = map(results, ~ {
        .x |>
          mutate("{var}" := as.character(.data[[var]])) |>  
          group_by(.data[[var]]) |>  
          metrics(
            truth = truth,
            estimate = .pred_class,
            .pred_1,
            event_level = "first"
          ) |>
          ungroup() |>
          mutate(across(.estimate, as.numeric)) |>
          mutate(across(where(is.factor), as.character)) |>
          tibble::as_tibble()
      }),
      subgroup_var = var
    ) |>
    unnest(cols = metrics) |>
    mutate(across(where(is.factor), as.character))
}) |>
  dplyr::select(-c(model, pred_prob, pred_class, truth, subgroups, results))


B <- 500

boots_sub <- rsample::bootstraps(subgroup_df, times = B)

boot_subgroup_metrics_long <- purrr::map_dfr(subgroup_vars, function(var) {
  
  purrr::map_dfr(boots_sub$splits, function(spl) {
    set.seed(1)
    d_boot <- rsample::analysis(spl)
    
    pred_prob  <- predict(final_model, new_data = d_boot, type = "prob")
    pred_class <- predict(final_model, new_data = d_boot, type = "class")
    
    res <- bind_cols(
      pred_prob,
      .pred_class = pred_class,
      truth = d_boot$mort_90d,
      d_boot |> dplyr::select(all_of(subgroup_vars))
    ) |>
      mutate("{var}" := as.character(.data[[var]])) |>
      group_by(.data[[var]]) |>
      metrics(
        truth = truth,
        estimate = .pred_class,
        .pred_1,
        event_level = "first"
      ) |>
      ungroup() |>
      mutate(subgroup_var = var) |>
      rename(subgroup_level = !!var)
    
    res
  })
  
})

boot_subgroup_ci_long <- boot_subgroup_metrics_long |>
  group_by(subgroup_var, subgroup_level, .metric) |>
  summarise(
    estimate_lo = quantile(.estimate, probs = alpha/2, na.rm = TRUE),
    estimate_hi = quantile(.estimate, probs = 1 - alpha/2, na.rm = TRUE),
    .groups = "drop"
  )

apparent_subgroup_point_long <- apparent_subgroup_results_long |>
  mutate(
    subgroup_level = case_when(
      subgroup_var == "barthel_category" ~ as.character(barthel_category),
      subgroup_var == "cfs_category"     ~ as.character(cfs_category),
      subgroup_var == "cps_category"     ~ as.character(cps_category),
      subgroup_var == "sms_category"     ~ as.character(sms_category),
      T ~ NA
    )
  ) |>
  select(subgroup_var, subgroup_level, .metric, .estimate)

apparent_subgroup_with_ci_long <- apparent_subgroup_point_long |>
  left_join(boot_subgroup_ci_long,
            by = c("subgroup_var", "subgroup_level", ".metric"))

apparent_subgroup_with_ci_wide <- apparent_subgroup_with_ci_long |>
  rename(estimate = .estimate) |>
  pivot_longer(
    cols = c(estimate, estimate_lo, estimate_hi),
    names_to = "which",
    values_to = "value"
  ) |>
  mutate(
    suffix = case_when(
      which == "estimate"    ~ "",
      which == "estimate_lo" ~ "_lo",
      which == "estimate_hi" ~ "_hi"
    ),
    out_name = paste0(.metric, suffix)
  ) |>
  select(-which, -suffix, -.metric) |>
  pivot_wider(
    names_from = out_name,
    values_from = value
  ) |>
  mutate(
    oe   = oe + 1,
    oe_lo = oe_lo + 1,
    oe_hi = oe_hi + 1,
    cals   = cals + 1,
    cals_lo = cals_lo + 1,
    cals_hi = cals_hi + 1
  )

estimator_cols <- paste(names(apparent_subgroup_with_ci_wide |>
                                select(-c(subgroup_level, subgroup_var))))

subgroup_results <- apparent_subgroup_with_ci_wide |>
  left_join(subgroup_counts, by = c("subgroup_var", "subgroup_level")) |>
  rename_with(~ paste0(.x, "_apparent"), all_of(estimator_cols)) |>
  mutate(
    accuracy_optimism_adjusted = accuracy_apparent - mean(optimism_df$accuracy_optimism),
    accuracy_lo_optimism_adjusted = accuracy_lo_apparent - mean(optimism_df$accuracy_optimism),
    accuracy_hi_optimism_adjusted = accuracy_hi_apparent - mean(optimism_df$accuracy_optimism),
    
    roc_auc_optimism_adjusted = roc_auc_apparent - mean(optimism_df$roc_auc_optimism),
    roc_auc_lo_optimism_adjusted = roc_auc_lo_apparent - mean(optimism_df$roc_auc_optimism),
    roc_auc_hi_optimism_adjusted = roc_auc_hi_apparent - mean(optimism_df$roc_auc_optimism),
    
    
    brier_class_optimism_adjusted = brier_class_apparent - mean(optimism_df$brier_class_optimism),
    brier_class_lo_optimism_adjusted = brier_class_lo_apparent - mean(optimism_df$brier_class_optimism),
    brier_class_hi_optimism_adjusted = brier_class_hi_apparent - mean(optimism_df$brier_class_optimism),
    
    bal_accuracy_optimism_adjusted = bal_accuracy_apparent - mean(optimism_df$bal_accuracy_optimism),
    bal_accuracy_lo_optimism_adjusted = bal_accuracy_lo_apparent - mean(optimism_df$bal_accuracy_optimism),
    bal_accuracy_hi_optimism_adjusted = bal_accuracy_hi_apparent - mean(optimism_df$bal_accuracy_optimism),
    
    
    pr_auc_optimism_adjusted = pr_auc_apparent - mean(optimism_df$pr_auc_optimism),
    pr_auc_lo_optimism_adjusted = pr_auc_lo_apparent - mean(optimism_df$pr_auc_optimism),
    pr_auc_hi_optimism_adjusted = pr_auc_hi_apparent - mean(optimism_df$pr_auc_optimism),
    
    csr2_optimism_adjusted = csr2_apparent - mean(optimism_df$csr2_optimism),
    csr2_lo_optimism_adjusted = csr2_lo_apparent - mean(optimism_df$csr2_optimism),
    csr2_hi_optimism_adjusted = csr2_hi_apparent - mean(optimism_df$csr2_optimism),
    
    oe_optimism_adjusted = oe_apparent - mean(optimism_df$oe_optimism),
    oe_lo_optimism_adjusted = oe_lo_apparent - mean(optimism_df$oe_optimism),
    oe_hi_optimism_adjusted = oe_hi_apparent - mean(optimism_df$oe_optimism),
    
    cals_optimism_adjusted = cals_apparent - mean(optimism_df$cals_optimism),
    cals_lo_optimism_adjusted = cals_lo_apparent - mean(optimism_df$cals_optimism),
    cals_hi_optimism_adjusted = cals_hi_apparent - mean(optimism_df$cals_optimism),
    
    cali_optimism_adjusted = cali_apparent - mean(optimism_df$cali_optimism),
    cali_lo_optimism_adjusted = cali_lo_apparent - mean(optimism_df$cali_optimism),
    cali_hi_optimism_adjusted = cali_hi_apparent - mean(optimism_df$cali_optimism)
  ) |>
  relocate(n, .after = subgroup_level)

make_ci_strings <- function(df,
                            sets = c("apparent", "optimism_adjusted"),
                            digits = 3,
                            id_cols = c("subgroup_var", "subgroup_level"),
                            out_suffix = "fmt",
                            drop_original = TRUE) {
  
  out_cols <- character()
  
  for (s in sets) {
    est_cols <- names(df) |> str_subset(paste0("_", s, "$"))
    
    metrics <- est_cols |>
      str_remove(paste0("_", s, "$")) |>
      keep(~ paste0(.x, "_lo_", s) %in% names(df) && paste0(.x, "_hi_", s) %in% names(df))
    
    for (m in metrics) {
      est <- paste0(m, "_", s)
      lo  <- paste0(m, "_lo_", s)
      hi  <- paste0(m, "_hi_", s)
      
      out <- paste0(m, "_", s, "_", out_suffix)
      out_cols <- c(out_cols, out)
      
      fmt <- paste0("%.", digits, "f (%.", digits, "f-%.", digits, "f)")
      df[[out]] <- sprintf(
        fmt,
        as.numeric(df[[est]]),
        as.numeric(df[[lo]]),
        as.numeric(df[[hi]])
      )
    }
  }
  
  if (!drop_original) return(df)
  
  keep_cols <- c(id_cols, out_cols)
  keep_cols <- keep_cols[keep_cols %in% names(df)]
  
  df |> select(all_of(keep_cols))
}

subgroup_results_export <- make_ci_strings(subgroup_results,sets = c("apparent", "optimism_adjusted"),
                                           digits = 3,
                                           id_cols = c("subgroup_var", "subgroup_level"),
                                           out_suffix = "pretty",
                                           drop_original = TRUE
)

write.csv2(subgroup_results, file = 'results/90d/model_training/final_performance/combined_subgroup_analyses.csv')
write.csv2(subgroup_results_export, file = 'results/90d/model_training/final_performance/combined_subgroup_analyses_export.csv')

