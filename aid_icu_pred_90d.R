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
    T ~ 0
  ),
  mort_1y = factor(mort_1y, levels = c(0,1)),
  mort_90d = case_when(
    prim_90mort == T ~ 1,
    T ~ 0),
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
  )
  ) |>
  # select columns used in the study
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

pred_1ym_recpie <- recipe(basic_formula, data = df) |>
  # Update levels in the outcome
  step_relevel(mort_90d, ref_level = "1") |>
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
pred_1ym_wflow <-
  workflow() |>
  add_recipe(pred_1ym_recpie)

## bake df dev -------------------------------------------------------------
prep_1ym_recpie <- prep(pred_1ym_recpie)
baked_df <- bake(prep_1ym_recpie, df)

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
    preproc = list(standard = pred_1ym_recpie),
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

bootstrap_en_wflow <- pred_1ym_wflow |>
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

final_model_tidy <- bootstrap_metrics$.extracts[[1]]$.extracts[[1]] |> tidy()

saveRDS(final_model, file = 'results/90d/model_training/final_model/final_model.rds')

saveRDS(final_model_tidy, file = 'results/90d/model_training/final_model/final_model_tidy.rds')

### extract performance metrics on models trained on bootstrapped resamples
bootstrapValMetrics <- bootstrap_metrics |>
  dplyr::select(id, .metrics) |>
  unnest(cols = .metrics) |>
  dplyr::select(-c(.estimator, .config)) |>
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
apparent_performance <- bootstrapValMetrics |>
  filter(id == 'Apparent')

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
    roc_auc = roc_auc - mean(optimism_df$roc_auc_optimism),
    brier_class = brier_class - mean(optimism_df$brier_class_optimism),
    bal_accuracy = bal_accuracy - mean(optimism_df$bal_accuracy_optimism),
    pr_auc = pr_auc - mean(optimism_df$pr_auc_optimism),
    csr2 = csr2 - mean(optimism_df$csr2_optimism),
    oe = oe - mean(optimism_df$oe_optimism),
    cals = cals - mean(optimism_df$cals_optimism),
    cali = cali - mean(optimism_df$cali_optimism)
  )

saveRDS(optimism_adjused_performance, file = 'results/90d/model_training/final_performance/optimism_adjusted_performance.rds')

### Write model performance csv
model_performance <- rbind(apparent_performance, optimism_adjused_performance)

write_csv2(model_performance, file = 'results/90d/model_training/final_performance/combined_performance.csv')

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
    title = "Calibration interceptt optimism plot",
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
calibration_df <- bootstrap_metrics |>
  dplyr::select(id, .predictions) |>
  unnest(.predictions) |>
  dplyr::select(id, .pred_1, mort_90d) |>
  mutate(group = ifelse(id == "Apparent", "apparent", "bootstrapped"),
         mort_90d = case_when(
           mort_90d == 0 ~ 0,
           mort_90d == 1 ~ 1,
           .ptype = numeric()
         )) 

cal_bootstrapped <- subset(calibration_df, group == "bootstrapped")
cal_apparent <- subset(calibration_df, group == "apparent")

cal_plot <- ggplot() +
  geom_smooth(
    data   = cal_bootstrapped,
    aes(x = .pred_1, y = mort_90d, group = id),
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

scaling_values <- tidy(prep_1ym_recpie, number = 3)

write_rds(scaling_values, 'results/90d/model_training/final_model/scaling_values.rds')

# subgroup analyses -------------------------------------------------------
subgroup_df <- baked_df |>
  mutate(
    barthel_category = case_when(
      dfRaw$Pre_Barthel >= 17 ~ 'BarthelHigh',
      dfRaw$Pre_Barthel <= 16 ~ 'BarthelLow',
      T ~ NA,
      .ptype = factor(levels = c('BarthelHigh','BarthelLow'))
    ),
    cfs_category = case_when(
      dfRaw$Pre_CFS <= 4 ~ 'CFSLow',
      dfRaw$Pre_CFS >= 5 ~ 'CFSHigh',
      T ~ NA,
      .ptype = factor(levels = c('CFSLow', 'CFSHigh'))
    ),
    cps_category = case_when(
      dfRaw$Pre_CPS <= 13 ~ 'CPSLow',
      dfRaw$Pre_CPS >= 14 ~ 'CPSHigh',
      T ~ NA,
      .ptype = factor(levels = c('CPSLow', 'CPSHigh'))
    ),
    sms_category = case_when(
      dfRaw$sms <= 25 ~ 'SMSlow',
      dfRaw$sms >= 26 ~ 'SMShigh',
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

bootstrap_predictions_subgroups_df <- extracted_boots_models |>
  mutate(
    pred_prob = map(model, ~ predict(.x, new_data = subgroup_df, type = 'prob')),
    pred_class = map(model, ~ predict(.x, new_data = subgroup_df, type = 'class')),
    truth = list(subgroup_df$mort_90d),
    subgroups = list(subgroup_df |> dplyr::select(all_of(subgroup_vars)))
  ) |>
  mutate(
    results = pmap(list(pred_prob, pred_class, truth, subgroups), 
                   ~ bind_cols(..1, .pred_class = ..2, truth = ..3, ..4))
  )

subgroup_results_long <- purrr::map_dfr(subgroup_vars, function(var) {
  bootstrap_predictions_subgroups_df |>
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

subgroup_results <- subgroup_results_long |>
  mutate(subgroup_level = case_when(
    !is.na(barthel_category) ~ barthel_category, 
    !is.na(cfs_category) ~ cfs_category, 
    !is.na(cps_category) ~ cps_category, 
    !is.na(sms_category) ~ sms_category,
    T ~ NA
  )
  ) |>
  dplyr::select(-c(barthel_category, cfs_category, cps_category, sms_category)) |>
  pivot_wider(
    names_from = .metric,
    values_from = .estimate
  )  |>
  dplyr::select(-c(`.estimator`))

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

apparent_subgroup_results <- apparent_subgroup_results_long |>
  mutate(subgroup_level = case_when(
    !is.na(barthel_category) ~ barthel_category, 
    !is.na(cfs_category) ~ cfs_category, 
    !is.na(cps_category) ~ cps_category, 
    !is.na(sms_category) ~ sms_category,
    T ~ NA
  )
  ) |>
  dplyr::select(-c(barthel_category, cfs_category, cps_category, sms_category)) |>
  pivot_wider(
    names_from = .metric,
    values_from = .estimate
  ) |>
  dplyr::select(-c(`.estimator`)) |>
  mutate(performance = 'apparent')

common_keys <- c("subgroup_var", "subgroup_level")

subgroup_boot_means <- subgroup_results |>
  group_by(across(all_of(common_keys))) |>
  summarise(across(where(is.numeric), ~ mean(.x, na.rm = TRUE)), .groups = "drop") |>
  left_join(subgroup_counts, by = c("subgroup_var", "subgroup_level"))

subgroup_joined <- apparent_subgroup_results |>
  left_join(subgroup_boot_means, by = common_keys, suffix = c("_apparent", "_bootstrap"))

performance_subgroup_combined <- subgroup_joined |>
  mutate(across(
    ends_with("_apparent"),
    ~ .x - (get(sub("\\_apparent$", "_bootstrap", cur_column())) - .x),
    .names = "{.col}_optimism_adjusted"
  )) |>
  relocate(n, .after = subgroup_level)

write.csv2(performance_subgroup_combined, file = 'results/90d/model_training/final_performance/combined_subgroup_analyses.csv')
