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
source('r/custom_performance_metrics.R')


# read data ---------------------------------------------------------------
dfRaw <- read_csv2(file = 'data/aid_icu.csv')


# preproccess data --------------------------------------------------------
df <- dfRaw |>
  clean_names() |>
  mutate(limitation = case_when(
    limitations %in% c(0,3,4) ~ 'no treatment limitations',
    limitations %in% c(2,3) ~ 'treatment limitations',
    T ~ NA,
    .ptype = factor(levels = c('no treatment limitations', 'treatment limitations'))
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
    alcohol_abuse == T ~ 'alcohol abuse',
    T ~ 'no alcohol abuse',
    .ptype = factor(levels = c('no alcohol abuse', 'alcohol abuse'))
  ),
  hematologic_cancer = case_when(
    hematologic_cancer == T ~ 'hematologic cancer',
    T ~ 'no hematologic cancer',
    .ptype = factor(levels = c('no hematologic cancer', 'hematologic cancer'))
  ),
  metastatic_cancer = case_when(
    metastatic_cancer == T ~ 'metastatic cancer',
    T ~ 'no metastatic cancer',
    .ptype = factor(levels = c('no metastatic cancer', 'metastatic cancer'))
  ),
  elective_surgery = case_when(
    elective_surg == T ~ 'elective surgery',
    T ~ 'no elective surgery',
    .ptype = factor(levels = c('no elective surgery', 'elective surgery'))
  ),
  emergency_surgery = case_when(
    emergency_surg == T ~ 'emergency surgery',
    T ~ 'no emergency surgery',
    .ptype = factor(levels = c('no emergency surgery', 'emergency surgery'))
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
    strata_site %in% c('DK14') ~ 'secondary',
    strata_site %in% c('DK01', 'DK04') ~ 'tertiary',
    T ~ NA,
    .ptype = factor(levels = c('secondary', 'tertiary'))
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
    pre_cps <= 25 ~ 'SMSlow',
    pre_cps >= 26 ~ 'SMShigh',
    T ~ NA,
    .ptype = factor(levels = c('SMSlow', 'SMShigh'))
  ),
  advanced_cancer = case_when(
    hematologic_cancer == T | metastatic_cancer == T ~ 'advanced cancer',
    T ~ 'no advanced cancer',
    .ptype = factor(levels = c('no advanced cancer', 'advanced cancer'))
  ),
  surgery = case_when(
    elective_surg == T | emergency_surg == T ~ 'surgical patient',
    T ~ 'not surgical patient',
    .ptype = factor(levels = c('surgical patient', 'not surgical patient'))
  ),
  )

# Explore outcome ---------------------------------------------------------
# 90 d mortality
table(df$mort_90d)
prop.table(table(df$mort_90d))

# 1 year mortality
table(df$mort_1y)
prop.table(table(df$mort_1y))


# missing data ------------------------------------------------------------
df |>
  summarise(across(everything(),
                   list(missing_count = ~ sum(is.na(.)),
                        missing_prop = ~ mean(is.na(.))))) |>
  pivot_longer(everything(), 
               names_to = c("variable", "metric"), 
               names_pattern = "^(.*)_(missing_count|missing_prop)$") %>%
  pivot_wider(names_from = metric, values_from = value)

# sex | mortality ---------------------------------------------------------
ggplot_sex_mortality <- df |>
  count(sex, mort_90d) |>
  group_by(sex) |>
  mutate(pct = n / sum(n)) |>
  ggplot(aes(x = sex,
             y = pct,
             fill = mort_90d)) +
  geom_col(position = 'dodge2') +
  viridis::scale_fill_viridis(discrete = T) +
  scale_y_continuous(labels = percent_format()) +
  theme_classic() +
  labs(
    title = 'Percentage: mortality | sex',
    fill = '90 day mortality:',
    x = 'sex',
    y = 'fraction'
  ) +
  theme(
    plot.title = element_text(color="black", size=14, face="bold", hjust = 0.5),
    legend.position = 'bottom'
  )

# age | mortality ---------------------------------------------------------
ggplot_age_mortality <- df |>
  ggplot(aes(age)) +
  geom_density(aes(fill = mort_90d), alpha = 0.4) +
  viridis::scale_fill_viridis(discrete = T) +
  scale_y_continuous(labels = percent_format()) +
  theme_classic() +
  labs(
    title = 'Count: Age | mortality',
    fill = '90 day mortality:',
    x = 'age (years)',
    y = 'fraction'
  ) +
  theme(
    plot.title = element_text(color="black", size=14, face="bold", hjust = 0.5),
    legend.position = 'bottom'
  )


# alcohol abuse | mortality -----------------------------------------------
ggplot_alcohol_mortality <- df |>
  count(alcohol_abuse, mort_90d) |>
  group_by(alcohol_abuse) |>
  mutate(pct = n / sum(n)) |>
  ggplot(aes(x = alcohol_abuse,
             y = pct,
             fill = mort_90d)) +
  geom_col(position = 'dodge2') +
  viridis::scale_fill_viridis(discrete = T) +
  scale_y_continuous(labels = percent_format()) +
  theme_classic() +
  labs(
    title = 'Percentage: mortality | alcohol abuse',
    fill = '90 day mortality:',
    x = 'alcohol abuse',
    y = 'fraction'
  ) +
  theme(
    plot.title = element_text(color="black", size=14, face="bold", hjust = 0.5),
    legend.position = 'bottom'
  )


# smoking | mortality -----------------------------------------------------
ggplot_smoking_mortality <- df |>
  count(smoking, mort_90d) |>
  group_by(smoking) |>
  mutate(pct = n / sum(n)) |>
  ggplot(aes(x = smoking,
             y = pct,
             fill = mort_90d)) +
  geom_col(position = 'dodge2') +
  viridis::scale_fill_viridis(discrete = T) +
  scale_y_continuous(labels = percent_format()) +
  theme_classic() +
  labs(
    title = 'Percentage: mortality | smoking',
    fill = '90 day mortality:',
    x = 'smoking',
    y = 'fraction'
  ) +
  theme(
    plot.title = element_text(color="black", size=14, face="bold", hjust = 0.5),
    legend.position = 'bottom'
  )


# icu type | mortality --------------------------------------------------------
ggplot_icu_type_mortality <- df |>
  count(icu_type, mort_90d) |>
  group_by(icu_type) |>
  mutate(pct = n / sum(n)) |>
  ggplot(aes(x = icu_type,
             y = pct,
             fill = mort_90d)) +
  geom_col(position = 'dodge2') +
  viridis::scale_fill_viridis(discrete = T) +
  scale_y_continuous(labels = percent_format()) +
  theme_classic() +
  labs(
    title = 'Percentage: mortality | icu type',
    fill = '90 day mortality:',
    x = 'icu type',
    y = 'fraction'
  ) +
  theme(
    plot.title = element_text(color="black", size=14, face="bold", hjust = 0.5),
    legend.position = 'bottom'
  )

# strata site | mortality --------------------------------------------------------
ggplot_strata_site_mortality <- df |>
  count(strata_site, mort_90d) |>
  group_by(strata_site) |>
  mutate(pct = n / sum(n)) |>
  ggplot(aes(x = strata_site,
             y = pct,
             fill = mort_90d)) +
  geom_col(position = 'dodge2') +
  viridis::scale_fill_viridis(discrete = T) +
  scale_y_continuous(labels = percent_format()) +
  theme_classic() +
  labs(
    title = 'Percentage: mortality | site',
    fill = '90 day mortality:',
    x = 'site',
    y = 'fraction'
  ) +
  theme(
    plot.title = element_text(color="black", size=14, face="bold", hjust = 0.5),
    legend.position = 'bottom'
  )

# limitations | mortality -------------------------------------------------
ggplot_limitation_mortality <- df |>
  count(limitation, mort_90d) |>
  group_by(limitation) |>
  mutate(pct = n / sum(n)) |>
  ggplot(aes(x = limitation,
             y = pct,
             fill = mort_90d)) +
  geom_col(position = 'dodge2') +
  viridis::scale_fill_viridis(discrete = T) +
  scale_y_continuous(labels = percent_format()) +
  theme_classic() +
  labs(
    title = 'Percentage: mortality | limitation type',
    fill = '90 day mortality:',
    x = 'limitation type',
    y = 'fraction'
  ) +
  theme(
    plot.title = element_text(color="black", size=14, face="bold", hjust = 0.5),
    legend.position = 'bottom'
  )

# delir type --------------------------------------------------------------
ggplot_delir_type_mortality <- df |>
  count(strata_delir, mort_90d) |>
  group_by(strata_delir) |>
  mutate(pct = n / sum(n)) |>
  ggplot(aes(x = strata_delir,
             y = pct,
             fill = mort_90d)) +
  geom_col(position = 'dodge2') +
  viridis::scale_fill_viridis(discrete = T) +
  scale_y_continuous(labels = percent_format()) +
  theme_classic() +
  labs(
    title = 'Percentage: mortality | delirium type',
    fill = '90 day mortality:',
    x = 'delir type',
    y = 'fraction'
  ) +
  theme(
    plot.title = element_text(color="black", size=14, face="bold", hjust = 0.5),
    legend.position = 'bottom'
  )

# hematological cancer ----------------------------------------------------
ggplot_hematological_cancer_mortality <- df |>
  count(hematologic_cancer, mort_90d) |>
  group_by(hematologic_cancer) |>
  mutate(pct = n / sum(n)) |>
  ggplot(aes(x = hematologic_cancer,
             y = pct,
             fill = mort_90d)) +
  geom_col(position = 'dodge2') +
  viridis::scale_fill_viridis(discrete = T) +
  scale_y_continuous(labels = percent_format()) +
  theme_classic() +
  labs(
    title = 'Percentage: mortality | hematological cancer',
    fill = '90 day mortality:',
    x = 'hematological cancer',
    y = 'fraction'
  ) +
  theme(
    plot.title = element_text(color="black", size=14, face="bold", hjust = 0.5),
    legend.position = 'bottom'
  )


# metastatic cancer -------------------------------------------------------
ggplot_metastatic_cancer_mortality <- df |>
  count(metastatic_cancer, mort_90d) |>
  group_by(metastatic_cancer) |>
  mutate(pct = n / sum(n)) |>
  ggplot(aes(x = metastatic_cancer,
             y = pct,
             fill = mort_90d)) +
  geom_col(position = 'dodge2') +
  viridis::scale_fill_viridis(discrete = T) +
  scale_y_continuous(labels = percent_format()) +
  theme_classic() +
  labs(
    title = 'Percentage: mortality | metastatic cancer',
    fill = '90 day mortality:',
    x = 'metastatic cancer',
    y = 'fraction'
  ) +
  theme(
    plot.title = element_text(color="black", size=14, face="bold", hjust = 0.5),
    legend.position = 'bottom'
  )


# any advanced cancer -----------------------------------------------------
ggplot_any_advanced_cancer_mortality <- df |>
  count(advanced_cancer, mort_90d) |>
  group_by(advanced_cancer) |>
  mutate(pct = n / sum(n)) |>
  ggplot(aes(x = advanced_cancer,
             y = pct,
             fill = mort_90d)) +
  geom_col(position = 'dodge2') +
  viridis::scale_fill_viridis(discrete = T) +
  scale_y_continuous(labels = percent_format()) +
  theme_classic() +
  labs(
    title = 'Percentage: mortality | any advanced cancer',
    fill = '90 day mortality:',
    x = 'any advanced cancer',
    y = 'fraction'
  ) +
  theme(
    plot.title = element_text(color="black", size=14, face="bold", hjust = 0.5),
    legend.position = 'bottom'
  )


# emergency surgery -------------------------------------------------------
ggplot_emergency_surgery_mortality <- df |>
  count(emergency_surgery, mort_90d) |>
  group_by(emergency_surgery) |>
  mutate(pct = n / sum(n)) |>
  ggplot(aes(x = emergency_surgery,
             y = pct,
             fill = mort_90d)) +
  geom_col(position = 'dodge2') +
  viridis::scale_fill_viridis(discrete = T) +
  scale_y_continuous(labels = percent_format()) +
  theme_classic() +
  labs(
    title = 'Percentage: mortality | emergency surgery',
    fill = '90 day mortality:',
    x = 'emergency surgery',
    y = 'fraction'
  ) +
  theme(
    plot.title = element_text(color="black", size=14, face="bold", hjust = 0.5),
    legend.position = 'bottom'
  )

# elective surgery -------------------------------------------------------
ggplot_elective_surgery_mortality <- df |>
  count(elective_surgery, mort_90d) |>
  mutate(pct = n / sum(n)) |>
  ggplot(aes(x = elective_surgery,
             y = pct,
             fill = mort_90d)) +
  geom_col(position = 'dodge2') +
  viridis::scale_fill_viridis(discrete = T) +
  scale_y_continuous(labels = percent_format()) +
  theme_classic() +
  labs(
    title = 'Percentage: mortality | elective surgery',
    fill = '90 day mortality:',
    x = 'elective surgery',
    y = 'fraction'
  ) +
  theme(
    plot.title = element_text(color="black", size=14, face="bold", hjust = 0.5),
    legend.position = 'bottom'
  )


# any surgery -------------------------------------------------------------
ggplot_any_surgery_mortality <- df |>
  count(surgery, mort_90d) |>
  mutate(pct = n / sum(n)) |>
  ggplot(aes(x = surgery,
             y = pct,
             fill = mort_90d)) +
  geom_col(position = 'dodge2') +
  viridis::scale_fill_viridis(discrete = T) +
  scale_y_continuous(labels = percent_format()) +
  theme_classic() +
  labs(
    title = 'Percentage: mortality | any surgery',
    fill = '90 day mortality:',
    x = 'any surgery',
    y = 'fraction'
  ) +
  theme(
    plot.title = element_text(color="black", size=14, face="bold", hjust = 0.5),
    legend.position = 'bottom'
  )


# pre barthel -------------------------------------------------------------
ggplot_pre_barthel_mortality <- df |>
  count(pre_barthel, mort_90d) |>
  group_by(pre_barthel) |>
  mutate(pct = n / sum(n)) |>
  ggplot(aes(x = pre_barthel,
             y = pct,
             fill = mort_90d)) +
  geom_col(position = 'dodge2') +
  viridis::scale_fill_viridis(discrete = T) +
  scale_y_continuous(labels = percent_format()) +
  theme_classic() +
  labs(
    title = 'Percentage: mortality | pre barthel',
    fill = '90 day mortality:',
    x = 'pre barthel',
    y = 'fraction'
  ) +
  theme(
    plot.title = element_text(color="black", size=14, face="bold", hjust = 0.5),
    legend.position = 'bottom'
  )

ggplot_pre_barthel_na_mortality <- df |>
  filter(is.na(pre_barthel)) |>
  count(pre_barthel, mort_90d) |>
  group_by(pre_barthel) |>
  mutate(pct = n / sum(n)) |>
  ggplot(aes(x = mort_90d,
             y = pct,
             fill = mort_90d)) +
  geom_col(position = 'dodge2') +
  viridis::scale_fill_viridis(discrete = T) +
  scale_y_continuous(labels = percent_format()) +
  theme_classic() +
  labs(
    title = 'Percentage: mortality | pre barthel = NA',
    fill = '90 day mortality:',
    x = '90 day mortality',
    y = 'fraction'
  ) +
  theme(
    plot.title = element_text(color="black", size=14, face="bold", hjust = 0.5),
    legend.position = 'bottom'
  )

# cfs ---------------------------------------------------------------------
ggplot_pre_cfs_mortality <- df |>
  count(pre_cfs, mort_90d) |>
  group_by(pre_cfs) |>
  mutate(pct = n / sum(n)) |>
  ggplot(aes(x = pre_cfs,
             y = pct,
             fill = mort_90d)) +
  geom_col(position = 'dodge2') +
  viridis::scale_fill_viridis(discrete = T) +
  scale_y_continuous(labels = percent_format()) +
  theme_classic() +
  labs(
    title = 'Percentage: mortality | pre cfs',
    fill = '90 day mortality:',
    x = 'pre cfs',
    y = 'fraction'
  ) +
  theme(
    plot.title = element_text(color="black", size=14, face="bold", hjust = 0.5),
    legend.position = 'bottom'
  )

ggplot_pre_cfs_na_mortality <- df |>
  filter(is.na(pre_cfs)) |>
  count(pre_cfs, mort_90d) |>
  group_by(pre_cfs) |>
  mutate(pct = n / sum(n)) |>
  ggplot(aes(x = mort_90d,
             y = pct,
             fill = mort_90d)) +
  geom_col(position = 'dodge2') +
  viridis::scale_fill_viridis(discrete = T) +
  scale_y_continuous(labels = percent_format()) +
  theme_classic() +
  labs(
    title = 'Percentage: mortality | pre cfs = NA',
    fill = '90 day mortality:',
    x = '90 day mortality',
    y = 'fraction'
  ) +
  theme(
    plot.title = element_text(color="black", size=14, face="bold", hjust = 0.5),
    legend.position = 'bottom'
  )

# cps ---------------------------------------------------------------------
ggplot_pre_cps_mortality <- df |>
  count(pre_cps, mort_90d) |>
  group_by(pre_cps) |>
  mutate(pct = n / sum(n)) |>
  ggplot(aes(x = pre_cps,
             y = pct,
             fill = mort_90d)) +
  geom_col(position = 'dodge2') +
  viridis::scale_fill_viridis(discrete = T) +
  scale_y_continuous(labels = percent_format()) +
  theme_classic() +
  labs(
    title = 'Percentage: mortality | pre cps',
    fill = '90 day mortality:',
    x = 'pre cfs',
    y = 'fraction'
  ) +
  theme(
    plot.title = element_text(color="black", size=14, face="bold", hjust = 0.5),
    legend.position = 'bottom'
  )

ggplot_pre_cps_na_mortality <- df |>
  filter(is.na(pre_cps)) |>
  count(pre_cps, mort_90d) |>
  group_by(pre_cps) |>
  mutate(pct = n / sum(n)) |>
  ggplot(aes(x = mort_90d,
             y = pct,
             fill = mort_90d)) +
  geom_col(position = 'dodge2') +
  viridis::scale_fill_viridis(discrete = T) +
  scale_y_continuous(labels = percent_format()) +
  theme_classic() +
  labs(
    title = 'Percentage: mortality | pre cps = NA',
    fill = '90 day mortality:',
    x = '90 day mortality',
    y = 'fraction'
  ) +
  theme(
    plot.title = element_text(color="black", size=14, face="bold", hjust = 0.5),
    legend.position = 'bottom'
  )

# sms ---------------------------------------------------------------------
ggplot_sms_mortality <- df |>
  count(sms, mort_90d) |>
  group_by(sms) |>
  mutate(pct = n / sum(n)) |>
  ggplot(aes(x = sms,
             y = pct,
             fill = mort_90d)) +
  geom_col(position = 'dodge2') +
  viridis::scale_fill_viridis(discrete = T) +
  scale_y_continuous(labels = percent_format()) +
  theme_classic() +
  labs(
    title = 'Percentage: mortality | sms',
    fill = '90 day mortality:',
    x = 'sms',
    y = 'fraction'
  ) +
  theme(
    plot.title = element_text(color="black", size=14, face="bold", hjust = 0.5),
    legend.position = 'bottom'
  )


# haloperidol -------------------------------------------------------------
ggplot_haloperidol_mortality <- df |>
  count(intervention, mort_90d) |>
  group_by(intervention) |>
  mutate(pct = n / sum(n)) |>
  ggplot(aes(x = intervention,
             y = pct,
             fill = mort_90d)) +
  geom_col(position = 'dodge2') +
  viridis::scale_fill_viridis(discrete = T) +
  scale_y_continuous(labels = percent_format()) +
  theme_classic() +
  labs(
    title = 'Percentage: mortality | Intervention',
    fill = '90 day mortality:',
    x = 'Intervention',
    y = 'fraction'
  ) +
  theme(
    plot.title = element_text(color="black", size=14, face="bold", hjust = 0.5),
    legend.position = 'bottom'
  )

# Correlations ------------------------------------------------------------
df_num <- df |>
  dplyr::select(c(mort_90d, sex, age, alcohol_abuse, strata_delir, advanced_cancer, surgery, sms, icu_type, pre_barthel, pre_cps, pre_cfs, limitation, intervention, emergency_surgery, elective_surgery, hematologic_cancer, metastatic_cancer, smoking, barthel_category, cfs_category, cps_category)) |>
  mutate(across(where(is.factor), ~ as.numeric(as.factor(.))))

cor_matrix <- df_num %>%
  cor(use = "pairwise.complete.obs", method = "spearman")

ggplot_spearman_corr <- ggcorrplot::ggcorrplot(cor_matrix, type = "lower", lab = TRUE)


# save visuals ------------------------------------------------------------
outdir <- "results/data_viz/90_d/"

plot_names <- ls(pattern = "^ggplot_")

for(nm in plot_names) {
  plt <- get(nm)                             
  ggsave(
    filename = file.path(outdir, paste0(nm, ".png")),
    plot     = plt,
    width    = 6, height = 4,                
    dpi      = 300
  )
}
