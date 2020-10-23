# Title     : Analysis of RT
# Objective : Test effect of cue-probe incongrunecy
# Created by: Jose C. Garcia Alanis
# Created on: 23.07.20
# R version : 4.0.2 (2020-06-22), Taking Off Again

# source function for fast loading and installing of packages
source('./r_functions/getPacks.R')
source('./r_functions/spr2.R')


# 1) define the path to behavioral data directory -----------------------------

# get system and user information
host <- Sys.info()

# set default path or promt user for other path
if (host['nodename'] == "josealanis-desktop") {

  # defaut path in project structure
  path_to_rt <- '../data/derivatives/results'

} else {

  path_to_rt <- readline('Please provide path to behavioral data: ')

}


# 2) import in the data -------------------------------------------------------
# this part requires the package 'dplyr'
getPacks('dplyr')

# files in dir
rt_files <- list.files(path = paste(path_to_rt, 'rt', sep = '/'),
                       full.names = T)

# read in the files
rt_list <- lapply(rt_files, read.table, sep = '\t', header = T)

# put the in a data frame
rt_df <- bind_rows(rt_list, .id = "column_label")

# recode block variable
rt_df <- rt_df %>% mutate(block =  ifelse(block == 0, 1, 2)) %>%
  mutate(block = factor(block, labels = c('Block 1', 'Block 2')))

# exclude subject 51
rt_df <- rt_df %>% filter(!subject == 51)


# 3) exploratory analyses correct reactions -----------------------------------

# extract trials with correct reponses
corrects <- rt_df %>%
  filter(reaction_cues == 'Correct' & reaction_probes == 'Correct')

# plasibility checks
# e.g., rt min should be > 0.0, max < 0.750
# e.g., only probes AX, AY, BX and BY should be present
summary(corrects); unique(corrects$probe)

# check distribution of rt
hist(corrects$rt)

# get rid of extreme values, e.g., rt values very close to 0
# using winsorsed scores
getPacks('psych')
corrects <- corrects %>%
  group_by(subject, block, probe) %>%
  mutate(w_rt = winsor(rt, trim = 0.1))

# ** some descriptive statistics **
# mean rt and sd (trial type by block)
corrects %>% group_by(probe, block) %>%
  summarise(m = mean(rt), sd = sd(rt))
# by trial type
corrects %>% group_by(probe) %>%
  summarise(m = mean(rt), sd = sd(rt))
# by block
corrects %>% group_by(block) %>%
  summarise(m = mean(rt), sd = sd(rt))
# average number of trials
corrects %>%
  group_by(subject, cue) %>%
  summarise(n = sum(!is.na(rt))) %>%
  group_by(cue) %>%
  summarise(mn = mean(n), sd = sd(n))

# *** plot distribution of reaction time ***
getPacks(c('ggplot2', 'viridis', 'Hmisc'))

# data for plot
m_rt <- corrects %>%
  group_by(subject, probe, block) %>%
  summarise(m = mean(rt))

m_rt <- corrects %>%
  group_by(subject, probe) %>%
  summarise(m = mean(rt))

# add some space between geoms
#pjd <- position_jitterdodge(
#  jitter.width = 0.5,
#  jitter.height = 0,
#  dodge.width = 0,
#  seed = 2)
pn <- position_nudge(x = 0.4)
pd <- position_jitter(0.2)


# create the plot
rt_plot <- ggplot(data = m_rt,
                  aes(x = probe, y = m,
                      fill = probe, shape = probe)) +
  geom_line(aes(group = subject), position = pd, alpha = 0.1, size = 0.4) +
  geom_point(position = pd) +
  scale_shape_manual(values = c(25, 24, 23, 21)) +
  geom_boxplot(position = pn, width = 0.15, alpha = 0.8) +
  scale_fill_viridis(discrete = T, begin = 0.05, end = .95) +
  labs(x = 'Cue-Probe',
       y = 'RT (ms)',
       fill = 'Cue-Probe',
       shape = 'Cue-Probe') +
  scale_y_continuous(limits = c(0.1, 0.7),
                     breaks = seq(0.1, 0.7, 0.1),
                     labels = seq(100, 700, 100)) +
  geom_segment(aes(x = -Inf, y = 0.1, xend = -Inf, yend = 0.7),
               color = 'black', size = rel(0.5), linetype = 1) +
  geom_segment(aes(x = 'AX', y = -Inf, xend = 'BY', yend = -Inf),
               color = 'black', size = rel(0.5), linetype = 1) +
  theme(axis.title.x = element_text(color = 'black', size = 12,
                                    margin = margin(t = 10)),
        axis.title.y= element_text(color = 'black', size = 12,
                                   margin = margin(r = 10)),
        axis.text = element_text(color = 'black', size = 10),
        panel.background = element_rect(fill = 'gray97'),
        strip.text = element_blank(),
        strip.background = element_blank(),
        legend.position='bottom',
        legend.title = element_blank(),
        legend.text = element_text(size = 10),
        panel.spacing = unit(1, "lines")); rt_plot
# save to disk
ggsave(filename = '../data/derivatives/results/figures/rt_distribution.pdf',
       plot = rt_plot, width = 4, height = 5, dpi = 300)

#
## create the plot
#rt_plot <- ggplot(data = m_rt,
#                  aes(x = block, y = m,
#                      fill = probe, shape = block)) +
#  facet_wrap(~ probe, ncol = 4, scales = 'free_y') +
#  geom_line(aes(group = subject), position = pjd, alpha = 0.25, size = 0.33,
#            color='black') +
#  geom_jitter(position = pjd, size = 1.75, color='black', alpha = 0.25) +
#  stat_summary(fun.data = "mean_cl_boot", position = pn,
#               fun.args = list(B = 10000, conf.int = 0.99), size = 0.8) +
#  scale_fill_viridis(discrete = T, end = .95) +
#  scale_shape_manual(values = c(21, 23), guide = FALSE) +
#  labs(x = 'Cue-Probe',
#       y = 'RT (sec)',
#       color = F) +
#  guides(fill = guide_legend(override.aes = list(shape = 21))) +
#  scale_y_continuous(limits = c(0.1, 0.7),
#                     breaks = seq(0.1, 0.7, 0.1)) +
#  geom_segment(aes(x = -Inf, y = 0.1, xend = -Inf, yend = 0.7),
#               color = 'black', size = rel(0.5), linetype = 1) +
#  geom_segment(aes(x = 'Block 1', y = -Inf, xend = 'Block 2', yend = -Inf),
#               color = 'black', size = rel(0.5), linetype = 1) +
#  theme(axis.title.x = element_text(color = 'black', size = 12,
#                                    margin = margin(t = 10)),
#        axis.title.y= element_text(color = 'black', size = 12,
#                                   margin = margin(r = 10)),
#        axis.text = element_text(color = 'black', size = 10),
#        panel.background = element_rect(fill = 'gray99'),
#        strip.text = element_blank(),
#        strip.background = element_blank(),
#        legend.position='bottom',
#        legend.title = element_blank(),
#        legend.text = element_text(size = 10),
#        panel.spacing = unit(1, "lines")); rt_plot
## save to disk
#ggsave(filename = '../data/derivatives/results/figures/rt_distribution.pdf',
#       plot = rt_plot, width = 7.5, height = 5, dpi = 300)


# 4) model rt of correct reactions --------------------------------------------
# this part requires the following packages
getPacks(c('lme4', 'car',
           'sjPlot',
           'tidyr'))

# summarise to the level of trial types by block
corrects <- corrects %>%
  group_by(subject, probe) %>%
  summarise(m_rt = mean(w_rt))

# transform variables to factors
corrects$probe <- as.factor(corrects$probe)
# corrects$block <- as.factor(corrects$block)

rt_mod <- lmer(data = corrects,
               m_rt ~ probe + (1|subject),
               contrasts = list(probe = 'contr.sum'))
# anova for model
rt_anova <- car::Anova(rt_mod, test = 'F', type = 'III'); rt_anova
# semi-partial R-squared for predictors
spr2 <- spR2(rt_anova)

# create a teble of the model summary
tab_model(rt_mod, digits = 3,
          file = '../data/derivatives/results/tables/rt_model.html',
          pred.labels = c('(Intercept)',
                          'AX', 'AY', 'BX'))

# create a teble of the model anova
tab_df(title = 'Anova RT Model',
       file = '../data/derivatives/results/tables/anova_rt_mod.html',
       cbind(row.names(rt_anova),
             as.data.frame(rt_anova),
             c(NA, spr2)), digits = 3)


# 5) interaction analysis for rt data -----------------------------------------
# this section requires the following packages
getPacks(c('emmeans', 'ggplot2', 'dplyr',  'tidyr'))

# levels of the probe variable by block (e.g. AX in block 0 = AX 0)
#probe_levels <- c('AX Block 1', 'AY Block 1', 'BX Block 1', 'BY Block 1',
#                  'AX Block 2', 'AY Block 2', 'BX Block 2', 'BY Block 2')

# compute estimated marginal means
rt_means <- emmeans(rt_mod, ~ probe)
# compute significance of pairwise contrasts
pwpm(rt_means, adjust = 'mvt', flip = TRUE)

# prepare pairwise data for plot
rt_means_df <- as.data.frame(rt_means)
#rt_means_df <- rt_means_df %>%
#  arrange(block) %>%
#  mutate(var1 = paste(probe, block, sep = ' '),
#         var2 = paste(probe, block, sep = ' '))

## reorder levels
#rt_means_df$var1 <- factor(rt_means_df$var1,
#                           levels = probe_levels)
#rt_means_df$var2 <- factor(rt_means_df$var2,
#                           levels = probe_levels)

# model variance estimates and residual degrees of freedom
mod_var <- VarCorr(rt_mod)
totSD <- sqrt(sum(as.data.frame(mod_var)$vcov))
edf <- df.residual(rt_mod)

# compute effect sizes for the pairwise contrasts
probe_means <- emmeans(rt_mod, ~ probe)
es_probes <- eff_size(probe_means, sigma = totSD, edf = edf); es_probes
#
#block_means <- emmeans(rt_mod, ~ block)
#es_block <- eff_size(block_means, sigma = totSD, edf = edf); es_block

# compute effect sizes
es <- eff_size(rt_means, sigma = totSD, edf = edf)
es <- as.data.frame(es)
#es <- es %>%
#  separate(contrast, into = c('var1', 'var2'), sep = ' - ')
  #%>%
  #mutate(var1 = gsub(var1, pattern = ',', replacement = ' '),
  #       var2 = gsub(var2, pattern = ',', replacement = ' '))
# reorder levels
#es$var1 <- factor(es$var1, levels = probe_levels)
#es$var2 <- factor(es$var2, levels = probe_levels)

# adjust p-values and add "*" for significance coding
pairwise_rt <- as.data.frame(pairs(rt_means, adjust='bonferroni'))
es$p_val <- pairwise_rt$p.value
es <- es %>%
  mutate(sig = ifelse(p_val > 0.05, ' ',
                      ifelse(p_val > 0.01, '*',
                             ifelse(p_val > 0.001, '**', '***'))))

contrast_plot <- ggplot(es, aes(contrast, effect.size)) +
  geom_point(size = 1.5) +
  geom_errorbar(aes(ymin = lower.CL, ymax = upper.CL),
                width = 0.15, size = 0.8) +
  geom_hline(yintercept = 0, linetype = 'dotted') +
  geom_text(aes(label = sig), angle = 0, vjust = 0, size = 5) +
  geom_segment(aes(x = -Inf, y = -4,
                   xend = -Inf, yend = 6),
               color = 'black', size = rel(0.5), linetype = 1) +
  geom_segment(aes(x = 'AX - AY', y = -Inf,
                   xend = 'BX - BY', yend = -Inf),
               color = 'black', size = rel(0.5), linetype = 1) +
  labs(x = 'Contrast', y = "Effect size (Cohen's d") +
  theme(axis.title.x = element_text(color = 'black', size = 14,
                                    margin = margin(t = 10)),
        axis.title.y= element_text(color = 'black', size = 14,
                                   margin = margin(r = 10)),
        axis.text.y = element_text(color = 'black', size = 11),
        axis.text.x = element_text(color = 'black', size = 11,
                                   hjust = 1),
        panel.background = element_rect(fill = 'gray97'),
        strip.text = element_blank(),
        strip.background = element_blank(),
        legend.position='bottom',
        panel.spacing = unit(1, "lines")) +
  coord_flip(); contrast_plot
# save to disk
ggsave(filename = '../data/derivatives/results/figures/contrast_rt_plot.pdf',
       plot = contrast_plot, width = 6, height = 3)


## create the plot
#pw_rt_plot <- ggplot(rt_means_df, aes(var1, var2)) +
#  geom_tile(fill = 'white') +
#  geom_text(aes(label = round(emmean, 2)), size = 5,
#            vjust = -.05) +
#  geom_text(aes(label = paste(round(lower.CL, 2),
#                              round(upper.CL, 2),
#                              sep = ' - ')),
#            vjust = 2.5, size = 2.75) +
#  geom_tile(data = es, aes(var1, var2, fill=effect.size),
#            colour = 'white', size = 0.8) +
#  geom_text(data = es, aes(var1, var2, label = sig),
#            colour = 'black', size = 7,  vjust = 1) +
#  # scale_fill_viridis(option = 'B', limits = c(-0.5, 0.5)) +
#  scale_fill_distiller(palette = "RdBu", limits = c(-5.0, 5.0),
#                       breaks = seq(-5.0, 5.0, 2.5)) +
#  geom_segment(aes(x = -Inf, y = 'AX Block 1',
#                   xend = -Inf, yend = 'BY Block 2'),
#               color = 'black', size = rel(0.5), linetype = 1) +
#  geom_segment(aes(x = 'AX Block 1', y = -Inf,
#                   xend = 'BY Block 2', yend = -Inf),
#               color = 'black', size = rel(0.5), linetype = 1) +
#  labs(x = 'Trial type by block', y = 'Trial type by block',
#       fill = "Difference (Cohen's D)") +
#  theme(axis.title.x = element_text(color = 'black', size = 16,
#                                    margin = margin(t = 10)),
#        axis.title.y= element_text(color = 'black', size = 16,
#                                   margin = margin(r = 10)),
#        axis.text.y = element_text(color = 'black', size = 14),
#        axis.text.x = element_text(color = 'black', size = 14, angle = 45,
#                                   hjust = 1),
#        panel.background = element_rect(fill = 'gray97'),
#        strip.text = element_blank(),
#        strip.background = element_blank(),
#        legend.position='bottom',
#        panel.spacing = unit(1, "lines")) +
#  coord_flip(); pw_rt_plot


# 6) exploratory analysis of error rates --------------------------------------
getPacks('dplyr')

# compute number of trials per condition
total <- rt_df %>%
  mutate(probe =
           ifelse(nchar(probe) > 1, substr(probe, 2, 2), probe),
         trial_type = paste0(cue, probe)) %>%
  group_by(subject, block, trial_type) %>%
  mutate(n_trials = sum(!is.na(trial))) %>%
  select(subject, block, trial_type, n_trials) %>%
  arrange(subject, block, trial_type) %>%
  unique()

# compute number of errors per condition
errors <- rt_df %>%
  mutate(probe =
           ifelse(nchar(probe) > 1, substr(probe, 2, 2), probe),
         trial_type = paste0(cue, probe)) %>%
  filter(reaction_probes == 'Incorrect') %>%
  group_by(subject, block, trial_type) %>%
  mutate(n_errors = sum(!is.na(trial))) %>%
  summarise(n_errors = mean(n_errors)) %>%
  arrange(subject, block, trial_type)

# merge data frames
errors <- merge(total, errors, c('subject', 'block', 'trial_type'), all.x = T)
# replace missing values with zeros
errors[is.na(errors)] <- 0

# ** compute error rates for summary **
errors <- errors %>%
  mutate(error_rate=(n_errors+0.5)/(n_trials+1))

errors %>% group_by(trial_type) %>%
  summarise(m = mean(error_rate), sd = sd(error_rate))

errors %>% group_by(block, trial_type) %>%
  summarise(m = mean(error_rate), sd = sd(error_rate))

# plot distribution of error rates
getPacks(c('ggplot2', 'viridis'))
pjd <- position_jitterdodge(
  jitter.width = 0.5,
  jitter.height = 0,
  dodge.width = 0,
  seed = 2)
pn <- position_nudge(x = 0.25)
errors_plot <- ggplot(errors,
                  aes(y = error_rate, x = block,
                      fill = trial_type, shape = block)) +
  facet_wrap(~ trial_type, ncol = 4, scales = 'free_y') +
  geom_line(aes(group = subject), position = pjd, alpha = 0.25, size = 0.33,
            color='black') +
  geom_jitter(position = pjd, size = 1.75, color='black', alpha = 0.2) +
  stat_summary(fun.data = "mean_cl_boot", position = pn,
               fun.args = list(B = 10000, conf.int = 0.99), size = 1) +
  scale_fill_viridis(discrete = T, end = .95) +
  scale_shape_manual(values = c(21, 23), guide = FALSE) +
  labs(x = 'Cue-Probe',
       y = 'Error rate',
       color = F) +
  guides(fill = guide_legend(override.aes = list(shape = 21))) +
  scale_y_continuous(limits = c(0.0, 0.9),
                     breaks = seq(-0.0, 0.9, 0.1)) +
  geom_segment(aes(x = -Inf, y = 0.0, xend = -Inf, yend = 0.9),
               color = 'black', size = rel(0.5), linetype = 1) +
  geom_segment(aes(x = 'Block 1', y = -Inf, xend = 'Block 2', yend = -Inf),
               color = 'black', size = rel(0.5), linetype = 1) +
  theme(axis.title.x = element_text(color = 'black', size = 12,
                                    margin = margin(t = 10)),
        axis.title.y= element_text(color = 'black', size = 12,
                                   margin = margin(r = 10)),
        axis.text = element_text(color = 'black', size = 10),
        panel.background = element_rect(fill = 'gray97'),
        strip.text = element_blank(),
        strip.background = element_blank(),
        legend.position='bottom',
        legend.title = element_blank(),
        legend.text = element_text(size = 10),
        panel.spacing = unit(1, "lines")); errors_plot
# save to disk
ggsave(filename = '../data/derivatives/results/figures/errors_distribution.pdf',
       plot = errors_plot, width = 7.5, height = 5, dpi = 300)


# 7) fit model for error rates ------------------------------------------------
# this part requires the following packages
getPacks(c('lme4', 'car',
           'sjPlot',
           'tidyr'))

# transform variables to factors
errors$trial_type <- as.factor(errors$trial_type)
errors$block <- as.factor(errors$block)

errors_mod <- lmer(data = errors,
               log(error_rate) ~ block * trial_type + (1|subject),
                   contrasts = list(trial_type = 'contr.sum',
                                    block = 'contr.sum'))
# anova for model
errors_anova <- car::Anova(errors_mod, test = 'F', type = 'III')
# semi-partial R-squared for predictors
err_spr2 <- spR2(errors_anova)

# create a teble of the model summary
tab_model(errors_mod, digits = 3,
          file = '../data/derivatives/results/tables/errors_model.html',
          pred.labels = c('(Intercept)',
                          'Block 1',
                          'AX', 'AY', 'BX',
                          'Block 1 * AX',
                          'Block 1 * AY',
                          'Block 1 * BX'))

# create a teble of the model anova
tab_df(title = 'Anova Errors Model',
       file = '../data/derivatives/results/tables/anova_errors_mod.html',
       cbind(row.names(errors_anova),
             as.data.frame(errors_anova),
             c(NA, err_spr2)), digits = 3)


# 8) interaction analysis for errors data -------------------------------------
# this section requires the following packages
getPacks(c('emmeans', 'ggplot2', 'dplyr',  'tidyr'))

# levels of the probe variable by block (e.g. AX in block 0 = AX 0)
probe_levels <- c('AX Block 1', 'AY Block 1', 'BX Block 1', 'BY Block 1',
                  'AX Block 2', 'AY Block 2', 'BX Block 2', 'BY Block 2')

# compute estimated marginal means
error_means <- emmeans(errors_mod, ~ trial_type * block, type = 'response')
# compute significance of pairwise contrasts
pwpm(error_means, adjust = 'mvt', flip = TRUE)

# prepare pairwise data for plot
error_means_df <- as.data.frame(error_means)
error_means_df <- error_means_df %>%
  arrange(block) %>%
  mutate(var1 = paste(trial_type, block, sep = ' '),
         var2 = paste(trial_type, block, sep = ' '))

# reorder levels
error_means_df$var1 <- factor(error_means_df$var1,
                              levels = probe_levels)
error_means_df$var2 <- factor(error_means_df$var2,
                              levels = probe_levels)

# model variance estimates and residual degrees of freedom
err_mod_var <- VarCorr(errors_mod)
err_totSD <- sqrt(sum(as.data.frame(err_mod_var)$vcov))
err_edf <- df.residual(errors_mod)

# compute effect sizes for the pairwise contrasts of cue-probe combination
probe_error_means <- emmeans(errors_mod, ~ trial_type)
es_error_probes <- eff_size(probe_error_means, sigma = err_totSD, edf = err_edf)
es_error_probes

# compute effect sizes for the pairwise contrasts of block
block_error_means <- emmeans(errors_mod, ~ block)
es_error_block <- eff_size(block_error_means, sigma = err_totSD, edf = err_edf)
es_error_block

# effect sizes for plot
es_error <- eff_size(error_means, sigma = err_totSD, edf = err_edf)
es_error <- as.data.frame(es_error)
es_error <- es_error %>%
  separate(contrast, into = c('var1', 'var2'), sep = ' - ') %>%
  mutate(var1 = gsub(var1, pattern = ',', replacement = ' '),
         var2 = gsub(var2, pattern = ',', replacement = ' '))
# reorder levels
es_error$var1 <- factor(es_error$var1, levels = probe_levels)
es_error$var2 <- factor(es_error$var2, levels = probe_levels)

# adjust p-values and add "*" for coding significance
pairwise_err <- as.data.frame(pairs(error_means, adjust='bonferroni'))
es_error$p_val <- pairwise_err$p.value
es_error <- es_error %>%
  mutate(sig = ifelse(p_val < 0.001, '***',
                     ifelse(p_val < 0.01, '**',
                            ifelse(p_val < 0.05, '*', ' '))))

# create the plot
pw_error_plot <- ggplot(error_means_df, aes(var1, var2)) +
  geom_tile(fill = 'white') +
  geom_text(aes(label = round(response, 2)), size = 5,
            vjust = -.05) +
  geom_text(aes(label = paste(round(lower.CL, 2),
                              round(upper.CL, 2),
                              sep = ' - ')),
            vjust = 2.5, size = 2.75) +
  geom_tile(data = es_error, aes(var1, var2, fill=effect.size),
            colour = 'white', size = 0.8) +
  geom_text(data = es_error, aes(var1, var2, label = sig),
            colour = 'black', size = 7,  vjust = 1) +
  # scale_fill_viridis(option = 'B', limits = c(-0.5, 0.5)) +
  scale_fill_distiller(palette = "RdBu", limits = c(-4.0, 4.0),
                       breaks = seq(-4.0, 4.0, 2.0)) +
  geom_segment(aes(x = -Inf, y = 'AX Block 1',
                   xend = -Inf, yend = 'BY Block 2'),
               color = 'black', size = rel(0.5), linetype = 1) +
  geom_segment(aes(x = 'AX Block 1', y = -Inf,
                   xend = 'BY Block 2', yend = -Inf),
               color = 'black', size = rel(0.5), linetype = 1) +
  labs(x = 'Trial type by block', y = 'Trial type by block',
       fill = "Difference (Cohen's D)") +
  theme(axis.title.x = element_text(color = 'black', size = 16,
                                    margin = margin(t = 10)),
        axis.title.y= element_text(color = 'black', size = 16,
                                   margin = margin(r = 10)),
        axis.text.y = element_text(color = 'black', size = 14),
        axis.text.x = element_text(color = 'black', size = 14, angle = 45,
                                   hjust = 1),
        panel.background = element_rect(fill = 'gray97'),
        strip.text = element_blank(),
        strip.background = element_blank(),
        legend.position='bottom',
        panel.spacing = unit(1, "lines")) +
  coord_flip(); pw_error_plot
# save to disk
ggsave(filename = '../data/derivatives/results/figures/pairwise_error_plot.pdf',
       plot = pw_error_plot, width = 6.5, height = 7, dpi = 300)


# 9) compute proactive control measures ---------------------------------------

n_corrects <-rt_df %>%
  mutate(probe =
           ifelse(nchar(probe) > 1, substr(probe, 2, 2), probe),
         trial_type = paste0(cue, probe)) %>%
  filter(reaction_probes == 'Correct') %>%
  group_by(subject, block, trial_type) %>%
  mutate(n_corrects = sum(!is.na(trial))) %>%
  summarise(n_corrects = mean(n_corrects)) %>%
  arrange(subject, block, trial_type)

n_corrects <- merge(total, n_corrects,
                    c('subject', 'block', 'trial_type'), all.x = T)

n_corrects <- n_corrects %>%
  mutate(correct_rate=(n_corrects+0.5)/(n_trials+1))

n_corrects <- merge(n_corrects, select(errors, subject, block, trial_type,
                                       error_rate),
                    c('subject', 'block', 'trial_type'))

a_bias <-  n_corrects %>%
  filter(trial_type == 'AX' | trial_type == 'AY') %>%
  group_by(subject, block) %>%
  mutate(a_bias = ifelse(trial_type == 'AX',
                         0.5*(qnorm(correct_rate) + qnorm(lead(error_rate))), NA )) %>%
  select(subject, block, a_bias) %>% filter(!is.na(a_bias)) %>%
  group_by(subject) %>%
  summarise(a_bias = mean(a_bias)) %>%
  mutate(a_bias = a_bias - mean(a_bias))

a_bias_file <- paste(path_to_rt, 'a_bias.tsv', sep = '/')
write.table(a_bias,
            file = a_bias_file,
            sep = '\t')

pbi_rt <- corrects %>%
  group_by(subject, block, probe) %>%
  filter(probe == 'BX' | probe == 'AY') %>%
  group_by(subject, block) %>%
  mutate(pbi_rt = ifelse(probe == 'AY',
                          (m_rt-lead(m_rt)) / (m_rt+lead(m_rt)), NA)) %>%
  select(subject, block, pbi_rt) %>% filter(!is.na(pbi_rt)) %>%
  group_by(subject) %>%
  summarise(pbi_rt = mean(pbi_rt)) %>%
  mutate(pbi_rt = pbi_rt - mean(pbi_rt))

pbi_file <- paste(path_to_rt, 'pbi.tsv', sep = '/')
write.table(pbi_rt,
            file = pbi_file,
            sep = '\t')