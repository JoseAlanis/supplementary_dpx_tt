# --- author: jose c. garcia alanis
# --- encoding: utf-8
# --- r version: 3.5.1 (2018-07-02) -- "Feather Spray"
# --- script version: Jan 2018
# --- content: analysis of correct reactions RT


# --- 1) Run first, then move on to anylsis section ----------------------------
# Set working directory

# Interactive setting of working directory
set_path <- function() {
  
  x <- readline("Working on a macOS? ")
  
  if (length(grep(pattern = 'y', x = x)) > 0) {
    
    answ <- readline('is input path  "/Volumes/TOSHIBA/manuscrips_and_data/dpx_tt/" correct? ')
    
    if (length(grep(pattern = 'y', x = answ)) > 0) {
      setwd("/Volumes/TOSHIBA/manuscrips_and_data/dpx_tt/") 
    } else {
      path <- readline('Set input path: ')
      setwd(path)
    } 
    
  } else {
    
    answ <- readline('is input path  "D:/manuscrips_and_data/dpx_tt/" correct? ')
    if (length(grep(pattern = 'y', answ)) > 0) {
      setwd("D:/manuscrips_and_data/dpx_tt/") 
    } else {
      path <- readline('Set input path: ')
      setwd(path)
      
    }
    
  }
}

# Set working deirectory
set_path()


# --- 2) Define workflow funktions  --------------------------------------------
# Source function for fast requiring and installing packages
source('./r_functions/getPacks.R')
source('./r_functions/stdResid.R')

# # If missing, get it from jose's gists
# devtools::source_gist('https://gist.github.com/JoseAlanis/86da75bf223e10344b7c16791f45bafe', 
#                       filename = 'getPacks.R')

# Standard error function (controls for NAs)
stderr <- function(x) sqrt(var(x,na.rm=TRUE)/length(na.omit(x)))

# Semi-partion R2
R2 <- function(model) {
  anov_mod <- as.data.frame(model)
  
  rr <- ((anov_mod$NumDF / anov_mod$DenDF) * anov_mod[, grepl(names(anov_mod), pattern = 'F*value')] ) / 
    (1+((anov_mod$NumDF / anov_mod$DenDF) * anov_mod[, grepl(names(anov_mod), pattern = 'F*value')] ))
  
  print(rr)
  
}


# --- 3) Load R packages needed for analyses -----------------------------------
# clean up
rm(set_path)

# Load packages necessary for analysis
getPacks(c('dplyr',
           'ggplot2', 'viridis', 'gridExtra', 'sjPlot',
           'psych'))


# --- 4) Import behavioral data ------------------------------------------------
# Get data
load('./data_frames/corrects_rt.Rda')

# Set order of levels of factor block
corrects$block <- factor(corrects$block, 
                         levels = c("Practice", "Performance"))


# --- 5) Inspect distribution --------------------------------------------------
# Plot RT
ggplot(corrects, aes(x = rt, fill = trial_type)) +
  geom_histogram(binwidth=1, 
                 position = position_dodge(.1)) + 
  facet_wrap(~ trial_type) +
  coord_cartesian(xlim = c(0, 800)) +
  scale_fill_viridis(option = 'B', discrete = T)

# Plot RT (dessity)
rt_dens <- ggplot(corrects, aes(x = rt, fill = trial_type)) + 
  geom_density(alpha=.3) + 
  facet_wrap(~ trial_type) +
  coord_cartesian(xlim = c(0, 800), ylim = c(0, .008)) +
  scale_fill_viridis(option = 'B', discrete = T) +
  labs(title = 'Raw-RT Distribution', x = 'Reaction Time (ms)', y = 'Density', fill = 'Trial Type') + 
  theme(plot.title = element_text(color = 'black', face = 'bold', size = 13,
                                  margin = margin(b = 15), hjust = .5),
        strip.text = element_text(color = 'black', face = 'bold', size = 13),
        axis.title.y = element_text(color = 'black', face = 'bold', size = 13,
                                    margin = margin(r = 15)),
        axis.title.x = element_text(color = 'black', face = 'bold', size = 13,
                                    margin = margin(t = 15)),
        axis.text = element_text(color = 'black', size = 12),
        legend.title = element_text(color = 'black', face = 'bold', size = 13)); rt_dens
# Save plot
ggsave(rt_dens, filename = './results/figs/rt_raw_dens.pdf',
       width = 8, height = 5)

# Min and max reaction time
min_max_rt <- corrects %>%
  group_by(block, trial_type) %>% 
  summarise(mean = mean(rt), 
            se = stderr(rt), 
            min = min(rt), 
            max = max(rt)); min_max_rt
# Save table
tab_df(data.frame(mutate_if(min_max_rt, is.numeric, round, 2)), 
       file = './results/tables/min_max_rt.html')

# Number of correct trials per subject
n_corrects <- corrects %>% 
  group_by(id, block, trial_type) %>% 
  summarise(N = sum(!is.na(rt))); n_corrects
# Save table
tab_df(data.frame(n_corrects), file = './results/tables/n_corrects.html')


# --- 6) Winsorise RT ----------------------------------------------------------
# Winsorise
corrects <- corrects %>% 
  group_by(id, block, trial_type) %>%
  mutate(win_rt = winsor(rt, trim = .1))

# Plot adjusted RT
ggplot(corrects, aes(x = win_rt, fill = trial_type)) +
  geom_histogram(binwidth=1, 
                 position = position_dodge(.1)) + 
  facet_wrap(~ trial_type) +
  scale_fill_viridis(option = 'B', discrete = T) +
  coord_cartesian(xlim = c(0, 800)) +
  theme(axis.title = element_text(size = 13))

# Plot adjusted density
rt_win_dens <- ggplot(corrects, aes(x = win_rt, fill = trial_type)) + 
  geom_density(alpha=.3) + 
  facet_wrap(~ trial_type) +
  coord_cartesian(xlim = c(0, 800), ylim = c(0, .008)) +
  scale_fill_viridis(option = 'B', discrete = T) +
  labs(title = 'Winsorised RT Distribution', x = 'Reaction Time (ms)', y = 'Density', fill = 'Trial Type') + 
  theme(plot.title = element_text(color = 'black', face = 'bold', size = 13,
                                  margin = margin(b = 15), hjust = .5),
        strip.text = element_text(color = 'black', face = 'bold', size = 13),
        axis.title.y = element_text(color = 'black', face = 'bold', size = 13,
                                    margin = margin(r = 15)),
        axis.title.x = element_text(color = 'black', face = 'bold', size = 13,
                                    margin = margin(t = 15)),
        axis.text = element_text(color = 'black', size = 12),
        legend.title = element_text(color = 'black', face = 'bold', size = 13)); rt_win_dens
# Save plot
ggsave(rt_win_dens, filename = './results/figs/rt_win_dens.pdf',
       width = 8, height = 5)

# Plot raw and adjusted distributions side by side
pdf(file = './results/figs/rt_distribution.pdf', height = 4, width = 12)
par(mfrow=c(1,2))
hist(corrects$rt, 
     breaks = 60, 
     xlim = c(0, 800), 
     ylim = c(0, 500), 
     main = 'RT (raw)', xlab = 'RT [ms]')
hist(corrects$win_rt, 
     breaks = 60, 
     xlim = c(0, 800), 
     ylim = c(0, 500), 
     main = 'RT (winsor)', xlab = 'RT [ms]')
dev.off()


# --- 7) Analyse single trials' RT ---------------------------------------------
# # Only keep trials with RTs > 100
# corrects <- corrects %>% filter(rt >= 100)

# Set order of levels of factor block
corrects$block <- factor(corrects$block, levels = c("Performance", "Practice"))

# Effect code cathegorical variables
contrasts(corrects$block) <-  contr.sum(2); contrasts(corrects$block)
contrasts(corrects$trial_type) <-  contr.sum(4); contrasts(corrects$trial_type)

# -- Change default contrasts options ! --
options(contrasts = c("contr.sum","contr.poly"))

# Require packages for analysis
getPacks(c('lme4', 'lmerTest',
           'sjPlot',
           'emmeans'))

# Model single trials' RT
mod_rt0 <- lmer(data = corrects,
               win_rt ~ block * trial_type + (1|id))
#anova(mod_rt0, ddf = 'Kenward-Roger')
anova(mod_rt0)
plot_model(mod_rt0, 'diag')

# Model single trials' log RT
log_mod_rt0 <- lmer(data = corrects,
                    log(win_rt) ~ block * trial_type + (1|id))
#anova(log_mod_rt0, ddf = 'Kenward-Roger')
anova(log_mod_rt0)
plot_model(log_mod_rt0, 'diag')

# Compare models
anova(mod_rt0, log_mod_rt0) # LogRT model has better fit


# --- 8) Analyse aggregated RT -------------------------------------------------
# --- USE AGGREGATED RT TO REDUCE COMPUTATIONAL LOAD ---

# Summarise RT by id, block and trial type
corrects_mean <- corrects %>% 
  group_by(id, block, trial_type) %>% 
  summarise(mean_rt = mean(win_rt))

# Set order of levels of factor block
corrects_mean$block <- factor(corrects_mean$block, 
                              levels = c("Performance", "Practice"))

# Effect code cathegorical variables
contrasts(corrects_mean$block) <-  contr.sum(2); contrasts(corrects_mean$block)
contrasts(corrects_mean$trial_type) <-  contr.sum(4); contrasts(corrects_mean$trial_type)

# Require packages for analysis
getPacks(c('lme4', 'lmerTest',
           'sjPlot',
           'emmeans'))

# Model aggregated RT + random intercept for subjects
mod_agg_rt0 <- lmer(data = data.frame(corrects_mean),
                log(mean_rt) ~ block * trial_type + (1|id))
anova(mod_agg_rt0, ddf = 'Kenward-Roger')
plot_model(mod_rt0, 'diag')

# Model aggregated RT + random intercept for subjects + random interceüt for trial types
mod_agg_rt1 <- lmer(data = data.frame(corrects_mean),
                    log(mean_rt) ~ trial_type * block + (1|id/trial_type))
anova(mod_agg_rt1, ddf = 'Kenward-Roger')
# Model summary
summary(mod_agg_rt1, ddf = 'Kenward-Roger')
# Model diagnostics
plot_model(mod_agg_rt1, 'diag')

# Compare models
anova_mods <- anova(mod_agg_rt0, mod_agg_rt1); anova_mods # By subject and trial type random effects have better fit

# Save results of model comparissons
tab_df(data.frame(attributes(anova_mods)$heading),
       title = 'Comparison of models with different random structure for log-RT',
       file = './results/tables/mod_comparisons_rt_models.html')
tab_df(round(data.frame(anova_mods), digits = 3),
       file = './results/tables/mod_comparisons_rt_estimates.html')

# --- 9) Remove outliers (optional) --------------------------------------------
# --- REFIT MODEL WITHOUT OUTLIERS (RESULTS DON'T CHANGE MUCH)
dat_rm <- stdResid(data = data.frame(corrects_mean),
         model = mod_agg_rt1,
         return.data = T,
         plot = T,
         show.bound = T)

# Model refitted without outliers
corrects_no_out <- filter(dat_rm, Outlier == 0)
mod_agg_rt1 <- lmer(data = corrects_no_out,
                    log(mean_rt) ~ trial_type * block + (1|id/trial_type))
# Anova table for final model
anova(mod_agg_rt1, ddf='Kenward-Roger')
# Regression table for model
summary(mod_agg_rt1, ddf='Kenward-Roger')
# model diagnostics for final model
plot_model(mod_agg_rt1, 'diag') 
# forest plot for standardised erstimates
std_est <- plot_model(mod_agg_rt1, 'std2', order.terms = c(7:1)); std_est


# --- 10) Compute sumary statistics for final model ----------------------------
# Compute effect sizes (semi partial R2)
amod <- anova(mod_agg_rt1, ddf = 'Kenward-Roger'); amod
amod <-  as.data.frame(amod); amod
amod$sp.R2 <- R2(amod); amod

# Save anova table
tab_df(round(amod, 5), 
       title = 'Anova results for linear mixed effects regression analysis of log-RT',
       file = './results/tables/anova_rt.html')

# Save model summary
tab_model(mod_agg_rt1,
          title = 'Model estimates for linear mixed effects regression analysis of log-RT',
          file = './results/tables/summary_rt.html')


# Descriptives
mean_rt_0 <- corrects_mean %>%
  group_by(block) %>% 
  summarise(mean = mean(mean_rt), 
            sd = sd(mean_rt)); mean_rt_0
# Descriptives
mean_rt_1 <- corrects_mean %>%
  group_by(trial_type) %>% 
  summarise(mean = mean(mean_rt), 
            sd = sd(mean_rt)); mean_rt_1

# Descriptives
mean_rt_2 <- corrects_mean %>%
  group_by(block, trial_type) %>% 
  summarise(mean = mean(mean_rt), 
            sd = sd(mean_rt)); mean_rt_2

# Save table
tab_df(data.frame(mutate_if(mean_rt_0, is.numeric, round, 2)), 
       file = './results/tables/decriptives_rt_block.html')
tab_df(data.frame(mutate_if(mean_rt_1, is.numeric, round, 2)), 
       file = './results/tables/decriptives_rt_trial_type.html')
tab_df(data.frame(mutate_if(mean_rt_2, is.numeric, round, 2)), 
       file = './results/tables/decriptives_rt_trial_type_by_block.html')


# --- 11) Interaction analysis  ------------------------------------------------
# Quick interaction plot
emmip(mod_agg_rt1,  block ~ trial_type, CIs = T, type = 'response')

# Pairwise contrasts
# By trial type
tt_means <- emmeans(mod_agg_rt1,  pairwise ~ trial_type, 
                    lmer.df = 'kenward-roger', 
                    adjust = 'holm', 
                    transform = 'response'); tt_means
confint(tt_means)
# By trial type
emmeans(mod_agg_rt1,  pairwise ~ block,
        lmer.df = 'kenward-roger', 
        adjust = 'holm', 
        transform = 'response')

# Interaction between block by trial type
b_by_tt <- emmeans(mod_agg_rt1,  
                       pairwise ~ block | trial_type, 
                       lmer.df = 'kenward-roger', 
                       adjust = 'holm',
                       transform = 'response'); tt_by_block
# CIs
confint(b_by_tt)

# Interaction between trial type by block
tt_by_block <- emmeans(mod_agg_rt1,  
                       pairwise ~ trial_type | block, 
                       lmer.df = 'kenward-roger', 
                       adjust = 'holm',
                       transform = 'response')
# CIs
confint(tt_by_block)

# Save means for plot
rt_emmeans <- emmeans(mod_agg_rt1,  pairwise ~ block | trial_type,
                      transform = 'response', 
                      lmer.df = 'kenward-roger', adjust='fdr'); rt_emmeans


# --- Create interaction plot ---
pd = position_dodge(.25)
rt_est <- ggplot(data.frame(rt_emmeans$emmeans), 
                aes(x = trial_type, y = response, 
                    color = block, group = block, shape = block)) +
  geom_line(position = pd, size = 1) +
  geom_errorbar(aes(ymin = lower.CL, ymax = upper.CL), 
                position = pd,  width = .15, 
                size = 0.8, linetype = 1, color = 'black') +
  geom_linerange(aes(ymin = response-SE, ymax = response+SE), 
                 position = pd, size = 2) +
  geom_point(position = pd, size = 3, color = 'black') +
  
  coord_cartesian(ylim = c(250, 550)) +
  scale_y_continuous(breaks = c(250, 300, 350, 400, 450, 500, 550)) +
  
  scale_color_viridis(option = 'D', begin = .4, discrete = T) +
  
  geom_segment(aes(x = -Inf, y = 250, xend = -Inf, yend = 550), 
               color = 'black', size = rel(1), linetype = 1) +
  geom_segment(aes(x = 'AX', y = -Inf, xend = 'BY', yend = -Inf), 
               color = 'black', size = rel(1), linetype = 1) +
  
  labs(title = 'C) Back tranformed marginal means for Log-RT model', 
       x = 'Levels of Trial Type', 
       y = 'Estimtaed RT [ms]', 
       color = 'Task Block', shape = 'Task Block') + 
  
  theme_classic() +
  theme(plot.title = element_text(color = 'black', size = 13, face = 'bold',
                                  margin = margin(b = 15)),
        axis.line = element_blank(),
        axis.title.x = element_text(color = 'black', size = 13, face = 'bold', 
                                    margin = margin(t = 15)),
        axis.title.y = element_text(color = 'black', size = 13, face = 'bold', 
                                    margin = margin(r = 15)),
        axis.text = element_text(color = 'black', size = 12),
        legend.position = 'bottom',
        legend.direction = 'horizontal',
        legend.text = element_text(size = 11),
        legend.title = element_text(size = 12, face = 'bold'),
        legend.key.size = unit(.6, 'cm')); rt_est

# Save plot
ggsave(rt_est, filename = './results/figs/emmeans_rt.pdf',
       width = 6, height = 7)


# Plot means used for analysis
vio_rt <- ggplot(corrects_mean, 
       aes(x = trial_type, y = mean_rt, 
           group = interaction(trial_type, block), fill = block)) + 
  
  geom_violin(trim = FALSE, position = position_dodge(.6), scale = 'area') +
  geom_boxplot(width = 0.1, position = position_dodge(.6), fill = 'white') +
  
  geom_segment(aes(x = -Inf, y = 0, xend = -Inf, yend = 800), 
               color = 'black', size = rel(1), linetype = 1) +
  geom_segment(aes(x = 'AX', y = -Inf, xend = 'BY', yend = -Inf), 
               color = 'black', size = rel(1), linetype = 1) +
  
  labs(title = 'B) Distribution of RT means',
       fill = 'Task Block',
       y = 'Mean RT [ms]', 
       x = 'Levels of Trial Type') + 
  
  scale_fill_viridis(option = 'D', begin = .4, discrete = T, alpha = .9) +
  scale_y_continuous(breaks = c(0, 200, 400, 600, 800)) +
  
  theme_classic() + 
  theme(axis.line = element_blank(),
        plot.title = element_text(color = 'black', size = 13, face = 'bold',
                                  margin = margin(b = 15)),
        axis.title.x = element_text(color = 'black', size = 13, face = 'bold',
                                    margin = margin(t = 15)),
        axis.title.y = element_text(color = 'black', size = 13, face = 'bold',
                                    margin = margin(r = 15)),
        axis.text = element_text(color = 'black', size = 12),
        legend.position = 'bottom',
        legend.direction = 'horizontal',
        legend.text = element_text(size = 11),
        legend.title = element_text(size = 12, face = 'bold'),
        legend.key.size = unit(.6, 'cm')); vio_rt

# Save plot
ggsave(vio_rt, filename = './results/figs/vio_rt.pdf',
       width = 8.5, height = 5)

# --- 12) Final figure for submission ------------------------------------------
# Standardised model estimates
mod_est <- std_est + geom_hline(yintercept = 0, linetype = 2) +
  scale_y_continuous(limits = c(-1, 1.2)) +
  scale_x_discrete(labels = c('AX', 'AY', 'BX', 
                              'Perfomance', 'AX:Perfomance',
                              'AY:Perfomance', 'BX:Perfomance')) +
  scale_color_viridis(option = 'A', discrete = T, direction = -1, end = .55) +
  
  geom_segment(aes(x = -Inf, y = -1, xend = -Inf, yend = 1), 
               color = 'black', size = rel(1), linetype = 1) +
  geom_segment(aes(x = 1, y = -Inf, xend = 7, yend = -Inf), 
               color = 'black', size = rel(1), linetype = 1) +
  
  labs(title = 'A) Standardised beta-weights for Log-RT model',
       x = 'Predictors',
       y = 'Estimates') + 
  
  theme_classic() + 
  theme(plot.title = element_text(color = 'black', size = 13, face = 'bold',
                                  margin = margin(b = 15)),
        axis.line = element_blank(),
        axis.title.x = element_text(color = 'black', face = 'bold', size = 13,
                                    margin = margin(t = 15)),
        axis.title.y = element_text(color = 'black', face = 'bold', size = 13,
                                    margin = margin(r = 15)),
        axis.text.x = element_text(color = 'black', size = 12),
        axis.text.y = element_text(color = 'black', size = 11, 
                                   margin = margin(r = 5))); mod_est
# Save plot
ggsave(mod_est, filename = './results/figs/mod_rt_est.pdf',
       width = 8, height = 4)

# Create margins for each plot in fugure
margin_1 = theme(plot.margin = unit(c(1.25, 0.25, 1.25, .25), "cm"))
margin_2 = theme(plot.margin = unit(c(0, 0.25, 1, 2.3), "cm"))
margin_3 = theme(plot.margin = unit(c(1.25, 0.25, 1, 0.5), "cm"))


# Arrange plot in single figure
fig_rt <- grid.arrange(grobs = list(mod_est + margin_1,
                                    vio_rt + margin_2, 
                                    rt_est + margin_3), 
                       layout_matrix = rbind(c(1,1,3,3),
                                             c(2,2,3,3))); fig_rt

# Save figure
ggsave(fig_rt, filename = './results/figs/fig_rt.pdf',
       width = 13, height = 8)
