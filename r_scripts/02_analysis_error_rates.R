# --- author: jose C. garcia alanis
# --- encoding: utf-8
# --- r version: 3.4.4 (2018-03-15)
# --- script version: Mon Nov 19 13:10:42 2018
# --- content: analysis of errors (behavioral data)


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

# Set working directory
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
           'ggplot2', 'viridis',
           'sjPlot', 'gridExtra'))


# --- 4) Import behavioral data ------------------------------------------------
# Get data
load('./data_frames/errors.Rda')


# --- 5) Inspect distribution --------------------------------------------------
# Histogram of error rates frequency
hist(errors$error_rate)

# Plot error rates
ggplot(errors, aes(x = error_rate, fill = trial_type)) +
  geom_histogram(binwidth=0.01, 
                 position = position_dodge(.1)) + 
  facet_wrap(~ trial_type) +
  scale_fill_viridis(option = 'B', discrete = T)

# Plot error rates
ggplot(errors, aes(x = error_rate, fill = block)) +
  geom_histogram(binwidth=0.01, 
                 position = position_dodge(.1)) + 
  facet_wrap(~ block) +
  scale_fill_viridis(option = 'B', discrete = T, end = 0.6)

# Min and max error rates
min_max_error_rate <- errors %>%
  group_by(block, trial_type) %>% 
  summarise(mean = mean(error_rate), 
            se = stderr(error_rate), 
            min = min(error_rate), 
            max = max(error_rate)); min_max_error_rate
# Save table
tab_df(data.frame(mutate_if(min_max_error_rate, is.numeric, round, 3)), 
       file = './results/tables/min_max_error_rates.html')

# desciptives errors RT
mean(errors$err_rt, na.rm = T)
sd(errors$err_rt, na.rm = T)


# --- 6) Analyse error rates ---------------------------------------------------
# Set order of levels of factor block
errors$block <- factor(errors$block, levels = c("Performance", "Practice"))

# Effect code cathegorical variables
contrasts(errors$block) <-  contr.sum(2); contrasts(errors$block)
contrasts(errors$trial_type) <-  contr.sum(4); contrasts(errors$trial_type)

# Load packages for analysis
getPacks(c('lme4', 'lmerTest', 'sjPlot', 'emmeans'))

# -- Change default contrasts options ! --
options(contrasts = c("contr.sum","contr.poly"))

# Model error rates
mod_err <- lmer(data = errors,
                log(error_rate) ~ trial_type * block + (1|id/trial_type))
# model summary
summary(mod_err, ddf = 'Kenward-Roger')
# ANOVA table
anova(mod_err, ddf = 'Kenward-Roger')

# --- 7) Remove outliers (optional) --------------------------------------------
# --- REFIT MODEL WITHOUT OUTLIERS (RESULTS DON'T CHANGE MUCH)
dat_rm <- stdResid(errors, mod_err, plot = T, show.bound = T)

# Model refitted without outliers
errors_no_out <- filter(dat_rm, Outlier == 0)
mod_err <- lmer(data = errors_no_out,
                log(error_rate) ~ trial_type * block + (1|id/trial_type))
# model summary
summary(mod_err, ddf = 'Kenward-Roger')
# ANOVA table
anova(mod_err, ddf = 'Kenward-Roger')
# model duiagnostics
plot_model(mod_err, 'diag')
# forest plot for standardised erstimates
std_est <- plot_model(mod_err, 'std2', order.terms = c(7:1)); std_est


# --- 8) Compute sumary statistics for final model -----------------------------
# Compute effect sizes (semi partial R2)
amod <- anova(mod_err, ddf = 'Kenward-Roger'); amod
amod <-  as.data.frame(amod); amod
amod$sp.R2 <- R2(amod); amod

# Save anova table
tab_df(round(amod, 5), 
       title = 'Anova results for linear mixed effects regression analysis of log-error-rates',
       file = './results/tables/anova_error_rates.html')


# Descriptives
mean_err_0 <- errors %>%
  group_by(block) %>% 
  summarise(mean = mean(error_rate), 
            sd = sd(error_rate)); mean_err_0
# Descriptives
mean_err_1 <- errors %>%
  group_by(trial_type) %>% 
  summarise(mean = mean(error_rate), 
            sd = sd(error_rate)); mean_err_1

# Descriptives
mean_err_2 <- errors %>%
  group_by(block, trial_type) %>% 
  summarise(mean = mean(error_rate), 
            sd = sd(error_rate)); mean_err_2

# Save table
tab_df(data.frame(mutate_if(mean_err_0, is.numeric, round, 2)), 
       file = './results/tables/decriptives_errors_block.html')
tab_df(data.frame(mutate_if(mean_err_1, is.numeric, round, 2)), 
       file = './results/tables/decriptives_errors_trial_type.html')
tab_df(data.frame(mutate_if(mean_err_2, is.numeric, round, 2)), 
       file = './results/tables/decriptives_errors_trial_type_by_block.html')

# --- 9) Interaction analysis --------------------------------------------------
# Quick interaction plot
emmip(mod_err,  block ~ trial_type, CIs = T, type = 'response')

# Pairwise contrasts
# By trial type
tt_means <- emmeans(mod_err,  pairwise ~ trial_type, 
                    lmer.df = 'kenward-roger', 
                    adjust = 'holm', 
                    transform = 'response'); tt_means
# CIs
confint(tt_means)

# By block
emmeans(mod_err,  pairwise ~ block,
        lmer.df = 'kenward-roger', 
        adjust = 'holm', 
        transform = 'response')

# Interaction between block by trial type
b_by_tt <- emmeans(mod_err,  
                   pairwise ~ block | trial_type, 
                   lmer.df = 'kenward-roger', 
                   adjust = 'holm',
                   transform = 'response'); b_by_tt
# CIs
confint(b_by_tt)

# Interaction between trial type by block
tt_by_block <- emmeans(mod_err,  
                       pairwise ~ trial_type | block, 
                       lmer.df = 'kenward-roger', 
                       adjust = 'holm',
                       transform = 'response')
# CIs
confint(tt_by_block)

# Save means for plot
err_emmeans <- emmeans(mod_err,  pairwise ~ block | trial_type,
                      transform = 'response', 
                      lmer.df = 'kenward-roger', adjust = 'holm'); err_emmeans


# --- Create interaction plot ---
pd = position_dodge(.25)
err_est <- ggplot(data.frame(err_emmeans$emmeans), 
                 aes(x = trial_type, y = response, 
                     color = block, group = block, shape = block)) +
  geom_line(position = pd, size = 1) +
  geom_errorbar(aes(ymin = lower.CL, ymax = upper.CL), 
                position = pd,  width = .15, 
                size = 0.8, linetype = 1, color = 'black') +
  geom_linerange(aes(ymin = response-SE, ymax = response+SE), 
                 position = pd, size = 2) +
  geom_point(position = pd, size = 3, color = 'black') +
  
  coord_cartesian(ylim = c(0, 0.25)) +
  scale_y_continuous(breaks = c(0, 0.05, 0.1, 0.15, 0.2, 0.25)) +
  
  scale_color_viridis(option = 'D', begin = .4, discrete = T) +
  
  geom_segment(aes(x = -Inf, y = 0, xend = -Inf, yend = .25), 
               color = 'black', size = rel(1), linetype = 1) +
  geom_segment(aes(x = 'AX', y = -Inf, xend = 'BY', yend = -Inf), 
               color = 'black', size = rel(1), linetype = 1) +
  
  labs(title = 'C) Back tranformed marginal means for error-rates model', 
       x = 'Levels of Trial Type', 
       y = 'Estimtaed error-rate', 
       color = 'Condition', shape = 'Condition') + 
  
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
        legend.key.size = unit(.6, 'cm')); err_est

# Save plot
ggsave(err_est, filename = './results/figs/emmeans_err.pdf',
       width = 6, height = 7)


# Plot means used for analysis
vio_err <- ggplot(errors, 
                 aes(x = trial_type, y = error_rate, 
                     group = interaction(trial_type, block), fill = block)) + 
  
  geom_violin(trim = FALSE, position = position_dodge(.6), scale = 'area') +
  geom_boxplot(width = 0.1, position = position_dodge(.6), fill = 'white') +
  
  scale_fill_viridis(option = 'D', begin = .4, discrete = T, alpha = .9) +
  scale_y_continuous(breaks = c(0, 0.2, 0.4, 0.6, 0.8, 1)) +
  
  geom_segment(aes(x = -Inf, y = 0, xend = -Inf, yend = 1),
               color = 'black', size = rel(1), linetype = 1) +
  geom_segment(aes(x = 'AX', y = -Inf, xend = 'BY', yend = -Inf),
               color = 'black', size = rel(1), linetype = 1) +
  
  labs(title = 'B) Distribution of error-rates',
       fill = 'Condition',
       y = 'Error-rate', 
       x = 'Levels of Trial Type') + 
  
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
        legend.key.size = unit(.6, 'cm')); vio_err

# Save plot
ggsave(vio_err, filename = './results/figs/vio_err.pdf',
       width = 8.5, height = 5)


# --- 10) Final figure for submission ------------------------------------------
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
  
  labs(title = 'A) Standardised beta-weights for error-rates model',
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
ggsave(mod_est, filename = './results/figs/mod_err_est.pdf',
       width = 8, height = 4)


# Create margins for each plot in fugure
margin_1 = theme(plot.margin = unit(c(1.25, 0.25, 1.25, .25), "cm"))
margin_2 = theme(plot.margin = unit(c(0, 0.25, 1, 2.3), "cm"))
margin_3 = theme(plot.margin = unit(c(1.25, 0.25, 1, 0.5), "cm"))

# Arrange plot in single figure
fig_err <- grid.arrange(grobs = list(mod_est + margin_1,
                                    vio_err + margin_2, 
                                    err_est + margin_3), 
                       layout_matrix = rbind(c(1,1,3,3),
                                             c(2,2,3,3))); fig_err

# Save figure
ggsave(fig_err, filename = './results/figs/fig_errors.pdf',
       width = 13, height = 8)
