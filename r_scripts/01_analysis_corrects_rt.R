# --- author: jose C. garcia alanis
# --- encoding: utf-8
# --- r version: 3.4.4 (2018-03-15)
# --- script version: Wed Nov 14 12:02:01 2018"
# --- content: analyse behavioral data


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
  
  rr <- ((anov_mod$NumDF / anov_mod$DenDF) * anov_mod$`F.value`) / 
    (1+((anov_mod$NumDF / anov_mod$DenDF) * anov_mod$`F.value`))
  
  print(rr)
  
}


# --- 3) Load R packages needed for analyses -----------------------------------
# clean up
rm(set_path)

# Packages
packs <- c('dplyr',
           'ggplot2', 'viridis',
           'sjPlot')
# Load them
getPacks(packs)


# --- 4) Import behavioral data ------------------------------------------------
# Get data
load('./data_frames/corrects_rt.Rda')


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
  scale_fill_viridis(option = 'B', discrete = T); rt_dens
# Save plot
ggsave(rt_dens, filename = './results/figs/rt_raw_dens.pdf',
       width = 7, height = 5)

# Min and max reaction time
min_max_rt <- corrects %>%
  group_by(block, trial_type) %>% 
  summarise(mean = mean (rt), 
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
  scale_fill_viridis(option = 'B', discrete = T); rt_win_dens
# Save plot
ggsave(rt_win_dens, filename = './results/figs/rt_win_dens.pdf',
       width = 7, height = 5)

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


# --- 7) Analyse RT ------------------------------------------------------------
# # Only keep trials with RTs > 100
# corrects <- corrects %>% filter(rt >= 100)

corrects_mean <- corrects %>% 
  group_by(id, block, trial_type) %>% 
  summarise(mean_rt = mean(win_rt))

# Effect code cathegorical variables
contrasts(corrects$block) <-  contr.sum(2); contrasts(corrects$block)
contrasts(corrects$trial_type) <-  contr.sum(4); contrasts(corrects$trial_type)

# Require packages for analysis
packs <- c('lme4', 'lmerTest',
           'sjPlot',
           'emmeans'); getPacks(packs)

# Model single trials' RT
mod_rt0 <- lmer(data = corrects,
               win_rt ~ block * trial_type + (1+trial_type+block|id))
anova(mod_rt0, ddfs = 'Kenward-Roger')
plot_model(mod_rt0, 'diag')

# Model single trials' log RT
log_mod_rt0 <- lmer(data = corrects,
                    log(win_rt) ~ block * trial_type + (1+trial_type+block|id))
anova(log_mod_rt0, ddf = 'Kenward-Roger')
plot_model(log_mod_rt0, 'diag')

# Compare models
anova(mod_rt0, log_mod_rt0)

# Model aggregated RT
mod_rt0 <- lmer(data = corrects_mean,
                mean_rt ~ block * trial_type + (1+trial_type+block|id))
anova(mod_rt0, ddf = 'Kenward-Roger')
plot_model(mod_rt0, 'diag')

# Model aggregated log RT
log_mod_rt0 <- lmer(data = corrects_mean,
                    log(mean_rt) ~ block * trial_type + (1+trial_type+block|id))
anova(log_mod_rt0, ddf = 'Kenward-Roger')
plot_model(log_mod_rt0, 'diag')

# # UNCOMMENT TO REFIT MODEL WITHOUT OUTLIERS (RESULTS DON'T CHANGE)
# dat_rm <- stdResid(data = data.frame(corrects_mean), 
#          model = log_mod_rt0, 
#          return.data = T, 
#          plot = T,
#          show.bound = T)
# 
# log_mod_rt0 <- lmer(data = filter(dat_rm, Outlier == 0),
#                     log(mean_rt) ~ block * trial_type + (1+trial_type+block|id))
# anova(log_mod_rt0, ddf='Kenward-Roger')
# plot_model(log_mod_rt0, 'diag')

# Compare models
anova(mod_rt0, log_mod_rt0)

# Compute effect sizes (semi partial R2)
amod <- anova(log_mod_rt0, ddf = 'Kenward-Roger'); amod
amod <-  as.data.frame(amod); amod
amod$sp.R2 <- R2(amod); amod

# Regression table for model
summary(log_mod_rt0)

# Save anova table
tab_df(round(amod, 4), 
       title = 'Anova table for linear mixed effects regression analysis of logRT',
       file = './results/tables/anova_rt.html')


# --- 8) Plot results ------------------------------------------------------------
# Quick interaction plot
emmip(log_mod_rt0,  block ~ trial_type, CIs = T, type = 'response')

# Pairwise contrasts
emmeans(log_mod_rt0,  pairwise ~ trial_type | block, 
        transform = 'response', lmer.df = 'kenward-roger')
emmeans(log_mod_rt0,  pairwise ~ block | trial_type, 
        transform = 'response', lmer.df = 'kenward-roger')

# Save means for plot
rt_emmeans <- emmeans(log_mod_rt0,  pairwise ~ block | trial_type,
                      transform = 'response', 
                      lmer.df = 'kenward-roger'); rt_emmeans


# Create interaction plot
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
  
  labs(title = 'RT Model estimates (marginal means)', 
       x = 'Levels of Trial Type', 
       y = 'Estimtaed RT [ms]', 
       color = 'Block', shape = 'Block') + 
  
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
        legend.text = element_text(size = 12),
        legend.title = element_text(size = 13),
        legend.key.size = unit(1, 'cm')); rt_est
# Save plot
ggsave(rt_est, filename = './results/figs/emmeans_rt.pdf',
       width = 6, height = 6)


# Plot means used for analysis
vio_rt <- ggplot(corrects_mean, 
       aes(x = trial_type, y = mean_rt, 
           group = interaction(trial_type, block), fill = block)) + 
  
  geom_violin(trim = FALSE, position = position_dodge(.6), scale = 'area') +
  geom_boxplot(width = 0.1, position = position_dodge(.6), fill = 'white') +
  
  geom_segment(aes(x = -Inf, y = 100, xend = -Inf, yend = 800), 
               color = 'black', size = rel(1), linetype = 1) +
  geom_segment(aes(x = 'AX', y = -Inf, xend = 'BY', yend = -Inf), 
               color = 'black', size = rel(1), linetype = 1) +
  
  labs(title = 'Distribution of RT means',
       fill = 'Task Block',
       y = 'Mean RT [ms]', 
       x = 'Levels of Trial Type') + 
  
  scale_fill_viridis(option = 'D', begin = .4, discrete = T, alpha = .9) +
  scale_y_continuous(breaks = c(100, 200, 300, 400, 500, 600, 700, 800)) +
  
  theme_classic() + 
  theme(axis.line = element_blank(),
        plot.title = element_text(color = 'black', size = 13, face = 'bold',
                                  margin = margin(b = 15)),
        legend.title = element_text(color = 'black', size = 13, face = 'bold'),
        legend.text = element_text(color = 'black', size = 12),
        axis.title.x = element_text(color = 'black', size = 13, face = 'bold',
                                    margin = margin(t = 15)),
        axis.title.y = element_text(color = 'black', size = 13, face = 'bold',
                                    margin = margin(r = 15)),
        axis.text = element_text(color = 'black', size = 12)); vio_rt
# Save plot
ggsave(vio_rt, filename = './results/figs/vio_rt.pdf',
       width = 8.5, height = 5)
