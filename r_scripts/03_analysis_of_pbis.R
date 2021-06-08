# --- author: Jose C. Garcia Alanis
# --- encoding: utf-8
# --- r version: 3.4.4 (2018-03-15) -- "Someone to Lean On"
# --- content: analysis
# --- version: "Mon Sep 24 10:40:32 2018"



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
source('../r_functions/pkgcheck.R')
source('./r_functions/stdResid.R')

# # If missing, get it from jose's gists
# devtools::source_gist('https://gist.github.com/JoseAlanis/86da75bf223e10344b7c16791f45bafe', 
#                       filename = 'pkgcheck.R')

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


# --- 4) Import pci-data -------------------------------------------------------
# PCI = proactive control index, 
# in manuscript refered to as PBI = proactive behaviour index. 
# -- Get data --
pci <- read.table('./data_frames/pc_index.txt', header = T)

# Recode blokc variable
pci <- pci %>% mutate(block = ifelse(block == 1, 'Practice', 'Performance'))
# Set order of levels of factor block
pci$block <- factor(pci$block, 
                    levels = c("Practice", "Performance"))

# ID to factor
pci$id <- as.factor(pci$id)


# --- 5) Inspect distribution --------------------------------------------------
# histograms of each pci/pbi
hist(pci$dprime)
hist(pci$a_bias)
hist(pci$pbi_err)
hist(pci$pbi_rt)

# Save them to figure
# Open plotting device
pdf(file = './results/figs/pbi_distribution.pdf', height = 8, width = 12)
par(mfrow=c(2,2)) # set up plot layout
# dprime
hist(pci$dprime, 
     breaks = 20, 
     xlim = c(2, 5), 
     main = 'Histogramm of d prime', xlab = 'd prime')
# a-cue response-bias
hist(pci$a_bias,
      breaks = 20, 
      xlim = c(-0.5, 1.5), 
      main = 'Histogramm of A-cue response-bias', xlab = 'A-bias')
# PBI based on errors
hist(pci$pbi_err,
     breaks = 20, 
     xlim = c(-1, 1), 
     main = 'Histogramm of PBI based on errors', xlab = 'PBI errors')
# PBI based on RT
hist(pci$pbi_rt,
     breaks = 20, 
     xlim = c(0, 0.5), 
     main = 'Histogramm of PBI based on RT', xlab = 'PBI RT')
# close device
dev.off()


# --- 5.1) A-cue reponse bias --------------------------------------------------
# Visualise change from practice to perfomance block
# Select a_bias
a_bias_df <- select(pci, block, a_bias, id)
# code if a_bias goes up or down in performance block
a_bias_df <- a_bias_df %>% 
  group_by(id) %>% 
  mutate(up = ifelse(a_bias < lead(a_bias), 'up', 'down')) 
# replace NAs
a_bias_df <- a_bias_df %>% 
  mutate(up = ifelse(is.na(up), lag(up), up))

# -- Create plot a_bias -- 
change_abias <- ggplot(data = data.frame(a_bias_df)) +
  
  geom_violin(trim = F, inherit.aes = F, 
              aes(x = block, 
                  y = a_bias, 
                  fill = block), size = 0.8) +
  geom_point(inherit.aes = F,
             aes(x = block, 
                 y = a_bias, 
                 color = up, 
                 shape = up), size = 1.5) +
  geom_line(inherit.aes = F,
            aes(x = block, 
                y = a_bias, 
                group = id, 
                color = up,
                linetype = up), size = 0.8) +
  
  coord_cartesian(ylim = c(-1, 2)) +
  
  scale_fill_viridis(option = 'D', begin = .4, discrete = T, alpha = .25) +
  scale_color_viridis(option = 'B', begin = .1, end = .6, discrete = T, direction = -1) +
  
  labs(color = 'Change', linetype = 'Change', shape = 'Change',
       fill = 'Condition',
       x = 'Levels of condition', 
       y = bquote("0.5 x ( Correct "["AX"] ~ " - Incorrect "["AY"] ~ ")"), 
       title = 'A-cue response bias') +
  
  theme_classic() +

  geom_segment(aes(x = -Inf, y = -1, xend = -Inf, yend = 2), 
               color = 'black', size = rel(1), linetype = 1) +
  geom_segment(aes(x = 'Practice', y = -Inf, xend = 'Performance', yend = -Inf), 
               color = 'black', size = rel(1), linetype = 1) +
  
  theme(axis.line = element_blank(),
        plot.title = element_text(color = 'black', size = 13, face = 'bold',
                                  hjust = 0.5, margin = margin(b = 15)),
        axis.title.x = element_text(color = 'black', size = 13, face = 'bold',
                                    margin = margin(t = 15)),
        axis.title.y = element_text(color = 'black', size = 13, face = 'bold',
                                    margin = margin(r = 15)),
        axis.text = element_text(color = 'black', size = 12),
        legend.position = 'bottom',
        legend.direction = 'vertical',
        legend.text = element_text(size = 11),
        legend.title = element_text(size = 12, face = 'bold'),
        legend.key.size = unit(.6, 'cm')); change_abias
# Save plot
ggsave(change_abias, filename = './results/figs/change_abias.pdf',
       width = 5.5, height = 7)

# --- 5.2) d prime  ------------------------------------------------------------
# Visualise change from practice to perfomance block
# Select a_bias
dprime_df <- select(pci, block, dprime, id)
# code if a_bias goes up or down in performance block
dprime_df <- dprime_df %>% 
  group_by(id) %>% 
  mutate(up = ifelse(dprime < lead(dprime), 'up', 'down')) 
# replace NAs
dprime_df <- dprime_df %>% 
  mutate(up = ifelse(is.na(up), lag(up), up))

# -- Create plot -- 
change_dprime <- ggplot(data = data.frame(dprime_df)) +
  
  geom_violin(trim = F, inherit.aes = F, 
              aes(x = block, 
                  y = dprime, 
                  fill = block), size = 0.8) +
  geom_point(inherit.aes = F,
             aes(x = block, 
                 y = dprime, 
                 color = up, 
                 shape = up), size = 1.5) +
  geom_line(inherit.aes = F,
            aes(x = block, 
                y = dprime, 
                group = id, 
                color = up,
                linetype = up), size = 0.8) +
  
  coord_cartesian(ylim = c(0, 6)) +
  
  scale_fill_viridis(option = 'D', begin = .4, discrete = T, alpha = .25) +
  scale_color_viridis(option = 'B', begin = .1, end = .6, discrete = T, direction = -1) +
  
  labs(color = 'Change', linetype = 'Change', shape = 'Change',
       fill = 'Condition',
       x = 'Levels of condition', 
       # bquote("Hello" ~ r[xy] == .(cor) ~ "and" ~ B^2))
       y = bquote("0.5 x ( Correct "["AX"] ~ " - Incorrect "["AY"] ~ ")"), 
       title = "d' Context") +
  
  theme_classic() +
  
  geom_segment(aes(x = -Inf, y = 0, xend = -Inf, yend = 6), 
               color = 'black', size = rel(1), linetype = 1) +
  geom_segment(aes(x = 'Practice', y = -Inf, xend = 'Performance', yend = -Inf), 
               color = 'black', size = rel(1), linetype = 1) +
  
  theme(axis.line = element_blank(),
        plot.title = element_text(color = 'black', size = 13, face = 'bold',
                                  hjust = 0.5, margin = margin(b = 15)),
        axis.title.x = element_text(color = 'black', size = 13, face = 'bold',
                                    margin = margin(t = 15)),
        axis.title.y = element_text(color = 'black', size = 13, face = 'bold',
                                    margin = margin(r = 15)),
        axis.text = element_text(color = 'black', size = 12),
        legend.position = 'bottom',
        legend.direction = 'vertical',
        legend.text = element_text(size = 11),
        legend.title = element_text(size = 12, face = 'bold'),
        legend.key.size = unit(.6, 'cm')); change_dprime
# Save plot
ggsave(change_dprime, filename = './results/figs/change_dprime.pdf',
       width = 5.5, height = 7)


# --- 5.3) PBI errors  ---------------------------------------------------------
# Visualise change from practice to perfomance block
# Select a_bias
pbie_df <- select(pci, block, pbi_err, id)
# code if a_bias goes up or down in performance block
pbie_df <- pbie_df %>% 
  group_by(id) %>% 
  mutate(up = ifelse(pbi_err < lead(pbi_err), 'up', 'down')) 
# replace NAs
pbie_df <- pbie_df %>% 
  mutate(up = ifelse(is.na(up), lag(up), up))

# -- Create plot -- 
change_pbie <- ggplot(data = data.frame(pbie_df)) +
  
  geom_violin(trim = F, inherit.aes = F, 
              aes(x = block, 
                  y = pbi_err, 
                  fill = block), size = 0.8) +
  geom_point(inherit.aes = F,
             aes(x = block, 
                 y = pbi_err, 
                 color = up, 
                 shape = up), size = 1.5) +
  geom_line(inherit.aes = F,
            aes(x = block, 
                y = pbi_err, 
                group = id, 
                color = up,
                linetype = up), size = 0.8) +
  
  coord_cartesian(ylim = c(-2, 2)) +
  
  scale_fill_viridis(option = 'D', begin = .4, discrete = T, alpha = .25) +
  scale_color_viridis(option = 'B', begin = .1, end = .6, discrete = T, direction = -1) +
  
  labs(color = 'Change', linetype = 'Change', shape = 'Change',
       fill = 'Condition',
       x = 'Levels of condition', 
       y = '0.5 x (Correct AX + Incorrect AY)', 
       title = "PBI based on errors") +
  
  theme_classic() +
  
  geom_segment(aes(x = -Inf, y = -2, xend = -Inf, yend = 2), 
               color = 'black', size = rel(1), linetype = 1) +
  geom_segment(aes(x = 'Practice', y = -Inf, xend = 'Performance', yend = -Inf), 
               color = 'black', size = rel(1), linetype = 1) +
  
  theme(axis.line = element_blank(),
        plot.title = element_text(color = 'black', size = 13, face = 'bold',
                                  hjust = 0.5, margin = margin(b = 15)),
        axis.title.x = element_text(color = 'black', size = 13, face = 'bold',
                                    margin = margin(t = 15)),
        axis.title.y = element_text(color = 'black', size = 13, face = 'bold',
                                    margin = margin(r = 15)),
        axis.text = element_text(color = 'black', size = 12),
        legend.position = 'bottom',
        legend.direction = 'vertical',
        legend.text = element_text(size = 11),
        legend.title = element_text(size = 12, face = 'bold'),
        legend.key.size = unit(.6, 'cm')); change_pbie
# Save plot
ggsave(change_pbie, filename = './results/figs/change_pbi_errors.pdf',
       width = 5.5, height = 7)


# --- 5.4) PBI RT  -------------------------------------------------------------
# Visualise change from practice to perfomance block
# Select a_bias
pbirt_df <- select(pci, block, pbi_rt, id)
# code if a_bias goes up or down in performance block
pbirt_df <- pbirt_df %>% 
  group_by(id) %>% 
  mutate(up = ifelse(pbi_rt < lead(pbi_rt), 'up', 'down')) 
# replace NAs
pbirt_df <- pbirt_df %>% 
  mutate(up = ifelse(is.na(up), lag(up), up))

# -- Create plot -- 
change_pbirt <- ggplot(data = data.frame(pbirt_df)) +
  
  geom_violin(trim = F, inherit.aes = F, 
              aes(x = block, 
                  y = pbi_rt, 
                  fill = block), size = 0.8) +
  geom_point(inherit.aes = F,
             aes(x = block, 
                 y = pbi_rt, 
                 color = up, 
                 shape = up), size = 1.5) +
  geom_line(inherit.aes = F,
            aes(x = block, 
                y = pbi_rt, 
                group = id, 
                color = up,
                linetype = up), size = 0.8) +
  
  coord_cartesian(ylim = c(-0.2, 0.6)) +
  
  scale_fill_viridis(option = 'D', begin = .4, discrete = T, alpha = .25) +
  scale_color_viridis(option = 'B', begin = .1, end = .6, discrete = T, direction = -1) +
  
  labs(color = 'Change', linetype = 'Change', shape = 'Change',
       fill = 'Condition',
       x = 'Levels of condition', 
       y = '0.5 x (Correct AX + Incorrect AY)', 
       title = "PBI based on RT") +
  
  theme_classic() +
  
  geom_segment(aes(x = -Inf, y = -0.2, xend = -Inf, yend = 0.6), 
               color = 'black', size = rel(1), linetype = 1) +
  geom_segment(aes(x = 'Practice', y = -Inf, xend = 'Performance', yend = -Inf), 
               color = 'black', size = rel(1), linetype = 1) +
  
  theme(axis.line = element_blank(),
        plot.title = element_text(color = 'black', size = 13, face = 'bold',
                                  hjust = 0.5, margin = margin(b = 15)),
        axis.title.x = element_text(color = 'black', size = 13, face = 'bold',
                                    margin = margin(t = 15)),
        axis.title.y = element_text(color = 'black', size = 13, face = 'bold',
                                    margin = margin(r = 15)),
        axis.text = element_text(color = 'black', size = 12),
        legend.position = 'bottom',
        legend.direction = 'vertical',
        legend.text = element_text(size = 11),
        legend.title = element_text(size = 12, face = 'bold'),
        legend.key.size = unit(.6, 'cm')); change_pbirt
# Save plot
ggsave(change_pbirt, filename = './results/figs/change_pbi_rt.pdf',
       width = 5.5, height = 7)



# --- 6) Analyse A-cue bias ----------------------------------------------------

# Set order of levels of factor block
pci$block <- factor(pci$block, levels = c("Performance", "Practice"))

# Effect code cathegorical variables
contrasts(pci$block) <-  contr.sum(2); contrasts(pci$block)

# -- Change default contrasts options ! --
options(contrasts = c("contr.sum", "contr.poly"))

# Require packages for analysis
getPacks(c('lme4', 'lmerTest',
           'sjPlot',
           'emmeans'))

# Model change in A-cue bias
mod_pc1 <- lmer(data = pci, 
                a_bias ~ block + (1|id)) 
# ANOVA table
anova(mod_pc1, ddf = 'Kenward-Roger')
# model diagnostics
plot_model(mod_pc1, 'diag')

# --- 6.1) Remove outliers (optional) ------------------------------------------
# --- REFIT MODEL WITHOUT OUTLIERS (RESULTS DON'T CHANGE MUCH)
dat1_rm <- stdResid(pci, mod_pc1, plot = T, show.bound = T)

# Model refitted without outliers
pci_no_out <- filter(dat1_rm, Outlier == 0)
mod_pc1 <- lmer(data = pci_no_out,
                a_bias ~ block + (1|id))
# Anova table for final model
anova(mod_pc1, ddf = 'Kenward-Roger')
# Regression table for model
summary(mod_pc1, ddf = 'Kenward-Roger')
# model diagnostics for final model
plot_model(mod_pc1, 'diag')
# forest plot for standardised erstimates
std_est <- plot_model(mod_pc1, 'std2'); std_est

# --- 6.2) Compute summary statistics for A_bias model -------------------------
# Compute effect sizes (semi partial R2)
amod <- anova(mod_pc1, ddf = 'Kenward-Roger'); amod
amod <-  as.data.frame(amod); amod
amod$sp.R2 <- R2(amod); amod

# Save anova table
tab_df(round(amod, 5), 
       title = 'Anova results for linear mixed effects regression analysis of A-cue bias',
       file = './results/tables/anova_abias.html')

# Save model summary
tab_model(mod_pc1,
          title = 'Model estimates for linear mixed effects regression analysis of A-cue bias',
          file = './results/tables/summary_abias.html')



# --- 7) Analyse d prime -------------------------------------------------------

# Set order of levels of factor block
pci$block <- factor(pci$block, levels = c("Performance", "Practice"))

# Effect code cathegorical variables
contrasts(pci$block) <-  contr.sum(2); contrasts(pci$block)

# -- Change default contrasts options ! --
options(contrasts = c("contr.sum", "contr.poly"))

# Require packages for analysis
getPacks(c('lme4', 'lmerTest',
           'sjPlot',
           'emmeans'))

# Model change in A-cue bias
mod_pc2 <- lmer(data = pci, 
                dprime ~ block + (1|id)) 
# ANOVA table
anova(mod_pc2, ddf = 'Kenward-Roger')
# model diagnostics
plot_model(mod_pc2, 'diag')

# --- 7.1) Remove outliers (optional) ------------------------------------------
# --- REFIT MODEL WITHOUT OUTLIERS (RESULTS DON'T CHANGE MUCH)
dat2_rm <- stdResid(pci, mod_pc2, plot = T, show.bound = T)

# Model refitted without outliers
pci_no_out <- filter(dat2_rm, Outlier == 0)
mod_pc2 <- lmer(data = pci_no_out,
                dprime ~ block + (1|id))
# Anova table for final model
anova(mod_pc2, ddf = 'Kenward-Roger')
# Regression table for model
summary(mod_pc2, ddf = 'Kenward-Roger')
# model diagnostics for final model
plot_model(mod_pc2, 'diag')
# forest plot for standardised erstimates
std_est_2 <- plot_model(mod_pc2, 'std2'); std_est_2

# --- 7.2) Compute summary statistics for A_bias model -------------------------
# Compute effect sizes (semi partial R2)
amod <- anova(mod_pc2, ddf = 'Kenward-Roger'); amod
amod <-  as.data.frame(amod); amod
amod$sp.R2 <- R2(amod); amod

# Save anova table
tab_df(round(amod, 5), 
       title = "Anova results for linear mixed effects regression analysis of d' Context",
       file = './results/tables/anova_dprime.html')

# Save model summary
tab_model(mod_pc2,
          title = "Model estimates for linear mixed effects regression analysis of d' Contex",
          file = './results/tables/summary_dprime.html')


# --- 8) Analyse PBI errors ----------------------------------------------------

# Set order of levels of factor block
pci$block <- factor(pci$block, levels = c("Performance", "Practice"))

# Effect code cathegorical variables
contrasts(pci$block) <-  contr.sum(2); contrasts(pci$block)

# -- Change default contrasts options ! --
options(contrasts = c("contr.sum", "contr.poly"))

# Require packages for analysis
getPacks(c('lme4', 'lmerTest',
           'sjPlot',
           'emmeans'))

# Model change in A-cue bias
mod_pc3 <- lmer(data = pci, 
                pbi_err ~ block + (1|id)) 
# ANOVA table
anova(mod_pc3, ddf = 'Kenward-Roger')
# model diagnostics
plot_model(mod_pc3, 'diag')

# --- 8.1) Remove outliers (optional) ------------------------------------------
# --- REFIT MODEL WITHOUT OUTLIERS (RESULTS DON'T CHANGE MUCH)
dat3_rm <- stdResid(pci, mod_pc3, plot = T, show.bound = T)

# Model refitted without outliers
pci_no_out <- filter(dat3_rm, Outlier == 0)
mod_pc3 <- lmer(data = pci_no_out,
                pbi_err ~ block + (1|id))
# Anova table for final model
anova(mod_pc3, ddf = 'Kenward-Roger')
# Regression table for model
summary(mod_pc3, ddf = 'Kenward-Roger')
# model diagnostics for final model
plot_model(mod_pc3, 'diag')
# forest plot for standardised erstimates
std_est_3 <- plot_model(mod_pc3, 'std2'); std_est_3

# --- 8.2) Compute summary statistics for A_bias model -------------------------
# Compute effect sizes (semi partial R2)
amod <- anova(mod_pc3, ddf = 'Kenward-Roger'); amod
amod <-  as.data.frame(amod); amod
amod$sp.R2 <- R2(amod); amod

# Save anova table
tab_df(round(amod, 5), 
       title = "Anova results for linear mixed effects regression analysis of PBI errors",
       file = './results/tables/anova_pbi_errors.html')

# Save model summary
tab_model(mod_pc3,
          title = "Model estimates for linear mixed effects regression analysis of PBI errors",
          file = './results/tables/summary_pbi_errors.html')


# --- 9) Analyse PBI RT --------------------------------------------------------

# Set order of levels of factor block
pci$block <- factor(pci$block, levels = c("Performance", "Practice"))

# Effect code cathegorical variables
contrasts(pci$block) <-  contr.sum(2); contrasts(pci$block)

# -- Change default contrasts options ! --
options(contrasts = c("contr.sum", "contr.poly"))

# Require packages for analysis
getPacks(c('lme4', 'lmerTest',
           'sjPlot',
           'emmeans'))

# Model change in A-cue bias
mod_pc4 <- lmer(data = pci, 
                pbi_rt ~ block + (1|id)) 
# ANOVA table
anova(mod_pc4, ddf = 'Kenward-Roger')
# model diagnostics
plot_model(mod_pc4, 'diag')

# --- 9.1) Remove outliers (optional) ------------------------------------------
# --- REFIT MODEL WITHOUT OUTLIERS (RESULTS DON'T CHANGE MUCH)
dat4_rm <- stdResid(pci, mod_pc4, plot = T, show.bound = T)

# Model refitted without outliers
pci_no_out <- filter(dat4_rm, Outlier == 0)
mod_pc4 <- lmer(data = pci_no_out,
                pbi_rt ~ block + (1|id))
# Anova table for final model
anova(mod_pc4, ddf = 'Kenward-Roger')
# Regression table for model
summary(mod_pc4, ddf = 'Kenward-Roger')
# model diagnostics for final model
plot_model(mod_pc4, 'diag')
# forest plot for standardised erstimates
std_est_4 <- plot_model(mod_pc4, 'std2'); std_est_4

# --- 9.2) Compute summary statistics for A_bias model -------------------------
# Compute effect sizes (semi partial R2)
amod <- anova(mod_pc4, ddf = 'Kenward-Roger'); amod
amod <-  as.data.frame(amod); amod
amod$sp.R2 <- R2(amod); amod

# Save anova table
tab_df(round(amod, 5), 
       title = "Anova results for linear mixed effects regression analysis of PBI RT",
       file = './results/tables/anova_pbi_rt.html')

# Save model summary
tab_model(mod_pc4,
          title = "Model estimates for linear mixed effects regression analysis of PBI RT",
          file = './results/tables/summary_pbi_rt.html')


# --- 10) Final figure for submission ------------------------------------------
# Standardised model estimates
# A bias
mod_est <- std_est + geom_hline(yintercept = 0, linetype = 2) +
  scale_y_continuous(limits = c(-1, 1)) +
  scale_x_discrete(labels = c('Performance-Practice')) +
  scale_color_viridis(option = 'A', discrete = T, direction = 1, end = .55) +
  
  geom_segment(aes(x = -Inf, y = -1, xend = -Inf, yend = 1), 
               color = 'black', size = rel(1), linetype = 1) +
  geom_segment(aes(x = 1, y = -Inf, xend = 1, yend = -Inf), 
               color = 'black', size = rel(1), linetype = 1) +
  
  labs(title = 'A) Standardised beta-weights of Condition \n (Performance - Practice)',
       subtitle = 'A-bias',
       y = 'Estimate') + 
  
  theme_classic() + 
  theme(plot.title = element_text(color = 'black', size = 13, face = 'bold',
                                  margin = margin(b = 10), hjust = .5),
        plot.subtitle = element_text(color = 'black', size = 13, face = 'bold',
                                     margin = margin(b = 10), hjust=.5),
        axis.line = element_blank(),
        axis.title.x = element_blank(),
        axis.title.y = element_text(color = 'black', face = 'bold', size = 13,
                                    margin = margin(r = 15)),
        axis.text.x = element_text(color = 'black', size = 12),
        axis.text.y = element_blank(), 
        axis.ticks.y = element_blank()); mod_est

# d prime
mod_est2 <- std_est_2 + geom_hline(yintercept = 0, linetype = 2) +
  scale_y_continuous(limits = c(-1, 1)) +
  scale_x_discrete(labels = c('Performance-Practice')) +
  scale_color_viridis(option = 'A', discrete = T, direction = -1, end = .55) +
  
  geom_segment(aes(x = -Inf, y = -1, xend = -Inf, yend = 1), 
               color = 'black', size = rel(1), linetype = 1) +
  geom_segment(aes(x = 1, y = -Inf, xend = 1, yend = -Inf), 
               color = 'black', size = rel(1), linetype = 1) +
  
  labs(subtitle = "d' Context",
       y = 'Estimate') + 
  
  theme_classic() + 
  theme(plot.title = element_blank(),
        plot.subtitle = element_text(color = 'black', size = 13, face = 'bold',
                                     margin = margin(b = 10), hjust=.5),
        axis.line = element_blank(),
        axis.title.x = element_blank(),
        axis.title.y = element_text(color = 'black', face = 'bold', size = 13,
                                    margin = margin(r = 15)),
        axis.text.x = element_text(color = 'black', size = 12),
        axis.text.y = element_blank(), 
        axis.ticks.y = element_blank()); mod_est2

# PBI errors
mod_est3 <- std_est_3 + geom_hline(yintercept = 0, linetype = 2) +
  scale_y_continuous(limits = c(-1, 1)) +
  scale_x_discrete(labels = c('Performance-Practice')) +
  scale_color_viridis(option = 'A', discrete = T, direction = 1, end = .55) +
  
  geom_segment(aes(x = -Inf, y = -1, xend = -Inf, yend = 1), 
               color = 'black', size = rel(1), linetype = 1) +
  geom_segment(aes(x = 1, y = -Inf, xend = 1, yend = -Inf), 
               color = 'black', size = rel(1), linetype = 1) +
  
  labs(subtitle = 'PBI errors',
       y = 'Estimate') + 
  
  theme_classic() + 
  theme(plot.title = element_blank(),
        plot.subtitle = element_text(color = 'black', size = 13, face = 'bold',
                                     margin = margin(b = 10), hjust=.5),
        axis.line = element_blank(),
        axis.title.x = element_blank(),
        axis.title.y = element_text(color = 'black', face = 'bold', size = 13,
                                    margin = margin(r = 15)),
        axis.text.x = element_text(color = 'black', size = 12),
        axis.text.y = element_blank(), 
        axis.ticks.y = element_blank()); mod_est3

# PBI RT
mod_est4 <- std_est_4 + geom_hline(yintercept = 0, linetype = 2) +
  scale_y_continuous(limits = c(-1, 1)) +
  scale_x_discrete(labels = c('Performance-Practice')) +
  scale_color_viridis(option = 'A', discrete = T, direction = 1, end = .55) +
  
  geom_segment(aes(x = -Inf, y = -1, xend = -Inf, yend = 1), 
               color = 'black', size = rel(1), linetype = 1) +
  geom_segment(aes(x = 1, y = -Inf, xend = 1, yend = -Inf), 
               color = 'black', size = rel(1), linetype = 1) +
  
  labs(subtitle = "PBI RT",
       y = 'Estimate') + 
  
  theme_classic() + 
  theme(plot.title = element_blank(),
        plot.subtitle = element_text(color = 'black', size = 13, face = 'bold',
                                     margin = margin(b = 10), hjust=.5),
        axis.line = element_blank(),
        axis.title.x = element_text(color = 'black', face = 'bold', size = 13,
                                    margin = margin(t = 15)),
        axis.title.y = element_text(color = 'black', face = 'bold', size = 13,
                                    margin = margin(r = 15)),
        axis.text.x = element_text(color = 'black', size = 12),
        axis.text.y = element_blank(), 
        axis.ticks.y = element_blank()); mod_est4


# Create margins for each plot in fugure
margin_1 = theme(plot.margin = unit(c(0.01, 0.25, 0, 0.25), "cm"))
margin_2 = theme(plot.margin = unit(c(1, 0.25, 0.5, 0.25), "cm"))
margin_3 = theme(plot.margin = unit(c(0.6, 0.25, 0.5, 0.25), "cm"))
margin_4 = theme(plot.margin = unit(c(0.5, 0.25, 0, 0.25), "cm"))


# Arrange plot in single figure
fig_pbis <- grid.arrange(grobs = list(mod_est + margin_1,
                                      mod_est2 + margin_2, 
                                      mod_est3 + margin_3,
                                      mod_est4 + margin_4), 
                         layout_matrix = rbind(c(1),
                                               c(2),
                                               c(3),
                                               c(4))); fig_pbis

# Save figure
ggsave(fig_pbis, filename = './results/figs/fig_pbis.pdf',
       width = 5, height = 6)
