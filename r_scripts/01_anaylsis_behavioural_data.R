# Title     : Analysis of RT
# Objective : Test effect of cue-probe incongrunecy
# Created by: Jose C. Garcia Alanis
# Created on: 23.07.20
# R version : 4.0.2 (2020-06-22), Taking Off Again

# source function for fast loading and installing of packages
source('./r_functions/getPacks.R')
source('./r_functions/spr2.R')


# 1) define the path to behavioral data directory ------------------------------

# get system and user information
host <- Sys.info()

# set default path or promt user for other path
if (host['nodename'] == "josealanis-desktop") {

  # defaut path in project structure
  path_to_rt <- '../data/derivatives/results/rt/'

} else {

  path_to_rt <- readline('Please provide path to behavioral data: ')

}


# 2) import in the data --------------------------------------------------------
# this part requires the package 'dplyr'
getPacks('dplyr')

# files in dir
rt_files <- list.files(path=path_to_rt, full.names = T)

# read in the files
rt_list <- lapply(rt_files, read.table, sep = '\t', header = T)

# put the in a data frame
rt_df <- bind_rows(rt_list, .id = "column_label")

# # clean up if desired
# rm(rt_list, rt_files)


# 3) exploratory analyses ------------------------------------------------------

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

# plot distribution of reaction time
getPacks(c('ggplot2', 'viridis'))
pd <- position_dodge(1.15)
rt_plot <- ggplot(corrects,
                  aes(x = w_rt, fill = probe, colour = probe)) +
  geom_density(alpha = 0.25) +
  # facet_wrap(~ probe, scales = 'free', nrow = 4) +
  labs(x = 'RT (sec)',
       y = 'Density',
       fill = 'Cue-Probe',
       colour = 'Cue-Probe') +
  scale_y_continuous(limits = c(0.0, 8.75),
                     breaks = seq(0.0, 8.0, 2.0)) +
  scale_x_continuous(limits = c(0.0, 0.8)) +
  scale_fill_viridis(discrete = T) +
  scale_color_viridis(discrete = T) +
  geom_segment(aes(x = -Inf, y = 0.0, xend = -Inf, yend = 8.0),
               color = 'black', size = rel(0.5), linetype = 1) +
  geom_segment(aes(x = 0.0, y = -Inf, xend = 0.8, yend = -Inf),
               color = 'black', size = rel(0.5), linetype = 1) +
  theme(axis.title.x = element_text(color = 'black', size = 12,
                                    margin = margin(t = 10)),
        axis.title.y= element_text(color = 'black', size = 12,
                                   margin = margin(r = 10)),
        axis.text = element_text(color = 'black', size = 10),
        panel.background = element_rect(fill = 'gray95'),
        strip.text = element_blank(),
        strip.background = element_blank(),
        legend.position='bottom',
        legend.title = element_text(size = 10, face = 'bold'),
        legend.text = element_text(size = 10),
        panel.spacing = unit(1, "lines")) +
  # geom_density(data = corrects, aes(x = w_rt),
  #              alpha = 0.0, color = 'black', linetype = 1) +
  geom_boxplot(data = corrects,
               aes(rt, y = 8.0),
               color = 'black',
               width = 0.75,
               outlier.size = 0.25,
               alpha = 0.5, position = pd,
               show.legend = FALSE); rt_plot
ggsave(filename = '../data/derivatives/results/figures/rt_distribution.pdf',
       plot = rt_plot, width = 5, height = 4)


# 4) model rt data -------------------------------------------------------------

# this part requires the following packages
getPacks(c('lme4', 'car',
           'sjPlot',
           'tidyr'))

# summarise to the level of trial types by block
corrects %>%
  group_by(subject, block, probe) %>%
  summarise(m_rt = mean(w_rt))

# transform variables to factors
dat_for_mod$probe <- as.factor(dat_for_mod$probe)
dat_for_mod$block <- as.factor(dat_for_mod$block)

rt_mod <- lmer(data = dat_for_mod,
               m_rt ~ block * probe + (1 |subject),
               contrasts = list(probe = 'contr.sum',
                                block = 'contr.sum'))
# anova for model
rt_anova <- car::Anova(rt_mod, test = 'F', type = 'III')
# semi-partial R-squared for predictors
spr2 <- spR2(rt_anova)

# create a teble of the model summary
tab_model(rt_mod, digits = 3,
          file = '../data/derivatives/results/tables/rt_model.html',
          pred.labels = c('(Intercept)',
                          'Block 1',
                          'AX', 'AY', 'BX',
                          'Block 1 * AX',
                          'Block 1 * AY',
                          'Block 1 * BX'))

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
probe_levels <- c('AX 0', 'AY 0', 'BX 0', 'BY 0',
                  'AX 1', 'AY 1', 'BX 1', 'BY 1')

# compute estimated marginal means
rt_means <- emmeans(rt_mod, ~ probe * block)
# compute significance of pairwise contrasts
pwpm(rt_means, adjust = 'mvt', flip = TRUE)

# prepare pairwise data for plot
rt_means_df <- as.data.frame(rt_means)
rt_means_df <- rt_means_df %>%
  arrange(block) %>%
  mutate(var1 = paste(probe, block, sep = ' '),
         var2 = paste(probe, block, sep = ' '))

# reorder levels
rt_means_df$var1 <- factor(rt_means_df$var1,
                           levels = probe_levels)
rt_means_df$var2 <- factor(rt_means_df$var2,
                           levels = probe_levels)

# model variance estimates and residual degrees of freedom
mod_var <- VarCorr(rt_mod)
totSD <- sqrt(sum(as.data.frame(mod_var)$vcov))
edf <- df.residual(rt_mod)

# compute effect sizes for the pairwise contrasts
probe_means <- emmeans(rt_mod, ~ probe)
es_probes <- eff_size(probe_means, sigma = totSD, edf = edf); es_probes

block_means <- emmeans(rt_mod, ~ block)
es_block <- eff_size(block_means, sigma = totSD, edf = edf); es_block

es <- eff_size(rt_means, sigma = totSD, edf = edf)
es <- as.data.frame(es)
es <- es %>%
  separate(contrast, into = c('var1', 'var2'), sep = ' - ') %>%
  mutate(var1 = gsub(var1, pattern = ',', replacement = ' '),
         var2 = gsub(var2, pattern = ',', replacement = ' '))
# reorder levels
es$var1 <- factor(es$var1, levels = probe_levels)
es$var2 <- factor(es$var2, levels = probe_levels)

# create the plot
pw_rt_plot <- ggplot(rt_means_df, aes(var1, var2)) +
  geom_tile(fill = 'white') +
  geom_text(aes(label = round(emmean, 2)), size = 5,
            vjust = -.05) +
  geom_text(aes(label = paste(round(lower.CL, 2),
                              round(upper.CL, 2),
                              sep = ' - ')),
            vjust = 2.5, size = 3.25) +
  geom_tile(data = es, aes(var1, var2, fill=effect.size),
            colour = 'white', size = 0.8) +
  # scale_fill_viridis(option = 'B', limits = c(-0.5, 0.5)) +
  scale_fill_distiller(palette = "RdBu", limits = c(-5.0, 5.0)) +
  geom_segment(aes(x = -Inf, y = 'AX 0', xend = -Inf, yend = 'BY 1'),
               color = 'black', size = rel(0.5), linetype = 1) +
  geom_segment(aes(x = 'AX 0', y = -Inf, xend = 'BY 1', yend = -Inf),
               color = 'black', size = rel(0.5), linetype = 1) +
  labs(x = 'Trial type by block', y = 'Trial type by block',
       fill = "Difference (Cohen's D)") +
  theme(axis.title.x = element_text(color = 'black', size = 15,
                                    margin = margin(t = 10)),
        axis.title.y= element_text(color = 'black', size = 15,
                                   margin = margin(r = 10)),
        axis.text = element_text(color = 'black', size = 12),
        panel.background = element_rect(fill = 'gray95'),
        strip.text = element_blank(),
        strip.background = element_blank(),
        legend.position='bottom',
        panel.spacing = unit(1, "lines")) +
  coord_flip()
ggsave(filename = '../data/derivatives/results/figures/pairwise_rt_plot.pdf',
       plot = pw_rt_plot, width = 6.5, height = 7)
