# Title     : Analysis of RT
# Objective :
# Created by: Jose C. Garcia Alanis
# Created on: 23.07.20
# R version : 4.0.2 (2020-06-22), Taking Off Again

# source function for fast loading and installing of packages
source('./r_functions/getPacks.R')


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

# quick plot
getPacks(c('ggplot2', 'viridis'))
rt_plot <- ggplot(corrects, aes(x = rt, fill = probe, colour = probe)) +
  geom_density(alpha = 0.5, aes(y = ..scaled..)) +
  facet_wrap(~ probe, scales = 'free', nrow = 4) +
  labs(x = 'RT (sec)',
       y = 'Density distribution',
       fill = 'trial type',
       colour = 'trial type') +
  scale_y_continuous(limits = c(0.0, 1.25), breaks = seq(0.0 , 1.0, 0.5)) +
  scale_x_continuous(limits = c(0.0, 1.0)) +
  scale_fill_viridis(discrete = T) +
  scale_color_viridis(discrete = T) +
  geom_segment(aes(x = -Inf, y = 0.0, xend = -Inf, yend = 1.0),
               color = 'black', size = rel(0.5), linetype = 1) +
  geom_segment(aes(x = 0.0, y = -Inf, xend = 1.0, yend = -Inf),
               color = 'black', size = rel(0.5), linetype = 1) +
  theme(axis.title.x = element_text(color = 'black', size = 12,
                                    margin = margin(t = 10)),
        axis.title.y= element_text(color = 'black', size = 12,
                                   margin = margin(r = 10)),
        axis.text = element_text(color = 'black'),
        panel.background = element_rect(fill = 'gray95'),
        strip.text = element_blank(),
        strip.background = element_blank(),
        legend.position='bottom',
        panel.spacing = unit(1, "lines")) +
  geom_density(data = corrects, aes(x = w_rt, y = ..scaled..),
               alpha = 0.0, color = 'black', linetype = 1) +
  geom_boxplot(data = corrects, aes(w_rt, y = 1.2),
               color = 'black', width = 0.1, outlier.size = 0.5, alpha = 0.5,
               show.legend = FALSE)
ggsave(filename = '../data/derivatives/results/figures/rt_distribution.pdf',
       plot = rt_plot, width = 4, height = 8)


# 3) model rt data -------------------------------------------------------------

# this part requires the package 'lme4' and 'car'
getPacks(c('lme4', 'car', 'sjPlot'))

# transform variables to factors
corrects$probe <- as.factor(corrects$probe)
corrects$block <- as.factor(corrects$block)

# setup and fit the model
rt_mod <- lmer(data = corrects, w_rt ~ block * probe + (1|subject),
               contrasts = list(probe = 'contr.sum', block = 'contr.sum'))
# anova table
Anova(rt_mod, test = 'F', type = 'III')
