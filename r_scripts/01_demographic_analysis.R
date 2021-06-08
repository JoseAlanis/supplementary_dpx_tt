# Title      : Analysis of sample demographics
# Objective  : Compute sample characteristics (e.g., mean age, etc.)
# Created by : Jose C. Garcia Alanis
# Created on : 23.07.20
# Last update: 20.05.21
# R version  : R version 4.0.5 (2021-03-31), Shake and Throw

# get custom fuctions
source('./r_functions/pkgcheck.R')

# 1) define the path to behavioral data directory -----------------------------

# get system and user information
host <- Sys.info()

check_sys <- grepl('jose', host['user']) & grepl('x|D', host['sysname'])

# set default path or promt user for other path
if (check_sys) {

  # defaut path in project structure
  path_to_rt <- '../data/derivatives/results'

} else {

  path_to_rt <- readline('Please provide path to behavioral data: ')

}

# 2) import in the data -------------------------------------------------------
# this part requires the package 'dplyr'
pkgcheck('dplyr')

# load data
perso <- read.table('../data/participants.tsv', sep = '\t', header = T)

# 3) compute summary statistics -----------------------------------------------
perso %>%
  summarise(
    n_subj = count(.),
    n_female = count(., sex)[count(perso, sex)$sex == 'F', ]$n,
    n_male = count(., sex)[count(perso, sex)$sex == 'M', ]$n,
    m_age = mean(age),
            sd_age = sd(age),
            min_age = min(age),
            max_age = max(age))

# 4) exclude outliers ---------------------------------------------------------
# person with a lot of errors in task
perso[perso$participant_id == 'sub-051', ]

# recompute summary stats
perso %>%
  filter(!participant_id == 'sub-051') %>%
  summarise(
    n_subj = count(.),
    n_female = count(., sex)[count(perso, sex)$sex == 'F', ]$n,
    n_male = count(., sex)[count(perso, sex)$sex == 'M', ]$n,
    m_age = mean(age),
            sd_age = sd(age),
            min_age = min(age),
            max_age = max(age))

perso %>%
  filter(!participant_id == 'sub-051') %>%
  group_by(sex) %>%
  summarise(
    m_age = mean(age),
            sd_age = sd(age),
            min_age = min(age),
            max_age = max(age)) %>%
  as.data.frame()

# 5) check for random within sample effects -----------------------------------
pkgcheck(c('psych', 'ggplot2'))

# exclude outlier
perso <- perso %>%
  filter(!participant_id == 'sub-051')

# plot age by gender
ggplot(data = perso, aes(x = age, fill = sex)) +
  geom_histogram(binwidth = 1,
                 color = "black") +
  annotate('text', x = mean(perso$age) + sd(perso$age),
           y = max(count(perso, age)$n)-1,
           label = paste('Skew:', round(skew(perso$age), 2))) +
  theme(panel.background = element_rect(fill = 'white'),
        panel.grid.major = element_line(size = 0.5,
                                        linetype = 'solid',
                                        colour = "gray97"),
        panel.grid.minor = element_line(size = 0.25,
                                        linetype = 'solid',
                                        colour = "gray97"))

# Q&D test of normality
shapiro.test(perso$age)

# gender effect on age
wilcox.test(data = perso, age ~ sex, paired = F, conf.int = T)
