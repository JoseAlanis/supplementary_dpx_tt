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
  
  rr <- ((anov_mod$NumDF / anov_mod$DenDF) * anov_mod$`F.value`) / 
    (1+((anov_mod$NumDF / anov_mod$DenDF) * anov_mod$`F.value`))
  
  print(rr)
  
}



# --- 3) Load R packages needed for analyses -----------------------------------
# clean up
rm(set_path)

# Packages
packs <- c('dplyr')
# Load them
getPacks(packs)


# --- 4) Import behavioral data ------------------------------------------------
# Get data
load('./data_frames/errors.Rda')


contrasts(errors$block) <-  contr.treatment(2, base = 1); contrasts(errors$block)
contrasts(errors$trial_type) <-  contr.treatment(4, base = 1); contrasts(errors$trial_type)

hist(errors$error_rate)

# Packages
packs <- c('lme4', 'lmerTest', 'sjPlot', 'emmeans')
# Load them
getPacks(packs)

mean(errors$err_rt, na.rm=T)
sd(errors$err_rt, na.rm=T)

mod_err <- lmer(data = errors,
                log(error_rate) ~ block * trial_type + (1|id/trial_type))
anova(mod_err,  ddf = 'Kenward-Roger')

# UNCOMMENT TO REFIT MODEL WITHOUT OUTLIERS (RESULTS DON'T CHANGE MUCH)
dat_rm <- stdResid(errors, mod_err, plot = T, show.bound = T)

# Model refitted without outliers
mod_err <- lmer(data = filter(dat_rm, Outlier == 0),
                log(error_rate) ~ trial_type * block + (1|id/trial_type))
summary(mod_err)
anova(mod_err,  ddf = 'Kenward-Roger')

plot_model(mod_err, 'diag')
x <- plot_model(mod_err, type = 'std2', order.terms = c(7:1))
tab_model(mod_err)

require(ggplot2)
require(viridis)
x + geom_hline(yintercept = 0, linetype = 2) +
  scale_y_continuous(limits = c(-.5, 1)) + 
  scale_x_discrete(labels = c('AY', 'BX', 'BY', 'Perfomance', 'AY:Perfomance', 'BX:Perfomance', 'BY:Perfomance')) +
  scale_color_viridis(option = 'A', discrete = T, direction = -1, end = .55) +
  
  geom_segment(aes(x = -Inf, y = -.5, xend = -Inf, yend = 1), 
               color = 'black', size = rel(1), linetype = 1) +
  geom_segment(aes(x = 1, y = -Inf, xend = 7, yend = -Inf), 
               color = 'black', size = rel(1), linetype = 1) +
  
  theme_classic() + 
  theme(axis.line = element_blank(),
        axis.title.x = element_text(color = 'black', face = 'bold', size = 13,
                                    margin = margin(t = 15)),
        axis.title.y = element_text(color = 'black', face = 'bold', size = 13,
                                    margin = margin(r = 15)),
        axis.text.x = element_text(color = 'black', size = 12),
        axis.text.y = element_text(color = 'black', size = 12, 
                                   margin = margin(r = 5)))

