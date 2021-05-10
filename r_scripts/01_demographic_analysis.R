# Title     : Analysis of demographics
# Objective : Get sample characteristics
# Created by: Jose C. Garcia Alanis
# Created on: 23.07.20
# R version : 4.0.2 (2020-06-22), Taking Off Again


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
getPacks('dplyr')

perso <- read.table('../data/participants.tsv', sep = '\t', header = T)