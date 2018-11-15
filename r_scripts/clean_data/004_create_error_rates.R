# --- author: jose C. garcia alanis
# --- encoding: utf-8
# --- r version: 3.5.1 (2018-07-02) -- "Feather Spray"
# --- content: create error rates data frame
# --- version: Thu Nov 15 16:28:51 2018


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

# # If missing, get it from jose's gists
# devtools::source_gist('https://gist.github.com/JoseAlanis/86da75bf223e10344b7c16791f45bafe', 
#                       filename = 'getPacks.R')

# # Define other paths if necessary
# project_path <- '/Volumes/TOSHIBA/manuscrips_and_data/dpx_tt/'
# upload_path <- '/Volumes/TOSHIBA/manuscrips_and_data/dpx_tt/upload/'
# setwd(project_path)


# --- 3) Load necessary packages -----------------------------------------------
pkgs <- c('dplyr')
getPacks(pkgs)


# --- 4) Import and clean behavioural data -------------------------------------
rt_data <- read.table('./data_frames/dpx_tt_behavioral.txt', header = T)

# Get ID key
IDs <- read.table('./meta/IDs.txt', header= T, sep = ',')
names(IDs) <- c('id', 'ID')

# Merge rt data and prson id
rt_data <- merge(rt_data, IDs, 'ID')
rt_data <- arrange(rt_data, ID, block, trial)


# --- 5) Computer nr of trials and nr of errors --------------------------------
# Compute total number of trials
total_trials <- rt_data %>% 
  group_by(id, block, trial_type) %>% 
  summarise(total_trials = sum(!is.na(trial)))

# Compute number of incorrect reactions
total_incorrect <- rt_data %>% 
  filter(reaction == 'incorrect') %>% 
  group_by(id, block, trial_type) %>% 
  summarise(total_incorrect = sum(!is.na(trial)), err_rt = mean(rt))


# --- 6) Calculate error rates -------------------------------------------------
# Merge total trials and number of errors
total <- merge(total_trials, 
               total_incorrect, 
               c('id', 'block', 'trial_type'),
               all.x = T)

# Replace NAs with zeros
total <- total %>% mutate_all(funs(replace(., is.na(total_incorrect), 0)))
# Calculate error rate controling for no errors and 100% errors
total <- total %>% mutate(error_rate=(total_incorrect+0.5)/(total_trials+1))

# --- 7) Calculate error rates -------------------------------------------------
# Save data frame
errors <- total
save(errors, file = './data_frames/errors.Rda')
write.table(total,'./data_frames/error_rates.txt', row.names = F, sep ='\t')
