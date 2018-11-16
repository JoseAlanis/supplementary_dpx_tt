# --- author: jose c. garcia alanis
# --- encoding: utf-8
# --- r version: 3.5.1 (2018-07-02) -- "Feather Spray"
# --- content: create errors data frame for analysis
# --- version: Fri Nov 16 12:11:33 2018


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
pkgs <- c('dplyr', 'tidyr')
getPacks(pkgs)


# --- 4) Import and clean behavioural data -------------------------------------
rt_data <- read.table('./data_frames/dpx_tt_behavioral.txt', header = T)

# Get ID key
IDs <- read.table('./meta/IDs.txt', header= T, sep = ',')
names(IDs) <- c('id', 'ID')

# Merge rt data and prson id
rt_data <- merge(rt_data, IDs, 'ID')
rt_data <- arrange(rt_data, ID, block, trial)


# --- 5) Create informative variables and rearrange levels ---------------------
# Add cue and probe variables for each trial
rt_data$cue <- recode_factor(as.factor(rt_data$trial_type), 
                             `1` = 'A', `2`= 'B', `3` = 'A', `4` = 'B')
rt_data$probe <- recode_factor(as.factor(rt_data$trial_type), 
                               `1` = 'X', `2`= 'X', `3` = 'Y', `4` = 'Y')

# Recode Trial type
rt_data$trial_type <- as.factor(rt_data$trial_type); levels(rt_data$trial_type)
rt_data$trial_type <- plyr::revalue(rt_data$trial_type, c('1' = 'AX',
                                                          '2' = 'BX',
                                                          '3' = 'AY',
                                                          '4' = 'BY'))
# Rearrange levels of trial type
rt_data$trial_type <- factor(rt_data$trial_type, 
                             levels(rt_data$trial_type) [c(1, 3, 2, 4)]); levels(rt_data$trial_type)

# Recode Reaction
rt_data$reaction <- factor(rt_data$reaction); levels(rt_data$reaction)
# Rearrange levels of reaction
rt_data$reaction <- plyr::revalue(rt_data$reaction, 
                                  c('hit' = 'Correct',
                                    'incorrect' = 'Incorrect')); levels(rt_data$reaction)

# Recode block (i.e., condition variable)
rt_data$block <- as.factor(rt_data$block); levels(rt_data$block)
# Rearrange levels of block
rt_data$block <- plyr::revalue(rt_data$block, c('1' = 'Practice',
                                                '2' = 'Performance'))

# ID to factor
rt_data$id <- as.factor(rt_data$id); levels(rt_data$id)


# --- 6) Computer nr of trials and nr of errors --------------------------------
# Compute total number of trials
total_trials <- rt_data %>% 
  group_by(id, block, trial_type, cue, probe) %>% 
  summarise(total_trials = sum(!is.na(trial)))

# Compute number of incorrect reactions
total_incorrect <- rt_data %>% 
  filter(reaction == 'Incorrect') %>% 
  group_by(id, block, trial_type, cue, probe) %>% 
  summarise(n_errors = sum(!is.na(trial)), err_rt = mean(rt))


# --- 7) Calculate error rates -------------------------------------------------
# Merge total trials and number of errors
total <- merge(total_trials, 
               total_incorrect, 
               c('id', 'block', 'trial_type', 'cue', 'probe'),
               all.x = T)

# Replace NAs with zeros
total <- total %>% mutate(n_errors = tidyr::replace_na(n_errors, 0))
# Calculate error rate controling for no errors and 100% errors
total <- total %>% mutate(error_rate=(n_errors+0.5)/(total_trials+1))

# --- 7) Calculate error rates -------------------------------------------------
# rename data frame
errors <- total
# Save data frame
save(errors, file = './data_frames/errors.Rda')
write.table(total,'./data_frames/error_rates.txt', row.names = F, sep ='\t')
