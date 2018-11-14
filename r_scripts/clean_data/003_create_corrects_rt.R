# --- author: jose c. garcia alanis
# --- encoding: utf-8
# --- r version: 3.4.4 (2018-03-15) -- "Someone to Lean On"
# --- content: get correct reactions data
# --- version: "Mon Sep 24 10:40:32 2018"

# --- 1) Run first, then move on to anylsis section ----------------------------
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
key <- read.table('./meta/IDs.txt', sep = ',', header = T)
names(key) <- c('id', 'ID')

# Merge with rt_data
rt_data <- merge(rt_data, key, 'ID')
rt_data$ID <-  NULL

# Keep correct reactions
for (i in 1:32) {
  # Save time for runtime summary
  start.time <- Sys.time()
  
  # --- 1) Load particpnat in question
  ID <- i
  print(ID)
  
  # --- 2) Behavioral data
  dat_id <- dplyr::filter(rt_data, id == ID)
  dat_id <- dplyr::arrange(dat_id, block, trial)
  dat_id$epoch_origin <- 1:nrow(dat_id)
  dat_id$cue <- recode_factor(as.factor(dat_id$trial_type), 
                              `1` = 'A', `2`= 'B', `3` = 'A', `4` = 'B')
  dat_id$probe <- recode_factor(as.factor(dat_id$trial_type), 
                                `1` = 'X', `2`= 'X', `3` = 'Y', `4` = 'Y')
  
  # --- 3) Get invalid epochs
  file_path = paste('./eeg/dpx_tt_mne_summary/data', ID, '_epochs_summary.txt', sep = '')
  vtrials <-  read.table(file_path, header = F, sep = ',')
  
  # too soon
  f_soon <- as.numeric(as.character(grep("^Too", vtrials$V1)))+1
  l_soon <- as.numeric(as.character(grep("^Miss", vtrials$V1)))-1
  
  # miss
  f_miss <- as.numeric(as.character(grep("^Miss", vtrials$V1)))+1
  l_miss <- as.numeric(as.character(grep("^Rejected", vtrials$V1)))-1
  
  if (l_soon < f_soon) {
    too_soon <- NA
  } else {
    too_soon <- as.numeric(as.character(vtrials$V1[f_soon:l_soon]))
  }
  
  if (l_miss < f_miss) {
    miss <- NA
  } else {
    miss <- as.numeric(as.character(vtrials$V1[f_miss:l_miss]))
  }
  
  # --- 4) 
  dat_id <- filter(dat_id, !epoch_origin %in% too_soon)
  dat_id <- filter(dat_id, !epoch_origin %in% miss)
  dat_id <- filter(dat_id, reaction == 'hit')
  print(length(dat_id$id))
  
  # --- 5)
  if (!exists('all_rt')) {
    all_rt <- data.frame()
  }
  
  all_rt <- rbind(all_rt, data.frame(dat_id))
  
  time.taken <- Sys.time() - start.time
  
  # --- 10)
  print(time.taken)
  
}

# --- 5) Rename levels ---------------------------------------------------------
getPacks('plyr')

# Trial type
all_rt$trial_type <- as.factor(all_rt$trial_type); levels(all_rt$trial_type)
all_rt$trial_type <- plyr::revalue(all_rt$trial_type, c('1' = 'AX',
                                                        '2' = 'BX',
                                                        '3' = 'AY',
                                                        '4' = 'BY'))
# Rearrange
all_rt$trial_type <- factor(all_rt$trial_type, 
                            levels(all_rt$trial_type) [c(1, 3, 2, 4)]); levels(all_rt$trial_type)

# Reaction
all_rt$reaction <- factor(all_rt$reaction); levels(all_rt$reaction)
# Rearrange
all_rt$reaction <- plyr::revalue(all_rt$reaction, c('hit' = 'Correct'))

# Block
all_rt$block <- as.factor(all_rt$block); levels(all_rt$block)
# Rearrange
all_rt$block <- plyr::revalue(all_rt$block, c('1' = 'Practice',
                                              '2' = 'Performance'))

# ID
all_rt$id <- as.factor(all_rt$id); levels(all_rt$id)


# --- 6) Save data frame -------------------------------------------------------
write.table(all_rt, './data_frames/correct_rt.txt', row.names = F, sep = '\t')
save(all_rt, file="./data_frames/corrects_rt.Rda")
