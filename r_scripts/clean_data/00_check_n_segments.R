##### ##### #####     Analysis scrips for Alanis et al., 2018   ##### ##### #####
#                          Check number of segments
#

setwd('~/Documents/Experiments/dpx_tt/')

# ----- DEBUG number of segments per person
for (i in 1:32) {
  
  if (!exists('temp')) {
    temp <- data.frame(NA, NA)
    names(temp) <- c('Subject', 'Segments')
  }

  temp[i, 1] <- i
  temp[i, 2]<- length(count.fields(paste('./eeg/rt_cue/RT', i, '_DPX.txt', 
                            sep = ''))) - 1 # <- first line is header
  
}; rm(i)

# ----- Save data
write.table(temp, './meta/n_segments.txt')