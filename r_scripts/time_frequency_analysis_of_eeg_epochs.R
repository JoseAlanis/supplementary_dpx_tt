# --- author: Jose C. Garcia Alanis
# --- encoding: utf-8
# --- r version: 3.4.4 (2018-03-15) -- "Someone to Lean On"
# --- content: time-frequency analysys of cue epochs
# --- version: "Wed Oct 10 08:57:38 2018"

# --- 1) Set paths ---------------------------------------------
project_path <- '/Users/Josealanis/Documents/Experiments/dpx_tt/'
upload_path <- '/Users/Josealanis/Documents/GitHub/supplementary_dpx_tt/'
setwd(project_path)

# --- 2) Load necessary packages -------------------------------
source('/Users/josealanis/Documents/GitHub/supplementary_dpx_tt/r_functions/getPacks.R')
source('~/Documents/GitHub/supplementary_dpx_tt/r_functions/logspace.R')
pkgs <- c('dplyr')
getPacks(pkgs)

# --- 3) Import behavioural data -------------------------------
rt_data <- read.table('./data_frames/dpx_tt_behavioral.txt', 
                      header = T)
# Import ID key
key <- read.table('./meta/IDs.txt', 
                  sep = ',', 
                  header = T)
names(key) <- c('id', 'ID')

# Merge ID to rt data frame
rt_data <- merge(rt_data, key, 'ID')
rt_data$ID <-  NULL


# --- 4) Main loop for tf analyises ----------------------------
# 1. Get behavioral data.
# 2. Recode trial types / which cue and probe were presented.
# 3. Import MNE summary and check for invalid trials (e.g., misses)
# 4. Remove artifact distorted epochs (if any)
# 5. Import EEG epochs
# 6. Remove invalid epochs
# 7. Condition to be used for TF-analyses
# 8. Time-frequency decomposition
# 9. Save results
# 10. Print function runtime
for (i in 1:32) {
  
  # Save time for runtime summary
  start.time <- Sys.time()
  
  # Particant in question
  ID <- i
  print(ID)
  
  # ===== 1. Get behavioural data ======
  dat_id <- dplyr::filter(rt_data, id == ID)
  # Arrange by task block and trial number
  dat_id <- dplyr::arrange(dat_id, block, trial)
  # Create column with original trial number
  dat_id$epoch_origin <- 1:nrow(dat_id)
  
  
  # ===== 2. Recode trial types =====
  # Cue presented
  dat_id$cue <- recode_factor(as.factor(dat_id$trial_type), 
                              `1` = 'A', `2` = 'B', 
                              `3` = 'A', `4` = 'B')
  # Probe presented
  dat_id$probe <- recode_factor(as.factor(dat_id$trial_type), 
                                `1` = 'X', `2` = 'X', 
                                `3` = 'Y', `4` = 'Y')
  
  
  # ===== 3. Import MNE EEG-epoch summary =====
  # Path to file
  file_path = paste('./eeg/dpx_tt_mne_summary/data', 
                    ID, 
                    '_epochs_summary.txt', 
                    sep = '')
  # Valid trials
  vtrials <-  read.table(file_path, header = F, sep = ',')
  f_valid <- as.numeric(as.character(grep("^Extracted", vtrials$V1)))+1
  l_valid <- as.numeric(nrow(vtrials))
  extracted <- as.numeric(as.character(vtrials$V1[f_valid:l_valid]))
  
  # Trials were participants pressed button too soon / invalid reaction
  f_soon <- as.numeric(as.character(grep("^Too", vtrials$V1)))+1
  l_soon <- as.numeric(as.character(grep("^Miss", vtrials$V1)))-1
  
  # Were any 'too-soon trials' found?
  if (l_soon < f_soon) {
    too_soon <- NA
  } else {
    too_soon <- as.numeric(as.character(vtrials$V1[f_soon:l_soon]))
  }
  
  # Trials were participants didn't press the button / invalid reaction
  f_miss <- as.numeric(as.character(grep("^Miss", vtrials$V1)))+1
  l_miss <- as.numeric(as.character(grep("^Rejected", vtrials$V1)))-1
  
  # Were any 'miss trials' found?
  if (l_miss < f_miss) {
    miss <- NA
  } else {
    miss <- as.numeric(as.character(vtrials$V1[f_miss:l_miss]))
  }
  
  
  # ===== 4. Remove bad (artefact distorted) epochs =====
  dat_id <- dplyr::filter(dat_id, epoch_origin %in% extracted)
  dat_id$epoch <- 1:nrow(dat_id) 
  
  # ===== 5. Import EEG epochs =====
  # Read file
  epochs <- read.table(paste('./eeg/dpx_epochs_no_base/data', 
                             ID, 
                             '_dpx_epochs.txt', 
                             sep = ''), 
                       header = T, 
                       sep = ',', 
                       colClasses = c(c('factor', 'numeric', 'numeric'), 
                                      rep('numeric', 64)))
  # epochs +1 (python arrays start at 0)
  epochs$epoch <- epochs$epoch + 1
  # Check if beahvioural and eeg epochs match and print messsge (TRUE = ok)
  print(sum(unique(epochs$epoch) == unique(dat_id$epoch)) == nrow(dat_id))
  
  # ===== 6. Remove invalid epochs =====
  epochs <- merge(epochs, dat_id, 'epoch')
  epochs <- filter(epochs, !epoch_origin %in% too_soon)
  epochs <- filter(epochs, !epoch_origin %in% miss)
  
  
  # ===== 7. Select condition for TF-analyses =====
  # Select Trials 
  EEG <- filter(epochs, condition == 'Correct B')
  rm(epochs)
  
  # ===== 8. TIME FREQUENCY DECOMPOSITION =====
  # Channel to use
  chan = 'PO8'
  # Baseline for divisive correction
  base_ix <- which(unique(EEG$time) >= -1000 & unique(EEG$time) <= -500)
  
  
  # WAVELET PARAMETERS
  # EEG sampling rate and points per segment
  s.rate = 256
  points = as.numeric(nrow( EEG[EEG$epoch == EEG$epoch[1], ] ))
  
  # Number of frequencies
  n_freq = 25
  # Min and max frequencies
  min_freq <- 1
  max_freq <- 50
  
  # Frequencies for wavelet (logarithmically sapced)
  frex <- logspace(log10(min_freq), log10(max_freq), n_freq)
  # Time space
  time = seq(-1, 1, by = 1/s.rate)
  
  # -- ***** WAVELET PARAMETER ***** --
  s_par <- logspace(log10(3), log10(10), n_freq) / (2*pi*frex)
  
  # Wavelet convolution parameters
  n_wavelet = length(time)
  n_data = points * length(unique(EEG$epoch))
  
  # FFT parameters for total and induced power
  n_convolution_1  = n_wavelet+n_data-1
  n_convolution_2  = n_wavelet+n_data-1
  # FFT parameters for ERP (only one 'trial')
  n_convolution_3  = n_wavelet+points-1
  half_of_wavelet_size <- (n_wavelet-1)/2
  
  # *** Compute erp ***
  dat_erp <- EEG %>% 
    dplyr::select(time, epoch, chan) %>% 
    dplyr::group_by(time) %>% 
    dplyr::summarise_at(vars(contains(chan)), funs(erp = mean))
  
  # *** Compute total and induced (i.e., total - erp) activity ***
  dat_induced <- EEG %>% 
    dplyr::select(time, epoch, Total = chan) %>% 
    dplyr::group_by(time) %>% 
    dplyr::right_join(dat_erp, by = 'time') %>% 
    dplyr::arrange(epoch, time) %>% dplyr::mutate(Induced = Total - erp)
  
  # *** Data for FFT ***
  # Add zeros to the end of the signal
  eeg_total <- c(dat_induced$Total, 
                 rep(0, 
                     (n_convolution_1 - length(dat_induced$Total))) )
  eeg_induced <- c(dat_induced$Induced, 
                   rep(0, 
                       (n_convolution_2 - length(dat_induced$Induced))) )
  eeg_erp <- c(dat_erp$erp, 
               rep(0, 
                   (n_convolution_3 - length(dat_erp$erp))) )
  
  # *** Compute FFT of EEG signal ***
  ffttotal <- fft(eeg_total)
  fftinduced <- fft(eeg_induced)
  ffterp <- fft(eeg_erp)
  
  # Create empty matrix for results
  tf_total <- data.frame(matrix(nrow = points, ncol = n_freq))
  tf_induced <- data.frame(matrix(nrow = points, ncol = n_freq))
  tf_evoked <- data.frame(matrix(nrow = points, ncol = n_freq))
  tf_itc <- data.frame(matrix(nrow = points, ncol = n_freq))
  
  # Create wavelet and fit it to the total, induced activity, 
  # and evoked (i.e., erp) activity.
  for (j in 1:n_freq) {
    print(j)
    
    # Create the wavelet
    if (exists('s_par')) {
      wavelet = exp(2*1i*pi*frex[j]*time) * exp(-time^2 / ( 2*(s_par[j] ^2) ) ) / frex[j]
    } else {
      wavelet = exp(2*1i*pi*frex[j]*time) * exp(-time^2 / ( 2*((4/(2*pi*frex[j])) ^2) ) ) / frex[j]
    }
    
    # Run convolution for total, induced and evoked (i.e., ERP) activity
    for (k in 1:3) {
      
      if (k == 1) {
        # TOTAL
        # FFT of data
        fft_wavelet = fft( c(wavelet, rep(0, (n_convolution_1 - length(wavelet))) ) )
        # Convolution
        convolution_result_fft = fft(fft_wavelet * ffttotal, inverse = T) / (length(fft_wavelet * ffttotal))
        convolution_result_fft = convolution_result_fft[(half_of_wavelet_size+1):(length(convolution_result_fft)-half_of_wavelet_size)]
        # Compute power
        temp_power = rowMeans(abs(data.frame(matrix(convolution_result_fft, nrow = points)))^2)
        # db correct power
        tf_total[, j] <- 10*log10(temp_power/mean( temp_power[base_ix[1]:base_ix[length(base_ix)]]  )  )
        
        # Intertrial phase consistency of total tf-monulation
        tf_itc[, j] <- abs(rowMeans( data.frame(matrix(exp(1i*atan2(Im(convolution_result_fft), Re(convolution_result_fft))), 
                                                       nrow = points)) ))
      } else if (k == 2) {  
        # INDUCED (non phase-locked)
        # FFT of data
        fft_wavelet = fft(c(wavelet, rep(0, (n_convolution_2 - length(wavelet))) )   )
        # Convolution
        convolution_result_fft = fft(fft_wavelet * fftinduced, inverse = T) / (length(fft_wavelet * fftinduced))
        convolution_result_fft = convolution_result_fft[(half_of_wavelet_size+1):(length(convolution_result_fft)-half_of_wavelet_size)]
        # Compute power
        temp_power = rowMeans(abs(data.frame(matrix(convolution_result_fft, nrow = points)))^2)
        # db correct power
        tf_induced[, j] <- 10*log10(temp_power/mean(temp_power[base_ix[1]:base_ix[length(base_ix)]]  )  )
        
      } else if (k == 3) {
        # EVOKED (ERP)
        # FFT of data
        fft_wavelet = fft(c(wavelet, rep(0, (n_convolution_3 - length(wavelet))) ))
        # Convolution
        convolution_result_fft = fft(fft_wavelet * ffterp, inverse = T) / (length(fft_wavelet * ffterp))
        convolution_result_fft = convolution_result_fft[(half_of_wavelet_size+1):(length(convolution_result_fft)-half_of_wavelet_size)]
        # Compute power
        temp_power = abs(data.frame(matrix(convolution_result_fft, nrow = points)))^2
        # db correct power
        tf_evoked[, j] <- 10*log10(temp_power/mean(temp_power[base_ix[1]:base_ix[length(base_ix)], ]  )  )
      }
    }
  }
  
  # --- Store results in data frames
  names(tf_total) <- as.character(round(frex, digits = 2))
  tf_total$Time <- unique(EEG$time)
  tf_total$ID <- i
  
  names(tf_itc) <- as.character(round(frex, digits = 2))
  tf_itc$Time <- unique(EEG$time)
  tf_itc$ID <- i
  
  names(tf_induced) <- as.character(round(frex, digits = 2))
  tf_induced$Time <- unique(EEG$time)
  tf_induced$ID <- i
  
  names(tf_evoked) <- as.character(round(frex, digits = 2))
  tf_evoked$Time <- unique(EEG$time)
  tf_evoked$ID <- i
  # --- Results stored
  
  # ===== 9. Save results of all participants =====
  # Check if data.frames for storage already exist
  if (!exists('total')) {
    total <- data.frame()
    induced <- data.frame()
    evoked <- data.frame()
    itc <- data.frame()
  }
  # Bind all participants in one frame
  total <- rbind(total, tf_total)
  induced <- rbind(induced, tf_induced)
  evoked <- rbind(evoked, tf_evoked)
  itc <- rbind(itc, tf_itc)
  
  # ===== 10. Calculate and print function runtime =====
  time.taken <- Sys.time() - start.time
  print(time.taken)
  
}