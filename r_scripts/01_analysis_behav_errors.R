##### ##### #####     Analysis scrips for Alanis et al., 2018   ##### ##### #####
#                          GLMER models for behavioral
#                                 error data

# Get helper functions
source('./Documents/GitHub/Supplementary_Soc_ERN/R_Functions/getPacks.R')
#source('./Documents/GitHub/Supplementary_Soc_ERN/R_Functions/stdResid.R')
#source('./Documents/GitHub/Supplementary_Soc_ERN/R_Functions/overDisp.R')

# Install and load multiple R packages necessary for analysis.
pkgs <- c('dplyr', 
          'lme4', 'lmerTest',
          'effects', 'emmeans', 'car', 'MuMIn',
          'ggplot2', 'cowplot', 'viridis')

getPacks(pkgs)
rm(pkgs)

# ------ READ in the data  --------------------------------------
load('~/Documents/Experiments/dpx_tt/OLD_STUFF/R_FRAMES/INDEX_Behavioural/Data_for_binomial_GLMM_errors.RData')


# ------ 1) Descriptive statistics  -----------------------------
as.data.frame( all_data %>% group_by(Reaction, ID) %>% 
                 summarise(S = sum(!is.na(Error_vs_Correct))) %>% 
                 summarise(M = mean(S), SD=sd(S), Min = min(S), Max = max(S) ))  

as.data.frame( all_data %>% group_by(Reaction, Trial_Type, ID) %>% 
                 summarise(S = sum(!is.na(Error_vs_Correct))) %>% 
                 summarise(M = mean(S), SD=sd(S), Min = min(S), Max = max(S) ))  


# ------ 2) Effect code variables  ------------------------------
contrasts(all_data$Block) <- contr.sum(2); contrasts(all_data$Block)
contrasts(all_data$Trial_Type) <- contr.sum(4); contrasts(all_data$Trial_Type)


# ------ 2) Set up and fit the model  ---------------------------
mod_Int <- glmer(Error_vs_Correct ~ Trial_Type*Block + (1|ID), 
                 family = binomial(link = 'logit'), data = all_data, 
                  control = glmerControl(optimizer="bobyqa"))
Anova(mod_Int, type = 3)
summary(mod_Int)

mod_Int_rI <- glmer(Error_vs_Correct ~ Trial_Type*Block + (1|ID/Trial_Type), 
                 family = binomial(link = 'logit'), data = all_data, 
                 control = glmerControl(optimizer="bobyqa"))
Anova(mod_Int_rI, type = 3)
summary(mod_Int_rI)

# --- Model diagnostics ---
sjPlot::sjp.glmer(mod_Int_rI, 're.qq') # random intercept looks ok
# --- Test random effets ---
anova(mod_Int, mod_Int_rI) # Model with by trial type rand. intercept is better

# --- Coefficient of determination ---
MuMIn::r.squaredGLMM(mod_Int_rI)
sjstats::r2(mod_Int_rI)


# ------ 3) Follow up analyses  ---------------------------------
# --- Trial type estimates ----
bi_mod_emms_tt <- emmeans(mod_Int_rI, pairwise ~ Trial_Type, 
                          type = 'response', 
                          adjust = 'bonferroni')

# --- Estimated probability ---
as.data.frame(bi_mod_emms_tt$emmeans)
# --- Pairwise contrasts ---
bi_mod_emms_tt$contrasts
as.data.frame(bi_mod_emms_tt$contrasts)
# Compute CIs
mutate(as.data.frame(bi_mod_emms_tt$contrasts), 
       LCL = odds.ratio - SE * 1.96, 
       UCL = odds.ratio + SE * 1.96)


# --- Trial type by block ----
bi_mod_emms <- emmeans(mod_Int_rI, pairwise ~ Trial_Type | Block, 
        type = 'response', 
        adjust = 'bonferroni')

# --- Estimated probability ---
as.data.frame(bi_mod_emms$emmeans)
# --- Pairwise contrasts ---
bi_mod_emms$contrasts
as.data.frame(bi_mod_emms$contrasts)

