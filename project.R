# ==============================================================================
# STATISTICAL METHODS PROJECT - ZAMBIA MALNUTRITION ANALYSIS
# Elisa Tomasi
# Role: Non-Linear Analysis (GAM)
# Description: Comparison between Linear Models and Generalized Additive Models
#              to detect non-linear patterns in child growth (Z-scores).
# ==============================================================================

# --- 1. SETUP & LIBRARIES ---
rm(list = ls())         # Clear environment
graphics.off()          # Clear plots
options(scipen = 999)   # Suppress scientific notation

# Load libraries in correct order
library(mgcv)   # For Generalized Additive Models (Ref: Lab 7, Lecture 21)
library(haven)  # For reading .dta files (Ref: Lab 6)

# --- 2. DATA LOADING ---
filename <- "zambia_height92.dta"

if (file.exists(filename)) {
  message("Loading dataset from working directory...")
  raw_data <- read_dta(filename)
} else {
  message("Dataset not found. Please select 'zambia_height92.dta' manually.")
  raw_data <- read_dta(file.choose())
}

# --- 3. DATA PREPARATION ---
cat("\n=== DATA PREPARATION ===\n")
zambia_data <- as.data.frame(raw_data)

# 3.1 Scaling Correction
if (mean(abs(zambia_data$zscore), na.rm = TRUE) > 10) {
  zambia_data$zscore <- zambia_data$zscore / 100
  cat("NOTE: Z-scores rescaled (divided by 100).\n")
}

# 3.2 Factor Conversion (Ref: Lab 6)
zambia_data$c_gender    <- factor(zambia_data$c_gender, labels = c("Female", "Male"))
zambia_data$m_work      <- factor(zambia_data$m_work, labels = c("No", "Yes"))
zambia_data$m_education <- factor(zambia_data$m_education) 
zambia_data$region      <- factor(zambia_data$region)

# 3.3 Outlier Filtering
initial_n <- nrow(zambia_data)
zambia_data <- zambia_data[zambia_data$zscore >= -6 & zambia_data$zscore <= 6, ]
zambia_data <- na.omit(zambia_data)

cat("Data cleaning complete. Observations used:", nrow(zambia_data), 
    "(Removed", initial_n - nrow(zambia_data), "outliers/missing)\n")

# --- 4. MODEL FITTING ---
cat("\n=== FITTING MODELS ===\n")

# 4.1 LINEAR MODEL (Benchmark)
lm_model <- lm(zscore ~ c_age + c_gender + c_breastf + 
                 m_agebirth + m_height + m_bmi + 
                 m_education + m_work + region, 
               data = zambia_data)

# 4.2 GAM MODEL (Analysis of Non-Linearity)
# Ref: Lab 7 & Lecture 21 (Slide 28: method="REML")
gam_model <- gam(zscore ~ 
                   s(c_age, k = 20) +          
                   s(m_bmi) + 
                   s(m_agebirth) + 
                   s(c_breastf) + 
                   s(m_height) + 
                   c_gender + m_education + m_work + region,
                 data = zambia_data,
                 method = "REML") 

# --- 5. INTERPRETATION & DIAGNOSTICS ---

cat("\n=== CHECKING LINEARITY (EDF ANALYSIS) ===\n")
print(summary(gam_model)$s.table)

cat("\n=== VISUAL DIAGNOSTICS ===\n")
par(mfrow=c(2,2))
gam.check(gam_model)

par(mfrow=c(2,3))
plot(gam_model, pages=1, shade=TRUE, seWithMean=TRUE, scale=0, 
     main="Estimated Smooth Effects (REML)")

# --- 6. MODEL COMPARISON (AIC & ANOVA) ---
cat("\n=== MODEL COMPARISON ===\n")

# AIC Comparison
aic_lm <- AIC(lm_model)
aic_gam <- AIC(gam_model)

cat("AIC Linear Model:", aic_lm, "\n")
cat("AIC GAM Model:   ", aic_gam, "\n")

# Formal Test (ANOVA)
# FIXED: Using 'lm_model' instead of 'lm_simple'
cat("\n--- Analysis of Deviance Table (Chisq Test) ---\n")
print(anova(lm_model, gam_model, test = "Chisq"))

# --- 7. PREDICTIVE VALIDATION (Train/Test Split) ---
cat("\n=== PREDICTIVE PERFORMANCE (Test Set Validation) ===\n")

set.seed(123) 
sample_size <- floor(0.8 * nrow(zambia_data))
train_idx   <- sample(seq_len(nrow(zambia_data)), size = sample_size)

train_data <- zambia_data[train_idx, ]
test_data  <- zambia_data[-train_idx, ]

# Refitting on Training Data
lm_train  <- lm(formula(lm_model), data = train_data)
gam_train <- gam(formula(gam_model), data = train_data, method = "REML")

# Predictions
pred_lm  <- predict(lm_train, newdata = test_data)
pred_gam <- predict(gam_train, newdata = test_data)

# MSE Calculation
mse_lm  <- mean((test_data$zscore - pred_lm)^2)
mse_gam <- mean((test_data$zscore - pred_gam)^2)

cat("Linear Model MSE:", round(mse_lm, 5), "\n")
cat("GAM Model MSE:   ", round(mse_gam, 5), "\n")

improvement <- (mse_lm - mse_gam) / mse_lm * 100
cat("Percent Improvement:", round(improvement, 2), "%\n")

if (improvement > 0) {
  cat("CONCLUSION: GAM provides better out-of-sample predictions.\n")
} else {
  cat("CONCLUSION: GAM does not significantly improve predictions.\n")
}