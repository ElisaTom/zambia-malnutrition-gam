# ==============================================================================
# STATISTICAL METHODS PROJECT - ZAMBIA MALNUTRITION ANALYSIS
# Student: Elisa Tomasi
# Role: Non-Linear Analysis (GAM)
# Description: Comparison between Linear Models (GLM) and Generalized Additive Models (GAM)
#              to detect non-linear patterns in child growth (Z-scores).
# ==============================================================================

# --- 1. SETUP & LIBRARIES ---
rm(list = ls())         # Clean workspace to avoid conflicts
graphics.off()          # Close any open plot windows
options(scipen = 999)   # Suppress scientific notation for better readability

# Load Libraries
# mgcv: The core package for GAM analysis, as introduced in Lab 7 and Lecture 21.
# haven: Required to import the .dta dataset (Stata format), as seen in Lab 6.
library(mgcv)   
library(haven)  

# --- 2. DATA LOADING ---
filename <- "zambia_height92.dta"

# Robust loading mechanism:
# Allows the script to run on different machines without changing paths manually.
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
# NOTE: Preliminary inspection revealed values > 100 (e.g., 300 instead of 3.0).
# This is a unit error (missing decimal point). We correct it by dividing by 100.
if (mean(abs(zambia_data$zscore), na.rm = TRUE) > 10) {
  zambia_data$zscore <- zambia_data$zscore / 100
  cat("NOTE: Z-scores rescaled (divided by 100) to match standard units.\n")
}

# 3.2 Factor Conversion
# As per Lab 6 guidelines, categorical variables must be explicitly defined as factors
# to ensure R treats them as groups rather than numeric values.
zambia_data$c_gender    <- factor(zambia_data$c_gender, labels = c("Female", "Male"))
zambia_data$m_work      <- factor(zambia_data$m_work, labels = c("No", "Yes"))
zambia_data$m_education <- factor(zambia_data$m_education) 
zambia_data$region      <- factor(zambia_data$region)

# CRITICAL NOTE: 'c_breastf' (breastfeeding duration) is kept NUMERIC.
# This allows the GAM to estimate a smooth curve s(c_breastf) to detect
# non-linear trends over time, which would be lost if converted to a factor.

# 3.3 Outlier Filtering (WHO Standards)
# We remove biologically implausible Z-scores (outside +/- 6 SD) to prevent
# extreme measurement errors from biasing the model.
initial_n <- nrow(zambia_data)
zambia_data <- zambia_data[zambia_data$zscore >= -6 & zambia_data$zscore <= 6, ]
zambia_data <- na.omit(zambia_data)

cat("Data cleaning complete. Observations used:", nrow(zambia_data), 
    "(Removed", initial_n - nrow(zambia_data), "cases due to outliers/missing values)\n")

# --- 4. MODEL FITTING ---
cat("\n=== FITTING MODELS ===\n")

# 4.1 LINEAR MODEL (Benchmark)
# We first fit a standard Linear Model (Lab 6). This serves as a baseline
# to demonstrate that the linearity assumption is insufficient for this data.
lm_model <- lm(zscore ~ c_age + c_gender + c_breastf + 
                 m_agebirth + m_height + m_bmi + 
                 m_education + m_work + region, 
               data = zambia_data)

# 4.2 GAM MODEL (Non-Linear Analysis)
# We fit a Generalized Additive Model (Lab 7) with the following theoretical choices:
# 1. s(): Smooth functions allow for flexible, non-linear relationships.
# 2. k=15 for c_age: We increase the basis dimension to capture the rapid
#    "growth faltering" (weaning effect) expected in the first 24 months.
# 3. method="REML": As discussed in Lecture 21 (Slide 28), we use REML estimation
#    instead of GCV to avoid overfitting (undersmoothing) and ensure robust results.

gam_model <- gam(zscore ~ 
                   s(c_age, k = 15) +          # Increased flexibility for age
                   s(m_bmi) +                  # Smooth for mother's BMI
                   s(m_agebirth) +             # Smooth for mother's age
                   s(c_breastf) +              # Smooth for breastfeeding duration
                   s(m_height) +               # Smooth for mother's height
                   c_gender + m_education + m_work + region, # Parametric terms
                 data = zambia_data,
                 method = "REML") 

# --- 5. INTERPRETATION & DIAGNOSTICS ---

cat("\n=== CHECKING LINEARITY (EDF ANALYSIS) ===\n")
# We analyze the Effective Degrees of Freedom (EDF) from the summary.
# - EDF approx 1: The relationship is linear.
# - EDF >> 1: The relationship is non-linear (justifying the GAM).
print(summary(gam_model)$s.table)

cat("\n=== VISUAL DIAGNOSTICS 1: RESIDUALS ===\n")
cat("DESCRIPTION: Checking model assumptions (Normality and Homoscedasticity).\n")
cat("- QQ-Plot: Points should follow the line.\n")
cat("- Histogram: Should show a bell curve centered at zero.\n")
par(mfrow=c(2,2))
gam.check(gam_model)

cat("\n=== VISUAL DIAGNOSTICS 2: SMOOTH EFFECTS ===\n")
cat("DESCRIPTION: Visualizing the partial effects of smooth terms.\n")
cat("- c_age: Expecting a drop in Z-score (weaning effect).\n")
cat("- Shaded areas represent 95% Bayesian credible intervals.\n")
par(mfrow=c(2,3))
plot(gam_model, pages=1, shade=TRUE, seWithMean=TRUE, scale=0, 
     main="Estimated Smooth Effects (REML)")

# --- 6. MODEL COMPARISON ---
cat("\n=== MODEL COMPARISON ===\n")

# AIC Comparison (Lower is better)
aic_lm <- AIC(lm_model)
aic_gam <- AIC(gam_model)

cat("AIC Linear Model:", aic_lm, "\n")
cat("AIC GAM Model:   ", aic_gam, "\n")

# Additional Goodness-of-Fit Statistics
cat("\n--- Goodness-of-Fit Statistics ---\n")
cat("Linear Model R-squared:", round(summary(lm_model)$r.squared, 4), "\n")
cat("GAM Model Deviance Explained:", round(summary(gam_model)$dev.expl, 4), "\n")

# Formal Hypothesis Test (ANOVA)
cat("\n--- ANOVA Test (Chisq) ---\n")
# We formally test if the additional complexity of the GAM provides a
# statistically significant improvement over the Linear Model.
# H0: The Linear Model is sufficient.
print(anova(lm_model, gam_model, test = "Chisq"))

# --- 7. PREDICTIVE VALIDATION (Train/Test Split) ---
cat("\n=== PREDICTIVE PERFORMANCE (Test Set Validation) ===\n")
# As practiced in Lab 7 (final section), we validate the model on unseen data
# to ensure the GAM is not simply overfitting the training set.

set.seed(123) 
sample_size <- floor(0.8 * nrow(zambia_data))
train_idx   <- sample(seq_len(nrow(zambia_data)), size = sample_size)

train_data <- zambia_data[train_idx, ]
test_data  <- zambia_data[-train_idx, ]

# Refitting models on Training Data only
lm_train  <- lm(formula(lm_model), data = train_data)
gam_train <- gam(formula(gam_model), data = train_data, method = "REML")

# Generating predictions on Test Data
pred_lm  <- predict(lm_train, newdata = test_data)
pred_gam <- predict(gam_train, newdata = test_data)

# Safety check for NA predictions
if (any(is.na(pred_lm)) || any(is.na(pred_gam))) {
  warning("Warning: Some predictions are NA.")
}

# MSE Calculation (Mean Squared Error)
mse_lm  <- mean((test_data$zscore - pred_lm)^2, na.rm = TRUE)
mse_gam <- mean((test_data$zscore - pred_gam)^2, na.rm = TRUE)

cat("Linear Model MSE:", round(mse_lm, 5), "\n")
cat("GAM Model MSE:   ", round(mse_gam, 5), "\n")

improvement <- (mse_lm - mse_gam) / mse_lm * 100
cat("Percent Improvement:", round(improvement, 2), "%\n")

if (improvement > 0) {
  cat("CONCLUSION: GAM generalizes better (lower error on unseen data).\n")
} else {
  cat("CONCLUSION: GAM does not significantly improve predictions.\n")
}

# --- 8. COMPARATIVE VISUALIZATION (BONUS) ---
cat("\n=== GENERATING COMPARATIVE PLOTS ===\n")

# Plot 1: Actual vs Predicted
cat("DESCRIPTION PLOT 1 (Actual vs Predicted):\n")
cat("- Comparison of real values (X-axis) vs predicted values (Y-axis).\n")
cat("- Points should align on the red diagonal line.\n")
cat("- Note how the GAM predictions (Green) cluster slightly better than Linear (Blue).\n")

par(mfrow=c(1,2))
plot(test_data$zscore, pred_lm, 
     main="Linear Model: Actual vs Predicted", 
     xlab="Actual Z-score", ylab="Predicted Z-score",
     pch=16, col=rgb(0,0,1,0.5))
abline(0, 1, col="red", lwd=2, lty=2)
grid()

plot(test_data$zscore, pred_gam, 
     main="GAM Model: Actual vs Predicted", 
     xlab="Actual Z-score", ylab="Predicted Z-score",
     pch=16, col=rgb(0,1,0,0.5))
abline(0, 1, col="red", lwd=2, lty=2)
grid()

# Plot 2: Residuals Distribution
cat("\nDESCRIPTION PLOT 2 (Residual Histograms):\n")
cat("- Comparison of error distributions.\n")
cat("- A taller, narrower distribution around zero indicates a more precise model.\n")

par(mfrow=c(1,2))
hist(test_data$zscore - pred_lm, breaks=30, main="Linear Residuals", 
     xlab="Error", col="lightblue", border="black")
hist(test_data$zscore - pred_gam, breaks=30, main="GAM Residuals", 
     xlab="Error", col="lightgreen", border="black")

# --- END OF SCRIPT ---