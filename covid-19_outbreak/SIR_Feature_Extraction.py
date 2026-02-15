"""
sir_feature_extraction.py
Extract SIR features for all 953 windows
Robust estimation with epidemiological constraints
"""

import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("SIR FEATURE EXTRACTION FOR COVID-19 OUTBREAK")
print("=" * 60)

# ============================================
# 1. Load the data
# ============================================
print("\n[1] Loading data...")

case_data = pd.read_excel('temp/number1.xlsx')
print(f"  ✓ Loaded case data: {len(case_data)} days")
print(f"  Date range: {case_data['date'].min()} to {case_data['date'].max()}")

# ============================================
# 2. Define improved SIR estimation function
# ============================================
print("\n[2] Defining robust SIR estimation...")

def estimate_sir_robust(window_cases, population=52000000):
    """
    Estimate SIR parameters with:
    - Data smoothing to reduce noise
    - Multiple starting points
    - Epidemiological bounds
    - Penalty for unrealistic values
    """
    
    # Get calibration data (first 21 days)
    raw_cases = window_cases[:21].values
    
    # Smooth the data to reduce noise
    if len(raw_cases) >= 7:
        try:
            # Ensure window length is odd for Savitzky-Golay
            window_length = 7
            if window_length > len(raw_cases):
                window_length = len(raw_cases)
            if window_length % 2 == 0:
                window_length -= 1
            if window_length >= 5:
                cases = savgol_filter(raw_cases, window_length, polyorder=2)
            else:
                cases = raw_cases
        except:
            cases = raw_cases
    else:
        cases = raw_cases
    
    # Ensure no negative cases after smoothing
    cases = np.maximum(cases, 1)
    
    t = np.arange(len(cases))
    
    # Initial conditions
    I0 = max(cases[0], 1)
    S0 = population - I0
    R0_initial = 0
    y0 = [S0, I0, R0_initial]
    
    # SIR model differential equations
    def sir_model(y, t, beta, gamma):
        S, I, R = y
        N = population
        dS = -beta * S * I / N
        dI = beta * S * I / N - gamma * I
        dR = gamma * I
        return [dS, dI, dR]
    
    # Objective function with penalties for unrealistic values
    def objective(params):
        beta, gamma = params
        
        # Quick plausibility check
        if beta <= 0 or gamma <= 0:
            return 1e10
        
        try:
            # Solve SIR model
            sol = odeint(sir_model, y0, t, args=(beta, gamma))
            pred = sol[:, 1]
            
            # Weighted MSE (focus more on recent days)
            weights = np.linspace(0.5, 1.0, len(t))
            mse = np.average((pred - cases) ** 2, weights=weights)
            
            # Calculate R0
            R0 = beta / gamma
            
            # Penalties for epidemiologically implausible values
            penalty = 0
            
            # R0 penalty (typical COVID-19 R0: 0.8-8.0)
            if R0 < 0.5:
                penalty += 1000 * (0.5 - R0) ** 2
            elif R0 > 8.0:
                penalty += 1000 * (R0 - 8.0) ** 2
            
            # Growth rate penalty (can't grow more than 40% per day)
            growth_rate = beta - gamma
            if growth_rate > 0.4:
                penalty += 1000 * (growth_rate - 0.4) ** 2
            if growth_rate < -0.3:
                penalty += 1000 * (-0.3 - growth_rate) ** 2
            
            # Gamma penalty (recovery should take 3-20 days)
            if gamma < 0.05:
                penalty += 1000 * (0.05 - gamma) ** 2
            if gamma > 0.33:
                penalty += 1000 * (gamma - 0.33) ** 2
            
            return mse + penalty
            
        except:
            return 1e10
    
    # Multiple starting points to avoid local minima
    start_points = [
        (0.15, 0.07),  # Slow spread, slow recovery
        (0.25, 0.10),  # Medium spread, typical recovery
        (0.35, 0.12),  # Fast spread, faster recovery
        (0.20, 0.05),  # Slow spread, very slow recovery (Omicron-like)
        (0.40, 0.15),  # Very fast spread, fast recovery
        (0.30, 0.08),  # Delta-like
        (0.18, 0.09),  # Original strain-like
    ]
    
    # Epidemiological bounds
    bounds = [(0.01, 0.8),   # β: transmission rate per day
              (0.03, 0.2)]    # γ: recovery rate (5-33 days recovery)
    
    best_result = None
    best_error = 1e10
    
    for beta0, gamma0 in start_points:
        try:
            result = minimize(objective, [beta0, gamma0],
                            bounds=bounds,
                            method='L-BFGS-B',
                            options={'maxiter': 500})
            
            # Check if result is valid
            if result.success and result.fun < best_error:
                beta, gamma = result.x
                
                # Final plausibility check
                R0 = beta / gamma
                if 0.3 <= R0 <= 8.5:  # Accept if within plausible range
                    best_error = result.fun
                    best_result = result
                    
        except:
            continue
    
    # If no good fit found, use median of plausible values
    if best_result is None:
        beta, gamma = 0.25, 0.10  # Default values
    else:
        beta, gamma = best_result.x
    
    # Calculate all derived quantities
    R0 = beta / gamma
    
    # Effective reproduction number (accounts for susceptible depletion)
    total_infected = np.sum(cases)
    S_current = max(population - total_infected, population * 0.01)
    Re = R0 * (S_current / population)
    
    # Growth rate and doubling time
    growth_rate = beta - gamma
    
    if growth_rate > 0.001:
        doubling_time = np.log(2) / growth_rate
    elif growth_rate < -0.001:
        doubling_time = -np.log(2) / growth_rate  # Halving time
    else:
        doubling_time = np.inf
    
    # Get fitted values for R² calculation
    sol = odeint(sir_model, y0, t, args=(beta, gamma))
    fitted = sol[:, 1]
    
    # Calculate R² (fit quality)
    ss_res = np.sum((cases - fitted) ** 2)
    ss_tot = np.sum((cases - np.mean(cases)) ** 2)
    
    if ss_tot > 0:
        r2 = 1 - (ss_res / ss_tot)
    else:
        r2 = -1  # Perfectly constant data
    
    # Prediction residual
    residual = cases[-1] - fitted[-1]
    relative_residual = residual / (cases[-1] + 1)
    
    # Estimate trend in Re (split window into two halves)
    half = len(cases) // 2
    if half >= 5:
        cases_first = cases[:half]
        cases_second = cases[half:]
        
        # Rough Re estimate for each half (simplified)
        Re_first = np.mean(cases_first[5:]) / (np.mean(cases_first[:5]) + 1)
        Re_second = np.mean(cases_second[5:]) / (np.mean(cases_second[:5]) + 1)
        Re_trend = (Re_second - Re_first) / max(half, 1)
    else:
        Re_trend = 0
    
    # Peak estimate
    if growth_rate > 0:
        peak_estimate = max(cases) * (1 + growth_rate * 7)
    else:
        peak_estimate = max(cases)
    
    # Compile all features
    features = {
        'window_id': None,
        'beta': round(beta, 4),
        'gamma': round(gamma, 4),
        'R0': round(R0, 3),
        'Re': round(Re, 3),
        'growth_rate': round(growth_rate, 4),
        'doubling_time': round(doubling_time, 1) if np.isfinite(doubling_time) else np.inf,
        'sir_fit_r2': round(r2, 4),
        'sir_residual': round(residual, 1),
        'sir_rel_residual': round(relative_residual, 4),
        'Re_trend': round(Re_trend, 4),
        'peak_estimate': round(peak_estimate, 1),
        'initial_cases': int(cases[0]),
        'final_cases': int(cases[-1]),
        'case_ratio': round(cases[-1] / (cases[0] + 1), 3),
        'sir_reliable': 1 if r2 > 0 and 0.3 < R0 < 8.0 else 0
    }
    
    return features

# ============================================
# 3. Extract SIR features for ALL windows
# ============================================
print("\n[3] Extracting SIR features for 953 windows...")
print("    This will take a few minutes...")

all_sir_features = []

for window_start in range(953):
    window_end = window_start + 35
    window_cases = case_data['number'].iloc[window_start:window_end]
    
    try:
        features = estimate_sir_robust(window_cases)
        features['window_id'] = window_start
        all_sir_features.append(features)
    except Exception as e:
        print(f"    Warning: Window {window_start} failed - {e}")
        # Add placeholder with NaN for failed windows
        features = {
            'window_id': window_start,
            'beta': np.nan, 'gamma': np.nan, 'R0': np.nan, 'Re': np.nan,
            'growth_rate': np.nan, 'doubling_time': np.nan, 'sir_fit_r2': -999,
            'sir_residual': np.nan, 'sir_rel_residual': np.nan, 'Re_trend': np.nan,
            'peak_estimate': np.nan, 'initial_cases': np.nan, 'final_cases': np.nan,
            'case_ratio': np.nan, 'sir_reliable': 0
        }
        all_sir_features.append(features)
    
    if (window_start + 1) % 100 == 0:
        print(f"    Processed {window_start + 1} windows...")

sir_df = pd.DataFrame(all_sir_features)
print(f"\n  ✓ Extracted features for {len(sir_df)} windows")
print(f"  ✓ {len(sir_df.columns)} SIR features per window")

# ============================================
# 4. Load RI features from original paper
# ============================================
print("\n[4] Loading original RI features...")
print("    Recreating RI features for all windows...")

def calculate_ri_features(case_data, window_start):
    """Calculate the 8 RI features for a window"""
    
    window = case_data.iloc[window_start:window_start+35].copy()
    window['idx'] = range(len(window))
    
    # Scale total cases to [0,1]
    total_min = window['number'].min()
    total_max = window['number'].max()
    if total_max > total_min:
        window['N_total'] = (window['number'] - total_min) / (total_max - total_min)
    else:
        window['N_total'] = 0
    
    # Split into calibration (21 days) and prediction (14 days)
    calib = window.iloc[:21].copy()
    pred = window.iloc[21:35].copy()
    
    # Linear regression for slopes
    lr_calib = LinearRegression()
    lr_calib.fit(calib['idx'].values.reshape(-1, 1), calib['N_total'].values)
    calib_slope = lr_calib.coef_[0]
    calib_mean = calib['N_total'].mean()
    
    lr_pred = LinearRegression()
    lr_pred.fit(pred['idx'].values.reshape(-1, 1), pred['N_total'].values)
    pred_slope = lr_pred.coef_[0]
    
    calib_std = calib['N_total'].std()
    week = window['date'].iloc[0].weekday()
    
    features = {
        'data_num': window_start,
        'Week': week,
        r'$\mu^c$': round(calib_mean, 4),
        r'$\beta^c$': round(calib_slope, 4),
        r'$\sigma^c$': round(calib_std, 4),
        r'$Delta^c$': 0,  # Not in current data
        r'$Omicron^c$': 0,  # Not in current data
        r'$Policy^c$': 0,  # Not in current data
        r'$Policy^p$': 0  # Not in current data
    }
    
    return features

ri_features_list = []
for window_start in range(953):
    ri_features = calculate_ri_features(case_data, window_start)
    ri_features_list.append(ri_features)
    
    if (window_start + 1) % 100 == 0:
        print(f"    Processed {window_start + 1} RI windows...")

ri_df = pd.DataFrame(ri_features_list)
print(f"\n  ✓ Created RI features for {len(ri_df)} windows")

# ============================================
# 5. Load labels from number2.xlsx
# ============================================
print("\n[5] Loading outbreak labels...")

ri_values = pd.read_excel('temp/number2.xlsx')
print(f"  ✓ Loaded {len(ri_values)} RI values")

# Create labels using terciles (3 equal groups)
ri_values_sorted = ri_values.sort_values('RI').reset_index(drop=True)

n = len(ri_values_sorted)
labels = np.zeros(n, dtype=int)
labels[int(n/3):int(2*n/3)] = 1
labels[int(2*n/3):] = 2

ri_values_sorted['label'] = labels
ri_values_with_labels = ri_values_sorted.sort_index()

print(f"  Label distribution: {ri_values_with_labels['label'].value_counts().sort_index().to_dict()}")

# ============================================
# 6. Combine everything
# ============================================
print("\n[6] Combining all features...")

final_features = ri_df.copy()
final_features['RI'] = ri_values_with_labels['RI'].values
final_features['label'] = ri_values_with_labels['label'].values

final_features = pd.merge(final_features, sir_df, left_on='data_num', right_on='window_id', how='left')
final_features = final_features.drop('window_id', axis=1)

print(f"  Final feature matrix: {final_features.shape}")
print(f"  Number of reliable SIR windows: {final_features['sir_reliable'].sum()} / {len(final_features)}")

# ============================================
# 7. Save the enhanced dataset
# ============================================
print("\n[7] Saving enhanced dataset...")

final_features.to_csv('result/enhanced_pre_data.csv', index=False)
print(f"  ✓ Saved to result/enhanced_pre_data.csv")

# Create train/test split (70/30)
np.random.seed(42)
n_train = int(0.7 * len(final_features))
train_indices = np.random.choice(final_features.index, n_train, replace=False)
test_indices = [i for i in final_features.index if i not in train_indices]

train_enhanced = final_features.iloc[train_indices].copy()
test_enhanced = final_features.iloc[test_indices].copy()

train_enhanced.to_csv('data/train_enhanced.csv', index=False)
test_enhanced.to_csv('data/test_enhanced.csv', index=False)

print(f"  ✓ Train set: {len(train_enhanced)} windows")
print(f"  ✓ Test set: {len(test_enhanced)} windows")

# ============================================
# 8. Summary statistics
# ============================================
print("\n" + "=" * 60)
print("SIR FEATURES SUMMARY")
print("=" * 60)

reliable = final_features[final_features['sir_reliable'] == 1]

print("\nSIR Parameter Ranges (reliable windows only):")
print(f"  β (transmission rate): {reliable['beta'].min():.3f} - {reliable['beta'].max():.3f} (mean: {reliable['beta'].mean():.3f})")
print(f"  γ (recovery rate): {reliable['gamma'].min():.3f} - {reliable['gamma'].max():.3f} (mean: {reliable['gamma'].mean():.3f})")
print(f"  R₀ (basic R): {reliable['R0'].min():.2f} - {reliable['R0'].max():.2f} (mean: {reliable['R0'].mean():.2f})")
print(f"  Re (effective R): {reliable['Re'].min():.2f} - {reliable['Re'].max():.2f} (mean: {reliable['Re'].mean():.2f})")
print(f"  Growth rate: {reliable['growth_rate'].min():.4f} - {reliable['growth_rate'].max():.4f}")
print(f"  SIR fit quality (R²): {reliable['sir_fit_r2'].min():.3f} - {reliable['sir_fit_r2'].max():.3f}")

print("\nCorrelation with RI (reliable windows):")
ri_corr = reliable[['RI', 'Re', 'growth_rate', 'sir_fit_r2']].corr()['RI']
print(ri_corr)

print("\nFirst 5 rows of enhanced data:")
print(final_features[['data_num', 'RI', 'label', 'Re', 'growth_rate', 'sir_fit_r2', 'sir_reliable']].head())


print("SIR FEATURE EXTRACTION COMPLETE!")
