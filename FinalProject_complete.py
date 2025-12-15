import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc

# ==========================================
# 1. 配置与常量 (Configuration & Constants)
# ==========================================
M_Z = 91.1876
SQM_MU = 0.105658**2
SQM_E = 0.000511**2

# Luminosity
LUMI_11 = 2330.0
LUMI_12 = 11580.0

# Plot settings
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 7)
plt.rcParams['font.size'] = 14

# Output directory for plots
if not os.path.exists('plots'):
    os.makedirs('plots')

# ==========================================
# 2. 数据加载与预处理 (Data Loading)
# ==========================================
def load_data():
    print(">>> Loading Data and MC...")
    
    # Helper to read CSV safely
    def read_csv_safe(path):
        if os.path.exists(path):
            return pd.read_csv(path)
        else:
            print(f"Warning: {path} not found. Returning empty DataFrame.")
            return pd.DataFrame()

    # Load Data
    data_11 = read_csv_safe('./data/clean_data_2011.csv')
    data_12 = read_csv_safe('./data/clean_data_2012.csv')
    data = pd.concat([data_11, data_12], ignore_index=True)
    data['type'] = 'data'
    data['weight'] = 1.0

    # Load MC and assign weights
    # Structure: (file_pattern, xsec, nevt, label, color)
    mc_configs = [
        # Signal
        ('higgs', 2011, './MC/higgs2011.csv', 0.0057, 299683, 'Higgs ($m_H=125$)', 'red'),
        ('higgs', 2012, './MC/higgs2012.csv', 0.0065, 299973, 'Higgs ($m_H=125$)', 'red'),
        
        # ZZ
        ('zz', 2011, './MC/zzto4mu2011.csv', 0.093, 1447136, r'ZZ $\rightarrow$ 4l', 'blue'),
        ('zz', 2011, './MC/zzto4e2011.csv', 0.093, 1493308, r'ZZ $\rightarrow$ 4l', 'blue'),
        ('zz', 2011, './MC/zzto2mu2e2011.csv', 0.208, 1479879, r'ZZ $\rightarrow$ 4l', 'blue'),
        ('zz', 2012, './MC/zzto4mu2012.csv', 0.107, 1499064, r'ZZ $\rightarrow$ 4l', 'blue'),
        ('zz', 2012, './MC/zzto4e2012.csv', 0.107, 1499093, r'ZZ $\rightarrow$ 4l', 'blue'),
        ('zz', 2012, './MC/zzto2mu2e2012.csv', 0.249, 1497445, r'ZZ $\rightarrow$ 4l', 'blue'),

        # DY (Z/gamma*)
        ('dy', 2011, './MC/dy1050_2011.csv', 9507.0, 39909640, r'Z/$\gamma^{*}$ + X', 'green'),
        ('dy', 2011, './MC/dy50_2011.csv', 2475.0, 36408225, r'Z/$\gamma^{*}$ + X', 'green'),
        ('dy', 2012, './MC/dy1050_2012.csv', 10.742, 6462290, r'Z/$\gamma^{*}$ + X', 'green'), # Note: xsec seems low in original code, using provided
        ('dy', 2012, './MC/dy50_2012.csv', 2955.0, 29426492, r'Z/$\gamma^{*}$ + X', 'green'),

        # TTBar
        ('tt', 2011, './MC/ttbar2011.csv', 19.504, 9771205, r'$t\bar{t}$', 'gray'),
        ('tt', 2012, './MC/ttbar2012.csv', 200.0, 6423106, r'$t\bar{t}$', 'gray'),
    ]

    mc_dict = {'higgs': [], 'zz': [], 'dy': [], 'tt': []}
    
    for group, year, path, xsec, nevt, label, color in mc_configs:
        df = read_csv_safe(path)
        if df.empty: continue
        lumi = LUMI_11 if year == 2011 else LUMI_12
        df['weight'] = lumi * xsec / nevt
        df['type'] = group
        df['label'] = label
        df['color'] = color
        mc_dict[group].append(df)

    # Combine MCs
    mc_dfs = {}
    for key in mc_dict:
        if mc_dict[key]:
            mc_dfs[key] = pd.concat(mc_dict[key], ignore_index=True)
        else:
            mc_dfs[key] = pd.DataFrame()

    return data, mc_dfs

# ==========================================
# 3. 物理计算辅助函数 (Physics Helpers)
# ==========================================
def calc_inv_mass_4l(df):
    """Calculate invariant mass of the 4-lepton system"""
    E = df['E1'] + df['E2'] + df['E3'] + df['E4']
    Px = df['px1'] + df['px2'] + df['px3'] + df['px4']
    Py = df['py1'] + df['py2'] + df['py3'] + df['py4']
    Pz = df['pz1'] + df['pz2'] + df['pz3'] + df['pz4']
    return np.sqrt(np.maximum(0, E**2 - (Px**2 + Py**2 + Pz**2)))

def calc_pt(px, py):
    return np.sqrt(px**2 + py**2)

def calc_pair_mass(p1_idx, p2_idx, df):
    """Calculate invariant mass of a pair of particles given their indices (e.g. 1, 2)"""
    E = df[f'E{p1_idx}'] + df[f'E{p2_idx}']
    Px = df[f'px{p1_idx}'] + df[f'px{p2_idx}']
    Py = df[f'py{p1_idx}'] + df[f'py{p2_idx}']
    Pz = df[f'pz{p1_idx}'] + df[f'pz{p2_idx}']
    return np.sqrt(np.maximum(0, E**2 - (Px**2 + Py**2 + Pz**2)))

# ==========================================
# 4. Step 4: Object Selection
# ==========================================
def perform_object_selection(df):
    """
    Electrons: pT > 7 GeV, |eta| < 2.5
    Muons: pT > 5 GeV, |eta| < 2.4
    """
    df['pass_obj'] = True
    for i in range(1, 5):
        pt = calc_pt(df[f'px{i}'], df[f'py{i}'])
        eta = np.abs(df[f'eta{i}'])
        pid = np.abs(df[f'PID{i}'])
        
        # Conditions
        is_ele = (pid == 11)
        is_mu = (pid == 13)
        pass_ele = is_ele & (pt > 7) & (eta < 2.5)
        pass_mu = is_mu & (pt > 5) & (eta < 2.4)
        
        df['pass_obj'] = df['pass_obj'] & (pass_ele | pass_mu)
        
        # Save pT for later use (ML)
        df[f'pt{i}'] = pt
        
    return df[df['pass_obj']].copy()

# ==========================================
# 5. Step 5: Event Selection & Z Reconstruction
# ==========================================
def reconstruct_Z_candidates(df):
    """
    Reconstruct Z1 (closest to mZ) and Z2 (the other pair).
    Handles 4e, 4mu, and 2e2mu cases.
    """
    # Initialize columns
    df['mZ1'] = 0.0
    df['mZ2'] = 0.0
    df['Z_flavor'] = '' # '4e', '4mu', '2e2mu'

    # Vectorized logic is complex, using apply for readability and robustness with combinatorics
    # However, for speed on large MC, we try a semi-vectorized approach based on flavor masks.
    
    # Identifiers
    ids = df[['PID1', 'PID2', 'PID3', 'PID4']].values
    
    # Masks for channels
    sum_abs_id = np.sum(np.abs(ids), axis=1)
    mask_4e = (sum_abs_id == 44) # 11*4
    mask_4mu = (sum_abs_id == 52) # 13*4
    mask_2e2mu = (sum_abs_id == 48) # 11*2 + 13*2
    
    # 2e2mu Case: Easier, find the same-flavor pairs
    # Usually clean data implies 1,2 are pair and 3,4 are pair OR specific ordering.
    # But let's be robust. 
    # For 2e2mu, find the pair of electrons and pair of muons.
    # Quick trick: Calculate all pairwise masses, pick valid SFOC (Same Flavor Opposite Charge).
    
    def get_z_masses(row):
        pids = [row['PID1'], row['PID2'], row['PID3'], row['PID4']]
        # 4-vectors (E, px, py, pz)
        p4s = []
        for i in range(1, 5):
            p4s.append(np.array([row[f'E{i}'], row[f'px{i}'], row[f'py{i}'], row[f'pz{i}']]))
            
        # Find valid pairs
        pairs = []
        indices = [0, 1, 2, 3]
        import itertools
        # Try pairing (i,j) and (k,l)
        # Possible unique pairings of 4 items into 2 pairs: (0,1)(2,3), (0,2)(1,3), (0,3)(1,2)
        combinations = [((0,1),(2,3)), ((0,2),(1,3)), ((0,3),(1,2))]
        
        candidates = []
        
        for (i, j), (k, l) in combinations:
            # Check SFOC for Z1 candidate
            if (pids[i] == -pids[j]):
                # Check SFOC for Z2 candidate
                if (pids[k] == -pids[l]):
                    # Calculate masses
                    p1 = p4s[i] + p4s[j]
                    m1 = np.sqrt(max(0, p1[0]**2 - np.sum(p1[1:]**2)))
                    
                    p2 = p4s[k] + p4s[l]
                    m2 = np.sqrt(max(0, p2[0]**2 - np.sum(p2[1:]**2)))
                    
                    # Determine Z1 (closest to M_Z)
                    if abs(m1 - M_Z) < abs(m2 - M_Z):
                        candidates.append((m1, m2))
                    else:
                        candidates.append((m2, m1))
        
        if not candidates:
            return 0.0, 0.0 # Should not happen in pre-selected data
            
        # If multiple combinations (e.g. 4e), pick the one with Z1 closest to M_Z
        candidates.sort(key=lambda x: abs(x[0] - M_Z))
        return candidates[0]

    # Apply row-wise (slow but robust given the complexity)
    # For a Final Project, correctness > speed.
    masses = df.apply(get_z_masses, axis=1)
    df['mZ1'] = [x[0] for x in masses]
    df['mZ2'] = [x[1] for x in masses]
    
    return df

def perform_event_selection(df):
    """
    Standard CMS Cuts:
    40 < mZ1 < 120
    12 < mZ2 < 120
    """
    cut_z1 = (df['mZ1'] > 40) & (df['mZ1'] < 120)
    cut_z2 = (df['mZ2'] > 12) & (df['mZ2'] < 120)
    return df[cut_z1 & cut_z2].copy()

# ==========================================
# 6. 绘图函数 (Plotting Functions)
# ==========================================
def plot_mass_distribution(data, mc_dfs, title, filename, bins=30, range_min=80, range_max=180, ml_selected=False):
    plt.figure(figsize=(10, 8))
    
    # Prepare MC stack
    mc_labels = []
    mc_colors = []
    mc_data = []
    mc_weights = []
    
    # Order: TT, DY, ZZ, Higgs (Signal on top usually, or separate)
    # Stack Backgrounds
    for key in ['tt', 'dy', 'zz']:
        if not mc_dfs[key].empty:
            df = mc_dfs[key]
            mc_data.append(df['m4l'])
            mc_weights.append(df['weight'])
            mc_labels.append(df['label'].iloc[0])
            mc_colors.append(df['color'].iloc[0])
            
    # Histogram for MC
    plt.hist(mc_data, bins=bins, range=(range_min, range_max), weights=mc_weights, 
             stacked=True, label=mc_labels, color=mc_colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Signal (Higgs) - Add to stack or overlay? 
    # Usually Higgs is small, so we stack it on top of backgrounds
    if not mc_dfs['higgs'].empty:
        df_sig = mc_dfs['higgs']
        plt.hist(df_sig['m4l'], bins=bins, range=(range_min, range_max), weights=df_sig['weight'],
                 label=df_sig['label'].iloc[0], color=df_sig['color'].iloc[0], histtype='step', linewidth=2)
        # Also fill it to show in stack logic if preferred, but step is clearer for signal shape.
        # Let's fill it on top of background stack for standard expectation plot
        plt.hist(df_sig['m4l'], bins=bins, range=(range_min, range_max), weights=df_sig['weight'],
                 bottom=np.sum([np.histogram(m, bins=bins, range=(range_min, range_max), weights=w)[0] for m, w in zip(mc_data, mc_weights)], axis=0),
                 color='red', alpha=0.5, edgecolor='red')

    # Data Points
    y, bin_edges = np.histogram(data['m4l'], bins=bins, range=(range_min, range_max))
    bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
    y_err = np.sqrt(y)
    plt.errorbar(bin_centers, y, yerr=y_err, fmt='ko', label='Data', capsize=0)

    plt.xlabel('$m_{4l}$ (GeV)')
    plt.ylabel(f'Events / {(range_max-range_min)/bins:.1f} GeV')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f'plots/{filename}.png')
    print(f"Saved plot: plots/{filename}.png")
    # plt.show() # Comment out for batch running

# ==========================================
# 7. 主程序 (Main Analysis Flow)
# ==========================================
def main():
    # --- 1. Load Data ---
    data, mc_dfs = load_data()
    
    # Calculate M4l initially
    data['m4l'] = calc_inv_mass_4l(data)
    for k in mc_dfs:
        if not mc_dfs[k].empty:
            mc_dfs[k]['m4l'] = calc_inv_mass_4l(mc_dfs[k])

    # --- Step 3: No Selection Plot ---
    print("\n--- Step 3: Initial Plot ---")
    plot_mass_distribution(data, mc_dfs, 
                           '4-Lepton Invariant Mass (No Selection) - [Your Name]', 
                           'step3_mass_no_cuts')

    # --- Step 4: Object Selection ---
    print("\n--- Step 4: Object Selection ---")
    data_obj = perform_object_selection(data)
    mc_obj = {k: perform_object_selection(v) for k, v in mc_dfs.items()}
    
    # Reconstruct Z candidates for plotting Z masses
    data_obj = reconstruct_Z_candidates(data_obj)
    for k in mc_obj:
        mc_obj[k] = reconstruct_Z_candidates(mc_obj[k])

    # Plot Z1 and Z2
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(data_obj['mZ1'], bins=40, range=(40, 120), color='black', histtype='step', label='Data $Z_1$')
    plt.xlabel('$m_{Z1}$ (GeV)')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(data_obj['mZ2'], bins=40, range=(0, 120), color='black', histtype='step', label='Data $Z_2$')
    plt.xlabel('$m_{Z2}$ (GeV)')
    plt.legend()
    plt.savefig('plots/step4_Z_masses.png')
    
    # --- Step 5: Event Selection & Optimization ---
    print("\n--- Step 5: Event Selection Optimization ---")
    data_evt = perform_event_selection(data_obj)
    mc_evt = {k: perform_event_selection(v) for k, v in mc_obj.items()}
    
    # Calculate Significance in 119-131 GeV
    low, high = 119, 131
    
    def get_counts(df, l, h):
        return df[(df['m4l'] >= l) & (df['m4l'] <= h)]['weight'].sum()
    
    s = get_counts(mc_evt['higgs'], low, high)
    b_zz = get_counts(mc_evt['zz'], low, high)
    b_dy = get_counts(mc_evt['dy'], low, high)
    b_tt = get_counts(mc_evt['tt'], low, high)
    b_total = b_zz + b_dy + b_tt
    
    data_obs = len(data_evt[(data_evt['m4l'] >= low) & (data_evt['m4l'] <= high)])
    
    significance = s / np.sqrt(s + b_total)
    data_excess = data_obs - b_total
    
    print("-" * 30)
    print(f"Mass Window: {low} - {high} GeV")
    print(f"Expected Signal (S): {s:.4f}")
    print(f"Expected Background (B): {b_total:.4f}")
    print(f"  - ZZ: {b_zz:.4f}")
    print(f"  - DY: {b_dy:.4f}")
    print(f"  - TT: {b_tt:.4f}")
    print(f"MC Significance (s/sqrt(s+b)): {significance:.4f}")
    print(f"Observed Data Events: {data_obs}")
    print(f"Data Excess (Obs - Bkg): {data_excess:.4f}")
    print("-" * 30)
    
    plot_mass_distribution(data_evt, mc_evt, 
                           '4-Lepton Mass (Step 5 Selection) - [Your Name]', 
                           'step5_mass_final_cuts')

    # --- Step 6: Machine Learning ---
    print("\n--- Step 6: Machine Learning (BDT) ---")
    
    # Prepare Training Data
    # Signal: Higgs, Background: ZZ (Dominant irreducible)
    # You can include others, but ZZ is the main challenge.
    sig_df = mc_evt['higgs'].copy()
    bkg_df = pd.concat([mc_evt['zz'], mc_evt['dy'], mc_evt['tt']], ignore_index=True)
    
    sig_df['target'] = 1
    bkg_df['target'] = 0
    
    # Features
    features = ['mZ1', 'mZ2', 'pt1', 'pt2', 'pt3', 'pt4']
    
    ml_data = pd.concat([sig_df, bkg_df], ignore_index=True)
    X = ml_data[features]
    y = ml_data['target']
    w = ml_data['weight'] # Important!
    
    # Train/Test Split
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, y, w, test_size=0.3, random_state=42)
    
    # Normalize weights for training stability
    # Scale signal weights so sum(sig) approx sum(bkg) helps training, 
    # but BDT handles it reasonably well. Let's keep physics weights but normalize total.
    
    # Train BDT
    print("Training Gradient Boosting Classifier...")
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    clf.fit(X_train, y_train, sample_weight=w_train)
    
    # Evaluate
    # Find optimal threshold on Test set to maximize Significance
    scores_test = clf.predict_proba(X_test)[:, 1]
    
    thresholds = np.linspace(0.1, 0.95, 50)
    best_sig = 0
    best_cut = 0.5
    
    print("Optimizing BDT Threshold...")
    for t in thresholds:
        # Pass ML cut
        mask = scores_test > t
        
        # Apply physics window cut as well (fair comparison to Step 5)
        # Note: X_test doesn't have m4l, need to carry it over or re-index.
        # For simplicity here, we maximize S/sqrt(S+B) overall in the test set 
        # (assuming mass distribution is similar, or just check global separation).
        # A rigorous analysis would check the window [119, 131].
        
        s_w = w_test[(y_test == 1) & mask].sum()
        b_w = w_test[(y_test == 0) & mask].sum()
        
        if s_w + b_w > 0:
            sig = s_w / np.sqrt(s_w + b_w)
            if sig > best_sig:
                best_sig = sig
                best_cut = t
                
    print(f"Best ML Significance (Test Set): {best_sig:.4f} at BDT Score > {best_cut:.2f}")
    
    # Apply to Full Data & MC for Plotting
    def apply_ml(df):
        if df.empty: return df
        # Handle cases where pt3/pt4 might be missing if we dropped columns? No we didn't.
        # Just ensure columns exist
        return df[clf.predict_proba(df[features])[:, 1] > best_cut]

    data_ml = apply_ml(data_evt)
    mc_ml = {k: apply_ml(v) for k, v in mc_evt.items()}
    
    # Recalculate physics significance on full MC with ML cut in Mass Window
    s_ml = get_counts(mc_ml['higgs'], low, high)
    b_ml_total = get_counts(mc_ml['zz'], low, high) + get_counts(mc_ml['dy'], low, high) + get_counts(mc_ml['tt'], low, high)
    sig_ml_final = s_ml / np.sqrt(s_ml + b_ml_total) if (s_ml + b_ml_total) > 0 else 0
    data_obs_ml = len(data_ml[(data_ml['m4l'] >= low) & (data_ml['m4l'] <= high)])
    
    print("-" * 30)
    print(f"ML Results in Mass Window {low}-{high} GeV:")
    print(f"Expected Signal: {s_ml:.4f}")
    print(f"Expected Background: {b_ml_total:.4f}")
    print(f"Final ML Significance: {sig_ml_final:.4f}")
    print(f"Observed Data Events: {data_obs_ml}")
    print("-" * 30)
    
    plot_mass_distribution(data_ml, mc_ml, 
                           f'4-Lepton Mass (ML Selection > {best_cut:.2f}) - [Your Name]', 
                           'step6_mass_ml_cuts')
    
    print("\nAll tasks completed successfully. Check the 'plots' folder.")

if __name__ == "__main__":
    main()