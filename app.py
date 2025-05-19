import streamlit as st
import os
import io
import base64
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.signal import savgol_filter
import pandas as pd
from baseline_targets import JM1_5128, JM1_711_DELTA, DF_5128, DF_711_DELTA
from dataclasses import dataclass, asdict
import json

# Set page config
st.set_page_config(page_title="Unsquigger", layout="wide")

# Constants
TREBLE_START_HZ = 5000
REF_FREQ_HZ = 630

@dataclass
class EQFilter:
    type: str
    fc: float
    gain: float
    Q: float

# Helper functions
def peak_eq(f, fc, gain_db, Q):
    return gain_db / (1 + ((np.log2(f / fc))**2) / Q**2)

def high_shelf(f, fc, gain_db, Q):
    return gain_db / (1 + (fc / f)**(2 * Q))

def load_txt(uploaded_file):
    try:
        if isinstance(uploaded_file, str):
            lines = uploaded_file.splitlines()
        else:
            content = uploaded_file.read()
            if isinstance(content, bytes):
                lines = content.decode('utf-8').splitlines()
            else:
                lines = content.splitlines()

        start_idx = 0
        for i, line in enumerate(lines):
            if "Freq(Hz) SPL(dB)" in line:
                start_idx = i + 1
                break

        data_lines = []
        for line in lines[start_idx:]:
            if line.strip() and not line.strip().startswith('*'):
                values = line.strip().split()[:2]
                if len(values) == 2:
                    data_lines.append(values)

        df = pd.DataFrame(data_lines, columns=['freq', 'value'])
        df['freq'] = pd.to_numeric(df['freq'], errors='coerce')
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df.dropna()
        df = df.sort_values(by='freq')

        freq_array = df['freq'].to_numpy()
        value_array = df['value'].to_numpy()

        if len(freq_array) == 0 or len(value_array) == 0:
            raise ValueError("Empty data after processing")

        return freq_array, value_array
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None, None

def smooth_measurement(freq, value, points_per_octave=24):
    num_points = len(freq)
    if num_points < 10:
        return value

    log_freq = np.log10(freq)
    interp = interp1d(log_freq, value, kind='linear', fill_value='extrapolate')
    log_uniform = np.linspace(log_freq[0], log_freq[-1], num_points)
    value_uniform = interp(log_uniform)

    window_len = max(5, int(num_points / points_per_octave) | 1)
    smoothed = savgol_filter(value_uniform, window_length=window_len, polyorder=2)

    interp_back = interp1d(log_uniform, smoothed, kind='linear', fill_value='extrapolate')
    return interp_back(np.log10(freq))

def apply_filters(freq, filters):
    total = np.zeros_like(freq)
    for f in filters:
        if f['type'] == 'peak':
            total += peak_eq(freq, f['fc'], f['gain'], f['Q'])
        elif f['type'] == 'shelf':
            total += high_shelf(freq, f['fc'], f['gain'], f['Q'])
    return total

def generate_target(meas_freq, meas_val, jm1_freq, jm1_val, rig_type="5128"):
    # Create frequency points for interpolation
    freq = np.logspace(np.log10(20), np.log10(20000), 1000)
    
    # Create interpolation functions with extrapolation
    interp_meas = interp1d(meas_freq, meas_val, kind='linear', fill_value='extrapolate')
    interp_jm1 = interp1d(jm1_freq, jm1_val, kind='linear', fill_value='extrapolate')

    # Interpolate measurements to the common frequency points
    meas = interp_meas(freq)
    baseline = interp_jm1(freq)

    # Find reference values at 630 Hz
    meas_ref = float(interp_meas(REF_FREQ_HZ))
    baseline_ref = float(interp_jm1(REF_FREQ_HZ))
    
    # Normalize measurement to baseline at reference frequency
    ref_diff = meas_ref - baseline_ref
    meas_aligned = meas - ref_diff  # Align measurement to baseline

    # Create resonance mask
    resonance_mask = np.ones_like(freq, dtype=bool)
    
    # Mask out resonance regions based on rig type
    if rig_type == "5128":
        # For 5128, handle rocking mode differently
        resonance_regions = []
        
        # Create separate mask for rocking mode region
        rocking_mode_region = (freq >= 100) & (freq <= 450)
        
        # For the rocking mode region, create a smoothed reference line
        if np.any(rocking_mode_region):
            rocking_indices = np.where(rocking_mode_region)[0]
            rocking_freq = freq[rocking_indices]
            rocking_response = meas[rocking_indices]
            
            # Use a larger window for smoother averaging
            smoothed_rocking = savgol_filter(rocking_response, 
                                           window_length=51, 
                                           polyorder=2)
            
            # Replace the measurement values in this region with smoothed version
            meas[rocking_indices] = smoothed_rocking
    else:  # 711
        resonance_regions = [
            (7500, 8500),    # 8k resonance
            (16500, 17500)   # 17k resonance
        ]
    
    # Mask out high frequency resonances
    for low, high in resonance_regions:
        mask_region = (freq >= low) & (freq <= high)
        resonance_mask[mask_region] = False

    # Calculate error between aligned measurement and baseline
    raw_error = meas_aligned - baseline
    raw_error_masked = raw_error[resonance_mask]
    freq_masked = freq[resonance_mask]
    
    # Smooth the error
    smoothed_error = savgol_filter(raw_error_masked, window_length=31, polyorder=3)

    def apply_filters(freq, filters):
        total = np.zeros_like(freq)
        for f in filters:
            if f['type'] == 'peak':
                total += peak_eq(freq, f['fc'], f['gain'], f['Q'])
            elif f['type'] == 'shelf':
                total += high_shelf(freq, f['fc'], f['gain'], f['Q'])
        return total

    def loss(params):
        bass_peak = {'type': 'peak', 'fc': params[0], 'gain': params[1], 'Q': params[2]}
        bass_shelf = {'type': 'shelf', 'fc': params[3], 'gain': params[4], 'Q': params[5]}
        pinna_peak1 = {'type': 'peak', 'fc': params[6], 'gain': params[7], 'Q': params[8]}
        pinna_peak2 = {'type': 'peak', 'fc': params[9], 'gain': params[10], 'Q': params[11]}
        treble_shelf = {'type': 'shelf', 'fc': params[12], 'gain': params[13], 'Q': params[14]}
        
        filters = [bass_peak, bass_shelf, pinna_peak1, pinna_peak2, treble_shelf]
        filtered_baseline = baseline[resonance_mask] + apply_filters(freq_masked, filters)
        return np.mean((meas_aligned[resonance_mask] - filtered_baseline)**2)

    # Initial filters
    initial_filters = [
        50, 3, 0.7,      # Bass peak 1
        80, 4, 0.7,      # Bass shelf
        1500, 2, 0.8,    # Pinna peak 1
        3000, 2, 0.8,    # Pinna peak 2
        4500, 5, 1.2     # Treble shelf
    ]

    bounds = [
        (20, 20), (-30, 30), (0.2, 6),     # Bass peak 1
        (20, 500), (-30, 30), (0.2, 1),     # Low shelf
        (1250, 2000), (-30, 30), (.5, 3),          # Pinna peak 1
        (2000, 4000), (-30, 30), (.5, 3),          # Pinna peak 2
        (4500, 20000), (-30, 30), (0.2, 1)    # Treble shelf
    ]

    # Optimize filters
    np.random.seed(42)
    result = minimize(loss, initial_filters, bounds=bounds, method="L-BFGS-B", 
         options={'ftol': 1e-8, 'gtol': 1e-8, 'maxiter': 500})

    # Create filters from optimized parameters
    filters = [
        {'type': 'peak', 'fc': result.x[0], 'gain': result.x[1], 'Q': result.x[2]},
        {'type': 'shelf', 'fc': result.x[3], 'gain': result.x[4], 'Q': result.x[5]},
        {'type': 'peak', 'fc': result.x[6], 'gain': result.x[7], 'Q': result.x[8]},
        {'type': 'peak', 'fc': result.x[9], 'gain': result.x[10], 'Q': result.x[11]},
        {'type': 'shelf', 'fc': result.x[12], 'gain': result.x[13], 'Q': result.x[14]}
    ]

    # Generate target by applying filters to baseline
    target = baseline + apply_filters(freq, filters)

    return filters, freq, target, meas_aligned, baseline

def dynamic_eq_adjustment(st, filters, freq, baseline, meas, rig_type, target_type):
    st.header("Dynamic EQ Filter Adjustment")
    
    # Create columns for each filter
    cols = st.columns(len(filters))
    
    adjusted_filters = []
    for i, (col, filt) in enumerate(zip(cols, filters)):
        with col:
            st.subheader(f"Filter {i+1}")
            filter_type = st.selectbox(f"Type {i+1}", 
                                       ["peak", "shelf"], 
                                       index=0 if filt['type'] == 'peak' else 1,
                                       key=f"type_{i}")
            
            fc = st.number_input(f"Frequency {i+1} (Hz)", 
                                 min_value=20.0, 
                                 max_value=20000.0, 
                                 value=filt['fc'],
                                 key=f"fc_{i}")
            
            gain = st.number_input(f"Gain {i+1} (dB)", 
                                   min_value=-30.0, 
                                   max_value=30.0, 
                                   value=filt['gain'],
                                   key=f"gain_{i}")
            
            q = st.number_input(f"Q {i+1}", 
                                min_value=0.2, 
                                max_value=6.0, 
                                value=filt['Q'],
                                step=0.1,
                                key=f"Q_{i}")
            
            adjusted_filters.append({
                'type': filter_type,
                'fc': fc,
                'gain': gain,
                'Q': q
            })
    
    # Additional baselines for cross-rig application
    additional_targets = {}
    
    if rig_type == "5128":
        # Load 711 baseline for cross-application
        if target_type == "jm1":
            with io.StringIO(JM1_711_DELTA) as f:
                cross_rig_freq, cross_rig_val = load_txt(f)
            label_prefix = "JM-1"
        else:
            with io.StringIO(DF_711_DELTA) as f:
                cross_rig_freq, cross_rig_val = load_txt(f)
            label_prefix = "DF"
        
        if cross_rig_freq is not None:
            # Interpolate 711 baseline to match frequency points
            interp_711 = interp1d(cross_rig_freq, cross_rig_val, kind='linear', fill_value='extrapolate')
            baseline_711 = interp_711(freq)
            
            additional_targets['711'] = (baseline_711, label_prefix)
    
    # Create a placeholder for dynamic plots
    plot_placeholder = st.empty()
    
    # Find reference values at 630 Hz
    ref_idx = np.abs(freq - REF_FREQ_HZ).argmin()
    baseline_ref = baseline[ref_idx]
    meas_ref = meas[ref_idx]
    
    # Reapply the filters to original baseline
    adjusted_target = baseline + apply_filters(freq, adjusted_filters)
    
    # Find the new reference value at 630 Hz for the adjusted target
    adjusted_ref = adjusted_target[ref_idx]
    
    # Normalize all curves to their respective 630 Hz reference points
    normalized_meas = meas - meas_ref
    normalized_baseline = baseline - baseline_ref
    normalized_adjusted_target = adjusted_target - adjusted_ref
    
    # Define BA bass compensation filter - corrected to use low shelf
    ba_filter = {'type': 'shelf', 'fc': 100.0, 'gain': 1.2, 'Q': 0.6}
    
    # Function for low shelf filter
    def low_shelf(f, fc, gain_db, Q):
        return gain_db / (1 + (f / fc)**(2 * Q))
    
    # Get the target type label for plot titles and labels
    target_label = "JM-1" if target_type == "jm1" else "Diffuse Field"
    
    # Create plots based on rig type
    if rig_type == "5128":
        # Create two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        
        # First plot: Original Rig Target
        ax1.semilogx(freq, normalized_meas, color='#888888', alpha=0.6, linewidth=1.5, label="Measurement (1/24 Octave)")
        ax1.semilogx(freq, normalized_baseline, color='#AAAAAA', linestyle='--', linewidth=2, label=f"{target_label} Target")
        ax1.semilogx(freq, normalized_adjusted_target, color='#000000', linewidth=2.5, label=f"Adjusted {target_label} Target")
        
        # Set plot properties for first plot
        ax1.set_title(f"Adjusted {rig_type} {target_label} Target (Normalized at {REF_FREQ_HZ} Hz)", fontweight='bold')
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Relative Amplitude (dB)")
        ax1.grid(True, which="both", linestyle='--', color='#E0E0E0', alpha=0.7)
        ax1.legend(frameon=False)
        ax1.set_xlim(20, 20000)
        ax1.set_ylim(-20, 20)
        ax1.set_xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
        ax1.set_xticklabels(['20', '50', '100', '200', '500', '1k', '2k', '5k', '10k', '20k'])
        
        # Second plot: Cross-Rig Targets with BA compensation
        for rig, (target, label_prefix) in additional_targets.items():
            # Apply adjusted filters to target
            cross_target = target + apply_filters(freq, adjusted_filters)
            
            # Create BA compensated version (using low shelf filter)
            ba_compensated_target = cross_target + low_shelf(freq, ba_filter['fc'], ba_filter['gain'], ba_filter['Q'])
            
            # Find reference values at 630 Hz for normalization
            cross_ref = target[ref_idx]
            filtered_cross_ref = cross_target[ref_idx]
            ba_cross_ref = ba_compensated_target[ref_idx]
            
            # Normalize all targets
            normalized_original_cross = target - cross_ref
            normalized_cross_target = cross_target - filtered_cross_ref
            normalized_ba_target = ba_compensated_target - ba_cross_ref
            
            # Plot cross-rig curves
            ax2.semilogx(freq, normalized_original_cross, color='#AAAAAA', linestyle='--', linewidth=2, label=f"{label_prefix} {rig} Target")
            ax2.semilogx(freq, normalized_cross_target, color='#000000', linewidth=2.5, label=f"Adjusted {label_prefix} {rig} Target")
            ax2.semilogx(freq, normalized_ba_target, color='#FF5733', linewidth=2.5, linestyle='-.', label=f"Adjusted {label_prefix} {rig} (with BA Bass Compensation)")
            
            # Set plot properties for second plot
            ax2.set_title(f"Filters Applied to {rig} {label_prefix} Target (Normalized at {REF_FREQ_HZ} Hz)", fontweight='bold')
            ax2.set_xlabel("Frequency (Hz)")
            ax2.set_ylabel("Relative Amplitude (dB)")
            ax2.grid(True, which="both", linestyle='--', color='#E0E0E0', alpha=0.7)
            ax2.legend(frameon=False)
            ax2.set_xlim(20, 20000)
            ax2.set_ylim(-20, 20)
            ax2.set_xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
            ax2.set_xticklabels(['20', '50', '100', '200', '500', '1k', '2k', '5k', '10k', '20k'])
        
        # Adjust layout
        plt.tight_layout()
    else:
        # For 711, create a single plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot curves
        ax.semilogx(freq, normalized_meas, color='#888888', alpha=0.6, linewidth=1.5, label="Measurement (1/24 Octave)")
        ax.semilogx(freq, normalized_baseline, color='#AAAAAA', linestyle='--', linewidth=2, label=f"{target_label} {rig_type} Target")
        ax.semilogx(freq, normalized_adjusted_target, color='#000000', linewidth=2.5, label=f"Adjusted {target_label} Target")
        
        # Set plot properties
        ax.set_title(f"Adjusted {rig_type} {target_label} Target (Normalized at {REF_FREQ_HZ} Hz)", fontweight='bold')
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Relative Amplitude (dB)")
        ax.grid(True, which="both", linestyle='--', color='#E0E0E0', alpha=0.7)
        ax.legend(frameon=False)
        ax.set_xlim(20, 20000)
        ax.set_ylim(-20, 20)
        ax.set_xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
        ax.set_xticklabels(['20', '50', '100', '200', '500', '1k', '2k', '5k', '10k', '20k'])
        
        # Adjust layout
        plt.tight_layout()
    
    # Display the plot
    plot_placeholder.pyplot(fig)
    
    # Export options
    if rig_type == "5128":
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            # Export Filters as plain text
            filter_text = "# Unsquigger EQ Filters\n"
            filter_text += "# Frequency (Hz) | Gain (dB) | Q | Type\n"
            filter_text += "-" * 50 + "\n"

            for i, f in enumerate(adjusted_filters, 1):
                filter_text += f"Filter {i}: {f['fc']:.1f} Hz | {f['gain']:.1f} dB | Q: {f['Q']:.2f} | {f['type']}\n"

            if st.download_button(
                label="Export EQ Filters",
                data=filter_text,
                file_name="eq_filters.txt",
                mime="text/plain"
            ):
                st.success("EQ Filters exported!")
        
        with col2:
            # Export Original Target
            target_df = pd.DataFrame({"Frequency (Hz)": freq, "Target (dB)": adjusted_target})
            csv = target_df.to_csv(sep='\t', index=False)
            if st.download_button(
                label=f"Export {rig_type} {target_label} Target",
                data=csv,
                file_name=f"{rig_type}_{target_type}_adjusted_target.txt",
                mime="text/tab-separated-values"
            ):
                st.success(f"{rig_type} {target_label} Target exported!")
        
        with col3:
            # Export Cross-Rig Targets
            for rig, (target, label_prefix) in additional_targets.items():
                cross_target = target + apply_filters(freq, adjusted_filters)
                cross_df = pd.DataFrame({"Frequency (Hz)": freq, "Target (dB)": cross_target})
                cross_csv = cross_df.to_csv(sep='\t', index=False)
                if st.download_button(
                    label=f"Export {rig} {label_prefix} Target",
                    data=cross_csv,
                    file_name=f"{rig}_{target_type}_baseline_target.txt",
                    mime="text/tab-separated-values"
                ):
                    st.success(f"{rig} {label_prefix} Target exported!")
        
        with col4:
            # Export Adjusted Cross-Rig Targets
            for rig, (target, label_prefix) in additional_targets.items():
                cross_target = target + apply_filters(freq, adjusted_filters)
                cross_df = pd.DataFrame({"Frequency (Hz)": freq, "Target (dB)": cross_target})
                cross_csv = cross_df.to_csv(sep='\t', index=False)
                if st.download_button(
                    label=f"Export Adjusted {rig} {label_prefix} Target",
                    data=cross_csv,
                    file_name=f"{rig}_{target_type}_adjusted_target.txt",
                    mime="text/tab-separated-values"
                ):
                    st.success(f"Adjusted {rig} {label_prefix} Target exported!")
        
        with col5:
            # Export BA Compensated Adjusted Cross-Rig Targets
            for rig, (target, label_prefix) in additional_targets.items():
                cross_target = target + apply_filters(freq, adjusted_filters)
                ba_compensated_target = cross_target + low_shelf(freq, ba_filter['fc'], ba_filter['gain'], ba_filter['Q'])
                ba_cross_df = pd.DataFrame({"Frequency (Hz)": freq, "Target (dB)": ba_compensated_target})
                ba_cross_csv = ba_cross_df.to_csv(sep='\t', index=False)
                if st.download_button(
                    label=f"Export Adjusted {rig} {label_prefix} + BA Bass Comp",
                    data=ba_cross_csv,
                    file_name=f"{rig}_{target_type}_adjusted_ba_target.txt",
                    mime="text/tab-separated-values"
                ):
                    st.success(f"Adjusted {rig} {label_prefix} + BA Bass Comp Target exported!")
    
    else:  # 711 rig
        col1, col2 = st.columns(2)
        
        with col1:
            # Export Filters as plain text
            filter_text = "# Unsquigger EQ Filters\n"
            filter_text += "# Frequency (Hz) | Gain (dB) | Q | Type\n"
            filter_text += "-" * 50 + "\n"

            for i, f in enumerate(adjusted_filters, 1):
                filter_text += f"Filter {i}: {f['fc']:.1f} Hz | {f['gain']:.1f} dB | Q: {f['Q']:.2f} | {f['type']}\n"

            if st.download_button(
                label="Export EQ Filters",
                data=filter_text,
                file_name="eq_filters.txt",
                mime="text/plain"
            ):
                st.success("EQ Filters exported!")
        
        with col2:
            # Export Original Target
            target_df = pd.DataFrame({"Frequency (Hz)": freq, "Target (dB)": adjusted_target})
            csv = target_df.to_csv(sep='\t', index=False)
            if st.download_button(
                label=f"Export {rig_type} {target_label} Target",
                data=csv,
                file_name=f"{rig_type}_{target_type}_adjusted_target.txt",
                mime="text/tab-separated-values"
            ):
                st.success(f"{rig_type} {target_label} Target exported!")
    
    return adjusted_filters

# Streamlit UI
st.title("Unsquigger")

# Sidebar for inputs
with st.sidebar:
    st.header("Trace Your Graph")
    
    # Add a button to link to the measurement tracer
    st.markdown("""
    <a href="https://usyless.uk/trace/" target="_blank">
        <button style="width:100%; padding:10px; background-color:#4CAF50; color:white; border:none; border-radius:5px;">
            UsyTrace
        </button>
    </a>
    """, unsafe_allow_html=True)
    
    st.divider()  # Optional: adds a visual separator
    
    # Add target type selection
    target_type = st.selectbox("Select HRTF Type", ["jm1", "df"], 
                              format_func=lambda x: "JM-1" if x == "jm1" else "Diffuse Field")
    
    rig_type = st.selectbox("Select Rig Type", ["5128", "711"])
    uploaded_file = st.file_uploader("Upload Measurement Text File", type=['txt'])

# Main content
if uploaded_file is not None:
    # Load measurement data
    meas_freq, meas_val = load_txt(uploaded_file)
    
    if meas_freq is not None:
        # Select baseline data based on rig type and target type
        if rig_type == "5128":
            if target_type == "jm1":
                baseline_data = JM1_5128
                target_label = "JM-1"
            else:
                baseline_data = DF_5128
                target_label = "Diffuse Field"
        else:  # 711
            if target_type == "jm1":
                baseline_data = JM1_711_DELTA
                target_label = "JM-1"
            else:
                baseline_data = DF_711_DELTA
                target_label = "Diffuse Field"
        
        # Create a temporary file for baseline data
        with io.StringIO(baseline_data) as f:
            baseline_freq, baseline_val = load_txt(f)
        
        if baseline_freq is not None:
            meas_val_smoothed = smooth_measurement(meas_freq, meas_val, points_per_octave=24)

            filters, freq, target, meas, baseline = generate_target(
                meas_freq, meas_val_smoothed, baseline_freq, baseline_val, rig_type
            )

            # Normalize the generated target to match baseline at 630 Hz
            ref_idx = np.abs(freq - REF_FREQ_HZ).argmin()
            target_ref = target[ref_idx]
            baseline_ref = baseline[ref_idx]
            target = target - (target_ref - baseline_ref)

            fig, ax = plt.subplots(figsize=(12, 8))
            meas_ref = meas[ref_idx]
            meas_normalized = meas - (meas_ref - baseline_ref)

            ax.semilogx(freq, meas_normalized, color='#888888', alpha=0.6, linewidth=1.5, label="Measurement (1/24 Octave)")
            ax.semilogx(freq, baseline, color='#AAAAAA', linestyle='--', linewidth=2, label=f"{target_label} Baseline")
            ax.semilogx(freq, target, color='#000000', linewidth=2.5, label=f"Generated {target_label} Target")
            
            # Set plot properties
            ax.set_title(f"Generated {target_label} ({rig_type} Rig)", fontweight='bold')
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("dB SPL")
            ax.grid(True, which="both", linestyle='--', color='#E0E0E0', alpha=0.7)
            ax.legend(frameon=False)
            
            # Set axis limits
            ax.set_xlim(20, 20000)
            ax.set_ylim(-20, 20)
            ax.set_xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
            ax.set_xticklabels(['20', '50', '100', '200', '500', '1k', '2k', '5k', '10k', '20k'])
            
            # Set y-axis to 40 dB scale
            ax.set_ylim(baseline_ref - 20, baseline_ref + 20)
            ax.set_yticks(np.arange(baseline_ref - 20, baseline_ref + 21, 5))
            
            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Display plot
            st.pyplot(fig)
            
            # Dynamic EQ Adjustment - pass target_type to the function
            adjusted_filters = dynamic_eq_adjustment(st, filters, freq, baseline, meas, rig_type, target_type)
            
            # Create download button for original target data
            target_df = pd.DataFrame({"Frequency (Hz)": freq, "Target (dB)": target})
            csv = target_df.to_csv(sep='\t', index=False)
            st.download_button(
                label=f"Download Original {target_label} Target Data",
                data=csv,
                file_name=f"{target_type}_fitted_target.txt",
                mime="text/tab-separated-values"
            )
            
            # Display original filter parameters
            st.header("Original Filter Parameters")
            for f in filters:
                st.write(f"Type: {f['type']}, Frequency: {f['fc']:.1f} Hz, Gain: {f['gain']:.1f} dB, Q: {f['Q']:.2f}")

else:
    st.info("Please upload a measurement file to generate the target.")
