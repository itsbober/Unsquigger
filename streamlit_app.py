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
from baseline_targets import JM1_5128, JM1_711_DELTA

# Set page config
st.set_page_config(page_title="IEM Target Generator", layout="wide")

# Constants
TREBLE_START_HZ = 5000
Q_MIN, Q_MAX = 0.2, 3.0
REF_FREQ_HZ = 630

# Helper functions
def peak_eq(f, fc, gain_db, Q):
    return gain_db / (1 + ((np.log2(f / fc))**2) / Q**2)

def high_shelf(f, fc, gain_db, Q):
    return gain_db / (1 + (fc / f)**(2 * Q))

def load_txt(uploaded_file):
    try:
        # Read all lines from the file
        if isinstance(uploaded_file, str):
            lines = uploaded_file.splitlines()
        else:
            content = uploaded_file.read()
            if isinstance(content, bytes):
                lines = content.decode('utf-8').splitlines()
            else:
                lines = content.splitlines()

        # Find where the actual data starts (after "Freq(Hz) SPL(dB)" for REW files)
        start_idx = 0
        for i, line in enumerate(lines):
            if "Freq(Hz) SPL(dB)" in line:
                start_idx = i + 1
                break

        # Get only the data lines
        data_lines = []
        for line in lines[start_idx:]:
            if line.strip() and not line.strip().startswith('*'):
                # Split on whitespace and take first two columns
                values = line.strip().split()[:2]
                if len(values) == 2:  # Ensure we have both frequency and SPL values
                    data_lines.append(values)

        # Create DataFrame
        df = pd.DataFrame(data_lines, columns=['freq', 'value'])
        
        # Convert to numeric values
        df['freq'] = pd.to_numeric(df['freq'], errors='coerce')
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        # Remove any rows with NaN values
        df = df.dropna()
        
        # Sort by frequency
        df = df.sort_values(by='freq')
        
        # Convert to numpy arrays
        freq_array = df['freq'].to_numpy()
        value_array = df['value'].to_numpy()
        
        if len(freq_array) == 0 or len(value_array) == 0:
            raise ValueError("Empty data after processing")
            
        return freq_array, value_array
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None, None

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

    # Add debug prints
    print(f"Measurement frequency range: {meas_freq.min()} - {meas_freq.max()}")
    print(f"Measurement value range: {meas_val.min()} - {meas_val.max()}")
    print(f"Interpolated measurement range: {meas.min()} - {meas.max()}")
    
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
        50, 3, 0.7,     # Bass peak
        80, 4, 0.7,     # Bass shelf
        1500, 2, 0.8,   # Pinna peak 1
        3000, 2, 0.8,   # Pinna peak 2
        4500, 5, 1.2    # Treble shelf
    ]

    bounds = [
        (20, 100), (-6, 6), (Q_MIN, Q_MAX),     # Bass peak
        (20, 200), (-15, 15), (0.5, 1),     # Bass shelf
        (1000, 2000), (-5, 5), (0.2, 3),        # Pinna peak 1
        (2000, 3000), (-5, 5), (0.2, 3),        # Pinna peak 2
        (4500, 20000), (-8, 8), (Q_MIN, 1)    # Treble shelf
    ]

    # Optimize filters
    result = minimize(loss, initial_filters, bounds=bounds, method="L-BFGS-B")

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

    # Remove the file saving part since we're handling the download in the Streamlit UI
    return filters, freq, target, meas_aligned, baseline

# Streamlit UI
st.title("IEM Target Generator")

# Sidebar for inputs
with st.sidebar:
    st.header("Settings")
    
    # Add a button to link to the measurement tracer
    st.markdown("""
    <a href="https://usyless.uk/trace/" target="_blank">
        <button style="width:100%; padding:10px; background-color:#4CAF50; color:white; border:none; border-radius:5px;">
            Trace your measurement!
        </button>
    </a>
    """, unsafe_allow_html=True)
    
    st.divider()  # Optional: adds a visual separator
    
    rig_type = st.selectbox("Select Rig Type", ["5128", "711"])
    uploaded_file = st.file_uploader("Upload Measurement File", type=['txt'])

# Main content
if uploaded_file is not None:
    # Load measurement data
    meas_freq, meas_val = load_txt(uploaded_file)
    
    if meas_freq is not None:
        # Load baseline data
        baseline_data = JM1_5128 if rig_type == "5128" else JM1_711_DELTA
        
        # Create a temporary file for baseline data
        with io.StringIO(baseline_data) as f:
            jm1_freq, jm1_val = load_txt(f)
        
        if jm1_freq is not None:
            # Generate target
            filters, freq, target, meas, baseline = generate_target(
                meas_freq, meas_val, jm1_freq, jm1_val, rig_type
            )
            
            # Create plot
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Find reference values for normalization
            ref_idx = np.abs(freq - REF_FREQ_HZ).argmin()
            baseline_ref = baseline[ref_idx]
            
            # Plot curves
            ax.semilogx(freq, meas, color='#888888', alpha=0.6, linewidth=1.5, label="Measurement")
            ax.semilogx(freq, baseline, color='#AAAAAA', linestyle='--', linewidth=2, label="JM-1 Baseline")
            ax.semilogx(freq, target, color='#000000', linewidth=2.5, label="Generated Target")
            
            # Set plot properties
            ax.set_title(f"IEM Target Generation ({rig_type} Rig)", fontweight='bold')
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("dB SPL")
            ax.grid(True, which="both", linestyle='--', color='#E0E0E0', alpha=0.7)
            ax.legend(frameon=False)
            
            # Set axis limits
            ax.set_xlim(20, 20000)
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
            
            # Create download button for target data
            target_df = pd.DataFrame({"Frequency (Hz)": freq, "Target (dB)": target})
            csv = target_df.to_csv(sep='\t', index=False)
            st.download_button(
                label="Download Target Data",
                data=csv,
                file_name="fitted_target.txt",
                mime="text/tab-separated-values"
            )
            
            # Display filter parameters
            st.header("Filter Parameters")
            for f in filters:
                st.write(f"Type: {f['type']}, Frequency: {f['fc']:.1f} Hz, Gain: {f['gain']:.1f} dB, Q: {f['Q']:.2f}")

else:
    st.info("Please upload a measurement file to generate the target.")
