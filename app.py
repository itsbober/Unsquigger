from flask import Flask, render_template, request, send_file, jsonify
import os
from werkzeug.utils import secure_filename
import io
import base64
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.signal import savgol_filter
import pandas as pd
from baseline_targets import JM1_5128, JM1_711_DELTA

# Get the absolute path of the current directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)

# Configure folders using absolute paths
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'output')
BASELINE_FOLDER = os.path.join(BASE_DIR, 'baseline_targets')
ALLOWED_EXTENSIONS = {'txt'}

# Configure app
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['BASELINE_FOLDER'] = BASELINE_FOLDER

# Create necessary directories
for directory in [UPLOAD_FOLDER, OUTPUT_FOLDER, BASELINE_FOLDER]:
    os.makedirs(directory, exist_ok=True)

# Add this function here
def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Constants and Filters
TREBLE_START_HZ = 5000
Q_MIN, Q_MAX = 0.2, 3.0
REF_FREQ_HZ = 630


# Constants and Filters
TREBLE_START_HZ = 5000
Q_MIN, Q_MAX = 0.2, 3.0
REF_FREQ_HZ = 630

def peak_eq(f, fc, gain_db, Q):
    return gain_db / (1 + ((np.log2(f / fc))**2) / Q**2)

def high_shelf(f, fc, gain_db, Q):
    return gain_db / (1 + (fc / f)**(2 * Q))

def load_txt(path):
    try:
        # Read the file with pandas
        df = pd.read_csv(path, sep="\t", header=None, names=["freq", "value"])
        
        print(f"Loading file: {path}")
        print(f"Raw DataFrame shape: {df.shape}")
        
        # Ensure the data is numeric and handle any non-numeric values
        df["freq"] = pd.to_numeric(df["freq"], errors='coerce')
        df["value"] = pd.to_numeric(df["value"], errors='coerce')
        
        # Remove any rows with NaN values
        df = df.dropna()
        
        print(f"DataFrame after dropping NaNs shape: {df.shape}")
        
        # Sort by frequency
        df = df.sort_values(by="freq")
        
        # Convert to numpy arrays
        freq_array = df["freq"].to_numpy()
        value_array = df["value"].to_numpy()
        
        print(f"Frequency array - Min: {freq_array.min()}, Max: {freq_array.max()}")
        print(f"Value array - Min: {value_array.min()}, Max: {value_array.max()}")
        
        # Ensure the arrays are not empty
        if len(freq_array) == 0 or len(value_array) == 0:
            raise ValueError("Empty data after processing")
            
        return freq_array, value_array
    except Exception as e:
        print(f"Error loading file {path}: {str(e)}")
        import traceback
        traceback.print_exc()
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
    
    # Create resonance mask
    resonance_mask = np.ones_like(freq, dtype=bool)
    
    # Mask out resonance regions based on rig type
    if rig_type == "5128":
        # For 5128, handle rocking mode differently
        resonance_regions = [
            (7500, 8500),    # 8k resonance
            (15500, 17500)   # 16-17k resonance
        ]
        
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
        (20, 100), (-30, 30), (Q_MIN, Q_MAX),     # Bass peak
        (20, 200), (-30, 30), (Q_MIN, Q_MAX),     # Bass shelf
        (1000, 2000), (-15, 15), (0.2, 3),        # Pinna peak 1
        (2000, 3000), (-15, 15), (0.2, 3),        # Pinna peak 2
        (4500, 20000), (-6, 6), (Q_MIN, Q_MAX)    # Treble shelf
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

    # Save target data
    output_path = os.path.join(BASE_DIR, 'output', 'fitted_target.txt')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame({"Frequency (Hz)": freq, "Target (dB)": target}).to_csv(output_path, sep="\t", index=False)

    return filters, freq, target, meas, baseline

@app.route('/generate', methods=['POST'])
def generate_target_route():
    try:
        if 'measurement' not in request.files:
            return jsonify({'error': 'Missing measurement file'}), 400

        measurement_file = request.files['measurement']
        rig_type = request.form.get('rig_type', '5128')

        # DEBUG: Print request information
        print("\nRequest information:")
        print(f"Measurement file: {measurement_file.filename}")
        print(f"Rig type: {rig_type}")

        if measurement_file.filename == '':
            return jsonify({'error': 'No selected measurement file'}), 400

        if not allowed_file(measurement_file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        measurement_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(measurement_file.filename))
        measurement_file.save(measurement_path)

        try:
            meas_freq, meas_val = load_txt(measurement_path)
            if meas_freq is None or len(meas_freq) == 0:
                return jsonify({'error': 'Invalid or empty measurement file'}), 400

            baseline_path = os.path.join(app.config['BASELINE_FOLDER'], 
                                       'JM1_5128.txt' if rig_type == "5128" else 'JM1_711_DELTA.txt')
            
            jm1_freq, jm1_val = load_txt(baseline_path)
            if jm1_freq is None or len(jm1_freq) == 0:
                return jsonify({'error': 'Invalid or empty baseline file'}), 400

            # Generate target
            filters, freq, target, meas, baseline = generate_target(meas_freq, meas_val, jm1_freq, jm1_val, rig_type)

            # DEBUG: Check plotting data
            print("\nPlotting data verification:")
            print(f"freq shape: {freq.shape}")
            print(f"target shape: {target.shape}")
            print(f"meas shape: {meas.shape}")
            print(f"baseline shape: {baseline.shape}")

            matplotlib.use('Agg')
            plt.close('all')
            fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
            plt.rcParams['axes.facecolor'] = 'white'
            plt.rcParams['grid.color'] = '#E0E0E0'
            plt.rcParams['grid.linestyle'] = '--'

            # Find values at 630 Hz for normalization
            ref_freq = 630
            ref_idx = np.abs(freq - ref_freq).argmin()
            
            meas_ref = meas[ref_idx]
            baseline_ref = baseline[ref_idx]
            target_ref = target[ref_idx]

            # Normalize all curves to the baseline value at 630 Hz
            meas_normalized = meas - (meas_ref - baseline_ref)
            target_normalized = target - (target_ref - baseline_ref)

            # Plot normalized curves
            if not np.isnan(meas_normalized).any():
                ax.semilogx(freq, meas_normalized, color='#888888', alpha=0.6, linewidth=1.5, label="Measurement")
            if not np.isnan(baseline).any():
                ax.semilogx(freq, baseline, color='#AAAAAA', linestyle='--', linewidth=2, label="JM-1 Baseline")
            if not np.isnan(target_normalized).any():
                ax.semilogx(freq, target_normalized, color='#000000', linewidth=2.5, label="Generated Target")

            ax.legend(frameon=False)
            ax.set_title(f"IEM Target Generation ({rig_type} Rig)", fontweight='bold')
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("dB SPL")
            ax.grid(True, which="both", linestyle='--', color='#E0E0E0', alpha=0.7)

            # Set explicit axis limits
            ax.set_xlim(20, 20000)
            ax.set_xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
            ax.set_xticklabels(['20', '50', '100', '200', '500', '1k', '2k', '5k', '10k', '20k'])

            # Set y-axis to 40 dB scale
            y_mid = baseline_ref  # Use baseline reference as middle point
            ax.set_ylim(y_mid - 20, y_mid + 20)  # 40 dB range centered around reference
            ax.set_yticks(np.arange(y_mid - 20, y_mid + 21, 5))  # 5 dB steps

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()

            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png', dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            # DEBUG: Check buffer size
            print(f"\nPlot buffer size: {img_buf.getbuffer().nbytes} bytes")
            
            img_buf.seek(0)
            plot_url = base64.b64encode(img_buf.getvalue()).decode()

            return jsonify({
                'success': True,
                'plot': plot_url,
                'filters': [
                    {
                        'type': f['type'],
                        'fc': float(f['fc']),
                        'gain': float(f['gain']),
                        'Q': float(f['Q'])
                    } for f in filters
                ],
                'rig_type': rig_type,
                'target_data': pd.read_csv(os.path.join(BASE_DIR, 'output', 'fitted_target.txt'), sep="\t").to_dict('records')
            })

        finally:
            if os.path.exists(measurement_path):
                os.remove(measurement_path)

    except Exception as e:
        print(f"Error in target generation: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/download_target')
def download_target():
    try:
        file_path = os.path.join(BASE_DIR, 'output', 'fitted_target.txt')
        
        # DEBUG: Check file existence and size
        print("\nDownload target debug:")
        print(f"Checking file: {file_path}")
        print(f"File exists: {os.path.exists(file_path)}")
        if os.path.exists(file_path):
            print(f"File size: {os.path.getsize(file_path)} bytes")
            
        if not os.path.exists(file_path):
            return jsonify({'error': 'Target file not found'}), 404
            
        return send_file(
            file_path,
            as_attachment=True,
            download_name='fitted_target.txt'
        )
    except Exception as e:
        print(f"Error in download_target: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/debug/paths')
def debug_paths():
    # Enhanced debug information
    debug_info = {
        'base_dir': BASE_DIR,
        'upload_folder': UPLOAD_FOLDER,
        'output_folder': OUTPUT_FOLDER,
        'baseline_folder': BASELINE_FOLDER,
        'output_file': os.path.join(BASE_DIR, 'output', 'fitted_target.txt'),
        'directories_exist': {
            'upload': os.path.exists(UPLOAD_FOLDER),
            'output': os.path.exists(OUTPUT_FOLDER),
            'baseline': os.path.exists(BASELINE_FOLDER)
        },
        'directory_contents': {
            'upload': os.listdir(UPLOAD_FOLDER) if os.path.exists(UPLOAD_FOLDER) else [],
            'output': os.listdir(OUTPUT_FOLDER) if os.path.exists(OUTPUT_FOLDER) else [],
            'baseline': os.listdir(BASELINE_FOLDER) if os.path.exists(BASELINE_FOLDER) else []
        },
        'permissions': {
            'upload': os.access(UPLOAD_FOLDER, os.W_OK) if os.path.exists(UPLOAD_FOLDER) else False,
            'output': os.access(OUTPUT_FOLDER, os.W_OK) if os.path.exists(OUTPUT_FOLDER) else False,
            'baseline': os.access(BASELINE_FOLDER, os.W_OK) if os.path.exists(BASELINE_FOLDER) else False
        }
    }
    return jsonify(debug_info)

@app.route('/')
def index():
    return render_template('index.html')

def init_baseline_targets():
    """Initialize baseline target files with debug information"""
    try:
        print("\nInitializing baseline targets:")
        
        # Write JM1_5128 target
        jm1_5128_path = os.path.join(app.config['BASELINE_FOLDER'], 'JM1_5128.txt')
        with open(jm1_5128_path, 'w') as f:
            f.write(JM1_5128)
        print(f"Written JM1_5128 target to: {jm1_5128_path}")
        print(f"File exists: {os.path.exists(jm1_5128_path)}")
        print(f"File size: {os.path.getsize(jm1_5128_path)} bytes")
        
        # Write JM1_711_DELTA target
        jm1_711_path = os.path.join(app.config['BASELINE_FOLDER'], 'JM1_711_DELTA.txt')
        with open(jm1_711_path, 'w') as f:
            f.write(JM1_711_DELTA)
        print(f"Written JM1_711_DELTA target to: {jm1_711_path}")
        print(f"File exists: {os.path.exists(jm1_711_path)}")
        print(f"File size: {os.path.getsize(jm1_711_path)} bytes")
        
    except Exception as e:
        print(f"Error initializing baseline targets: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    # Print startup debug information
    print("\n=== IEM Target Generator ===")
    print("\nDirectory Information:")
    print(f"Base Directory: {BASE_DIR}")
    print(f"Upload Folder: {UPLOAD_FOLDER}")
    print(f"Output Folder: {OUTPUT_FOLDER}")
    print(f"Baseline Folder: {BASELINE_FOLDER}")
    
    # Create directories if they don't exist
    for directory in [UPLOAD_FOLDER, OUTPUT_FOLDER, BASELINE_FOLDER]:
        if not os.path.exists(directory):
            print(f"Creating directory: {directory}")
            os.makedirs(directory)
    
    # Initialize baseline targets
    init_baseline_targets()
    
    # Verify matplotlib backend
    print("\nMatplotlib Configuration:")
    print(f"Backend: {matplotlib.get_backend()}")
    print(f"Interactive mode: {plt.isinteractive()}")
    
    # Start the Flask application
    print("\nStarting Flask application...")
    app.run(debug=True)
