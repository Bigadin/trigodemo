"""
Configuration for conveyor belt object counting.
Adjust these parameters to tune the counting algorithm for your specific setup.
"""

# =============================================================================
# BACKGROUND SUBTRACTION
# =============================================================================

# Algorithm: "MOG2" or "KNN"
BG_SUBTRACTOR_TYPE = "MOG2"

# MOG2 parameters
MOG2_HISTORY = 500          # Number of frames for background model
MOG2_VAR_THRESHOLD = 16     # Variance threshold for pixel classification
MOG2_DETECT_SHADOWS = False # Detect shadows (disable for speed)

# KNN parameters (if using KNN)
KNN_HISTORY = 500
KNN_DIST2_THRESHOLD = 400.0
KNN_DETECT_SHADOWS = False

# =============================================================================
# MORPHOLOGY (mask cleanup)
# =============================================================================

# Kernel sizes for morphological operations (must be odd numbers)
MORPH_OPEN_KERNEL_SIZE = 3   # Removes small noise
MORPH_CLOSE_KERNEL_SIZE = 5  # Fills small gaps

# Number of iterations for each operation
MORPH_OPEN_ITERATIONS = 1
MORPH_CLOSE_ITERATIONS = 2

# =============================================================================
# THRESHOLD CALIBRATION
# =============================================================================

# Mode: "manual" or "auto"
THRESHOLD_MODE = "manual"  # Use manual for more predictable behavior

# Manual threshold (S_norm value, 0.0 to 1.0)
# Used when THRESHOLD_MODE = "manual"
MANUAL_THRESHOLD = 0.25  # 5% of counting zone filled = object present

# Auto threshold parameters
# threshold = baseline_mean + (THRESHOLD_K * baseline_std)
WARMUP_FRAMES = 30       # Frames to collect baseline statistics (reduced)
THRESHOLD_K = 3.0        # Multiplier for standard deviation (reduced)

# Minimum threshold to prevent false positives on very stable backgrounds
MIN_AUTO_THRESHOLD = 0.01  # 1% minimum

# =============================================================================
# COUNTING STATE MACHINE
# =============================================================================

# Minimum frames in OFF state before allowing new count (anti-double-counting)
MIN_OFF_FRAMES = 5

# Minimum frames in ON state before confirming presence (noise rejection)
MIN_ON_FRAMES = 2

# Noise floor: if S_norm < this value, treat as 0 (ignore tiny fluctuations)
NOISE_FLOOR = 0.001  # 0.1%

# =============================================================================
# PREPROCESSING
# =============================================================================

# Apply Gaussian blur before background subtraction
PREPROCESS_BLUR = True
PREPROCESS_BLUR_SIZE = 5  # Must be odd

# =============================================================================
# LEARNING RATE
# =============================================================================

# Background model learning rate (-1 = auto, 0 = frozen, 0.0-1.0 = manual)
# Lower values = slower adaptation, more stable
# Higher values = faster adaptation to changes
BG_LEARNING_RATE = -1  # Auto

# =============================================================================
# METHOD 6: MULTI-SIGNAL EVENT SCORING
# =============================================================================

# Enable Method 6 (multi-signal) instead of simple S(t) threshold
USE_METHOD_6 = True

# --- Conveyor Axis ---
# Flip conveyor direction (if objects move in opposite direction to detected axis)
FLIP_DIRECTION = False

# --- Multi-Band Configuration ---
# Number of bands along conveyor axis
NUM_BANDS = 4

# Number of upstream bands to sum for S1(t) signal
UPSTREAM_BANDS = 1  # Use first K bands as "upstream"

# --- Event Scoring Weights ---
# E(t) = W_DS1 * max(0, dS1) + W_DC * abs(dC) + W_DS * max(0, dS)
# NOTE: W_DS should be 0 to avoid double-counting (dS1 already captures entry)
W_DS1 = 1.0   # Weight for upstream band derivative (main signal for entry detection)
W_DC = 0.3    # Weight for centroid shift (reduced - helps but can cause noise)
W_DS = 0.0    # Weight for total foreground derivative (disabled to prevent double-count)

# --- EMA Smoothing ---
# Exponential moving average for E(t): E_smooth = alpha*E + (1-alpha)*E_smooth_prev
EMA_ALPHA = 0.3  # Higher = more responsive, lower = smoother

# --- Event Detection Thresholds ---
# Presence threshold: S_norm must exceed this for counting to be possible
PRESENCE_THRESHOLD = 0.02  # 2% of zone filled

# Event threshold: E_smooth must exceed this to trigger a count
EVENT_THRESHOLD = 50.0  # Adjust based on zone size and object size

# Reset threshold: E_smooth must fall below this before re-arming
RESET_THRESHOLD = 20.0  # Lower than EVENT_THRESHOLD

# Cooldown frames after a count before re-arming is possible
COOLDOWN_FRAMES = 30

# --- Debug ---
# Show detailed debug overlay (axis, bands, signals)
DEBUG_OVERLAY = True
