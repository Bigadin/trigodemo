"""
Conveyor belt object counter using background subtraction.
No object detection/classification - purely foreground-based counting.

Supports two modes:
- Simple S(t) threshold (original)
- Method 6: Multi-signal event scoring (handles touching objects)
"""

import cv2
import numpy as np
from enum import Enum
from typing import Optional, Tuple, List
import math

import conveyor_config as cfg


class CounterState(Enum):
    EMPTY = "empty"
    OCCUPIED = "occupied"


class ConveyorCounter:
    """
    Counts objects passing through a counting zone using background subtraction.

    Pipeline per frame:
    1. Apply include mask (ROI) if defined
    2. Extract counting zone region
    3. Background subtraction
    4. Morphological cleanup
    5. Compute S_norm (foreground ratio)
    6. Method 6: Compute additional signals (dS, C, dC, S1, dS1, E)
    7. Event scoring and counting
    """

    def __init__(self, frame_shape: Tuple[int, int]):
        """
        Initialize counter.

        Args:
            frame_shape: (height, width) of video frames
        """
        self.frame_h, self.frame_w = frame_shape

        # Background subtractor
        self._init_bg_subtractor()

        # Morphology kernels
        self.open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (cfg.MORPH_OPEN_KERNEL_SIZE, cfg.MORPH_OPEN_KERNEL_SIZE)
        )
        self.close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (cfg.MORPH_CLOSE_KERNEL_SIZE, cfg.MORPH_CLOSE_KERNEL_SIZE)
        )

        # Masks (set externally via set_masks)
        self.include_mask: Optional[np.ndarray] = None
        self.counting_mask: Optional[np.ndarray] = None
        self.counting_area: int = 0

        # Threshold calibration (for simple mode)
        self.threshold: float = cfg.MANUAL_THRESHOLD
        self.warmup_samples: List[float] = []
        self.warmup_complete: bool = (cfg.THRESHOLD_MODE == "manual")
        self.frames_processed: int = 0

        # State machine (simple mode)
        self.state = CounterState.EMPTY
        self.count: int = 0
        self.off_frame_count: int = 0
        self.on_frame_count: int = 0

        # Current frame stats
        self.current_s_norm: float = 0.0
        self.current_is_present: bool = False

        # =====================================================================
        # METHOD 6: Multi-signal state
        # =====================================================================

        # Conveyor axis (computed from counting polygon)
        self.axis_u: Tuple[float, float] = (0.0, 1.0)  # Unit direction vector
        self.axis_theta_deg: float = 90.0  # Angle in degrees
        self.min_area_rect: Optional[Tuple] = None  # cv2.minAreaRect result

        # Band boundaries along axis
        self.p_min: float = 0.0
        self.p_max: float = 1.0
        self.band_width: float = 1.0
        self.flipped: bool = False  # Track if direction has been flipped

        # Previous frame values for derivatives
        self.prev_s: float = 0.0
        self.prev_s1: float = 0.0
        self.prev_c: Optional[float] = None

        # Current signals
        self.current_s: float = 0.0  # Raw foreground pixel count
        self.current_ds: float = 0.0
        self.current_c: Optional[float] = None
        self.current_dc: float = 0.0
        self.current_s1: float = 0.0
        self.current_ds1: float = 0.0
        self.current_e: float = 0.0
        self.current_e_smooth: float = 0.0
        self.band_counts: List[int] = [0] * cfg.NUM_BANDS

        # Method 6 state machine
        self.armed: bool = True
        self.cooldown: int = 0

        # Counting polygon points (for axis computation)
        self.counting_polygon_pts: Optional[np.ndarray] = None

    def _init_bg_subtractor(self):
        """Initialize the background subtractor based on config."""
        if cfg.BG_SUBTRACTOR_TYPE == "MOG2":
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=cfg.MOG2_HISTORY,
                varThreshold=cfg.MOG2_VAR_THRESHOLD,
                detectShadows=cfg.MOG2_DETECT_SHADOWS
            )
        else:
            self.bg_subtractor = cv2.createBackgroundSubtractorKNN(
                history=cfg.KNN_HISTORY,
                dist2Threshold=cfg.KNN_DIST2_THRESHOLD,
                detectShadows=cfg.KNN_DETECT_SHADOWS
            )

    def _compute_conveyor_axis(self, polygon_pts: np.ndarray):
        """
        Compute conveyor axis from counting polygon using minAreaRect.

        Args:
            polygon_pts: Nx2 array of polygon vertices
        """
        if polygon_pts is None or len(polygon_pts) < 3:
            return

        # Compute minimum area rectangle
        rect = cv2.minAreaRect(polygon_pts.astype(np.float32))
        self.min_area_rect = rect

        center, (w, h), angle = rect

        # Determine longitudinal axis (longest side)
        if w >= h:
            theta = angle
        else:
            theta = angle + 90

        # Normalize to [-180, 180)
        while theta >= 180:
            theta -= 360
        while theta < -180:
            theta += 360

        self.axis_theta_deg = theta

        # Convert to radians and unit vector
        theta_rad = math.radians(theta)
        ux = math.cos(theta_rad)
        uy = math.sin(theta_rad)

        # Apply flip if configured
        if cfg.FLIP_DIRECTION:
            ux, uy = -ux, -uy

        self.axis_u = (ux, uy)

        # Compute band boundaries by projecting polygon vertices
        projections = polygon_pts[:, 0] * ux + polygon_pts[:, 1] * uy
        self.p_min = float(np.min(projections))
        self.p_max = float(np.max(projections))

        if self.p_max > self.p_min:
            self.band_width = (self.p_max - self.p_min) / cfg.NUM_BANDS
        else:
            self.band_width = 1.0

    def set_masks(self, include_polygons: List[List[List[float]]],
                  counting_polygons: List[List[List[float]]]):
        """
        Build masks from polygon coordinates.
        """
        # Include mask (ROI)
        if include_polygons:
            self.include_mask = np.zeros((self.frame_h, self.frame_w), dtype=np.uint8)
            for poly in include_polygons:
                if len(poly) >= 3:
                    pts = np.array(poly, dtype=np.int32)
                    cv2.fillPoly(self.include_mask, [pts], 255)
        else:
            self.include_mask = None

        # Counting mask
        if counting_polygons:
            self.counting_mask = np.zeros((self.frame_h, self.frame_w), dtype=np.uint8)
            all_pts = []
            for poly in counting_polygons:
                if len(poly) >= 3:
                    pts = np.array(poly, dtype=np.int32)
                    cv2.fillPoly(self.counting_mask, [pts], 255)
                    all_pts.extend(poly)
            self.counting_area = cv2.countNonZero(self.counting_mask)

            # Store polygon points and compute axis
            if all_pts:
                self.counting_polygon_pts = np.array(all_pts, dtype=np.float32)
                self._compute_conveyor_axis(self.counting_polygon_pts)
        else:
            self.counting_mask = None
            self.counting_area = 0
            self.counting_polygon_pts = None

    def reset_background(self):
        """Reset the background model."""
        self._init_bg_subtractor()
        self.warmup_samples.clear()
        self.warmup_complete = (cfg.THRESHOLD_MODE == "manual")
        self.frames_processed = 0

        # Reset Method 6 state
        self.prev_s = 0.0
        self.prev_s1 = 0.0
        self.prev_c = None
        self.current_e_smooth = 0.0
        self.armed = True
        self.cooldown = 0

    def reset_count(self):
        """Reset the object count to zero."""
        self.count = 0
        self.state = CounterState.EMPTY
        self.off_frame_count = 0
        self.on_frame_count = 0
        self.armed = True
        self.cooldown = 0

    def calibrate_threshold(self, samples: List[float]):
        """Compute threshold from collected samples."""
        if not samples:
            self.threshold = cfg.MANUAL_THRESHOLD
            return

        arr = np.array(samples)
        mean = np.mean(arr)
        std = np.std(arr)
        computed = mean + cfg.THRESHOLD_K * std
        self.threshold = max(computed, cfg.MIN_AUTO_THRESHOLD)

    def _compute_method6_signals(self, fg_in_counting: np.ndarray, s: float):
        """
        Compute Method 6 signals: dS, C, dC, S1, dS1, E.

        Args:
            fg_in_counting: Binary mask of foreground pixels in counting zone
            s: Current foreground pixel count (not normalized)
        """
        ux, uy = self.axis_u

        # --- dS ---
        self.current_ds = s - self.prev_s

        # --- C(t) and dC(t) ---
        ys, xs = np.where(fg_in_counting > 0)

        if len(xs) > 0:
            # Project to conveyor axis
            projections = ux * xs + uy * ys
            self.current_c = float(np.mean(projections))

            # dC only if previous C exists
            if self.prev_c is not None:
                self.current_dc = self.current_c - self.prev_c
            else:
                self.current_dc = 0.0
        else:
            self.current_c = self.prev_c  # Keep previous
            self.current_dc = 0.0

        # --- Band counts and S1(t) ---
        self.band_counts = [0] * cfg.NUM_BANDS

        if len(xs) > 0 and self.band_width > 0:
            # Compute band index for each pixel
            band_indices = ((projections - self.p_min) / self.band_width).astype(np.int32)
            band_indices = np.clip(band_indices, 0, cfg.NUM_BANDS - 1)

            # Count pixels per band
            for i in range(cfg.NUM_BANDS):
                self.band_counts[i] = int(np.sum(band_indices == i))

        # S1 = sum of upstream bands (first bands along axis direction)
        self.current_s1 = float(sum(self.band_counts[:cfg.UPSTREAM_BANDS]))

        # dS1
        self.current_ds1 = self.current_s1 - self.prev_s1

        # --- Event score E(t) ---
        e = (cfg.W_DS1 * max(0.0, self.current_ds1) +
             cfg.W_DC * abs(self.current_dc) +
             cfg.W_DS * max(0.0, self.current_ds))
        self.current_e = e

        # EMA smoothing
        self.current_e_smooth = (cfg.EMA_ALPHA * e +
                                  (1.0 - cfg.EMA_ALPHA) * self.current_e_smooth)

        # Update previous values for next frame
        self.prev_s = s
        self.prev_s1 = self.current_s1
        if self.current_c is not None:
            self.prev_c = self.current_c

    def _method6_counting(self, s_norm: float):
        """
        Method 6 counting logic with event scoring.

        Args:
            s_norm: Normalized foreground ratio
        """
        # Presence gate
        present = s_norm >= cfg.PRESENCE_THRESHOLD
        self.current_is_present = present

        # Update cooldown
        if self.cooldown > 0:
            self.cooldown -= 1

        # Re-arm check
        if self.cooldown == 0 and self.current_e_smooth <= cfg.RESET_THRESHOLD:
            self.armed = True

        # Count trigger
        if present and self.current_e_smooth >= cfg.EVENT_THRESHOLD and self.armed:
            self.count += 1
            self.armed = False
            self.cooldown = cfg.COOLDOWN_FRAMES
            self.state = CounterState.OCCUPIED
        elif not present:
            self.state = CounterState.EMPTY

    def _simple_counting(self, s_norm: float):
        """Original simple S(t) threshold counting."""
        is_present = s_norm >= self.threshold
        self.current_is_present = is_present

        if self.state == CounterState.EMPTY:
            if is_present:
                self.on_frame_count += 1
                self.off_frame_count = 0
                if self.on_frame_count >= cfg.MIN_ON_FRAMES:
                    self.state = CounterState.OCCUPIED
                    self.count += 1
            else:
                self.on_frame_count = 0
                self.off_frame_count += 1
        else:
            if not is_present:
                self.off_frame_count += 1
                self.on_frame_count = 0
                if self.off_frame_count >= cfg.MIN_OFF_FRAMES:
                    self.state = CounterState.EMPTY
            else:
                self.off_frame_count = 0
                self.on_frame_count += 1

    def update(self, frame: np.ndarray) -> dict:
        """
        Process a frame and update count.
        """
        self.frames_processed += 1

        if self.counting_mask is None or self.counting_area == 0:
            return self._make_result(warmup_progress=1.0, warning="No counting zone")

        # Step A: Apply include mask (ROI)
        if self.include_mask is not None:
            masked_frame = cv2.bitwise_and(frame, frame, mask=self.include_mask)
        else:
            masked_frame = frame

        # Step B: Convert to grayscale and blur
        gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
        if cfg.PREPROCESS_BLUR:
            gray = cv2.GaussianBlur(gray, (cfg.PREPROCESS_BLUR_SIZE, cfg.PREPROCESS_BLUR_SIZE), 0)

        # Step C: Background subtraction
        fg_mask = self.bg_subtractor.apply(gray, learningRate=cfg.BG_LEARNING_RATE)

        # Step D: Morphological cleanup
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.open_kernel,
                                    iterations=cfg.MORPH_OPEN_ITERATIONS)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.close_kernel,
                                    iterations=cfg.MORPH_CLOSE_ITERATIONS)

        # Step E: Compute S(t) within counting zone
        counting_fg = cv2.bitwise_and(fg_mask, fg_mask, mask=self.counting_mask)
        fg_pixels = cv2.countNonZero(counting_fg)
        s_norm = fg_pixels / self.counting_area if self.counting_area > 0 else 0.0

        if s_norm < cfg.NOISE_FLOOR:
            s_norm = 0.0

        self.current_s_norm = s_norm
        self.current_s = float(fg_pixels)

        # Warmup phase
        if not self.warmup_complete:
            self.warmup_samples.append(s_norm)
            warmup_progress = len(self.warmup_samples) / cfg.WARMUP_FRAMES

            if len(self.warmup_samples) >= cfg.WARMUP_FRAMES:
                self.calibrate_threshold(self.warmup_samples)
                self.warmup_complete = True
                warmup_progress = 1.0

            return self._make_result(warmup_progress=warmup_progress)

        # Step F: Counting
        if cfg.USE_METHOD_6:
            # Compute Method 6 signals
            self._compute_method6_signals(counting_fg, fg_pixels)
            self._method6_counting(s_norm)
        else:
            self._simple_counting(s_norm)

        return self._make_result(warmup_progress=1.0)

    def _make_result(self, warmup_progress: float, warning: str = None) -> dict:
        """Build result dictionary with all signals."""
        result = {
            "count": self.count,
            "s_norm": self.current_s_norm,
            "threshold": self.threshold,
            "state": self.state.value,
            "is_present": self.current_is_present,
            "warmup_progress": warmup_progress,
            "warning": warning,
        }

        # Add Method 6 signals
        if cfg.USE_METHOD_6:
            result.update({
                "s": self.current_s,
                "ds": self.current_ds,
                "c": self.current_c,
                "dc": self.current_dc,
                "s1": self.current_s1,
                "ds1": self.current_ds1,
                "e": self.current_e,
                "e_smooth": self.current_e_smooth,
                "band_counts": self.band_counts.copy(),
                "armed": self.armed,
                "cooldown": self.cooldown,
                "axis_theta": self.axis_theta_deg,
                "axis_u": self.axis_u,
                "p_min": self.p_min,
                "p_max": self.p_max,
            })

        return result

    def get_stats(self) -> dict:
        """Get current statistics without processing a frame."""
        stats = {
            "count": self.count,
            "s_norm": self.current_s_norm,
            "threshold": self.threshold,
            "state": self.state.value,
            "is_present": self.current_is_present,
            "warmup_complete": self.warmup_complete
        }

        if cfg.USE_METHOD_6:
            stats.update({
                "e_smooth": self.current_e_smooth,
                "armed": self.armed,
                "cooldown": self.cooldown,
            })

        return stats

    def get_debug_overlay_data(self) -> dict:
        """Get data needed for debug overlay rendering."""
        return {
            "min_area_rect": self.min_area_rect,
            "axis_u": self.axis_u,
            "axis_theta": self.axis_theta_deg,
            "p_min": self.p_min,
            "p_max": self.p_max,
            "band_width": self.band_width,
            "band_counts": self.band_counts.copy(),
            "counting_polygon_pts": self.counting_polygon_pts,
            "flipped": self.flipped,
        }


def build_mask_from_polygons(polygons: List[List[List[float]]],
                              height: int, width: int) -> Optional[np.ndarray]:
    """
    Utility function to build a mask from polygon list.
    """
    if not polygons:
        return None

    mask = np.zeros((height, width), dtype=np.uint8)
    valid = False

    for poly in polygons:
        if len(poly) >= 3:
            pts = np.array(poly, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
            valid = True

    return mask if valid else None
