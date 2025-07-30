# VapourSynth PSI Plugin

VapourSynth plugin implementing the PSI (Perceptual Sharpness Index) algorithm for measuring image sharpness perception.

## Overview

The PSI metric provides a perceptual measure of image sharpness based on local edge gradient analysis. It evaluates the perceptual quality of edges in images by analyzing edge width characteristics across different image blocks.

## Usage

### PSI

Calculates the Perceptual Sharpness Index of input frames:

```python
core.psi.PSI(clip clip[, float percentile=22.0, int blocksize=32, float threshold_w=2.0, float angle_tolerance=8.0, float w_jnb=3.0, float sobel_threshold=0.1, int output_mode=0])
```

#### Parameters:
- `clip`: Input clip (grayscale or YUV - only luma plane is processed) (format: 8-16 bit integer or 32 bit float)
- `percentile`: Percentage of sharpest blocks to use for metric calculation (range: 0-100, default: 22.0)
- `blocksize`: Size of blocks for averaging edge width measurements (default: 32)
- `threshold_w`: Minimum sum of edge widths in a block to process it further (default: 2.0)
- `angle_tolerance`: Tolerance for horizontal edge detection in degrees (range: 0-90, default: 8.0)
- `w_jnb`: Width threshold for Just Noticeable Blur correction (default: 3.0)
- `sobel_threshold`: Threshold for Sobel edge detection (default: 0.1)
- `output_mode`: Output mode selection (default: 0)
  - `0`: Return original input frame (copy)
  - `1`: Return sharpness distribution map (32-bit float grayscale)

## Frame Properties

The filter outputs the following frame property:
- `PSI`: Perceptual Sharpness Index score (higher values indicate sharper images)

## Building

```bash
meson setup build
ninja -C build install
```

## Algorithm Reference

Based on the research paper:
- **C. Feichtenhofer, H. Fassold, P. Schallauer**  
  *"A perceptual image sharpness metric based on local edge gradient analysis"*  
  IEEE Signal Processing Letters, 20 (4), 379-382, 2013

and github repository:
<https://github.com/feichtenhofer/PSI>