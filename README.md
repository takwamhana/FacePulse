# FacePulse
 is an experimental computer-vision project that analyzes subtle changes in facial color over time. It detects a face in a video, estimates forehead and cheek regions of interest (ROIs), records their average green-channel intensity, and exports the resulting signals for visualization and exploratory filtering.

> **Important:** FacePulse is a prototype for educational and research exploration. It is **not a medical device** and must not be used to diagnose hypertension, estimate blood pressure, or make healthcare decisions. Pixel intensity can be affected by lighting, camera settings, movement, skin appearance, and video compression.

## Features

- Detects frontal faces with OpenCV's pretrained Haar cascade.
- Selects the largest detected face in each frame.
- Estimates three rectangular facial regions:
  - Forehead
  - Left cheek
  - Right cheek
- Measures mean green-channel pixel intensity in each region.
- Displays a live preview with face and ROI rectangles.
- Saves an annotated video and a CSV file containing per-frame measurements.
- Generates a plot of raw facial signals.
- Optionally applies an exploratory 0.7–3.0 Hz Butterworth band-pass filter to the forehead signal when enough valid samples are available.

## How it works

1. **Load the video** from `data/raw/face_video.mp4`.
2. **Read frames** and retrieve the video's frame rate (FPS).
3. **Detect the face** using OpenCV's Haar cascade.
4. **Estimate ROIs** using proportions of the detected face bounding box. These are approximate rectangles, not landmark-based segmentation.
5. **Extract signals** by calculating the mean green-channel intensity for each ROI.
6. **Preview and save** annotated frames while processing the video.
7. **Export results** to CSV and generate signal plots.
8. **Filter the forehead signal** if the samples are complete and the filter can be applied safely.

## Project structure

```text
FacePulse/
├── data/
│   └── raw/
│       └── face_video.mp4       # Input video
├── outputs/                     # Created automatically
│   ├── annotated.mp4            # Annotated side-by-side video
│   ├── facial_signals.csv       # Per-frame measurements
│   ├── raw_signal.png           # Raw green-channel plots
│   └── filtered_signal.png      # Filtered forehead signal, if available
├── src/
│   └── main.py                  # Main processing script (example path)
└── README.md
```

The script uses `Path(__file__).resolve().parents[1]` as the project root, so place the script in a directory such as `src/` for the paths above to work as intended. If your script is stored elsewhere, update `ROOT` accordingly.

## Requirements

- Python 3.11 or 3.12 recommended for broad scientific-package compatibility.
- OpenCV
- NumPy
- pandas
- Matplotlib
- SciPy

## Installation

Create and activate a virtual environment from the project directory.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install opencv-python numpy pandas matplotlib scipy
```

If Python 3.12 is not installed, use another Python version supported by the installed versions of these packages.

## Add or change the input video

**You can use a different video instead of the example video.** Put the video you want to analyze in `data/raw/` and either:

- Rename it to `face_video.mp4`, or
- Change the `INPUT_VIDEO` variable in the script to point to your chosen file.

For example:

```python
INPUT_VIDEO = ROOT / "data" / "raw" / "my_video.mp4"
```

Use a video that clearly shows a frontal face, has reasonably steady lighting, and minimizes head movement. The script expects a video file; it does not currently accept a still image as the signal input. If you meant editing or annotating an uploaded video, FacePulse can create an annotated output video; it does not modify the original file in place.

## Run the project

From the project root, run:

```powershell
python src/main.py
```

Replace `src/main.py` with the actual path to your script if it is stored elsewhere.

A preview window opens during processing. Press **Q** or **Esc** to stop early. If you stop early, the outputs contain only the frames processed up to that point.

## Output files

| File | Description |
|---|---|
| `outputs/annotated.mp4` | Video frames annotated with the face box, ROI boxes, and available ROI intensity values. The current code attempts to show the original frame alongside a green-channel view. |
| `outputs/facial_signals.csv` | One row per processed frame, including timestamp, face-detection flag, and ROI green-channel measurements. A `forehead_filtered` column is added if filtering succeeds. |
| `outputs/raw_signal.png` | Plot of raw forehead and cheek green-channel intensities over time. |
| `outputs/filtered_signal.png` | Plot of the filtered forehead signal, created only when filtering succeeds. |

CSV columns include `time_seconds`, `face_detected`, `forehead_green`, `left_cheek_green`, and `right_cheek_green`.

## Configuration notes

- `INPUT_VIDEO`: path to the source video.
- `OUTPUT_DIR`: directory where generated files are saved.
- `minSize=(80, 80)`: minimum face size accepted by the detector.
- Filter range: 0.7–3.0 Hz, approximately 42–180 cycles per minute. This is an exploratory signal-processing range, **not a validated pulse or blood-pressure measurement**.
- The face regions are estimated from the face box and may include hair, eyes, shadows, or background if the face is not aligned as expected.

## Current code checks / troubleshooting

Before running, check these details in the current script:

1. **`green_bgr` is undefined.** The line `side_by_side = np.hstack([frame, green_bgr])` will raise a `NameError`. Create the green-channel preview first, for example:
   ```python
   green_view = frame[:, :, 1]
   green_bgr = cv2.cvtColor(green_view, cv2.COLOR_GRAY2BGR)
   side_by_side = np.hstack([frame, green_bgr])
   ```
   This assumes both images have the same height and width.
2. **Video writer codec and extension:** the code uses the `XVID` codec but writes to an `.mp4` filename. Codec/container compatibility varies by system. Use a codec suitable for the selected extension, or save as `.avi` when using `XVID`.
3. **ROI measurements:** `mean_green()` is called after the face rectangle is drawn, so the drawn rectangle may slightly affect measurements if it overlaps an ROI. For more reliable analysis, calculate measurements on the unannotated frame before drawing overlays.
4. **Face tracking:** the script detects faces independently per frame. It does not track identity or compensate for movement, exposure changes, or camera motion.
5. **Filtering:** the code skips filtering if there are missing signal samples or insufficient data. Short videos may not produce a filtered signal.

## Limitations and privacy

- Haar-cascade detection and proportional ROIs are approximate and can fail with non-frontal poses, occlusion, poor lighting, or diverse recording conditions.
- Changes in green-channel intensity are not specific to blood flow and can reflect many non-physiological factors.
- No clinical validation, blood-pressure estimation, or diagnosis is implemented.
- Use videos only with appropriate consent, and store face recordings and derived data securely.

## Future improvements

- Replace approximate rectangles with facial landmarks or a face mesh.
- Add face tracking and motion compensation.
- Improve lighting normalization and signal quality checks.
- Add robust peak detection and evaluation against a reference sensor for research purposes.
- Add a user interface for selecting a video and reviewing results.