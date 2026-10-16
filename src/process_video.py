from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt


# Project paths
ROOT = Path(__file__).resolve().parents[1]
INPUT_VIDEO = ROOT / "data" / "raw" / "face_video.mp4"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_face_detector():
    """Load OpenCV's pretrained frontal-face detector."""
    cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    detector = cv2.CascadeClassifier(cascade_path)

    if detector.empty():
        raise RuntimeError("Could not load the face detector.")

    return detector


def extract_regions(face):
    """
    Estimate forehead and cheek ROIs relative to a face box.

    These are approximate rectangular regions, not facial landmarks.
    """
    x, y, w, h = face

    return {
        "forehead": (
            x + int(0.25 * w),
            y + int(0.15 * h),
            int(0.50 * w),
            int(0.18 * h),
        ),
        "left_cheek": (
            x + int(0.12 * w),
            y + int(0.48 * h),
            int(0.30 * w),
            int(0.20 * h),
        ),
        "right_cheek": (
            x + int(0.58 * w),
            y + int(0.48 * h),
            int(0.30 * w),
            int(0.20 * h),
        ),
    }


def mean_green(frame, roi):
    """Return mean green-channel intensity for a rectangular ROI."""
    x, y, w, h = roi

    height, width = frame.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(width, x + w)
    y2 = min(height, y + h)

    if x2 <= x1 or y2 <= y1:
        return np.nan

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return np.nan

    # OpenCV uses BGR channel order: green is index 1.
    return float(np.mean(crop[:, :, 1]))


def filter_signal(signal, fps):
    """
    Apply a 0.7–3.0 Hz exploratory band-pass filter.
    This range corresponds roughly to 42–180 cycles/minute.
    """
    signal = np.asarray(signal, dtype=float)

    if len(signal) < 30 or fps <= 6.0:
        return None

    if not np.isfinite(signal).all():
        return None

    signal = signal - np.mean(signal)

    nyquist = fps / 2.0
    low = 0.7 / nyquist
    high = min(3.0 / nyquist, 0.99)

    if not 0 < low < high < 1:
        return None

    sos = butter(
        3,
        [low, high],
        btype="bandpass",
        output="sos",
    )

    try:
        return sosfiltfilt(sos, signal)
    except ValueError:
        return None


def main():
    if not INPUT_VIDEO.exists():
        raise FileNotFoundError(
            f"Video not found: {INPUT_VIDEO}\n"
            "Place your video at data/raw/face_video.mp4"
        )

    detector = get_face_detector()
    cap = cv2.VideoCapture(str(INPUT_VIDEO))

    if not cap.isOpened():
        raise RuntimeError("Could not open the input video.")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise RuntimeError("Could not determine video FPS.")

    rows = []
    frame_index = 0


    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    annotated_path = OUTPUT_DIR / "annotated.avi"  # extension .avi
    annotated_path = OUTPUT_DIR / "annotated.mp4"
    annotated_writer = None  

    cv2.namedWindow("FacePulse - live", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
    "FacePulse - live",
    cv2.WND_PROP_TOPMOST,
    1
)
    cv2.resizeWindow("FacePulse - live", 960, 540)

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
        )

        row = {
            "time_seconds": frame_index / fps,
            "face_detected": False,
            "forehead_green": np.nan,
            "left_cheek_green": np.nan,
            "right_cheek_green": np.nan,
        }


        green_view = frame[:, :, 1]

        if len(faces) > 0:
            # Use the largest detected face.
            x, y, w, h = max(
                faces,
                key=lambda box: box[2] * box[3],
            )

            row["face_detected"] = True

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2,
            )

            regions = extract_regions((x, y, w, h))

            for name, roi in regions.items():
                value = mean_green(frame, roi)
                row[f"{name}_green"] = value

                rx, ry, rw, rh = roi
                cv2.rectangle(
                    frame,
                    (rx, ry),
                    (rx + rw, ry + rh),
                    (255, 180, 0),
                    2,
                )

  
                if not np.isnan(value):
                    cv2.putText(
                        frame,
                        f"{name}: {value:.2f}",
                        (rx, max(20, ry - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

        rows.append(row)
        frame_index += 1


        side_by_side = np.hstack([frame, green_bgr])

        # On ajoute un petit bandeau de texte en haut à gauche.
        cv2.putText(
            side_by_side,
            f"Frame {frame_index} | t={frame_index / fps:.2f}s | fps={fps:.1f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )


        cv2.imshow("FacePulse - live", side_by_side)


        if annotated_writer is None:
            h_out, w_out = side_by_side.shape[:2]
            annotated_writer = cv2.VideoWriter(
                str(annotated_path), fourcc, fps, (w_out, h_out)
            )
        annotated_writer.write(side_by_side)


        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            print("Affichage interrompu par l'utilisateur.")
            break

        if frame_index % 100 == 0:
            print(f"Processed {frame_index} frames...")

    cap.release()
    if annotated_writer is not None:
        annotated_writer.release()
    cv2.destroyAllWindows()

    if not rows:
        raise RuntimeError("The video contains no readable frames.")

    df = pd.DataFrame(rows)

    csv_path = OUTPUT_DIR / "facial_signals.csv"
    df.to_csv(csv_path, index=False)

    print(f"Frames processed: {len(df)}")
    print(f"Video FPS: {fps:.2f}")
    print(f"Face detected in {df['face_detected'].mean():.1%} of frames")
    print(f"Signal data saved to: {csv_path}")
    print(f"Annotated video saved to: {annotated_path}")

    # Plot raw green-channel signals.
    fig, ax = plt.subplots(figsize=(12, 5))

    for column in [
        "forehead_green",
        "left_cheek_green",
        "right_cheek_green",
    ]:
        ax.plot(
            df["time_seconds"],
            df[column],
            label=column.replace("_", " ").title(),
            alpha=0.8,
        )

    ax.set_title("Facial Green-Channel Intensity")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Mean pixel intensity (0–255)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    plot_path = OUTPUT_DIR / "raw_signal.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    # Filter the forehead signal only when all samples are valid.
    forehead = df["forehead_green"].interpolate(
        limit_area="inside"
    )

    if forehead.notna().all():
        filtered = filter_signal(forehead.to_numpy(), fps)

        if filtered is not None:
            df["forehead_filtered"] = filtered

            df.to_csv(csv_path, index=False)

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(
                df["time_seconds"],
                filtered,
                label="Filtered forehead signal",
            )
            ax.set_title("Exploratory Band-Pass Filter")
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Filtered intensity")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()

            filtered_path = OUTPUT_DIR / "filtered_signal.png"
            fig.savefig(filtered_path, dpi=150)
            plt.close(fig)

            print(f"Filtered signal saved to: {filtered_path}")
        else:
            print("Not enough valid samples for filtering.")
    else:
        print(
            "Signal contains missing samples. "
            "Improve face tracking before filtering."
        )

    print(f"Raw signal plot saved to: {plot_path}")


if __name__ == "__main__":
    main()