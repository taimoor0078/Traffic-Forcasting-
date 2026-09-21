# Traffic Flow Forecasting from Video using Deep Learning and Computer Vision

Production-ready **backend** for an MSc Deep Learning project. Given raw traffic
video, the pipeline automatically detects and tracks road users with a
**pretrained** YOLOv8, builds a traffic time series, engineers features, and
trains/compares **LSTM, GRU, and Transformer Encoder** forecasting models —
end to end, with a single command.

No frontend, API, authentication, or database is included, per scope.

## Pipeline Overview

| Step | Module | Output |
|---|---|---|
| 1 | `video_loader.py` | Frames sampled at configurable FPS |
| 2 | `detector.py` | Pretrained YOLOv8 detections (never trained) |
| 3 | `roi_counter.py` | Tracked IDs, ROI in/out counts, no double-counting |
| 4 | `generator.py` | `timeseries/traffic_timeseries.csv` |
| 5 | `preprocessor.py` | Cleaned, scaled, windowed, sequentially-split data |
| 6 | `feature_engineer.py` | Density/ratio/rolling/peak-hour features |
| 7 | `{lstm,gru,transformer}_model.py` | Three Keras architectures |
| 8 | `trainer.py` | EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard, mixed precision |
| 9 | `evaluator.py` | MAE/MSE/RMSE/MAPE/R², diagnostic plots, prediction CSVs |
| 10 | `evaluator.py` (`compare_models`) | `results/comparison.csv`, `comparison.png`, `metrics.json` |
| 11 | `model_saver.py` | `saved_models/best_{lstm,gru,transformer}.keras`, scaler, metadata |
| 12 | `inference_pipeline.py` | Multi-horizon forecasts from a new raw video |

## Setup

```bash
pip install -r requirements.txt
```

TensorFlow, Ultralytics YOLOv8, OpenCV, pandas, NumPy, scikit-learn, and
Matplotlib are the core dependencies. YOLOv8 pretrained weights
(`yolov8x.pt` by default, configurable in `config/config.yaml`) are
auto-downloaded by `ultralytics` on first use.

## Dataset

Place raw video files under `videos/` (or point `dataset.root_path` /
`paths.videos_dir` in `config/config.yaml` at your AI City Challenge,
BDD100K, or UA-DETRAC video directory — the loader only needs a folder of
`.mp4`/`.avi`/`.mov`/`.mkv` files, so any of these sources work as-is).

## Run the full pipeline

```bash
python train.py
```

This automatically: discovers every video in `videos/` → extracts frames →
runs YOLOv8 detection+tracking → builds and saves the time series →
engineers features → cleans/scales/windows the data → trains LSTM, GRU, and
Transformer models with full callback support → evaluates and compares them
→ saves the best-performing model, scaler, and metadata to `saved_models/`.

Useful flags:

```bash
python train.py --video videos/intersection_01.mp4      # single video only
python train.py --timeseries-csv timeseries/traffic_timeseries.csv  # skip video processing, iterate on modeling
python train.py --config config/my_config.yaml           # alternate config
```

## Run inference on a new video

```bash
python infer.py --video path/to/new_traffic_video.mp4
```

Automatically extracts frames, detects/tracks objects, builds the time
series, loads the best saved model, and writes forecasts for the next
10 / 30 / 60 / 120 timestamps (configurable via `inference.forecast_steps`)
to `results/`.

## Configuration

Every path, hyperparameter, and pipeline setting lives in
`config/config.yaml` — nothing is hardcoded in source. Key sections:

- `dataset` / `paths` — dataset source and output directory layout
- `video` — extraction FPS, resize resolution
- `detection` — YOLOv8 weights, thresholds, class taxonomy mapping
- `tracking` — tracker type, ROI line for entry/exit counting
- `timeseries` — aggregation bucket size
- `preprocessing` — missing values, outliers, scaler, window size, splits
- `features` — rolling window sizes, peak-hour ranges
- `models` — LSTM/GRU/Transformer architecture hyperparameters
- `training` — batch size, epochs, LR, callback patience, mixed precision
- `inference` — forecast horizons, best-model selection metric

## Notes on class taxonomy

COCO (the dataset YOLOv8 is pretrained on) does not natively distinguish
ambulance / police vehicle / fire truck / taxi / van / pickup / cyclist /
motorcyclist / scooter rider as separate classes. `config.detection.class_map`
documents exactly which underlying COCO class each project-level class is
derived from. For genuine fine-grained detection of these classes, supply an
open-vocabulary detector (e.g. YOLO-World) checkpoint via
`detection.model_weights` — the rest of the pipeline is agnostic to this.

## Output directory layout

```
config/       dataset/     videos/      frames/      detections/
tracking/     timeseries/  models/      training/    evaluation/
plots/        results/     logs/        saved_models/
```

All created automatically on first run.
