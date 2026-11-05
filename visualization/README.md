# SparseOccVLA result visualizer

Local web viewer for the result pkl produced by `tools/test.py --out`.
Runs on the Python standard library + a bundled three.js (no pip/npm installs,
no internet required after setup).

## Run

```bash
bash visualization/run.sh
```

The script prints a URL (e.g. `http://127.0.0.1:8000`); open it in a browser.
Overrides: `PYTHON_BIN` (interpreter, needs `numpy`), `SPARSEOCC_VIZ_HOST`,
`SPARSEOCC_VIZ_PORT`.

## Usage

1. Type a pkl path in the text box (pre-filled with
   `/home/adt/sparseocc_occ_baseline.pkl`), or pick one from the dropdown
   (lists `*.pkl` under the home directory), then press **Load**.
2. **Prev / Next** step through samples; **Play** auto-advances at 1s/frame.
3. The four occupancy panels share one camera — rotate/zoom any of them with
   the mouse and the other three follow.

## Layout

- **Top**: language output (question / answer pairs).
- **Middle** (1.5x): current occupancy (left) and front camera view (right).
- **Bottom** (1x): three future occupancy frames
  (`occ_forecasting[1:4]`, i.e. fu-t2 / fu-t4 / fu-t6).
- The predicted trajectory is drawn as a polyline in every occupancy panel and,
  when extrinsics are available, projected into the front view.

## pkl schema (fields this viewer reads)

The result file is the custom `SPARSEVLA-PKL-1` stream written by
`tools/test.py --out`. Each sample record (`result`) is a dict. Fields produced
by the current test path (when available):

| key | type | meaning |
| --- | --- | --- |
| `occ_pred` | uint8 `(200,200,16)` | current occupancy; value = class id, `17` = empty |
| `occ_forecasting` | uint8 `(4,200,200,16)` | `[current, fu-t2, fu-t4, fu-t6]` |
| `generated_text` | list of `{Q, A, G}` | language output |
| `token` | str | nuScenes sample token |
| `traj_forecasting` | list of `[x, y]` | predicted ego-frame waypoints when the loaded model/config emits planning trajectories |
| `front_image` | str | first camera image path from the test sample (CAM_FRONT in standard nuScenes camera ordering) |
| `lidar2img` | 4x4 list | lidar-frame to front-camera pixel projection matrix, including test image transforms |
| `ego2img` | 4x4 list | ego-frame to front-camera pixel projection matrix, used for projecting ego-frame waypoints |

Optional fields you can add to each sample (the viewer degrades gracefully if
they are absent):

| key | type | meaning |
| --- | --- | --- |
| `traj_forecasting` | list of `[x, y]` | waypoints in ego frame, meters (x forward, y left) |
| `front_image` | str | absolute path to the CAM_FRONT image, or a `data:image/...;base64,...` URI |
| `intrinsics` | 3x3 list, or path to `.json`/`.npy` | camera intrinsic matrix `K` |
| `extrinsics` | 4x4 list, or path | ego→camera `[R|t]` (used with `intrinsics`) |
| `lidar2img` | 4x4 list, or path | combined `K@[R|t]` (used directly when `intrinsics` is absent) |

Notes:

- Trajectory waypoints are interpreted as `[x=forward, y=left]` and lifted to
  `z = 0.5 m` for the 3D overlay and front-view projection. Adjust
  `TRAJ.z` in `static/app.js` if your convention differs.
- `extrinsics` requires `intrinsics` for projection; `lidar2img` (the standard
  nuScenes matrix) is used on its own and maps an ego point
  `[x, y, z, 1]` straight to pixel coordinates.

## Files

- `server.py` — stdlib HTTP server + JSON API (`/api/list_pkls`,
  `/api/load`, `/api/sample`).
- `pkl_reader.py` — parses the `SPARSEVLA-PKL-1` format and converts occupancy
  grids to occupied-voxel point clouds.
- `static/` — three.js frontend (`index.html`, `app.js`, `style.css`).
- `run.sh` — launcher.
