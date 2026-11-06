# Annotation Tool

Web application for assisted annotation of image and video datasets, built around Meta's SAM 3. It runs as three Docker services: a Vue 3 frontend, a FastAPI gateway and a GPU-backed segmentation service.

## Features

- **Auto-labeling with SAM 3.** Label a whole image directory from a list of text prompts (concepts), with a configurable confidence threshold.
- **Annotation review and editing.** Visual editor for COCO annotations (boxes and polygons) with per-image navigation. Images can be read from a server path or uploaded from the browser.
- **Label manager.** Rename, merge or delete the categories of a dataset.
- **SAM 3 tools.** Convert bounding boxes into segmentation masks (optionally guided by the category name) and convert whole datasets. The gateway proxies these operations under `/sam3/` (`segment-image`, `segment-point`, `convert-dataset`, `jobs`).
- **Annotation comparison.** Evaluate a predictions COCO file against a ground-truth COCO file (`/comparison` view, `POST /comparison/evaluate`). Images are paired by `file_name` and categories by name; boxes are matched greedily by IoU at a configurable threshold, and the tool reports precision, recall and F1 globally and per category. Both files can be given as server paths or uploaded.
- **Interactive point segmentation.** Click to add foreground points, Shift+click for background points, Enter to confirm (`Sam3TrackerModel`).
- **Video tracking.** Extract frames from a video, track text-described objects with SAM 3 Video, and review, rename, merge or delete tracks.
- **One-shot propagation.** Use one annotated mask as the example and segment the same concept across a directory of images with INSID3 on DINOv3 features, without training.
- **Multi-format export and import** through Datumaro and native exporters: COCO, YOLO (detection, segmentation, OBB), CreateML, MOT, YouTube-VIS and CVAT video. The format registry is in `services/gateway/app/export/formats.py`.
- **Job tracking.** Long-running jobs (labeling, tracking, propagation, conversion) are stored in SQLite and can be monitored from the UI. `GET /jobs` aggregates the stored jobs with the live jobs of the segmentation service. In the Jobs view, each completed labeling job has a **Review** button (opens the result in the annotation editor, together with the source image directory) and a **Compare** button (opens the comparison view with the job output as predictions).
- **Interface languages.** English and Spanish.

## Architecture

```
Browser
   |
Frontend (Vue 3 + nginx)   host port 3001 (container port 80)
   |  /api/ proxied by nginx
Gateway (FastAPI)          port 8000
   |
Segmentation (FastAPI)     port 8002   SAM 3, INSID3, GPU
```

| Service | Port | Description |
|---------|------|-------------|
| `frontend` | 3001 | Vue 3, Pinia and Tailwind CSS, served by nginx. `/api/` is proxied to the gateway. |
| `gateway` | 8000 | Datasets, annotations, filesystem browsing, export/import, jobs, and proxy to the segmentation service. |
| `segmentation` | 8002 | SAM 3 inference, video tracking and INSID3 propagation. Requires an NVIDIA GPU. |

`services/shared` contains the job database, job lifecycle, job logger, VRAM monitor and legacy export code used by the backend services.

Docker volumes: `datasets` (data to annotate and exports), `jobs_db` (SQLite job database), `checkpoints` (DINOv3 weights) and `hf_cache` (Hugging Face cache).

## Requirements

- Docker with Docker Compose and the NVIDIA Container Toolkit.
- An NVIDIA GPU. The segmentation container reserves one GPU.
- A Hugging Face account with granted access to the gated model [`facebook/sam3`](https://huggingface.co/facebook/sam3), and an access token.
- For one-shot propagation only: the DINOv3 ViT weights, downloaded manually from Meta (gated). The expected file name depends on `INSID3_MODEL_SIZE`:
  - `small`: `dinov3_vits16_pretrain_lvd1689m-08c60483.pth`
  - `base`: `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`
  - `large` (default): `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`

## Installation and usage

```bash
git clone https://github.com/asferrer/annotation-tool.git
cd annotation-tool
cp .env.example .env
# edit .env and set HF_TOKEN

docker compose up -d --build
```

Open http://localhost:3001. The SAM 3 weights are downloaded into the `hf_cache` volume on first use; by default the model is loaded lazily on the first request (`SAM3_LAZY_LOAD=true`).

`docker-compose.override.yml` is applied automatically and enables hot reload for the gateway. To start without it:

```bash
docker compose -f docker-compose.yml up -d --build
```

For INSID3 propagation, copy the DINOv3 weight file into the `checkpoints` volume under `dinov3/` (default `DINOV3_WEIGHTS_DIR=/app/checkpoints/dinov3`). The code of the DINOv3 repository is fetched through `torch.hub` on first use and cached in the same volume. Datasets are read from and written to the `datasets` volume (`/app/datasets` inside the containers); the gateway file browser and the comparison endpoint only access the paths listed in `ALLOWED_FS_PATHS`.

### Configuration

Variables are read from `.env` (see `.env.example`).

| Variable | Default | Service | Description |
|----------|---------|---------|-------------|
| `HF_TOKEN` | empty | segmentation | Hugging Face token (required for `facebook/sam3`). |
| `SAM3_MODEL_ID` | `facebook/sam3` | segmentation | Hugging Face model id. |
| `SAM3_LAZY_LOAD` | `true` | segmentation | Load SAM 3 on the first request instead of at startup. |
| `SAM3_FP32` | `false` | segmentation | Run SAM 3 in float32 instead of FP16. |
| `LABELING_COOLDOWN_MS` | `200` | segmentation | Pause between images during labeling (GPU thermal control). |
| `INTER_CLASS_COOLDOWN_MS` | `300` | segmentation | Pause between classes within one image. |
| `VIDEO_MAX_FRAMES` | `3000` | segmentation | Maximum number of frames extracted per video. |
| `DINOV3_WEIGHTS_DIR` | `/app/checkpoints/dinov3` | segmentation | Directory with the DINOv3 weights. |
| `INSID3_MODEL_SIZE` | `large` | segmentation | DINOv3 size: `small`, `base` or `large`. |
| `SEGMENTATION_SERVICE_URL` | `http://segmentation:8002` | gateway | URL of the segmentation service. |
| `CORS_ORIGINS` | `http://localhost:3001,http://localhost:5173` | gateway | Allowed CORS origins (set by the override file). |
| `ALLOWED_FS_PATHS` | `/app/datasets` | gateway | Comma-separated paths the file browser and the comparison endpoint may access (set by the override file). |

### Frontend development

```bash
cd frontend
npm ci           # installs the exact versions from package-lock.json
npm run dev      # Vite dev server
npm run build    # type-check and production build
```

## Repository structure

```
frontend/                  Vue 3 application (views, components, composables, i18n)
services/
  gateway/                 FastAPI gateway
    app/routers/           REST routers (annotations, comparison, datasets, export, jobs, sam3, videos, tracks, ...)
    app/export/            Format registry, Datumaro engine, importer, native exporters
    tests/
  segmentation/            SAM 3 and INSID3 service
    app/routers/           labeling, sam3_tool, video, propagate, ...
    app/insid3/            INSID3 engine and loader (vendor/ holds third-party code)
    app/video/             Frame extraction and tracking
    tests/
  shared/                  Code shared by both services
docker-compose.yml
docker-compose.override.yml
.env.example
```

## Tests

Tests are run per service, from the service directory, with that service's dependencies installed. No GPU or model weights are required.

```bash
cd services/gateway
pip install -r requirements.txt -r requirements-test.txt
python -m pytest tests -q

cd ../segmentation
pip install -r requirements.txt -r requirements-test.txt
python -m pytest tests -q
```

The segmentation Docker build also runs its tests in a build stage and fails if they do not pass.

## Acknowledgements and third-party licenses

This project builds on the following work. See `NOTICE` for the license notices.

- **SAM 3** (Meta), used through Hugging Face `transformers`. Distributed under Meta's SAM License; the weights are gated and are not included.
- **DINOv3** (Meta), the encoder used by INSID3. Distributed under the DINOv3 License; the weights are not included and must be downloaded by the user.
- **INSID3** (Visual Inference Lab, TU Darmstadt), training-free in-context segmentation. Vendored in `services/segmentation/app/insid3/vendor/` under Apache-2.0, with its own `LICENSE` and `NOTICE`. Repository: https://github.com/visinf/INSID3
- **Datumaro** (Intel), dataset import and export. MIT license.
- **Hugging Face `transformers`**, Apache-2.0.

Check the license terms of SAM 3 and DINOv3 before using the weights, or annotations generated with them, in your own work.

## Citation

If you use this software, please cite the article (metadata also in `CITATION.cff`):

```bibtex
@article{sanchezferrer2026synthetic,
  title   = {When Does Synthetic Data Help Underwater Debris Detection? Gap Metrics, Data Regimes and Detector Paradigms},
  author  = {S{\'a}nchez-Ferrer, Alejandro and Gallego, Antonio Javier and Calvo-Zaragoza, Jorge},
  journal = {Neurocomputing},
  year    = {2026},
  note    = {Manuscript in preparation}
}
```

## License

Apache License 2.0. See `LICENSE`. Third-party components keep their own licenses, as described above.
