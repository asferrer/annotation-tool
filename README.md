# Annotation Tool

AI-powered dataset annotation and labeling tool with SAM3 support.

## Features

- **Auto Labeling**: Automatically label images using SAM3 (Segment Anything Model 3)
- **Annotation Review**: Visual editor for reviewing and editing COCO annotations
- **Label Manager**: Rename, merge, or delete categories in your datasets
- **SAM3 Tools**: Convert bounding boxes to precise segmentation masks

## Quick Start

```bash
# Set your Hugging Face token
cp .env.example .env
# Edit .env with your HF_TOKEN

# Start all services
docker-compose up -d

# Open in browser
open http://localhost:3001
```

## Architecture

```
Frontend (Vue 3) :3001
    |
Gateway (FastAPI) :8000
    |
Segmentation (SAM3) :8002
```

## Services

| Service | Port | Description |
|---------|------|-------------|
| Frontend | 3001 | Vue 3 + Tailwind CSS |
| Gateway | 8000 | FastAPI API gateway |
| Segmentation | 8002 | SAM3 segmentation service |
