// Labeling
export type LabelingTaskType = 'detection' | 'segmentation' | 'both'
export type RelabelMode = 'add' | 'replace' | 'improve_segmentation'

export interface QualityMetrics {
  avg_confidence: number
  images_with_detections: number
  images_without_detections: number
  low_confidence_count: number
}

export interface LabelingJob {
  job_id: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
  progress: number
  processed_images: number
  total_images: number
  annotations_created: number
  current_image?: string
  output_dir?: string
  image_directories?: string[]
  objects_by_class?: Record<string, number>
  quality_metrics?: QualityMetrics
  warnings?: string[]
  error?: string
}

export interface LabelingPreview {
  filename: string
  image_data: string
}

export interface PartialAnnotationsResponse {
  job_id: string
  available: boolean
  data?: {
    images: CocoImage[]
    annotations: CocoAnnotation[]
    categories: CocoCategory[]
  }
  processed_images: number
  total_images: number
}

// SAM3
export interface SegmentationResult {
  masks: any[]
  scores: number[]
}

export interface Job {
  job_id: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  progress: number
  error?: string
}

// COCO Types
export interface CocoImage {
  id: number
  file_name: string
  width: number
  height: number
}

export interface CocoAnnotation {
  id: number
  image_id: number
  category_id: number
  bbox: [number, number, number, number]
  area: number
  iscrowd?: number
  segmentation?: number[][]
  score?: number
}

export interface CocoCategory {
  id: number
  name: string
  supercategory?: string
}

// Datasets
export interface DatasetInfo {
  path: string
  name: string
  num_images: number
  num_annotations?: number
  num_categories?: number
}

export interface CategoryInfo {
  id: number
  name: string
  count: number
}

export interface DatasetAnalysis {
  total_images: number
  total_annotations: number
  categories: CategoryInfo[]
  annotations_per_image?: {
    mean: number
    min: number
    max: number
  }
}

// Health
export interface HealthStatus {
  status: string
  services: Record<string, { status: string }>
}

// Export engine
export interface ExportFormatSpec {
  id: string
  label: string
  backend: 'datumaro' | 'native'
  tasks: string[]
  supports_import: boolean
  description: string
}

export interface ExportJob {
  id: string
  job_type: string
  service: string
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled' | 'interrupted'
  output_path?: string
  error_message?: string
  result_summary?: Record<string, unknown>
  created_at?: string
  completed_at?: string
}

// Video
export interface VideoInfo {
  path: string
  width: number
  height: number
  fps: number
  frame_count: number
  duration_s: number
}

export interface VideoUploadResult {
  video_id: string
  video_path: string
  frames_dir: string
  size_bytes: number
  original_name: string
}

export interface ExtractFramesResult {
  frames_dir: string
  extracted_frames: number
  frame_stride: number
  effective_fps: number
  video: VideoInfo
}

export interface TrackSummary {
  track_id: number
  category_id: number
  label: string
  frame_count: number
  first_frame: number
  last_frame: number
  avg_score: number
}

export interface VideoMeta {
  id: number
  name: string
  source_path: string
  fps: number
  frame_stride: number
  frame_count: number
  width: number
  height: number
}
