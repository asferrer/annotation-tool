import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 120000,
})

// Health
export async function getHealthStatus() {
  const { data } = await api.get('/health')
  return data
}

// Annotations CRUD
export async function loadAnnotations(cocoJsonPath: string) {
  const { data } = await api.post('/annotations/load', { coco_json_path: cocoJsonPath })
  return data
}

export async function saveAnnotations(request: any) {
  const { data } = await api.post('/annotations/save', request)
  return data
}

export async function getImageAnnotations(imageId: number, cocoJsonPath: string) {
  const { data } = await api.get('/annotations/images/' + imageId, { params: { coco_json_path: cocoJsonPath } })
  return data
}

export async function createAnnotation(request: { coco_json_path: string; image_id: number; category_id: number; bbox: number[]; segmentation?: number[][]; area?: number }) {
  const { data } = await api.post('/annotations/create', request)
  return data
}

export async function updateAnnotation(request: { coco_json_path: string; annotation_id: number; bbox?: number[]; category_id?: number; segmentation?: number[][] }) {
  const { data } = await api.put('/annotations/update', request)
  return data
}

export async function deleteAnnotation(request: { coco_json_path: string; annotation_id: number }) {
  const { data } = await api.post('/annotations/delete', request)
  return data
}

// Labeling
export async function startLabeling(request: any) {
  const { data } = await api.post('/labeling/start', request)
  return data
}

export async function startRelabeling(request: any) {
  const { data } = await api.post('/labeling/relabel', request)
  return data
}

export async function listLabelingJobs() {
  const { data } = await api.get('/labeling/jobs')
  return data as { jobs: any[]; total: number }
}

export async function getLabelingJobStatus(jobId: string) {
  const { data } = await api.get(`/labeling/jobs/${jobId}`)
  return data
}

export async function cancelLabelingJob(jobId: string) {
  const { data } = await api.delete(`/labeling/jobs/${jobId}`)
  return data
}

export async function getLabelingJobResult(jobId: string) {
  const { data } = await api.get(`/labeling/jobs/${jobId}/result`)
  return data
}

export async function getLabelingJobPreviews(jobId: string) {
  const { data } = await api.get(`/labeling/jobs/${jobId}/previews`)
  return data
}

export async function getPartialAnnotations(jobId: string) {
  const { data } = await api.get(`/labeling/jobs/${jobId}/partial-annotations`)
  return data
}

// Export engine (multi-format, Datumaro-backed)
export async function getExportFormats() {
  const { data } = await api.get('/export/formats')
  return data as import('@/types/api').ExportFormatSpec[]
}

export async function startExport(request: { coco_json_path: string; format: string; include_images?: boolean; images_dir?: string | null }) {
  const { data } = await api.post('/export/start', request)
  return data as { job_id: string; status: string }
}

export async function getExportJob(jobId: string) {
  const { data } = await api.get(`/export/jobs/${jobId}`)
  return data as import('@/types/api').ExportJob
}

export function getExportDownloadUrl(jobId: string): string {
  return `/api/export/jobs/${jobId}/download`
}

export async function startImport(request: { input_path: string; format: string; task_name: string }) {
  const { data } = await api.post('/import/start', request)
  return data as { job_id: string; status: string }
}

// SAM3
export async function segmentWithText(
  imagePath: string,
  textPrompt: string,
): Promise<{ success: boolean; segmentation_coco: number[][] | null; bbox: number[] | null; confidence: number; error?: string }> {
  const { data } = await api.post('/annotations/segment-text', { image_path: imagePath, text_prompt: textPrompt })
  return data
}

export async function segmentBbox(imagePath: string, bbox: number[], textHint?: string): Promise<{ success: boolean; segmentation_coco: number[][] | null }> {
  const { data } = await api.post('/annotations/segment-bbox', { image_path: imagePath, bbox, text_hint: textHint || null })
  return data
}

export async function segmentPoint(
  imagePath: string,
  points: [number, number][],
  labels: number[],
  textHint?: string,
): Promise<{ success: boolean; segmentation_coco: number[][] | null; bbox: number[] | null; confidence: number }> {
  const { data } = await api.post('/annotations/segment-point', {
    image_path: imagePath,
    points,
    labels,
    text_hint: textHint || null,
  })
  return data
}

export async function sam3ConvertDataset(request: any) {
  const { data } = await api.post('/sam3/convert-dataset', request)
  return data
}

export async function getSam3JobStatus(jobId: string) {
  const { data } = await api.get(`/sam3/jobs/${jobId}`)
  return data
}

// Datasets
export async function listDatasets() {
  const { data } = await api.get('/datasets')
  return data
}

export async function analyzeDataset(datasetPath: string) {
  const { data } = await api.post('/datasets/analyze', { dataset_path: datasetPath })
  return data
}

export async function renameCategory(datasetPath: string, categoryId: number, newName: string) {
  const { data } = await api.post('/datasets/categories/rename', { dataset_path: datasetPath, category_id: categoryId, new_name: newName })
  return data
}

export async function deleteCategory(datasetPath: string, categoryId: number) {
  const { data } = await api.delete(`/datasets/categories/${categoryId}`, { params: { dataset_path: datasetPath } })
  return data
}

// Filesystem
export async function listDirectories(path: string) {
  const { data } = await api.get('/filesystem/browse', { params: { path, type: 'directories' } })
  return data
}

export async function listFiles(path: string, pattern?: string) {
  const { data } = await api.get('/filesystem/browse', { params: { path, type: 'files', pattern } })
  return data
}

export async function checkPathExists(path: string): Promise<{ exists: boolean; is_directory: boolean }> {
  try {
    const { data } = await api.get('/filesystem/check-path', { params: { path } })
    return data
  } catch {
    return { exists: false, is_directory: false }
  }
}

/**
 * Upload images in chunks to handle datasets of any size.
 * Sends batches of CHUNK_SIZE files sequentially to avoid memory issues.
 */
export async function uploadImages(
  taskName: string,
  files: File[],
  onProgress?: (uploaded: number, total: number) => void,
) {
  const CHUNK_SIZE = 50
  let totalUploaded = 0
  let directory = ''
  const allFiles: string[] = []

  for (let i = 0; i < files.length; i += CHUNK_SIZE) {
    const chunk = files.slice(i, i + CHUNK_SIZE)
    const formData = new FormData()
    formData.append('task_name', taskName)
    for (const file of chunk) {
      formData.append('files', file)
    }

    const { data } = await api.post('/filesystem/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 0, // no timeout
    })

    directory = data.directory
    totalUploaded += data.uploaded_count
    allFiles.push(...data.files)

    if (onProgress) onProgress(totalUploaded, files.length)
  }

  return { directory, uploaded_count: totalUploaded, files: allFiles, coco_json_path: '' }
}

export async function uploadCocoJson(taskName: string, file: File) {
  const formData = new FormData()
  formData.append('task_name', taskName)
  formData.append('file', file)
  const { data } = await api.post('/filesystem/upload-coco', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 120000,
  })
  return data as { path: string }
}

export async function scanImages(dirPath: string) {
  const { data } = await api.get('/filesystem/scan-images', { params: { path: dirPath } })
  return data
}

export function getImageUrl(imagePath: string): string {
  return `/api/filesystem/image?path=${encodeURIComponent(imagePath)}`
}

// Comparison
export async function evaluateComparison(request: {
  ground_truth_path: string
  predictions_path: string
  iou_threshold: number
}) {
  const { data } = await api.post('/comparison/evaluate', request)
  return data
}

// Jobs
export async function getAllJobs() {
  const { data } = await api.get('/jobs')
  return data
}

// Video
export async function uploadVideo(file: File, onProgress?: (percent: number) => void) {
  const formData = new FormData()
  formData.append('file', file)
  const { data } = await api.post('/videos/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 0,
    onUploadProgress: (e) => {
      if (onProgress && e.total) onProgress(Math.round((e.loaded / e.total) * 100))
    },
  })
  return data as import('@/types/api').VideoUploadResult
}

export async function probeVideo(videoPath: string) {
  const { data } = await api.post('/videos/probe', { video_path: videoPath })
  return data as import('@/types/api').VideoInfo
}

export async function extractFrames(request: { video_path: string; target_fps?: number; frame_stride?: number; max_frames?: number; output_dir?: string }) {
  const { data } = await api.post('/videos/extract-frames', request, { timeout: 600000 })
  return data as import('@/types/api').ExtractFramesResult
}

export async function startTracking(request: { frames_dir: string; prompts: string[]; min_confidence?: number; output_json: string; video_name?: string; source_video_path?: string; fps?: number; frame_stride?: number }) {
  const { data } = await api.post('/videos/track', request)
  return data as { success: boolean; job_id: string; status: string; total_frames: number }
}

export async function getVideoJob(jobId: string) {
  const { data } = await api.get(`/videos/jobs/${jobId}`)
  return data as import('@/types/api').ExportJob & { processed_items?: number; total_items?: number }
}

export async function cancelVideoJob(jobId: string) {
  const { data } = await api.delete(`/videos/jobs/${jobId}`)
  return data
}

// Tracks
export async function getTracks(cocoJsonPath: string) {
  const { data } = await api.get('/tracks', { params: { coco_json_path: cocoJsonPath } })
  return data as { tracks: import('@/types/api').TrackSummary[]; total: number; videos: import('@/types/api').VideoMeta[] }
}

export async function renameTrack(cocoJsonPath: string, trackId: number, newLabel: string) {
  const { data } = await api.post('/tracks/rename', { coco_json_path: cocoJsonPath, track_id: trackId, new_label: newLabel })
  return data
}

export async function deleteTrack(cocoJsonPath: string, trackId: number) {
  const { data } = await api.post('/tracks/delete', { coco_json_path: cocoJsonPath, track_id: trackId })
  return data
}

export async function mergeTracks(cocoJsonPath: string, sourceTrackId: number, targetTrackId: number) {
  const { data } = await api.post('/tracks/merge', { coco_json_path: cocoJsonPath, source_track_id: sourceTrackId, target_track_id: targetTrackId })
  return data
}

// INSID3 propagation (one annotated example -> whole dataset)
export async function getPropagateStatus() {
  const { data } = await api.get('/propagate/status')
  return data as { insid3: { model_size: string; weights_path: string; weights_present: boolean; instructions: string | null } }
}

export async function startPropagation(request: { reference_image_path: string; reference_polygon: number[][]; target_directory: string; label: string; merge_threshold?: number; min_area?: number; output_json: string; max_images?: number }) {
  const { data } = await api.post('/propagate/start', request)
  return data as { success: boolean; job_id: string; status: string }
}

export async function getPropagateJob(jobId: string) {
  const { data } = await api.get(`/propagate/jobs/${jobId}`)
  return data as import('@/types/api').ExportJob & { processed_items?: number; total_items?: number }
}

export async function cancelPropagateJob(jobId: string) {
  const { data } = await api.delete(`/propagate/jobs/${jobId}`)
  return data
}

export default api
