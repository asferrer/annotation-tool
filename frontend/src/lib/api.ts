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

export async function exportAnnotations(request: { coco_json_path: string; output_dir: string; formats: string[] }) {
  const { data } = await api.post('/annotations/export', request)
  return data
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

export default api
