<script setup lang="ts">
import { ref, computed, watch, nextTick, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { useUiStore } from '@/stores/ui'
import { useTaskStore } from '@/stores/task'
import { useSettingsStore } from '@/stores/settings'
import { loadAnnotations, saveAnnotations, getImageUrl, scanImages, startLabeling, getLabelingJobStatus, getHealthStatus, getPartialAnnotations, segmentBbox, segmentPoint, segmentWithText } from '@/lib/api'
import ExportDialog from '@/components/export/ExportDialog.vue'
import PropagateDialog from '@/components/annotate/PropagateDialog.vue'
import { useAnnotationEditor } from '@/composables/useAnnotationEditor'
import BaseButton from '@/components/ui/BaseButton.vue'
import {
  Save, ZoomIn, ZoomOut, Maximize2,
  Eye, EyeOff, Tag, Trash2, ImageIcon,
  AlertCircle, CheckCircle2, MousePointer2,
  Square, Hand, Undo2, Redo2, ChevronLeft, ChevronRight,
  Wand2, Plus, X, Download, ArrowLeft,
  SkipBack, SkipForward, Search, Layers, Loader2, List, Crosshair,
  ChevronUp, ChevronDown, Share2,
} from 'lucide-vue-next'

const router = useRouter()
const { t, locale } = useI18n()
const uiStore = useUiStore()
const taskStore = useTaskStore()
const settings = useSettingsStore()
const editor = useAnnotationEditor()

function toggleLanguage() {
  settings.language = settings.language === 'en' ? 'es' : 'en'
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const ZOOM_STEP = 0.15
const ZOOM_MIN = 0.1
const ZOOM_MAX = 10.0
const HANDLE_SIZE = 8

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const isLoading = ref(true)
const cocoJsonPath = ref('')
const imagesDirPath = ref('')
const isSaving = ref(false)

// Canvas
const canvasContainer = ref<HTMLDivElement | null>(null)
const imageEl = ref<HTMLImageElement | null>(null)
const zoom = ref(1.0)
const panX = ref(0)
const panY = ref(0)
const imgNatW = ref(0)
const imgNatH = ref(0)
const imgW = ref(0)
const imgH = ref(0)
const showBboxes = ref(true)
const showLabels = ref(true)
const showMasks = ref(true)
const drawCategoryId = ref<number>(1)

// Interaction state
const isDrawing = ref(false)
const drawStart = ref({ x: 0, y: 0 })
const drawCurrent = ref({ x: 0, y: 0 })
const isDragging = ref(false)
const isResizing = ref(false)
const resizeHandle = ref('')
const dragStart = ref({ x: 0, y: 0 })
const dragOrigBbox = ref<[number, number, number, number]>([0, 0, 0, 0])
const isPanning = ref(false)
const panStart = ref({ x: 0, y: 0 })
const panStartOffset = ref({ x: 0, y: 0 })

// Frame navigation
const activeImageId = ref<number | null>(null)
const frameInput = ref('')

// Right panel
const rightTab = ref<'objects' | 'labels'>('objects')
const aiPanelOpen = ref(true)
const selectedCategoryFilter = ref<number | null>(null)

// Category management
const newLabelName = ref('')

// Auto-labeling
const autoLabelClassSource = ref<'existing' | 'custom'>('existing')
const autoLabelClasses = ref('')
const autoLabelSelectedCats = ref<number[]>([])
const autoLabelConfidence = ref(0.5)
const autoLabelRunning = ref(false)
const autoLabelProgress = ref(0)
const autoLabelStatus = ref('')

// SAM3 status
const sam3Status = ref<'checking' | 'available' | 'loading' | 'unavailable'>('checking')
const sam3Error = ref('')

// Export
const showExport = ref(false)
const showPropagate = ref(false)

const propagateDefaults = computed(() => {
  const ann = editor.selectedAnnotation.value
  const cat = ann ? categories.value.find((c) => c.id === ann.category_id) : null
  return {
    polygon: ann?.segmentation ?? [],
    label: cat?.name ?? '',
  }
})

// Category popup after drawing bbox
const pendingBbox = ref<{ x: number; y: number; w: number; h: number } | null>(null)
const popupPos = ref({ x: 0, y: 0 })
const isSegmenting = ref(false)

// Point-click segmentation (Sam3TrackerModel)
const pointClickPoints = ref<{ x: number; y: number; label: number }[]>([])
const pointClickPreviewSeg = ref<number[][] | null>(null)
const pointClickPreviewBbox = ref<number[] | null>(null)
const isPointSegmenting = ref(false)
const pendingPointClick = ref(false)

// Auto-label live sync
const autoLabelJobId = ref<string | null>(null)

// Text-prompt segmentation (SAM3 PCS mode — segment current image by text)
const textPromptInput = ref('')
const textPromptRunning = ref(false)
const textPromptError = ref('')

// Panel visibility
const showRightPanel = ref(true)
const showImageList = ref(false)

// Image list panel
const imageSearch = ref('')
const imageListRef = ref<HTMLDivElement | null>(null)
const imageListWidth = ref(208) // px, matches w-52
const isResizingPanel = ref(false)
const resizePanelStartX = ref(0)
const resizePanelStartWidth = ref(0)

// Annotation edit mode (bbox resize handles vs polygon vertex drag)
const annotationEditMode = ref<'bbox' | 'mask'>('bbox')
const draggingPolyPoint = ref<{ polyIdx: number; ptIdx: number } | null>(null)
const editingPolygon = ref<number[][] | null>(null)
const hoveredPolyPoint = ref(false)

// ---------------------------------------------------------------------------
// Computed
// ---------------------------------------------------------------------------
const images = computed(() => editor.dataset.value?.images ?? [])
const categories = computed(() => editor.dataset.value?.categories ?? [])
const totalImages = computed(() => images.value.length)
const activeImage = computed(() => images.value.find((img) => img.id === activeImageId.value) ?? null)
const currentIndex = computed(() => images.value.findIndex((img) => img.id === activeImageId.value))

const filteredImageList = computed(() => {
  const q = imageSearch.value.trim().toLowerCase()
  if (!q) return images.value
  return images.value.filter((img) => img.file_name.toLowerCase().includes(q))
})

// O(1) lookup: imageId → 1-based global index
const imageIndexMap = computed(() => {
  const m = new Map<number, number>()
  images.value.forEach((img, i) => m.set(img.id, i + 1))
  return m
})

// O(n) once: annotation count per image
const annCountByImage = computed(() => {
  const m = new Map<number, number>()
  editor.annotations.value.forEach((ann) => m.set(ann.image_id, (m.get(ann.image_id) ?? 0) + 1))
  return m
})

// Polygon vertices of the selected annotation scaled to canvas coords (for mask edit mode)
const selectedPolygonPoints = computed(() => {
  const ann = editor.selectedAnnotation.value
  if (!ann?.segmentation?.length || annotationEditMode.value !== 'mask') return []
  const seg = editingPolygon.value ?? ann.segmentation
  const pts: { x: number; y: number; polyIdx: number; ptIdx: number }[] = []
  seg.forEach((poly, polyIdx) => {
    for (let i = 0; i + 1 < poly.length; i += 2) {
      pts.push({ x: poly[i] * scaleX.value, y: poly[i + 1] * scaleY.value, polyIdx, ptIdx: i })
    }
  })
  return pts
})

const cursorStyle = computed(() => {
  if (draggingPolyPoint.value) return 'move'
  if (hoveredPolyPoint.value) return 'move'
  if (editor.toolMode.value === 'draw-bbox') return 'crosshair'
  if (editor.toolMode.value === 'segment-point') return 'crosshair'
  if (editor.toolMode.value === 'pan') return isPanning.value ? 'grabbing' : 'grab'
  return 'default'
})

const activeImageUrl = computed(() => {
  if (!activeImage.value || !imagesDirPath.value) return ''
  const sep = imagesDirPath.value.endsWith('/') ? '' : '/'
  return getImageUrl(imagesDirPath.value + sep + activeImage.value.file_name)
})

const activeAnnotations = computed(() => {
  if (activeImageId.value === null) return []
  let anns = editor.getAnnotationsForImage(activeImageId.value)
  if (selectedCategoryFilter.value !== null) anns = anns.filter((a) => a.category_id === selectedCategoryFilter.value)
  return anns
})

function getCategoryColor(catId: number): string {
  const label = categories.value.find((c) => c.id === catId)
  if (label) {
    const idx = categories.value.indexOf(label)
    const colors = taskStore.LABEL_COLORS
    return colors[idx % colors.length]
  }
  return '#6366f1'
}

function getCategoryName(catId: number): string {
  return categories.value.find((c) => c.id === catId)?.name ?? `#${catId}`
}

const scaleX = computed(() => (imgNatW.value > 0 ? imgW.value / imgNatW.value : 1))
const scaleY = computed(() => (imgNatH.value > 0 ? imgH.value / imgNatH.value : 1))

const scaledBboxes = computed(() => {
  if (!showBboxes.value || imgNatW.value === 0) return []
  return activeAnnotations.value.map((ann) => {
    const [x, y, w, h] = ann.bbox
    return {
      id: ann.id, x: x * scaleX.value, y: y * scaleY.value,
      w: w * scaleX.value, h: h * scaleY.value,
      color: getCategoryColor(ann.category_id),
      label: getCategoryName(ann.category_id),
      selected: ann.id === editor.selectedId.value,
    }
  })
})

// Convert a flat COCO polygon [x0,y0,x1,y1,...] to SVG points string "x0,y0 x1,y1 ..."
function cocoPolyToSvgPoints(poly: number[], sx: number, sy: number): string {
  const pts: string[] = []
  for (let i = 0; i + 1 < poly.length; i += 2) {
    pts.push(`${poly[i] * sx},${poly[i + 1] * sy}`)
  }
  return pts.join(' ')
}

const scaledPolygons = computed(() => {
  if (!showMasks.value || imgNatW.value === 0) return []
  return activeAnnotations.value
    .filter((ann) => ann.segmentation && ann.segmentation.length > 0)
    .map((ann) => {
      // Show live editing polygon for the selected annotation while dragging a vertex
      const seg = (ann.id === editor.selectedId.value && editingPolygon.value) ? editingPolygon.value : ann.segmentation!
      return {
        id: ann.id,
        color: getCategoryColor(ann.category_id),
        selected: ann.id === editor.selectedId.value,
        polygons: seg.map((poly) => cocoPolyToSvgPoints(poly, scaleX.value, scaleY.value)),
      }
    })
})

const drawPreview = computed(() => {
  if (!isDrawing.value) return null
  const x = Math.min(drawStart.value.x, drawCurrent.value.x)
  const y = Math.min(drawStart.value.y, drawCurrent.value.y)
  const w = Math.abs(drawCurrent.value.x - drawStart.value.x)
  const h = Math.abs(drawCurrent.value.y - drawStart.value.y)
  return { x: x * scaleX.value, y: y * scaleY.value, w: w * scaleX.value, h: h * scaleY.value }
})

// ---------------------------------------------------------------------------
// Initialize
// ---------------------------------------------------------------------------
onMounted(async () => {
  if (!taskStore.config) { router.push('/'); return }

  imagesDirPath.value = taskStore.config.imagesDirPath
  cocoJsonPath.value = taskStore.config.cocoJsonPath

  try {
    if (cocoJsonPath.value) {
      const data = await loadAnnotations(cocoJsonPath.value)
      // Merge task labels with COCO categories
      const mergedCats = [...data.categories]
      for (const tl of taskStore.config.labels) {
        if (!mergedCats.some((c) => c.name === tl.name)) {
          mergedCats.push({ id: tl.id, name: tl.name })
        }
      }
      data.categories = mergedCats
      editor.loadDataset(data)
    } else {
      const result = await scanImages(imagesDirPath.value)
      const cats = taskStore.config.labels.map((l) => ({ id: l.id, name: l.name }))
      editor.loadDataset({ images: result.images, annotations: [], categories: cats })
    }

    // Restore saved image position if available
    const savedIdx = taskStore.lastSavedImageIndex
    if (savedIdx > 0 && savedIdx < images.value.length) {
      activeImageId.value = images.value[savedIdx].id
    } else {
      activeImageId.value = images.value[0]?.id ?? null
    }
    if (categories.value.length > 0) drawCategoryId.value = categories.value[0].id
    // Default to draw-bbox so the user can start annotating immediately
    editor.toolMode.value = 'draw-bbox'
  } catch (err: any) {
    uiStore.showError('Load failed', err?.message ?? 'Unknown error')
    router.push('/')
  } finally {
    isLoading.value = false
  }

  // Check SAM3 availability in background
  checkSam3Status()
})

// ---------------------------------------------------------------------------
// Frame navigation
// ---------------------------------------------------------------------------
function goToFrame(idx: number) {
  const clamped = Math.max(0, Math.min(totalImages.value - 1, idx))
  if (images.value[clamped]) selectImage(images.value[clamped].id)
}
function selectImage(id: number) {
  activeImageId.value = id; panX.value = 0; panY.value = 0; editor.select(null)
  frameInput.value = ''
  // Persist image position to resume session later
  taskStore.persistSession(currentIndex.value)
  // zoom will be auto-calculated in onImageLoad → fitToView
}
function prevFrame() { goToFrame(currentIndex.value - 1) }
function nextFrame() { goToFrame(currentIndex.value + 1) }
function firstFrame() { goToFrame(0) }
function lastFrame() { goToFrame(totalImages.value - 1) }
function jumpToFrame() {
  const n = parseInt(frameInput.value)
  if (!isNaN(n) && n >= 1 && n <= totalImages.value) goToFrame(n - 1)
}

// ---------------------------------------------------------------------------
// Save
// ---------------------------------------------------------------------------
async function handleSave(): Promise<boolean> {
  const data = editor.getDatasetForSave()
  if (!data) return false
  if (!cocoJsonPath.value) cocoJsonPath.value = imagesDirPath.value.replace(/\/$/, '') + '/annotations.json'
  isSaving.value = true
  try {
    await saveAnnotations({ coco_json_path: cocoJsonPath.value, data })
    editor.markSaved()
    uiStore.showSuccess('Saved', 'Annotations saved')
    return true
  } catch (err: any) {
    uiStore.showError('Save failed', err?.message ?? 'Error')
    return false
  } finally { isSaving.value = false }
}

// ---------------------------------------------------------------------------
// Label management
// ---------------------------------------------------------------------------
function addLabel() {
  const name = newLabelName.value.trim()
  if (!name || !editor.dataset.value) return
  if (categories.value.some((c) => c.name.toLowerCase() === name.toLowerCase())) return
  const maxId = categories.value.reduce((max, c) => Math.max(max, c.id), 0)
  editor.dataset.value = { ...editor.dataset.value, categories: [...categories.value, { id: maxId + 1, name }] }
  if (categories.value.length === 1) drawCategoryId.value = categories.value[0].id
  newLabelName.value = ''
  editor.hasUnsavedChanges.value = true
}

function removeLabel(id: number) {
  if (!editor.dataset.value) return
  editor.annotations.value = editor.annotations.value.filter((a) => a.category_id !== id)
  editor.dataset.value = { ...editor.dataset.value, categories: categories.value.filter((c) => c.id !== id) }
  editor.hasUnsavedChanges.value = true
}

// ---------------------------------------------------------------------------
// SAM3 health check
// ---------------------------------------------------------------------------
async function checkSam3Status() {
  sam3Status.value = 'checking'
  try {
    const health = await getHealthStatus()
    const seg = health.services?.segmentation
    if (!seg) { sam3Status.value = 'unavailable'; sam3Error.value = 'Segmentation service unreachable'; return }
    if (seg.sam3_available) { sam3Status.value = 'available'; sam3Error.value = ''; return }
    if (seg.sam3_loading) { sam3Status.value = 'loading'; sam3Error.value = seg.sam3_load_progress || 'Loading model...'; return }
    sam3Status.value = 'unavailable'
    sam3Error.value = seg.sam3_load_error || 'SAM3 model not available'
  } catch {
    sam3Status.value = 'unavailable'
    sam3Error.value = 'Cannot connect to gateway'
  }
}

// ---------------------------------------------------------------------------
// Auto-labeling
// ---------------------------------------------------------------------------
function getAutoLabelClasses(): string[] {
  if (autoLabelClassSource.value === 'existing') {
    // Use selected existing categories, or all if none selected
    const selected = autoLabelSelectedCats.value
    if (selected.length > 0) {
      return categories.value.filter((c) => selected.includes(c.id)).map((c) => c.name)
    }
    return categories.value.map((c) => c.name)
  }
  return autoLabelClasses.value.split(',').map((c) => c.trim()).filter(Boolean)
}

async function startAutoLabeling() {
  const classes = getAutoLabelClasses()
  if (classes.length === 0) {
    uiStore.showError('No classes', 'Select or define at least one class to detect')
    return
  }
  autoLabelRunning.value = true; autoLabelProgress.value = 0; autoLabelStatus.value = 'Starting...'
  try {
    const outputDir = imagesDirPath.value.replace(/\/$/, '') + '/auto_labeled'
    const resp = await startLabeling({
      image_directories: [imagesDirPath.value], classes, output_dir: outputDir,
      output_formats: ['coco'], task_type: 'segmentation', min_confidence: autoLabelConfidence.value, save_visualizations: false,
    })
    autoLabelJobId.value = resp.job_id
    autoLabelStatus.value = 'Processing...'
    // Poll status + live sync partial annotations into editor
    statusPollTimer = setInterval(async () => {
      try {
        const job = await getLabelingJobStatus(resp.job_id)
        autoLabelProgress.value = job.progress ?? 0
        autoLabelStatus.value = `${job.processed_images ?? 0}/${job.total_images ?? '?'} images`
        if (job.status === 'completed' || job.status === 'failed') {
          if (statusPollTimer) { clearInterval(statusPollTimer); statusPollTimer = null }
          if (livePollTimer) { clearInterval(livePollTimer); livePollTimer = null }
          autoLabelRunning.value = false; autoLabelJobId.value = null
          if (job.status === 'completed') {
            autoLabelStatus.value = `Done! ${job.annotations_created ?? 0} annotations`
            uiStore.showSuccess('Auto-labeling complete', `${job.annotations_created ?? 0} annotations`)
            try {
              const finalOutputDir = job.output_dir ?? outputDir
              const data = await loadAnnotations(finalOutputDir + '/annotations.json')
              editor.loadDataset(data); cocoJsonPath.value = finalOutputDir + '/annotations.json'
              activeImageId.value = images.value[0]?.id ?? null
              if (categories.value.length > 0) drawCategoryId.value = categories.value[0].id
            } catch { /* user can load manually */ }
          } else {
            autoLabelStatus.value = `Failed: ${job.error ?? 'Unknown'}`
          }
        }
      } catch { /* retry */ }
    }, 2000)
    // Live annotation sync every 5s
    livePollTimer = setInterval(async () => {
      try {
        const partial = await getPartialAnnotations(resp.job_id)
        if (partial.available && partial.data) {
          editor.loadDataset(partial.data)
          if (activeImageId.value === null && partial.data.images.length > 0) {
            activeImageId.value = partial.data.images[0].id
          }
          if (categories.value.length > 0 && !drawCategoryId.value) drawCategoryId.value = categories.value[0].id
        }
      } catch { /* ignore */ }
    }, 5000)
  } catch (err: any) {
    autoLabelRunning.value = false; autoLabelStatus.value = ''
    uiStore.showError('Failed', err?.message ?? 'Error')
  }
}

// ---------------------------------------------------------------------------
// Category popup
// ---------------------------------------------------------------------------
function startPopupDrag(e: MouseEvent) {
  e.preventDefault()
  e.stopPropagation()
  const container = canvasContainer.value
  if (!container) return
  const rect = container.getBoundingClientRect()
  // Offset = cursor position within container minus popup top-left (same coord system)
  const ox = e.clientX - rect.left - popupPos.value.x
  const oy = e.clientY - rect.top - popupPos.value.y
  const onMove = (ev: MouseEvent) => {
    popupPos.value = { x: ev.clientX - rect.left - ox, y: ev.clientY - rect.top - oy }
  }
  const onUp = () => {
    document.removeEventListener('mousemove', onMove, true)
    document.removeEventListener('mouseup', onUp, true)
  }
  document.addEventListener('mousemove', onMove, true)
  document.addEventListener('mouseup', onUp, true)
}

function getActiveImagePath(): string {
  if (!activeImage.value || !imagesDirPath.value) return ''
  const sep = imagesDirPath.value.endsWith('/') ? '' : '/'
  return imagesDirPath.value + sep + activeImage.value.file_name
}

async function trySegmentAndUpdate(annId: number, bbox: { x: number; y: number; w: number; h: number }, categoryId?: number) {
  if (sam3Status.value !== 'available') return
  isSegmenting.value = true
  try {
    const imagePath = getActiveImagePath()
    if (!imagePath) return
    const textHint = categoryId ? getCategoryName(categoryId) : undefined
    const result = await segmentBbox(imagePath, [bbox.x, bbox.y, bbox.w, bbox.h], textHint)
    if (result.success && result.segmentation_coco && result.segmentation_coco.length > 0) {
      editor.updateAnnotationSegmentation(annId, result.segmentation_coco)
    }
  } catch {
    // SAM3 failed — annotation stays as bbox-only, which is fine
  } finally {
    isSegmenting.value = false
  }
}

// ---------------------------------------------------------------------------
// Point-click segmentation (Sam3TrackerModel)
// ---------------------------------------------------------------------------
async function triggerPointSegmentation() {
  if (!pointClickPoints.value.length || sam3Status.value !== 'available') return
  isPointSegmenting.value = true
  try {
    const imagePath = getActiveImagePath()
    if (!imagePath) return
    const points = pointClickPoints.value.map((p) => [p.x, p.y] as [number, number])
    const labels = pointClickPoints.value.map((p) => p.label)
    // Pass category name as text_hint if a single annotation is selected (better PCS quality)
    const selAnn = editor.selectedAnnotation.value
    const textHint = selAnn
      ? categories.value.find((c) => c.id === selAnn.category_id)?.name
      : categories.value.length === 1 ? categories.value[0].name : undefined
    const result = await segmentPoint(imagePath, points, labels, textHint)
    if (result.success && result.segmentation_coco) {
      pointClickPreviewSeg.value = result.segmentation_coco
      pointClickPreviewBbox.value = result.bbox
    }
  } catch {
    // preview won't update — not critical
  } finally {
    isPointSegmenting.value = false
  }
}

function clearPointClick() {
  pointClickPoints.value = []
  pointClickPreviewSeg.value = null
  pointClickPreviewBbox.value = null
  isPointSegmenting.value = false
  pendingPointClick.value = false
}

// Single-click auto-segment: calls SAM3 with one foreground point and auto-shows category picker.
async function triggerSinglePointSegmentation(x: number, y: number) {
  clearPointClick()
  if (categories.value.length === 0) { uiStore.showError('No labels', 'Add labels first'); return }
  pointClickPoints.value = [{ x, y, label: 1 }]
  popupPos.value = { x: Math.min(x * scaleX.value * zoom.value + panX.value + 60, 300), y: Math.max(y * scaleY.value * zoom.value + panY.value - 20, 40) }
  isPointSegmenting.value = true
  try {
    const imagePath = getActiveImagePath()
    if (!imagePath) { clearPointClick(); return }
    const textHint = categories.value.length === 1 ? categories.value[0].name : undefined
    const result = await segmentPoint(imagePath, [[x, y]], [1], textHint)
    if (result.success && result.segmentation_coco) {
      pointClickPreviewSeg.value = result.segmentation_coco
      pointClickPreviewBbox.value = result.bbox
      if (categories.value.length === 1) {
        await confirmPointSegmentation(categories.value[0].id)
      } else {
        pendingPointClick.value = true
      }
    } else {
      clearPointClick()
    }
  } catch {
    clearPointClick()
  } finally {
    isPointSegmenting.value = false
  }
}

async function confirmPointSegmentation(catId: number) {
  if (!pointClickPoints.value.length || activeImageId.value === null) { clearPointClick(); return }
  const bbox = pointClickPreviewBbox.value
  if (!bbox) { clearPointClick(); return }
  const ann = editor.createAnnotation(activeImageId.value, catId, { x: bbox[0], y: bbox[1], w: bbox[2], h: bbox[3] })
  if (pointClickPreviewSeg.value) {
    editor.updateAnnotationSegmentation(ann.id, pointClickPreviewSeg.value)
  }
  clearPointClick()
}

async function confirmBboxCategory(catId: number) {
  if (!pendingBbox.value || activeImageId.value === null) {
    pendingBbox.value = null
    return
  }
  const bbox = pendingBbox.value
  pendingBbox.value = null
  drawCategoryId.value = catId
  const ann = editor.createAnnotation(activeImageId.value, catId, bbox)
  trySegmentAndUpdate(ann.id, bbox, catId)
}

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------
async function openExport() {
  // Save COCO first so the export reflects the latest edits
  const saved = await handleSave()
  if (!saved) return
  if (!cocoJsonPath.value) { uiStore.showError('Error', 'No save path'); return }
  showExport.value = true
}

// ---------------------------------------------------------------------------
// Canvas interaction (same logic as before, compacted)
// ---------------------------------------------------------------------------
function fitToView() {
  if (!canvasContainer.value || imgNatW.value === 0 || imgNatH.value === 0) {
    zoom.value = 1.0; panX.value = 0; panY.value = 0; return
  }
  const container = canvasContainer.value
  const padding = 32 // 16px margin each side
  const availW = container.clientWidth - padding
  const availH = container.clientHeight - padding
  const scaleToFit = Math.min(availW / imgNatW.value, availH / imgNatH.value, 1.0)
  zoom.value = Math.max(ZOOM_MIN, +scaleToFit.toFixed(3))
  panX.value = 0
  panY.value = 0
  nextTick(() => {
    if (imageEl.value) { imgW.value = imageEl.value.offsetWidth; imgH.value = imageEl.value.offsetHeight }
  })
}

function onImageLoad() {
  if (!imageEl.value) return
  imgNatW.value = imageEl.value.naturalWidth; imgNatH.value = imageEl.value.naturalHeight
  // Auto fit image to canvas viewport
  nextTick(() => {
    if (imageEl.value) { imgW.value = imageEl.value.offsetWidth; imgH.value = imageEl.value.offsetHeight }
    fitToView()
  })
}
watch(zoom, () => nextTick(() => { if (imageEl.value) { imgW.value = imageEl.value.offsetWidth; imgH.value = imageEl.value.offsetHeight } }))

watch(activeImageId, () => {
  if (!showImageList.value) return
  nextTick(() => {
    imageListRef.value?.querySelector('[data-active="true"]')?.scrollIntoView({ block: 'nearest' })
  })
})

function onWheel(e: WheelEvent) {
  e.preventDefault()
  zoom.value = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, +(zoom.value + (e.deltaY > 0 ? -ZOOM_STEP : ZOOM_STEP)).toFixed(2)))
}

function getCoords(e: MouseEvent) {
  if (!imageEl.value) return null
  const r = imageEl.value.getBoundingClientRect()
  return { x: Math.max(0, Math.min(imgNatW.value, (e.clientX - r.left) / scaleX.value / zoom.value)), y: Math.max(0, Math.min(imgNatH.value, (e.clientY - r.top) / scaleY.value / zoom.value)) }
}

// Reset edit mode when selection changes
watch(() => editor.selectedId.value, () => {
  annotationEditMode.value = 'bbox'
  editingPolygon.value = null
  draggingPolyPoint.value = null
  hoveredPolyPoint.value = false
})

// Compute bbox that tightly contains all polygon vertices
function bboxFromPolygons(polygons: number[][]): [number, number, number, number] {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
  for (const poly of polygons) {
    for (let i = 0; i + 1 < poly.length; i += 2) {
      if (poly[i] < minX) minX = poly[i]
      if (poly[i + 1] < minY) minY = poly[i + 1]
      if (poly[i] > maxX) maxX = poly[i]
      if (poly[i + 1] > maxY) maxY = poly[i + 1]
    }
  }
  return [minX, minY, maxX - minX, maxY - minY]
}

// Hit-test a polygon vertex. Returns {polyIdx, ptIdx} or null.
function hitTestPolyPoint(imgX: number, imgY: number) {
  const ann = editor.selectedAnnotation.value
  if (!ann?.segmentation?.length) return null
  const seg = editingPolygon.value ?? ann.segmentation
  const r = 8 / (scaleX.value * zoom.value) // 8px screen → image coords
  for (let polyIdx = 0; polyIdx < seg.length; polyIdx++) {
    const poly = seg[polyIdx]
    for (let i = 0; i + 1 < poly.length; i += 2) {
      const dx = poly[i] - imgX, dy = poly[i + 1] - imgY
      if (dx * dx + dy * dy < r * r) return { polyIdx, ptIdx: i }
    }
  }
  return null
}

// Resize handlers for the image list panel
function onPanelResizeStart(e: MouseEvent) {
  isResizingPanel.value = true
  resizePanelStartX.value = e.clientX
  resizePanelStartWidth.value = imageListWidth.value
  document.addEventListener('mousemove', onPanelResizeMove)
  document.addEventListener('mouseup', onPanelResizeEnd)
  e.preventDefault()
}
function onPanelResizeMove(e: MouseEvent) {
  imageListWidth.value = Math.max(150, Math.min(500, resizePanelStartWidth.value + e.clientX - resizePanelStartX.value))
}
function onPanelResizeEnd() {
  isResizingPanel.value = false
  document.removeEventListener('mousemove', onPanelResizeMove)
  document.removeEventListener('mouseup', onPanelResizeEnd)
}

// Re-segment the selected annotation using SAM3 (replaces mask, bbox stays)
async function resegmentSelected() {
  if (!editor.selectedAnnotation.value || editor.selectedId.value === null) return
  const ann = editor.selectedAnnotation.value
  await trySegmentAndUpdate(ann.id, { x: ann.bbox[0], y: ann.bbox[1], w: ann.bbox[2], h: ann.bbox[3] }, ann.category_id)
}

// Segment the current image by free-text prompt (SAM3 PCS mode).
// Creates a new annotation with the best-matching mask + asks the user to assign a category.
async function runTextPromptSegmentation() {
  const prompt = textPromptInput.value.trim()
  if (!prompt || activeImageId.value === null) return
  textPromptRunning.value = true
  textPromptError.value = ''
  try {
    const imagePath = getActiveImagePath()
    if (!imagePath) return
    const result = await segmentWithText(imagePath, prompt)
    if (!result.success || !result.segmentation_coco) {
      textPromptError.value = result.error ?? 'No mask found for that prompt'
      return
    }
    // Find or create a category matching the prompt text
    const lowerPrompt = prompt.toLowerCase()
    let cat = categories.value.find((c) => c.name.toLowerCase() === lowerPrompt)
    if (!cat) {
      // Create new label from prompt
      const newId = Math.max(0, ...categories.value.map((c) => c.id)) + 1
      cat = { id: newId, name: prompt }
      if (editor.dataset.value) {
        editor.dataset.value = { ...editor.dataset.value, categories: [...categories.value, cat] }
      }
      editor.hasUnsavedChanges.value = true
    }
    const bbox = result.bbox ?? [0, 0, 10, 10]
    const ann = editor.createAnnotation(activeImageId.value, cat.id, { x: bbox[0], y: bbox[1], w: bbox[2], h: bbox[3] })
    editor.updateAnnotationSegmentation(ann.id, result.segmentation_coco)
    textPromptInput.value = ''
  } catch (err: any) {
    textPromptError.value = err?.message ?? 'Segmentation failed'
  } finally {
    textPromptRunning.value = false
  }
}

function onDblClick(e: MouseEvent) {
  if (editor.toolMode.value !== 'select') return
  const c = getCoords(e)
  if (!c) return
  const hit = hitTest(c.x, c.y)
  if (hit === null) return
  const ann = editor.annotations.value.find((a) => a.id === hit)
  if (ann?.segmentation?.length) {
    editor.select(hit)
    annotationEditMode.value = 'mask'
  }
}

function hitTest(x: number, y: number): number | null {
  for (let i = activeAnnotations.value.length - 1; i >= 0; i--) {
    const a = activeAnnotations.value[i]; const [bx, by, bw, bh] = a.bbox
    if (x >= bx && x <= bx + bw && y >= by && y <= by + bh) return a.id
  }
  return null
}

function hitHandle(x: number, y: number): string | null {
  const ann = editor.selectedAnnotation.value; if (!ann) return null
  const [bx, by, bw, bh] = ann.bbox; const hs = HANDLE_SIZE / scaleX.value / zoom.value
  for (const [h, hx, hy] of [['nw', bx, by], ['ne', bx + bw, by], ['sw', bx, by + bh], ['se', bx + bw, by + bh], ['n', bx + bw / 2, by], ['s', bx + bw / 2, by + bh], ['w', bx, by + bh / 2], ['e', bx + bw, by + bh / 2]] as [string, number, number][]) {
    if (Math.abs(x - hx) < hs && Math.abs(y - hy) < hs) return h
  }
  return null
}

function onMouseDown(e: MouseEvent) {
  if (e.button === 1 || (e.button === 0 && editor.toolMode.value === 'pan')) {
    isPanning.value = true; panStart.value = { x: e.clientX, y: e.clientY }; panStartOffset.value = { x: panX.value, y: panY.value }; e.preventDefault(); return
  }
  if (e.button !== 0) return
  const c = getCoords(e); if (!c) return
  if (editor.toolMode.value === 'draw-bbox') {
    if (categories.value.length === 0) { uiStore.showError('No labels', 'Add labels first'); return }
    isDrawing.value = true; drawStart.value = { ...c }; drawCurrent.value = { ...c }; return
  }
  if (editor.toolMode.value === 'segment-point') {
    if (categories.value.length === 0) { uiStore.showError('No labels', 'Add labels first'); return }
    const label = e.shiftKey ? 0 : 1  // Shift+click = background (0), click = foreground (1)
    if (pointClickPoints.value.length === 0 && label === 1) {
      // First foreground click → instant single-point auto-segment (same as draw-bbox single click)
      triggerSinglePointSegmentation(c.x, c.y)
    } else {
      // Subsequent click or background click → multi-point refinement mode
      pendingPointClick.value = false  // close picker if it was open from previous single-point
      pointClickPoints.value = [...pointClickPoints.value, { x: c.x, y: c.y, label }]
      triggerPointSegmentation()
    }
    return
  }
  // In mask edit mode, check for polygon vertex hit first
  if (annotationEditMode.value === 'mask' && editor.selectedAnnotation.value?.segmentation?.length) {
    const hit = hitTestPolyPoint(c.x, c.y)
    if (hit) {
      draggingPolyPoint.value = hit
      editingPolygon.value = editor.selectedAnnotation.value.segmentation.map((p) => [...p])
      return
    }
  }
  const h = hitHandle(c.x, c.y)
  if (h && editor.selectedAnnotation.value) { isResizing.value = true; resizeHandle.value = h; dragStart.value = { ...c }; dragOrigBbox.value = [...editor.selectedAnnotation.value.bbox] as any; return }
  const hit = hitTest(c.x, c.y)
  if (hit !== null) { editor.select(hit); isDragging.value = true; dragStart.value = { ...c }; dragOrigBbox.value = [...editor.annotations.value.find((a) => a.id === hit)!.bbox] as any }
  else editor.select(null)
}

function onMouseMove(e: MouseEvent) {
  if (isPanning.value) { panX.value = panStartOffset.value.x + (e.clientX - panStart.value.x); panY.value = panStartOffset.value.y + (e.clientY - panStart.value.y); return }
  const c = getCoords(e); if (!c) return
  if (isDrawing.value) { drawCurrent.value = { ...c }; return }
  // Polygon vertex drag
  if (draggingPolyPoint.value && editingPolygon.value) {
    const { polyIdx, ptIdx } = draggingPolyPoint.value
    editingPolygon.value[polyIdx][ptIdx] = Math.max(0, Math.min(imgNatW.value, c.x))
    editingPolygon.value[polyIdx][ptIdx + 1] = Math.max(0, Math.min(imgNatH.value, c.y))
    editingPolygon.value = editingPolygon.value.map((p) => [...p]) // trigger reactivity
    return
  }
  // Update hover cursor for polygon points
  if (annotationEditMode.value === 'mask') {
    hoveredPolyPoint.value = hitTestPolyPoint(c.x, c.y) !== null
  } else {
    hoveredPolyPoint.value = false
  }
  if (isDragging.value && editor.selectedId.value !== null) {
    const ann = editor.annotations.value.find((a) => a.id === editor.selectedId.value)
    if (ann) { ann.bbox = [dragOrigBbox.value[0] + c.x - dragStart.value.x, dragOrigBbox.value[1] + c.y - dragStart.value.y, dragOrigBbox.value[2], dragOrigBbox.value[3]]; editor.annotations.value = [...editor.annotations.value] }
    return
  }
  if (isResizing.value && editor.selectedId.value !== null) {
    const [ox, oy, ow, oh] = dragOrigBbox.value; const dx = c.x - dragStart.value.x; const dy = c.y - dragStart.value.y
    let nx = ox, ny = oy, nw = ow, nh = oh; const h = resizeHandle.value
    if (h.includes('w')) { nx = ox + dx; nw = ow - dx } if (h.includes('e')) nw = ow + dx
    if (h.includes('n')) { ny = oy + dy; nh = oh - dy } if (h.includes('s')) nh = oh + dy
    if (nw < 5) nw = 5; if (nh < 5) nh = 5
    const ann = editor.annotations.value.find((a) => a.id === editor.selectedId.value)
    if (ann) { ann.bbox = [nx, ny, nw, nh]; ann.area = nw * nh; editor.annotations.value = [...editor.annotations.value] }
  }
}

function onMouseUp() {
  if (isPanning.value) { isPanning.value = false; return }
  // Commit polygon vertex drag
  if (draggingPolyPoint.value && editingPolygon.value && editor.selectedId.value !== null) {
    const newBbox = bboxFromPolygons(editingPolygon.value)
    editor.updateAnnotationMask(editor.selectedId.value, editingPolygon.value, newBbox)
    draggingPolyPoint.value = null
    editingPolygon.value = null
    return
  }
  if (isDrawing.value) {
    isDrawing.value = false
    const x = Math.min(drawStart.value.x, drawCurrent.value.x), y = Math.min(drawStart.value.y, drawCurrent.value.y)
    const w = Math.abs(drawCurrent.value.x - drawStart.value.x), h = Math.abs(drawCurrent.value.y - drawStart.value.y)
    if (w > 3 && h > 3 && activeImageId.value !== null) {
      if (categories.value.length <= 1) {
        // Only 1 or 0 categories — create immediately, then try to segment
        const ann = editor.createAnnotation(activeImageId.value, drawCategoryId.value, { x, y, w, h })
        trySegmentAndUpdate(ann.id, { x, y, w, h }, drawCategoryId.value)
      } else {
        // Multiple categories — show popup to pick
        pendingBbox.value = { x, y, w, h }
        const cx = (x + w / 2) * scaleX.value * zoom.value + panX.value
        const cy = (y + h / 2) * scaleY.value * zoom.value + panY.value
        popupPos.value = { x: Math.min(cx + 60, 300), y: Math.max(cy - 20, 40) }
      }
    }
    return
  }
  if (isDragging.value && editor.selectedId.value !== null) {
    const ann = editor.annotations.value.find((a) => a.id === editor.selectedId.value)
    if (ann) { const dx = ann.bbox[0] - dragOrigBbox.value[0], dy = ann.bbox[1] - dragOrigBbox.value[1]; ann.bbox = [...dragOrigBbox.value] as any; if (Math.abs(dx) > 0.5 || Math.abs(dy) > 0.5) editor.moveAnnotation(editor.selectedId.value, dx, dy) }
    isDragging.value = false; return
  }
  if (isResizing.value && editor.selectedId.value !== null) {
    const ann = editor.annotations.value.find((a) => a.id === editor.selectedId.value)
    if (ann) { const nb = { x: ann.bbox[0], y: ann.bbox[1], w: ann.bbox[2], h: ann.bbox[3] }; ann.bbox = [...dragOrigBbox.value] as any; editor.resizeAnnotation(editor.selectedId.value, nb) }
    isResizing.value = false
  }
}

// ---------------------------------------------------------------------------
// Keyboard
// ---------------------------------------------------------------------------
function onKeyDown(e: KeyboardEvent) {
  const t = e.target as HTMLElement; if (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT') return
  switch (e.key) {
    case 'd': case 'f': case 'ArrowRight': e.preventDefault(); nextFrame(); break
    case 'a': case 'ArrowLeft': e.preventDefault(); prevFrame(); break
    case 'v': editor.toolMode.value = 'select'; break
    case 'n': editor.toolMode.value = 'draw-bbox'; break
    case 'p': if (sam3Status.value === 'available') { editor.toolMode.value = 'segment-point' } break
    case '+': case '=': e.preventDefault(); zoom.value = Math.min(ZOOM_MAX, +(zoom.value + ZOOM_STEP).toFixed(2)); break
    case '-': e.preventDefault(); zoom.value = Math.max(ZOOM_MIN, +(zoom.value - ZOOM_STEP).toFixed(2)); break
    case 'b': showBboxes.value = !showBboxes.value; break
    case 'l': showLabels.value = !showLabels.value; break
    case 'm': showMasks.value = !showMasks.value; break
    case 'i': showImageList.value = !showImageList.value; break
    case 'Delete': case 'Backspace': if (editor.selectedId.value !== null) { e.preventDefault(); editor.deleteAnnotation(editor.selectedId.value) } break
    case 'Escape': editor.select(null); isDrawing.value = false; pendingBbox.value = null; clearPointClick(); editor.toolMode.value = 'select'; showExport.value = false; break
    case 'Enter': if (editor.toolMode.value === 'segment-point' && pointClickPoints.value.length > 0 && pointClickPreviewBbox.value) {
      pendingPointClick.value = true
      const lastPt = pointClickPoints.value[pointClickPoints.value.length - 1]
      popupPos.value = { x: Math.min(lastPt.x * scaleX.value * zoom.value + panX.value + 60, 300), y: Math.max(lastPt.y * scaleY.value * zoom.value + panY.value - 20, 40) }
    } break
    case 'z': if (e.ctrlKey || e.metaKey) { e.preventDefault(); e.shiftKey ? editor.redo() : editor.undo() } break
    case 'y': if (e.ctrlKey || e.metaKey) { e.preventDefault(); editor.redo() } break
    case 's': if (e.ctrlKey || e.metaKey) { e.preventDefault(); handleSave() } break
  }
}
// Responsive: refit image when window/container resizes
let resizeObserver: ResizeObserver | null = null
let statusPollTimer: ReturnType<typeof setInterval> | null = null
let livePollTimer: ReturnType<typeof setInterval> | null = null

onMounted(() => {
  window.addEventListener('keydown', onKeyDown)
  // Observe canvas container size changes
  nextTick(() => {
    if (canvasContainer.value) {
      resizeObserver = new ResizeObserver(() => {
        if (imgNatW.value > 0) fitToView()
      })
      resizeObserver.observe(canvasContainer.value)
    }
  })
})

onUnmounted(() => {
  window.removeEventListener('keydown', onKeyDown)
  resizeObserver?.disconnect()
  if (statusPollTimer) { clearInterval(statusPollTimer); statusPollTimer = null }
  if (livePollTimer) { clearInterval(livePollTimer); livePollTimer = null }
})
</script>

<template>
  <!-- Fullscreen CVAT-like layout -->
  <div v-if="isLoading" class="h-screen flex items-center justify-center bg-background">
    <div class="text-center text-gray-400"><div class="animate-spin h-8 w-8 border-2 border-primary border-t-transparent rounded-full mx-auto mb-3" /><p class="text-sm">Loading dataset...</p></div>
  </div>

  <div v-else class="h-screen flex flex-col bg-background text-white overflow-hidden">

    <!-- ====== TOP NAVBAR (CVAT-style) ====== -->
    <header class="flex-shrink-0 h-10 bg-background-secondary border-b border-gray-700/50 flex items-center px-2 gap-1 text-xs overflow-x-auto">

      <!-- Back -->
      <button @click="router.push('/')" class="p-1.5 text-gray-400 hover:text-white" title="Back to tasks"><ArrowLeft class="h-4 w-4" /></button>
      <span class="text-gray-500 mr-1 truncate max-w-32">{{ taskStore.config?.name }}</span>

      <div class="w-px h-5 bg-gray-700" />

      <!-- Save -->
      <button @click="handleSave" :disabled="isSaving || !editor.hasUnsavedChanges.value" class="p-1.5 text-gray-400 hover:text-white disabled:opacity-30" title="Save (Ctrl+S)"><Save class="h-4 w-4" /></button>

      <!-- Undo/Redo -->
      <button @click="editor.undo()" :disabled="!editor.canUndo.value" class="p-1.5 text-gray-400 hover:text-white disabled:opacity-30" title="Undo (Ctrl+Z)"><Undo2 class="h-4 w-4" /></button>
      <button @click="editor.redo()" :disabled="!editor.canRedo.value" class="p-1.5 text-gray-400 hover:text-white disabled:opacity-30" title="Redo (Ctrl+Y)"><Redo2 class="h-4 w-4" /></button>

      <div class="w-px h-5 bg-gray-700" />

      <!-- Frame navigation (CVAT-style, centered) -->
      <div class="flex-1 flex items-center justify-center gap-1">
        <button @click="firstFrame" :disabled="currentIndex <= 0" class="p-1 text-gray-400 hover:text-white disabled:opacity-30" title="First frame"><SkipBack class="h-3.5 w-3.5" /></button>
        <button @click="prevFrame" :disabled="currentIndex <= 0" class="p-1 text-gray-400 hover:text-white disabled:opacity-30" title="Previous (A/←)"><ChevronLeft class="h-4 w-4" /></button>

        <input v-model="frameInput" @keyup.enter="jumpToFrame" type="text" :placeholder="String(currentIndex + 1)" class="w-12 text-center text-xs bg-background-tertiary border border-gray-600 rounded px-1 py-0.5 text-white focus:outline-none focus:border-primary" />
        <span class="text-gray-500">/ {{ totalImages }}</span>

        <button @click="nextFrame" :disabled="currentIndex >= totalImages - 1" class="p-1 text-gray-400 hover:text-white disabled:opacity-30" title="Next (D/→)"><ChevronRight class="h-4 w-4" /></button>
        <button @click="lastFrame" :disabled="currentIndex >= totalImages - 1" class="p-1 text-gray-400 hover:text-white disabled:opacity-30" title="Last frame"><SkipForward class="h-3.5 w-3.5" /></button>
      </div>

      <div class="w-px h-5 bg-gray-700" />

      <!-- Right: filename + status + export + panel toggle -->
      <span v-if="activeImage" class="text-gray-500 truncate max-w-48 hidden sm:inline">{{ activeImage.file_name }}</span>

      <div v-if="editor.hasUnsavedChanges.value" class="flex items-center gap-1 text-yellow-400 ml-2 flex-shrink-0"><AlertCircle class="h-3.5 w-3.5" /></div>
      <div v-else class="flex items-center gap-1 text-green-500 ml-2 flex-shrink-0"><CheckCircle2 class="h-3.5 w-3.5" /></div>

      <div class="w-px h-5 bg-gray-700 ml-1 flex-shrink-0" />

      <button @click="toggleLanguage" class="p-1.5 text-gray-400 hover:text-white flex-shrink-0 text-xs font-medium w-7 text-center" :title="locale === 'en' ? 'Switch to Spanish' : 'Switch to English'">{{ locale === 'en' ? 'ES' : 'EN' }}</button>
      <button @click="openExport" class="p-1.5 text-gray-400 hover:text-white flex-shrink-0" :title="t('annotate.export', 'Export')"><Download class="h-4 w-4" /></button>

      <button @click="showRightPanel = !showRightPanel" :class="['p-1.5 flex-shrink-0', showRightPanel ? 'text-primary' : 'text-gray-400 hover:text-white']" title="Toggle panel">
        <ChevronRight v-if="showRightPanel" class="h-4 w-4" />
        <ChevronLeft v-else class="h-4 w-4" />
      </button>
    </header>

    <!-- ====== MAIN AREA ====== -->
    <div class="flex flex-1 min-h-0">

      <!-- LEFT TOOLS SIDEBAR (CVAT-style, narrow) -->
      <aside class="flex-shrink-0 w-11 bg-background-secondary border-r border-gray-700/50 flex flex-col items-center py-2 gap-1 overflow-visible z-40">

        <!-- Select -->
        <div class="relative group/tool">
          <button @click="editor.toolMode.value = 'select'" :class="['p-2 rounded-lg transition-colors', editor.toolMode.value === 'select' ? 'bg-primary text-white' : 'text-gray-400 hover:text-white hover:bg-gray-700/50']"><MousePointer2 class="h-4 w-4" /></button>
          <div class="absolute left-full ml-2 top-1/2 -translate-y-1/2 z-[100] pointer-events-none opacity-0 group-hover/tool:opacity-100 transition-opacity duration-150 delay-500 bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 shadow-2xl w-52">
            <p class="font-semibold text-white text-xs">{{ t('annotate.tools.select') }} <kbd class="ml-1 font-mono text-[10px] bg-gray-700 px-1.5 py-0.5 rounded text-gray-300">V</kbd></p>
            <p class="text-gray-400 text-[11px] leading-relaxed mt-1">{{ t('annotate.tools.selectDesc') }}</p>
          </div>
        </div>

        <!-- Pan -->
        <div class="relative group/tool">
          <button @click="editor.toolMode.value = 'pan'" :class="['p-2 rounded-lg transition-colors', editor.toolMode.value === 'pan' ? 'bg-primary text-white' : 'text-gray-400 hover:text-white hover:bg-gray-700/50']"><Hand class="h-4 w-4" /></button>
          <div class="absolute left-full ml-2 top-1/2 -translate-y-1/2 z-[100] pointer-events-none opacity-0 group-hover/tool:opacity-100 transition-opacity duration-150 delay-500 bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 shadow-2xl w-52">
            <p class="font-semibold text-white text-xs">{{ t('annotate.tools.pan') }}</p>
            <p class="text-gray-400 text-[11px] leading-relaxed mt-1">{{ t('annotate.tools.panDesc') }}</p>
          </div>
        </div>

        <div class="w-6 border-t border-gray-700 my-1" />

        <!-- Zoom in -->
        <div class="relative group/tool">
          <button @click="zoom = Math.min(ZOOM_MAX, +(zoom + ZOOM_STEP).toFixed(2))" class="p-2 text-gray-400 hover:text-white hover:bg-gray-700/50 rounded-lg"><ZoomIn class="h-4 w-4" /></button>
          <div class="absolute left-full ml-2 top-1/2 -translate-y-1/2 z-[100] pointer-events-none opacity-0 group-hover/tool:opacity-100 transition-opacity duration-150 delay-500 bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 shadow-2xl w-44">
            <p class="font-semibold text-white text-xs">{{ t('annotate.tools.zoomIn') }} <kbd class="ml-1 font-mono text-[10px] bg-gray-700 px-1.5 py-0.5 rounded text-gray-300">+</kbd></p>
            <p class="text-gray-400 text-[11px] mt-1">{{ t('annotate.tools.zoomInDesc') }}</p>
          </div>
        </div>

        <!-- Fit to view -->
        <div class="relative group/tool">
          <button @click="fitToView" class="p-2 text-gray-400 hover:text-white hover:bg-gray-700/50 rounded-lg"><Maximize2 class="h-4 w-4" /></button>
          <div class="absolute left-full ml-2 top-1/2 -translate-y-1/2 z-[100] pointer-events-none opacity-0 group-hover/tool:opacity-100 transition-opacity duration-150 delay-500 bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 shadow-2xl w-44">
            <p class="font-semibold text-white text-xs">{{ t('annotate.tools.fitView') }}</p>
            <p class="text-gray-400 text-[11px] mt-1">{{ t('annotate.tools.fitViewDesc') }}</p>
          </div>
        </div>

        <!-- Zoom out -->
        <div class="relative group/tool">
          <button @click="zoom = Math.max(ZOOM_MIN, +(zoom - ZOOM_STEP).toFixed(2))" class="p-2 text-gray-400 hover:text-white hover:bg-gray-700/50 rounded-lg"><ZoomOut class="h-4 w-4" /></button>
          <div class="absolute left-full ml-2 top-1/2 -translate-y-1/2 z-[100] pointer-events-none opacity-0 group-hover/tool:opacity-100 transition-opacity duration-150 delay-500 bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 shadow-2xl w-44">
            <p class="font-semibold text-white text-xs">{{ t('annotate.tools.zoomOut') }} <kbd class="ml-1 font-mono text-[10px] bg-gray-700 px-1.5 py-0.5 rounded text-gray-300">-</kbd></p>
            <p class="text-gray-400 text-[11px] mt-1">{{ t('annotate.tools.zoomOutDesc') }}</p>
          </div>
        </div>

        <div class="w-6 border-t border-gray-700 my-1" />

        <!-- Draw bbox / point-click -->
        <div class="relative group/tool">
          <button @click="editor.toolMode.value = 'draw-bbox'" :class="['p-2 rounded-lg transition-colors', editor.toolMode.value === 'draw-bbox' ? 'bg-primary text-white' : 'text-gray-400 hover:text-white hover:bg-gray-700/50']"><Square class="h-4 w-4" /></button>
          <div class="absolute left-full ml-2 top-1/2 -translate-y-1/2 z-[100] pointer-events-none opacity-0 group-hover/tool:opacity-100 transition-opacity duration-150 delay-500 bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 shadow-2xl w-56">
            <p class="font-semibold text-white text-xs">{{ t('annotate.tools.annotate') }} <kbd class="ml-1 font-mono text-[10px] bg-gray-700 px-1.5 py-0.5 rounded text-gray-300">N</kbd></p>
            <p class="text-gray-400 text-[11px] leading-relaxed mt-1"><span class="text-white">Click</span> — {{ t('annotate.tools.annotateClick') }}</p>
            <p class="text-gray-400 text-[11px] leading-relaxed mt-0.5"><span class="text-white">Click + drag</span> — {{ t('annotate.tools.annotateDrag') }}</p>
          </div>
        </div>

        <!-- Multi-point segmentation -->
        <div class="relative group/tool">
          <button @click="editor.toolMode.value = 'segment-point'" :disabled="sam3Status !== 'available'" :class="['p-2 rounded-lg transition-colors disabled:opacity-30', editor.toolMode.value === 'segment-point' ? 'bg-purple-600 text-white' : 'text-gray-400 hover:text-white hover:bg-gray-700/50']"><Crosshair class="h-4 w-4" /></button>
          <div class="absolute left-full ml-2 top-1/2 -translate-y-1/2 z-[100] pointer-events-none opacity-0 group-hover/tool:opacity-100 transition-opacity duration-150 delay-500 bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 shadow-2xl w-56">
            <p class="font-semibold text-white text-xs">{{ t('annotate.tools.multiPoint') }} <kbd class="ml-1 font-mono text-[10px] bg-gray-700 px-1.5 py-0.5 rounded text-gray-300">P</kbd></p>
            <p class="text-gray-400 text-[11px] leading-relaxed mt-1"><span class="text-white">Click</span> — {{ t('annotate.tools.multiPointFg') }}</p>
            <p class="text-gray-400 text-[11px] leading-relaxed mt-0.5"><span class="text-white">Shift+click</span> — {{ t('annotate.tools.multiPointBg') }}</p>
            <p class="text-gray-400 text-[11px] leading-relaxed mt-0.5"><span class="text-white">Enter</span> — {{ t('annotate.tools.multiPointConfirm') }}</p>
            <p v-if="sam3Status !== 'available'" class="text-yellow-400/80 text-[10px] mt-1.5">{{ t('annotate.tools.multiPointRequires') }}</p>
          </div>
        </div>

        <div class="w-6 border-t border-gray-700 my-1" />

        <!-- AI Tools panel -->
        <div class="relative group/tool">
          <button @click="aiPanelOpen = !aiPanelOpen; showRightPanel = true" :class="['p-2 rounded-lg transition-colors', aiPanelOpen ? 'bg-purple-600 text-white' : 'text-gray-400 hover:text-white hover:bg-gray-700/50']"><Wand2 class="h-4 w-4" /></button>
          <div class="absolute left-full ml-2 top-1/2 -translate-y-1/2 z-[100] pointer-events-none opacity-0 group-hover/tool:opacity-100 transition-opacity duration-150 delay-500 bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 shadow-2xl w-52">
            <p class="font-semibold text-white text-xs">{{ t('annotate.tools.aiTools') }}</p>
            <p class="text-gray-400 text-[11px] leading-relaxed mt-1">{{ t('annotate.tools.aiToolsDesc') }}</p>
          </div>
        </div>

        <div class="w-6 border-t border-gray-700 my-1" />

        <!-- Image list -->
        <div class="relative group/tool">
          <button @click="showImageList = !showImageList" :class="['p-2 rounded-lg transition-colors', showImageList ? 'bg-gray-600 text-white' : 'text-gray-400 hover:text-white hover:bg-gray-700/50']"><List class="h-4 w-4" /></button>
          <div class="absolute left-full ml-2 top-1/2 -translate-y-1/2 z-[100] pointer-events-none opacity-0 group-hover/tool:opacity-100 transition-opacity duration-150 delay-500 bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 shadow-2xl w-44">
            <p class="font-semibold text-white text-xs">{{ t('annotate.tools.imageList') }} <kbd class="ml-1 font-mono text-[10px] bg-gray-700 px-1.5 py-0.5 rounded text-gray-300">I</kbd></p>
            <p class="text-gray-400 text-[11px] mt-1">{{ t('annotate.tools.imageListDesc') }}</p>
          </div>
        </div>

        <!-- Spacer -->
        <div class="flex-1" />

        <!-- Visibility toggles -->
        <div class="relative group/tool">
          <button @click="showBboxes = !showBboxes" :class="['p-2 rounded-lg', showBboxes ? 'text-primary' : 'text-gray-600']"><Eye class="h-4 w-4" /></button>
          <div class="absolute left-full ml-2 top-1/2 -translate-y-1/2 z-[100] pointer-events-none opacity-0 group-hover/tool:opacity-100 transition-opacity duration-150 delay-500 bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 shadow-2xl w-44">
            <p class="font-semibold text-white text-xs">{{ t('annotate.tools.bboxes') }} <kbd class="ml-1 font-mono text-[10px] bg-gray-700 px-1.5 py-0.5 rounded text-gray-300">B</kbd></p>
            <p class="text-gray-400 text-[11px] mt-1">{{ t('annotate.tools.bboxesDesc') }}</p>
          </div>
        </div>
        <div class="relative group/tool">
          <button @click="showLabels = !showLabels" :class="['p-2 rounded-lg', showLabels ? 'text-primary' : 'text-gray-600']"><Tag class="h-4 w-4" /></button>
          <div class="absolute left-full ml-2 top-1/2 -translate-y-1/2 z-[100] pointer-events-none opacity-0 group-hover/tool:opacity-100 transition-opacity duration-150 delay-500 bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 shadow-2xl w-44">
            <p class="font-semibold text-white text-xs">{{ t('annotate.tools.labelsToggle') }} <kbd class="ml-1 font-mono text-[10px] bg-gray-700 px-1.5 py-0.5 rounded text-gray-300">L</kbd></p>
            <p class="text-gray-400 text-[11px] mt-1">{{ t('annotate.tools.labelsToggleDesc') }}</p>
          </div>
        </div>
        <div class="relative group/tool">
          <button @click="showMasks = !showMasks" :class="['p-2 rounded-lg', showMasks ? 'text-purple-400' : 'text-gray-600']"><Layers class="h-4 w-4" /></button>
          <div class="absolute left-full ml-2 top-1/2 -translate-y-1/2 z-[100] pointer-events-none opacity-0 group-hover/tool:opacity-100 transition-opacity duration-150 delay-500 bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 shadow-2xl w-44">
            <p class="font-semibold text-white text-xs">{{ t('annotate.tools.masks') }} <kbd class="ml-1 font-mono text-[10px] bg-gray-700 px-1.5 py-0.5 rounded text-gray-300">M</kbd></p>
            <p class="text-gray-400 text-[11px] mt-1">{{ t('annotate.tools.masksDesc') }}</p>
          </div>
        </div>
        <span v-if="isSegmenting || isPointSegmenting" class="flex items-center gap-1 px-2 text-xs text-purple-300 animate-pulse"><Loader2 class="h-3 w-3 animate-spin" /></span>
      </aside>

      <!-- IMAGE LIST PANEL (collapsible, resizable) -->
      <aside v-if="showImageList" class="flex-shrink-0 bg-background-secondary border-r border-gray-700/50 flex flex-col relative" :style="{ width: imageListWidth + 'px' }">
        <!-- Search header -->
        <div class="flex items-center gap-1.5 px-2 py-1.5 border-b border-gray-700/50 flex-shrink-0">
          <Search class="h-3 w-3 text-gray-500 flex-shrink-0" />
          <input v-model="imageSearch" type="text" placeholder="Filter images…" class="flex-1 text-xs bg-transparent text-white placeholder-gray-500 focus:outline-none" />
          <span class="text-xs text-gray-600 tabular-nums">{{ filteredImageList.length }}</span>
        </div>
        <!-- Image list -->
        <div ref="imageListRef" class="flex-1 overflow-y-auto">
          <div
            v-for="img in filteredImageList" :key="img.id"
            :data-active="img.id === activeImageId ? 'true' : undefined"
            @click="activeImageId = img.id"
            :class="['flex items-center gap-2 px-2 py-1.5 cursor-pointer text-xs transition-colors border-l-2 group',
              img.id === activeImageId
                ? 'bg-primary/15 border-l-primary text-white'
                : 'border-l-transparent text-gray-400 hover:bg-gray-700/20 hover:text-gray-200']">
            <span class="tabular-nums text-gray-600 w-7 text-right flex-shrink-0">{{ imageIndexMap.get(img.id) }}</span>
            <span class="flex-1 truncate" :title="img.file_name">{{ img.file_name }}</span>
            <span v-if="annCountByImage.get(img.id)" class="flex-shrink-0 tabular-nums bg-primary/20 text-primary rounded px-1 py-0.5 text-[10px]">{{ annCountByImage.get(img.id) }}</span>
          </div>
          <div v-if="filteredImageList.length === 0" class="flex items-center justify-center h-16 text-xs text-gray-600">No images found</div>
        </div>
        <!-- Drag handle -->
        <div class="absolute top-0 right-0 w-1.5 h-full cursor-col-resize hover:bg-primary/40 transition-colors z-10" @mousedown.stop="onPanelResizeStart" />
      </aside>

      <!-- CANVAS WRAPPER (flex item, sized by flex layout only) -->
      <div class="flex-1 min-w-0 min-h-0 relative">
      <!-- CANVAS (absolute fill, isolates content from flex sizing) -->
      <div class="absolute inset-0 overflow-hidden bg-gray-950"
        ref="canvasContainer"
        @mousedown="onMouseDown" @mousemove="onMouseMove" @mouseup="onMouseUp" @dblclick="onDblClick"
        @mouseleave="() => { isDragging = false; isResizing = false; isPanning = false }"
        @wheel="onWheel" @contextmenu.prevent
        :style="{ cursor: cursorStyle }">

        <!-- Draw category indicator (floating) -->
        <div v-if="editor.toolMode.value === 'draw-bbox' && categories.length > 0" class="absolute top-2 left-2 z-10 flex items-center gap-1.5 bg-black/70 rounded-lg px-2.5 py-1.5 backdrop-blur-sm">
          <div class="w-3 h-3 rounded-sm" :style="{ backgroundColor: getCategoryColor(drawCategoryId) }" />
          <select v-model="drawCategoryId" class="text-xs bg-transparent border-0 text-white focus:outline-none cursor-pointer">
            <option v-for="cat in categories" :key="cat.id" :value="cat.id" class="bg-gray-800">{{ cat.name }}</option>
          </select>
        </div>

        <!-- Point-click mode indicator (floating) -->
        <div v-if="editor.toolMode.value === 'segment-point'" class="absolute top-2 left-2 z-10 flex items-center gap-2 bg-black/70 rounded-lg px-2.5 py-1.5 backdrop-blur-sm text-xs">
          <Crosshair class="h-3.5 w-3.5 text-purple-400 flex-shrink-0" />
          <span class="text-purple-300">SAM3</span>
          <!-- Idle: no points yet -->
          <template v-if="pointClickPoints.length === 0 && !isPointSegmenting">
            <span class="text-gray-500">· click to segment · shift+click = background</span>
          </template>
          <!-- Segmenting in progress -->
          <template v-else-if="isPointSegmenting">
            <span class="text-purple-400 animate-pulse">· segmenting…</span>
          </template>
          <!-- 1 point: preview ready, offer refinement -->
          <template v-else-if="pointClickPoints.length === 1 && pointClickPreviewBbox">
            <span class="text-gray-400">· 1 point</span>
            <span class="text-green-400">· click to refine · Enter to confirm</span>
          </template>
          <!-- Multi-point: show counters and confirm hint -->
          <template v-else-if="pointClickPoints.length > 1">
            <span class="text-gray-400">· {{ pointClickPoints.filter(p => p.label === 1).length }} foreground / {{ pointClickPoints.filter(p => p.label === 0).length }} background</span>
            <span v-if="pointClickPreviewBbox" class="text-green-400">· Enter to confirm</span>
            <span v-else class="text-purple-400 animate-pulse">· processing…</span>
          </template>
        </div>

        <!-- Zoom indicator -->
        <div class="absolute bottom-2 left-2 z-10 text-xs text-gray-500 bg-black/50 rounded px-2 py-0.5">{{ Math.round(zoom * 100) }}%</div>

        <div v-if="!activeImage" class="flex items-center justify-center h-full text-gray-600">
          <div class="text-center"><ImageIcon class="h-16 w-16 mx-auto mb-3 opacity-20" /><p class="text-sm">No image selected</p></div>
        </div>
        <div v-else class="relative inline-block" :style="{ transform: `translate(${panX}px, ${panY}px) scale(${zoom})`, transformOrigin: 'top left', margin: '16px' }">
          <img ref="imageEl" :src="activeImageUrl" class="block max-w-none select-none" draggable="false" @load="onImageLoad" />
          <svg v-if="imgW > 0" :width="imgW" :height="imgH" class="absolute top-0 left-0 overflow-visible" :viewBox="`0 0 ${imgW} ${imgH}`" style="pointer-events: none;">
            <!-- Segmentation masks (rendered below bboxes) -->
            <g v-for="seg in scaledPolygons" :key="`seg-${seg.id}`">
              <polygon v-for="(pts, i) in seg.polygons" :key="i"
                :points="pts"
                :fill="seg.color"
                :fill-opacity="seg.selected ? 0.35 : 0.15"
                :stroke="seg.color"
                :stroke-width="seg.selected ? 2 : 1"
                stroke-opacity="0.6"
              />
            </g>
            <!-- Polygon vertex handles (mask edit mode) -->
            <g v-if="annotationEditMode === 'mask' && selectedPolygonPoints.length > 0">
              <circle v-for="(pt, i) in selectedPolygonPoints" :key="i"
                :cx="pt.x" :cy="pt.y" r="5"
                fill="white" stroke="#a855f7" stroke-width="1.5"
                :fill-opacity="draggingPolyPoint?.polyIdx === pt.polyIdx && draggingPolyPoint?.ptIdx === pt.ptIdx ? 1 : 0.85"
              />
            </g>
            <g v-for="bb in scaledBboxes" :key="bb.id">
              <rect :x="bb.x" :y="bb.y" :width="bb.w" :height="bb.h" :stroke="bb.color" :stroke-width="bb.selected ? 3 : 2" fill="none" :opacity="bb.selected ? 1 : 0.8" />
              <rect v-if="bb.selected" :x="bb.x" :y="bb.y" :width="bb.w" :height="bb.h" :fill="bb.color" opacity="0.1" />
              <template v-if="showLabels">
                <rect :x="bb.x" :y="bb.y-16" :width="bb.label.length*7+8" :height="16" :fill="bb.color" rx="2" opacity="0.9" />
                <text :x="bb.x+4" :y="bb.y-4" fill="white" font-size="10" font-family="ui-monospace,monospace" font-weight="600">{{ bb.label }}</text>
              </template>
              <template v-if="bb.selected">
                <rect v-for="(pos,h) in {nw:[bb.x-3,bb.y-3],ne:[bb.x+bb.w-3,bb.y-3],sw:[bb.x-3,bb.y+bb.h-3],se:[bb.x+bb.w-3,bb.y+bb.h-3],n:[bb.x+bb.w/2-3,bb.y-3],s:[bb.x+bb.w/2-3,bb.y+bb.h-3],w:[bb.x-3,bb.y+bb.h/2-3],e:[bb.x+bb.w-3,bb.y+bb.h/2-3]}" :key="h" :x="pos[0]" :y="pos[1]" width="6" height="6" fill="white" :stroke="bb.color" stroke-width="1.5" rx="1" />
              </template>
            </g>
            <rect v-if="drawPreview" :x="drawPreview.x" :y="drawPreview.y" :width="drawPreview.w" :height="drawPreview.h" :stroke="getCategoryColor(drawCategoryId)" stroke-width="2" fill="none" stroke-dasharray="6 3" />
            <!-- Point-click preview mask -->
            <g v-if="pointClickPreviewSeg">
              <polygon v-for="(poly, i) in pointClickPreviewSeg" :key="`pt-mask-${i}`"
                :points="cocoPolyToSvgPoints(poly, scaleX, scaleY)"
                fill="#a855f7" fill-opacity="0.3" stroke="#a855f7" stroke-width="2" stroke-dasharray="6 3" />
            </g>
            <!-- Point-click dots (foreground=green, background=red) -->
            <g v-if="pointClickPoints.length > 0">
              <g v-for="(pt, i) in pointClickPoints" :key="`pt-${i}`">
                <circle :cx="pt.x * scaleX" :cy="pt.y * scaleY" r="7" :fill="pt.label === 1 ? '#22c55e' : '#ef4444'" fill-opacity="0.85" :stroke="pt.label === 1 ? '#16a34a' : '#dc2626'" stroke-width="1.5" />
                <text :x="pt.x * scaleX" :y="pt.y * scaleY + 4" text-anchor="middle" fill="white" font-size="9" font-weight="bold">{{ pt.label === 1 ? '+' : '−' }}</text>
              </g>
            </g>
          </svg>
        </div>

        <!-- Category picker popup (after drawing a bbox) -->
        <div v-if="pendingBbox" class="absolute z-20 bg-background-secondary border border-gray-600 rounded-lg shadow-xl py-1 min-w-[140px]"
          :style="{ left: popupPos.x + 'px', top: popupPos.y + 'px' }"
          @mousedown.stop @mouseup.stop>
          <p class="px-3 py-1 text-xs text-gray-500 font-medium uppercase cursor-move select-none" @mousedown="startPopupDrag">{{ t('annotate.labelAs') }}</p>
          <button v-for="cat in categories" :key="cat.id" @click="confirmBboxCategory(cat.id)"
            class="w-full flex items-center gap-2 px-3 py-1.5 text-xs text-gray-200 hover:bg-primary/20 transition-colors">
            <span class="w-2.5 h-2.5 rounded-sm flex-shrink-0" :style="{ backgroundColor: getCategoryColor(cat.id) }" />
            {{ cat.name }}
          </button>
          <button @click="pendingBbox = null" class="w-full px-3 py-1.5 text-xs text-gray-500 hover:bg-gray-700/30 border-t border-gray-700/50 mt-1">
            {{ t('common.actions.cancel') }}
          </button>
        </div>

        <!-- Category picker popup (after point-click segmentation confirmation) -->
        <div v-if="pendingPointClick" class="absolute z-20 bg-background-secondary border border-purple-700/60 rounded-lg shadow-xl py-1 min-w-[140px]"
          :style="{ left: popupPos.x + 'px', top: popupPos.y + 'px' }"
          @mousedown.stop @mouseup.stop>
          <p class="px-3 py-1 text-xs text-purple-400 font-medium uppercase flex items-center gap-1.5 cursor-move select-none" @mousedown="startPopupDrag"><Crosshair class="h-3 w-3" /> {{ t('annotate.createAs') }}</p>
          <button v-for="cat in categories" :key="cat.id" @click="confirmPointSegmentation(cat.id)"
            class="w-full flex items-center gap-2 px-3 py-1.5 text-xs text-gray-200 hover:bg-purple-600/20 transition-colors">
            <span class="w-2.5 h-2.5 rounded-sm flex-shrink-0" :style="{ backgroundColor: getCategoryColor(cat.id) }" />
            {{ cat.name }}
          </button>
          <button @click="clearPointClick()" class="w-full px-3 py-1.5 text-xs text-gray-500 hover:bg-gray-700/30 border-t border-gray-700/50 mt-1">
            {{ t('common.actions.cancel') }}
          </button>
        </div>
      </div>
      </div>

      <!-- RIGHT SIDEBAR (CVAT Objects Sidebar) - collapsible -->
      <aside v-if="showRightPanel" class="flex flex-col w-60 flex-shrink-0 border-l border-gray-700/50 bg-background-secondary">
        <!-- Tabs: Objects + Labels only (AI panel is always visible at the bottom) -->
        <div class="flex border-b border-gray-700/50 flex-shrink-0">
          <button @click="rightTab = 'objects'" :class="['flex-1 py-2 text-xs font-medium', rightTab === 'objects' ? 'text-primary border-b-2 border-primary' : 'text-gray-500 hover:text-gray-300']">{{ t('annotate.objects') }}</button>
          <button @click="rightTab = 'labels'" :class="['flex-1 py-2 text-xs font-medium', rightTab === 'labels' ? 'text-primary border-b-2 border-primary' : 'text-gray-500 hover:text-gray-300']">{{ t('annotate.labels') }}</button>
        </div>

        <!-- OBJECTS TAB -->
        <template v-if="rightTab === 'objects'">
          <div class="p-2 border-b border-gray-700/50 flex-shrink-0">
            <select v-model="selectedCategoryFilter" class="w-full text-xs bg-background-tertiary border border-gray-600 rounded px-2 py-1 text-gray-300">
              <option :value="null">{{ t('annotations.allCategories') }} ({{ activeAnnotations.length }})</option>
              <option v-for="cat in categories" :key="cat.id" :value="cat.id">{{ cat.name }}</option>
            </select>
          </div>
          <div class="flex-1 overflow-y-auto">
            <div v-if="activeAnnotations.length === 0" class="flex flex-col items-center justify-center h-24 text-gray-600">
              <Tag class="h-6 w-6 mb-1 opacity-30" /><p class="text-xs">{{ t('annotate.noObjects') }}</p>
            </div>
            <div v-for="ann in activeAnnotations" :key="ann.id" @click="editor.select(ann.id)"
              :class="['flex items-center gap-2 px-2 py-1.5 cursor-pointer border-l-2 text-xs group transition-colors',
                ann.id === editor.selectedId.value ? 'bg-primary/15 border-l-primary' : 'border-l-transparent hover:bg-gray-700/20']">
              <div class="w-2 h-2 rounded-full flex-shrink-0" :style="{backgroundColor: getCategoryColor(ann.category_id)}" />
              <span class="flex-1 truncate text-gray-200">{{ getCategoryName(ann.category_id) }}</span>
              <Layers v-if="ann.segmentation?.length" class="h-3 w-3 text-purple-400 flex-shrink-0" title="Has segmentation mask" />
              <span class="text-gray-600 tabular-nums">#{{ ann.id }}</span>
              <button @click.stop="editor.deleteAnnotation(ann.id)" class="p-0.5 text-gray-600 hover:text-red-400 opacity-0 group-hover:opacity-100"><Trash2 class="h-3 w-3" /></button>
            </div>
          </div>
          <div v-if="editor.selectedAnnotation.value" class="flex-shrink-0 border-t border-gray-700/50 p-2.5 space-y-2">
            <div>
              <label class="text-xs text-gray-500">{{ t('annotate.category') }}</label>
              <select :value="editor.selectedAnnotation.value.category_id" @change="editor.updateCategory(editor.selectedId.value!, Number(($event.target as HTMLSelectElement).value))" class="w-full mt-0.5 text-xs bg-background-tertiary border border-gray-600 rounded px-2 py-1.5 text-gray-300">
                <option v-for="cat in categories" :key="cat.id" :value="cat.id">{{ cat.name }}</option>
              </select>
            </div>
            <!-- Edit mode toggle -->
            <div>
              <label class="text-xs text-gray-500 block mb-1">{{ t('annotate.editMode') }}</label>
              <div class="flex rounded overflow-hidden border border-gray-700 text-xs">
                <button @click="annotationEditMode = 'bbox'; editingPolygon = null"
                  :class="['flex-1 py-1 transition-colors', annotationEditMode === 'bbox' ? 'bg-primary text-white' : 'bg-background-tertiary text-gray-400 hover:text-white']">
                  {{ t('annotate.bbox') }}
                </button>
                <button @click="annotationEditMode = 'mask'"
                  :class="['flex-1 py-1 transition-colors', annotationEditMode === 'mask' ? 'bg-purple-600 text-white' : 'bg-background-tertiary text-gray-400 hover:text-white']"
                  :disabled="!editor.selectedAnnotation.value.segmentation?.length"
                  :title="editor.selectedAnnotation.value.segmentation?.length ? t('annotate.dragVertices') : t('annotate.noHasMask')">
                  {{ t('annotate.mask') }}
                </button>
              </div>
              <p v-if="annotationEditMode === 'mask'" class="text-[10px] text-purple-400 mt-1">{{ t('annotate.dragVertices') }}</p>
            </div>
            <!-- Re-segment button -->
            <button v-if="sam3Status === 'available'" @click="resegmentSelected" :disabled="isSegmenting"
              class="w-full text-xs py-1.5 rounded border border-purple-800/60 bg-purple-900/30 text-purple-300 hover:bg-purple-900/50 disabled:opacity-40 flex items-center justify-center gap-1.5 transition-colors">
              <Loader2 v-if="isSegmenting" class="h-3 w-3 animate-spin" /><Wand2 v-else class="h-3 w-3" />
              {{ isSegmenting ? t('annotate.segmenting') : t('annotate.resegment') }}
            </button>
            <!-- Propagate to dataset (INSID3) -->
            <button v-if="editor.selectedAnnotation.value.segmentation?.length" @click="showPropagate = true"
              class="w-full text-xs py-1.5 rounded border border-teal-800/60 bg-teal-900/30 text-teal-300 hover:bg-teal-900/50 flex items-center justify-center gap-1.5 transition-colors">
              <Share2 class="h-3 w-3" /> {{ t('annotate.propagate', 'Propagate to dataset') }}
            </button>
            <p class="text-xs text-gray-600">{{ Math.round(editor.selectedAnnotation.value.bbox[2]) }}×{{ Math.round(editor.selectedAnnotation.value.bbox[3]) }}px · {{ Math.round(editor.selectedAnnotation.value.area) }}px²</p>
            <p v-if="editor.selectedAnnotation.value.segmentation?.length" class="text-xs text-purple-400 flex items-center gap-1"><Layers class="h-3 w-3" /> {{ editor.selectedAnnotation.value.segmentation.length }} {{ t('annotate.hasPolygons') }}</p>
          </div>
        </template>

        <!-- LABELS TAB -->
        <template v-if="rightTab === 'labels'">
          <div class="p-2.5 border-b border-gray-700/50 flex-shrink-0">
            <div class="flex gap-1">
              <input v-model="newLabelName" @keyup.enter="addLabel" type="text" :placeholder="t('annotate.addLabelPlaceholder')" class="flex-1 text-xs bg-background-tertiary border border-gray-600 rounded px-2 py-1.5 text-white placeholder-gray-500 focus:outline-none focus:border-primary" />
              <button @click="addLabel" :disabled="!newLabelName.trim()" class="p-1.5 rounded bg-primary text-white disabled:opacity-30"><Plus class="h-3.5 w-3.5" /></button>
            </div>
          </div>
          <div class="flex-1 overflow-y-auto">
            <div v-if="categories.length === 0" class="p-4 text-center text-gray-600 text-xs">No labels. Add labels above.</div>
            <div v-for="(cat, idx) in categories" :key="cat.id" class="flex items-center gap-2 px-3 py-2 border-b border-gray-700/30 group">
              <div class="w-3 h-3 rounded-sm flex-shrink-0" :style="{backgroundColor: getCategoryColor(cat.id)}" />
              <span class="text-xs text-gray-200 flex-1">{{ cat.name }}</span>
              <span class="text-xs text-gray-600">{{ editor.annotations.value.filter(a => a.category_id === cat.id).length }}</span>
              <button @click="removeLabel(cat.id)" class="p-0.5 text-gray-600 hover:text-red-400 opacity-0 group-hover:opacity-100"><X class="h-3 w-3" /></button>
            </div>
          </div>
        </template>

        <!-- AI PANEL — always visible at the bottom of the right sidebar -->
        <div class="flex-shrink-0 border-t border-purple-900/60 bg-background-secondary">
          <!-- Collapsible header -->
          <button @click="aiPanelOpen = !aiPanelOpen"
            class="w-full flex items-center gap-2 px-3 py-2 hover:bg-purple-900/20 transition-colors">
            <Wand2 class="h-3.5 w-3.5 text-purple-400 flex-shrink-0" />
            <span class="text-xs font-semibold text-purple-300 flex-1 text-left">{{ t('annotate.ai') }}</span>
            <!-- SAM3 status dot -->
            <div :class="['w-2 h-2 rounded-full flex-shrink-0', sam3Status === 'available' ? 'bg-green-400' : sam3Status === 'loading' ? 'bg-yellow-400 animate-pulse' : sam3Status === 'checking' ? 'bg-gray-500 animate-pulse' : 'bg-red-400']" :title="sam3Status" />
            <ChevronUp v-if="aiPanelOpen" class="h-3.5 w-3.5 text-gray-500" />
            <ChevronDown v-else class="h-3.5 w-3.5 text-gray-500" />
          </button>

          <div v-if="aiPanelOpen" class="px-3 pb-3 space-y-3 max-h-[420px] overflow-y-auto">

            <!-- ── Text-prompt segmentation ── -->
            <div class="space-y-1.5">
              <p class="text-[10px] font-semibold text-gray-400 uppercase tracking-wide">Segment by text</p>
              <div class="flex gap-1">
                <input
                  v-model="textPromptInput"
                  @keyup.enter="runTextPromptSegmentation"
                  type="text"
                  placeholder="ej: fish, diver, debris…"
                  :disabled="textPromptRunning || sam3Status !== 'available'"
                  class="flex-1 text-xs bg-background-tertiary border border-gray-600 rounded px-2 py-1.5 text-white placeholder-gray-500 focus:outline-none focus:border-purple-500 disabled:opacity-50"
                />
                <button @click="runTextPromptSegmentation"
                  :disabled="textPromptRunning || !textPromptInput.trim() || sam3Status !== 'available'"
                  class="p-1.5 rounded bg-purple-600 hover:bg-purple-500 text-white disabled:opacity-40 transition-colors flex-shrink-0">
                  <Loader2 v-if="textPromptRunning" class="h-3.5 w-3.5 animate-spin" />
                  <Search v-else class="h-3.5 w-3.5" />
                </button>
              </div>
              <p v-if="textPromptError" class="text-[10px] text-red-400 bg-red-900/20 rounded px-2 py-1">{{ textPromptError }}</p>
              <p class="text-[10px] text-gray-600">SAM3 segments the best match in the current image. The label is created if it does not exist.</p>
            </div>

            <div class="border-t border-gray-700/50" />

            <!-- ── Auto-labeling (all images) ── -->
            <div class="space-y-2">
              <p class="text-[10px] font-semibold text-gray-400 uppercase tracking-wide">Label all images</p>

              <div>
                <label class="text-xs text-gray-400 mb-1.5 block">{{ t('annotate.objectClasses') }}</label>
                <div class="flex gap-1 bg-gray-800 rounded-lg p-0.5 mb-2">
                  <button @click="autoLabelClassSource = 'existing'" :disabled="autoLabelRunning"
                    :class="['flex-1 py-1.5 rounded-md text-xs font-medium transition-colors', autoLabelClassSource === 'existing' ? 'bg-purple-600 text-white' : 'text-gray-400 hover:text-white']">
                    {{ t('annotate.fromLabels') }}
                  </button>
                  <button @click="autoLabelClassSource = 'custom'" :disabled="autoLabelRunning"
                    :class="['flex-1 py-1.5 rounded-md text-xs font-medium transition-colors', autoLabelClassSource === 'custom' ? 'bg-purple-600 text-white' : 'text-gray-400 hover:text-white']">
                    {{ t('annotate.custom') }}
                  </button>
                </div>

                <template v-if="autoLabelClassSource === 'existing'">
                  <div v-if="categories.length === 0" class="text-xs text-gray-500 bg-gray-800/50 rounded-lg p-3 text-center">{{ t('annotate.noLabelsYet') }}</div>
                  <div v-else class="space-y-1 max-h-32 overflow-y-auto">
                    <label v-for="cat in categories" :key="cat.id" class="flex items-center gap-2 px-2 py-1 rounded-lg hover:bg-gray-700/30 cursor-pointer">
                      <input type="checkbox" :value="cat.id" v-model="autoLabelSelectedCats" :disabled="autoLabelRunning" class="w-3.5 h-3.5 rounded border-gray-600 bg-gray-700 text-purple-500 focus:ring-purple-500" />
                      <div class="w-2 h-2 rounded-sm flex-shrink-0" :style="{ backgroundColor: getCategoryColor(cat.id) }" />
                      <span class="text-xs text-gray-200 truncate">{{ cat.name }}</span>
                    </label>
                    <div class="flex items-center gap-2 pt-0.5">
                      <button @click="autoLabelSelectedCats = categories.map(c => c.id)" :disabled="autoLabelRunning" class="text-xs text-purple-400 hover:text-purple-300">{{ t('annotate.selectAll') }}</button>
                      <span class="text-gray-600">·</span>
                      <button @click="autoLabelSelectedCats = []" :disabled="autoLabelRunning" class="text-xs text-gray-500 hover:text-gray-400">{{ t('annotate.clear') }}</button>
                      <span class="ml-auto text-xs text-gray-600">{{ autoLabelSelectedCats.length }}/{{ categories.length }}</span>
                    </div>
                  </div>
                </template>

                <template v-if="autoLabelClassSource === 'custom'">
                  <input v-model="autoLabelClasses" type="text" placeholder="fish, coral, debris..." :disabled="autoLabelRunning"
                    class="w-full text-xs bg-background-tertiary border border-gray-600 rounded px-2 py-2 text-white placeholder-gray-500 focus:outline-none focus:border-primary" />
                  <p class="text-[10px] text-gray-600 mt-1">Comma-separated. Created automatically.</p>
                </template>
              </div>

              <div>
                <label class="text-xs text-gray-400 mb-1 block">{{ t('annotate.minConfidence') }}: {{ autoLabelConfidence.toFixed(2) }}</label>
                <input v-model.number="autoLabelConfidence" type="range" min="0.1" max="0.9" step="0.05" class="w-full accent-purple-500" :disabled="autoLabelRunning" />
              </div>

              <button @click="startAutoLabeling"
                :disabled="autoLabelRunning || sam3Status !== 'available' || (autoLabelClassSource === 'custom' && !autoLabelClasses.trim()) || (autoLabelClassSource === 'existing' && categories.length === 0)"
                class="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg text-xs font-medium bg-purple-600 hover:bg-purple-500 text-white disabled:opacity-40 transition-colors">
                <Wand2 class="h-3.5 w-3.5" />
                {{ autoLabelRunning ? 'Running...' : sam3Status !== 'available' ? 'SAM3 unavailable' : 'Label all images' }}
              </button>

              <div v-if="autoLabelRunning || autoLabelStatus" class="space-y-1">
                <div v-if="autoLabelRunning" class="w-full bg-gray-700 rounded-full h-1.5"><div class="bg-purple-500 h-1.5 rounded-full transition-all" :style="{width: autoLabelProgress+'%'}" /></div>
                <p class="text-xs text-gray-400">{{ autoLabelStatus }}</p>
              </div>
            </div>
          </div>
        </div>
      </aside>
    </div>

    <!-- ====== EXPORT MODAL ====== -->
    <ExportDialog
      :open="showExport"
      :coco-json-path="cocoJsonPath"
      :stats="{ images: totalImages, annotations: editor.annotations.value.length, categories: categories.length }"
      @close="showExport = false"
    />

    <!-- ====== PROPAGATE MODAL (INSID3) ====== -->
    <PropagateDialog
      :open="showPropagate"
      :reference-image-path="getActiveImagePath()"
      :reference-polygon="propagateDefaults.polygon"
      :default-label="propagateDefaults.label"
      @close="showPropagate = false"
    />
  </div>
</template>
