<script setup lang="ts">
import { ref, computed, watch, nextTick, onMounted, onUnmounted } from 'vue'
import { useRoute } from 'vue-router'
import { useUiStore } from '@/stores/ui'
import { loadAnnotations, saveAnnotations, getImageUrl, scanImages, startLabeling, getLabelingJobStatus, uploadImages } from '@/lib/api'
import { useAnnotationEditor, type ToolMode } from '@/composables/useAnnotationEditor'
import BaseButton from '@/components/ui/BaseButton.vue'
import AlertBox from '@/components/common/AlertBox.vue'
import LoadingSpinner from '@/components/common/LoadingSpinner.vue'
import DirectoryBrowser from '@/components/common/DirectoryBrowser.vue'
import ExportDialog from '@/components/export/ExportDialog.vue'
import {
  FolderOpen, Save, ZoomIn, ZoomOut, Maximize2,
  Eye, EyeOff, Tag, Trash2, Search, ImageIcon,
  AlertCircle, CheckCircle2, MousePointer2,
  Square, Hand, Undo2, Redo2, ChevronLeft, ChevronRight,
  Wand2, Plus, X, Loader2, FolderOpen as FolderIcon, Layers, Download, Upload, HardDrive,
} from 'lucide-vue-next'

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const CATEGORY_COLORS = [
  '#ef4444', '#f97316', '#eab308', '#22c55e', '#06b6d4',
  '#3b82f6', '#8b5cf6', '#ec4899', '#f43f5e', '#14b8a6',
]
const ZOOM_STEP = 0.15
const ZOOM_MIN = 0.1
const ZOOM_MAX = 10.0
const HANDLE_SIZE = 8

// ---------------------------------------------------------------------------
// Stores & editor
// ---------------------------------------------------------------------------
const route = useRoute()
const uiStore = useUiStore()
const editor = useAnnotationEditor()

// ---------------------------------------------------------------------------
// Dataset loader
// ---------------------------------------------------------------------------
const cocoJsonPath = ref('')
const imagesDirPath = ref('')
const isLoadingDataset = ref(false)
const loadError = ref<string | null>(null)
const loadMode = ref<'coco' | 'images'>('coco') // coco = load existing, images = scan directory
const imagesSource = ref<'path' | 'upload'>('path')
const imageFiles = ref<File[]>([])
const imageUploadProgress = ref(0)
const isUploadingImages = ref(false)

// ---------------------------------------------------------------------------
// Category management
// ---------------------------------------------------------------------------
const newCategoryName = ref('')
const showCategoryEditor = ref(false)

// ---------------------------------------------------------------------------
// Auto-labeling
// ---------------------------------------------------------------------------
const showAutoLabel = ref(false)
const autoLabelClasses = ref('')
const autoLabelConfidence = ref(0.5)
const autoLabelRunning = ref(false)
const autoLabelProgress = ref(0)
const autoLabelStatus = ref('')
const autoLabelJobId = ref<string | null>(null)
let autoLabelPollTimer: ReturnType<typeof setInterval> | null = null

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------
const showExport = ref(false)

async function openExport() {
  if (editor.hasUnsavedChanges.value) {
    await handleSave()
  }
  if (cocoJsonPath.value) showExport.value = true
}

// ---------------------------------------------------------------------------
// Navigator
// ---------------------------------------------------------------------------
const searchQuery = ref('')
const activeImageId = ref<number | null>(null)
const selectedCategoryFilter = ref<number | null>(null)

// ---------------------------------------------------------------------------
// Canvas
// ---------------------------------------------------------------------------
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
const isSaving = ref(false)
const drawCategoryId = ref<number>(1)

// Drawing state
const isDrawing = ref(false)
const drawStart = ref({ x: 0, y: 0 })
const drawCurrent = ref({ x: 0, y: 0 })

// Drag/resize state
const isDragging = ref(false)
const isResizing = ref(false)
const resizeHandle = ref('')
const dragStart = ref({ x: 0, y: 0 })
const dragOrigBbox = ref<[number, number, number, number]>([0, 0, 0, 0])

// Pan state
const isPanning = ref(false)
const panStart = ref({ x: 0, y: 0 })
const panStartOffset = ref({ x: 0, y: 0 })

// Right panel tab
const rightTab = ref<'objects' | 'categories' | 'auto'>('objects')

// ---------------------------------------------------------------------------
// Computed
// ---------------------------------------------------------------------------
const images = computed(() => editor.dataset.value?.images ?? [])
const categories = computed(() => editor.dataset.value?.categories ?? [])
const totalImages = computed(() => images.value.length)

const filteredImages = computed(() => {
  const q = searchQuery.value.trim().toLowerCase()
  return images.value.filter((img) => !q || img.file_name.toLowerCase().includes(q))
})

const activeImage = computed(() => images.value.find((img) => img.id === activeImageId.value) ?? null)

const activeImageUrl = computed(() => {
  if (!activeImage.value || !imagesDirPath.value) return ''
  const sep = imagesDirPath.value.endsWith('/') ? '' : '/'
  return getImageUrl(imagesDirPath.value + sep + activeImage.value.file_name)
})

const activeAnnotations = computed(() => {
  if (activeImageId.value === null) return []
  let anns = editor.getAnnotationsForImage(activeImageId.value)
  if (selectedCategoryFilter.value !== null) {
    anns = anns.filter((a) => a.category_id === selectedCategoryFilter.value)
  }
  return anns
})

const annotationCountByImage = computed(() => {
  const counts = new Map<number, number>()
  for (const ann of editor.annotations.value) {
    counts.set(ann.image_id, (counts.get(ann.image_id) ?? 0) + 1)
  }
  return counts
})

const categoryMap = computed(() => {
  const map = new Map<number, string>()
  categories.value.forEach((c) => map.set(c.id, c.name))
  return map
})

function getCategoryColor(categoryId: number): string {
  return CATEGORY_COLORS[(categoryId - 1) % CATEGORY_COLORS.length]
}
function getCategoryName(categoryId: number): string {
  return categoryMap.value.get(categoryId) ?? `cat_${categoryId}`
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
    .map((ann) => ({
      id: ann.id,
      color: getCategoryColor(ann.category_id),
      selected: ann.id === editor.selectedId.value,
      polygons: ann.segmentation!.map((poly) => cocoPolyToSvgPoints(poly, scaleX.value, scaleY.value)),
    }))
})

const drawPreview = computed(() => {
  if (!isDrawing.value) return null
  const x = Math.min(drawStart.value.x, drawCurrent.value.x)
  const y = Math.min(drawStart.value.y, drawCurrent.value.y)
  const w = Math.abs(drawCurrent.value.x - drawStart.value.x)
  const h = Math.abs(drawCurrent.value.y - drawStart.value.y)
  return { x: x * scaleX.value, y: y * scaleY.value, w: w * scaleX.value, h: h * scaleY.value }
})

const currentImageIndex = computed(() => filteredImages.value.findIndex((img) => img.id === activeImageId.value))

// ---------------------------------------------------------------------------
// Dataset loading
// ---------------------------------------------------------------------------
const IMAGE_RE = /\.(jpg|jpeg|png|bmp|webp|tiff|tif)$/i

function onImageFilesSelected(e: Event) {
  const input = e.target as HTMLInputElement
  if (input.files) {
    imageFiles.value = [...imageFiles.value, ...Array.from(input.files).filter(f => IMAGE_RE.test(f.name))]
  }
  input.value = ''
}

async function handleUploadImages(): Promise<boolean> {
  if (imageFiles.value.length === 0) {
    loadError.value = 'Select image files to upload.'
    return false
  }
  isUploadingImages.value = true
  try {
    const taskName = cocoJsonPath.value
      ? cocoJsonPath.value.split('/').pop()?.replace('.json', '') || 'review'
      : 'review_' + Date.now()
    const result = await uploadImages(taskName, imageFiles.value, (uploaded, total) => {
      imageUploadProgress.value = Math.round((uploaded / total) * 100)
    })
    imagesDirPath.value = result.directory
    uiStore.showSuccess('Images uploaded', `${result.uploaded_count} images uploaded`)
    return true
  } catch (e: any) {
    loadError.value = e.message || 'Image upload failed'
    return false
  } finally {
    isUploadingImages.value = false
  }
}

async function handleLoadDataset() {
  if (imagesSource.value === 'upload' && !imagesDirPath.value.trim()) {
    const ok = await handleUploadImages()
    if (!ok) return
  }
  if (!imagesDirPath.value.trim()) {
    loadError.value = 'Please provide the images directory path.'
    return
  }
  isLoadingDataset.value = true
  loadError.value = null

  try {
    if (loadMode.value === 'coco' && cocoJsonPath.value.trim()) {
      // Load existing COCO JSON
      const data = await loadAnnotations(cocoJsonPath.value.trim())
      editor.loadDataset(data)
      uiStore.showSuccess('Dataset loaded', `${data.images.length} images, ${data.annotations.length} annotations`)
    } else {
      // Scan directory for images, create empty COCO structure
      const result = await scanImages(imagesDirPath.value.trim())
      editor.loadDataset({
        images: result.images,
        annotations: [],
        categories: [],
      })
      loadMode.value = 'images'
      uiStore.showSuccess('Images loaded', `${result.images.length} images found. Add categories and start annotating.`)
    }

    activeImageId.value = images.value[0]?.id ?? null
    if (categories.value.length > 0) drawCategoryId.value = categories.value[0].id
  } catch (err: any) {
    loadError.value = err?.response?.data?.detail ?? err?.message ?? 'Failed to load.'
  } finally {
    isLoadingDataset.value = false
  }
}

async function handleSave() {
  const data = editor.getDatasetForSave()
  if (!data) return

  // If we loaded from images only and have no cocoJsonPath, generate one
  if (!cocoJsonPath.value.trim()) {
    cocoJsonPath.value = imagesDirPath.value.replace(/\/$/, '') + '/annotations.json'
  }

  isSaving.value = true
  try {
    await saveAnnotations({ coco_json_path: cocoJsonPath.value.trim(), data })
    editor.markSaved()
    uiStore.showSuccess('Saved', `Saved to ${cocoJsonPath.value}`)
  } catch (err: any) {
    uiStore.showError('Save failed', err?.response?.data?.detail ?? err?.message ?? 'Unknown error')
  } finally {
    isSaving.value = false
  }
}

// ---------------------------------------------------------------------------
// Category management
// ---------------------------------------------------------------------------
function addCategory() {
  const name = newCategoryName.value.trim()
  if (!name || !editor.dataset.value) return
  if (categories.value.some((c) => c.name.toLowerCase() === name.toLowerCase())) {
    uiStore.showError('Duplicate', `Category "${name}" already exists`)
    return
  }
  const maxId = categories.value.reduce((max, c) => Math.max(max, c.id), 0)
  editor.dataset.value = {
    ...editor.dataset.value,
    categories: [...categories.value, { id: maxId + 1, name }],
  }
  if (categories.value.length === 1) drawCategoryId.value = categories.value[0].id
  newCategoryName.value = ''
  editor.hasUnsavedChanges.value = true
}

function removeCategory(catId: number) {
  if (!editor.dataset.value) return
  // Remove category and its annotations
  editor.annotations.value = editor.annotations.value.filter((a) => a.category_id !== catId)
  editor.dataset.value = {
    ...editor.dataset.value,
    categories: categories.value.filter((c) => c.id !== catId),
  }
  editor.hasUnsavedChanges.value = true
}

// ---------------------------------------------------------------------------
// Auto-labeling with SAM3
// ---------------------------------------------------------------------------
async function startAutoLabeling() {
  if (!autoLabelClasses.value.trim() || !imagesDirPath.value) return

  const classes = autoLabelClasses.value.split(',').map((c) => c.trim()).filter(Boolean)
  if (classes.length === 0) return

  autoLabelRunning.value = true
  autoLabelProgress.value = 0
  autoLabelStatus.value = 'Starting auto-labeling...'

  try {
    // Determine output dir for labeling results
    const outputDir = imagesDirPath.value.replace(/\/$/, '') + '/auto_labeled'

    const response = await startLabeling({
      image_directories: [imagesDirPath.value],
      classes,
      output_dir: outputDir,
      output_formats: ['coco'],
      task_type: 'detection',
      min_confidence: autoLabelConfidence.value,
      save_visualizations: false,
    })

    autoLabelJobId.value = response.job_id
    autoLabelStatus.value = 'Processing images...'

    // Poll for status — stored at component level so onUnmounted can cancel it
    autoLabelPollTimer = setInterval(async () => {
      try {
        const job = await getLabelingJobStatus(response.job_id)
        autoLabelProgress.value = job.progress ?? 0
        autoLabelStatus.value = `Processing: ${job.processed_images ?? 0}/${job.total_images ?? '?'} images`

        if (job.status === 'completed') {
          clearInterval(autoLabelPollTimer!)
          autoLabelPollTimer = null
          autoLabelRunning.value = false
          autoLabelStatus.value = `Completed! ${job.annotations_created ?? 0} annotations created.`
          uiStore.showSuccess('Auto-labeling complete', `${job.annotations_created ?? 0} annotations created`)

          // Reload the dataset with the generated COCO file
          const cocoPath = outputDir + '/annotations.json'
          try {
            const data = await loadAnnotations(cocoPath)
            editor.loadDataset(data)
            cocoJsonPath.value = cocoPath
            activeImageId.value = images.value[0]?.id ?? null
            if (categories.value.length > 0) drawCategoryId.value = categories.value[0].id
          } catch {
            uiStore.showInfo('Load manually', `Auto-labeled file at: ${cocoPath}`)
          }
        } else if (job.status === 'failed') {
          clearInterval(autoLabelPollTimer!)
          autoLabelPollTimer = null
          autoLabelRunning.value = false
          autoLabelStatus.value = `Failed: ${job.error ?? 'Unknown error'}`
          uiStore.showError('Auto-labeling failed', job.error ?? 'Unknown error')
        }
      } catch {
        // Ignore poll errors, retry
      }
    }, 2000)
  } catch (err: any) {
    autoLabelRunning.value = false
    autoLabelStatus.value = ''
    uiStore.showError('Failed to start', err?.message ?? 'Unknown error')
  }
}

// ---------------------------------------------------------------------------
// Navigation
// ---------------------------------------------------------------------------
function selectImage(imageId: number) {
  activeImageId.value = imageId
  zoom.value = 1.0
  panX.value = 0
  panY.value = 0
  editor.select(null)
}

function navigateImage(dir: 'prev' | 'next') {
  const idx = currentImageIndex.value
  if (dir === 'next' && idx < filteredImages.value.length - 1) selectImage(filteredImages.value[idx + 1].id)
  else if (dir === 'prev' && idx > 0) selectImage(filteredImages.value[idx - 1].id)
}

// ---------------------------------------------------------------------------
// Zoom & pan
// ---------------------------------------------------------------------------
function zoomIn() { zoom.value = Math.min(ZOOM_MAX, +(zoom.value + ZOOM_STEP).toFixed(2)) }
function zoomOut() { zoom.value = Math.max(ZOOM_MIN, +(zoom.value - ZOOM_STEP).toFixed(2)) }
function fitToView() { zoom.value = 1.0; panX.value = 0; panY.value = 0 }

function onWheel(e: WheelEvent) {
  e.preventDefault()
  const delta = e.deltaY > 0 ? -ZOOM_STEP : ZOOM_STEP
  zoom.value = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, +(zoom.value + delta).toFixed(2)))
}

// ---------------------------------------------------------------------------
// Image load
// ---------------------------------------------------------------------------
function onImageLoad() {
  if (!imageEl.value) return
  imgNatW.value = imageEl.value.naturalWidth
  imgNatH.value = imageEl.value.naturalHeight
  updateRenderedSize()
}
function updateRenderedSize() {
  if (!imageEl.value) return
  imgW.value = imageEl.value.offsetWidth
  imgH.value = imageEl.value.offsetHeight
}
watch(zoom, () => nextTick(updateRenderedSize))

// ---------------------------------------------------------------------------
// Canvas mouse helpers
// ---------------------------------------------------------------------------
function getImageCoords(e: MouseEvent): { x: number; y: number } | null {
  if (!imageEl.value) return null
  const rect = imageEl.value.getBoundingClientRect()
  const px = (e.clientX - rect.left) / scaleX.value / zoom.value
  const py = (e.clientY - rect.top) / scaleY.value / zoom.value
  return { x: Math.max(0, Math.min(imgNatW.value, px)), y: Math.max(0, Math.min(imgNatH.value, py)) }
}

function hitTest(x: number, y: number): number | null {
  for (let i = activeAnnotations.value.length - 1; i >= 0; i--) {
    const a = activeAnnotations.value[i]
    const [bx, by, bw, bh] = a.bbox
    if (x >= bx && x <= bx + bw && y >= by && y <= by + bh) return a.id
  }
  return null
}

function hitTestHandle(x: number, y: number): string | null {
  const ann = editor.selectedAnnotation.value
  if (!ann) return null
  const [bx, by, bw, bh] = ann.bbox
  const hs = HANDLE_SIZE / scaleX.value / zoom.value
  const handles: Record<string, [number, number]> = {
    'nw': [bx, by], 'ne': [bx + bw, by], 'sw': [bx, by + bh], 'se': [bx + bw, by + bh],
    'n': [bx + bw / 2, by], 's': [bx + bw / 2, by + bh],
    'w': [bx, by + bh / 2], 'e': [bx + bw, by + bh / 2],
  }
  for (const [handle, [hx, hy]] of Object.entries(handles)) {
    if (Math.abs(x - hx) < hs && Math.abs(y - hy) < hs) return handle
  }
  return null
}

// ---------------------------------------------------------------------------
// Canvas mouse events
// ---------------------------------------------------------------------------
function onCanvasMouseDown(e: MouseEvent) {
  if (e.button === 1 || (e.button === 0 && editor.toolMode.value === 'pan')) {
    isPanning.value = true
    panStart.value = { x: e.clientX, y: e.clientY }
    panStartOffset.value = { x: panX.value, y: panY.value }
    e.preventDefault()
    return
  }
  if (e.button !== 0) return
  const coords = getImageCoords(e)
  if (!coords) return

  if (editor.toolMode.value === 'draw-bbox') {
    if (categories.value.length === 0) {
      uiStore.showError('No categories', 'Add at least one category before drawing')
      return
    }
    isDrawing.value = true
    drawStart.value = { ...coords }
    drawCurrent.value = { ...coords }
    return
  }

  const handle = hitTestHandle(coords.x, coords.y)
  if (handle && editor.selectedAnnotation.value) {
    isResizing.value = true
    resizeHandle.value = handle
    dragStart.value = { ...coords }
    dragOrigBbox.value = [...editor.selectedAnnotation.value.bbox] as [number, number, number, number]
    return
  }

  const hitId = hitTest(coords.x, coords.y)
  if (hitId !== null) {
    editor.select(hitId)
    isDragging.value = true
    dragStart.value = { ...coords }
    const ann = editor.annotations.value.find((a) => a.id === hitId)!
    dragOrigBbox.value = [...ann.bbox] as [number, number, number, number]
  } else {
    editor.select(null)
  }
}

function onCanvasMouseMove(e: MouseEvent) {
  if (isPanning.value) {
    panX.value = panStartOffset.value.x + (e.clientX - panStart.value.x)
    panY.value = panStartOffset.value.y + (e.clientY - panStart.value.y)
    return
  }
  const coords = getImageCoords(e)
  if (!coords) return

  if (isDrawing.value) { drawCurrent.value = { ...coords }; return }

  if (isDragging.value && editor.selectedId.value !== null) {
    const dx = coords.x - dragStart.value.x
    const dy = coords.y - dragStart.value.y
    const ann = editor.annotations.value.find((a) => a.id === editor.selectedId.value)
    if (ann) {
      ann.bbox = [dragOrigBbox.value[0] + dx, dragOrigBbox.value[1] + dy, dragOrigBbox.value[2], dragOrigBbox.value[3]]
      editor.annotations.value = [...editor.annotations.value]
    }
    return
  }

  if (isResizing.value && editor.selectedId.value !== null) {
    const [ox, oy, ow, oh] = dragOrigBbox.value
    const dx = coords.x - dragStart.value.x
    const dy = coords.y - dragStart.value.y
    let nx = ox, ny = oy, nw = ow, nh = oh
    const h = resizeHandle.value
    if (h.includes('w')) { nx = ox + dx; nw = ow - dx }
    if (h.includes('e')) { nw = ow + dx }
    if (h.includes('n')) { ny = oy + dy; nh = oh - dy }
    if (h.includes('s')) { nh = oh + dy }
    if (nw < 5) nw = 5
    if (nh < 5) nh = 5
    const ann = editor.annotations.value.find((a) => a.id === editor.selectedId.value)
    if (ann) { ann.bbox = [nx, ny, nw, nh]; ann.area = nw * nh; editor.annotations.value = [...editor.annotations.value] }
    return
  }
}

function onCanvasMouseUp() {
  if (isPanning.value) { isPanning.value = false; return }

  if (isDrawing.value) {
    isDrawing.value = false
    const x = Math.min(drawStart.value.x, drawCurrent.value.x)
    const y = Math.min(drawStart.value.y, drawCurrent.value.y)
    const w = Math.abs(drawCurrent.value.x - drawStart.value.x)
    const h = Math.abs(drawCurrent.value.y - drawStart.value.y)
    if (w > 3 && h > 3 && activeImageId.value !== null) {
      editor.createAnnotation(activeImageId.value, drawCategoryId.value, { x, y, w, h })
    }
    return
  }

  if (isDragging.value && editor.selectedId.value !== null) {
    const ann = editor.annotations.value.find((a) => a.id === editor.selectedId.value)
    if (ann) {
      const dx = ann.bbox[0] - dragOrigBbox.value[0]
      const dy = ann.bbox[1] - dragOrigBbox.value[1]
      ann.bbox = [...dragOrigBbox.value] as [number, number, number, number]
      if (Math.abs(dx) > 0.5 || Math.abs(dy) > 0.5) editor.moveAnnotation(editor.selectedId.value, dx, dy)
    }
    isDragging.value = false
    return
  }

  if (isResizing.value && editor.selectedId.value !== null) {
    const ann = editor.annotations.value.find((a) => a.id === editor.selectedId.value)
    if (ann) {
      const newBbox = { x: ann.bbox[0], y: ann.bbox[1], w: ann.bbox[2], h: ann.bbox[3] }
      ann.bbox = [...dragOrigBbox.value] as [number, number, number, number]
      editor.resizeAnnotation(editor.selectedId.value, newBbox)
    }
    isResizing.value = false
    return
  }
}

// ---------------------------------------------------------------------------
// Keyboard shortcuts
// ---------------------------------------------------------------------------
function handleKeyDown(e: KeyboardEvent) {
  const t = e.target as HTMLElement
  if (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT') return

  switch (e.key) {
    case 'ArrowLeft': case 'a': e.preventDefault(); navigateImage('prev'); break
    case 'ArrowRight': case 'd': e.preventDefault(); navigateImage('next'); break
    case '+': case '=': e.preventDefault(); zoomIn(); break
    case '-': e.preventDefault(); zoomOut(); break
    case 'b': showBboxes.value = !showBboxes.value; break
    case 'l': showLabels.value = !showLabels.value; break
    case 'm': showMasks.value = !showMasks.value; break
    case 'v': editor.toolMode.value = 'select'; break
    case 'n': editor.toolMode.value = 'draw-bbox'; break
    case 'Delete': case 'Backspace':
      if (editor.selectedId.value !== null) { e.preventDefault(); editor.deleteAnnotation(editor.selectedId.value) }
      break
    case 'Escape': editor.select(null); isDrawing.value = false; editor.toolMode.value = 'select'; break
    case 'z': if (e.ctrlKey || e.metaKey) { e.preventDefault(); if (e.shiftKey) editor.redo(); else editor.undo() } break
    case 'y': if (e.ctrlKey || e.metaKey) { e.preventDefault(); editor.redo() } break
    case 's': if (e.ctrlKey || e.metaKey) { e.preventDefault(); if (editor.hasUnsavedChanges.value) handleSave() } break
  }
}

onMounted(async () => {
  window.addEventListener('keydown', handleKeyDown)

  const qCocoPath = route.query.coco_json_path as string | undefined
  const qImagesDir = route.query.images_dir as string | undefined
  if (qCocoPath) {
    cocoJsonPath.value = qCocoPath
    if (qImagesDir) imagesDirPath.value = qImagesDir
    loadMode.value = 'coco'
    await handleLoadDataset()
  }
})
onUnmounted(() => {
  window.removeEventListener('keydown', handleKeyDown)
  if (autoLabelPollTimer) {
    clearInterval(autoLabelPollTimer)
    autoLabelPollTimer = null
  }
})

function formatBbox(bbox: [number, number, number, number]): string {
  return `[${bbox.map((v) => Math.round(v)).join(', ')}]`
}
</script>

<template>
  <div class="flex flex-col h-full min-h-0" :data-unsaved="editor.hasUnsavedChanges.value">

    <!-- ============================================================== -->
    <!-- DATASET LOADER -->
    <!-- ============================================================== -->
    <div v-if="!editor.dataset.value" class="p-6 space-y-4 max-w-3xl mx-auto w-full">
      <div>
        <h2 class="text-2xl font-bold text-white">Annotation Workspace</h2>
        <p class="mt-1 text-sm text-gray-400">Load an existing dataset or start from a directory of images.</p>
      </div>

      <!-- Mode tabs -->
      <div class="flex gap-1 bg-background-tertiary rounded-lg p-1 w-fit">
        <button @click="loadMode = 'coco'" :class="['px-4 py-2 rounded-md text-sm font-medium transition-colors', loadMode === 'coco' ? 'bg-primary text-white' : 'text-gray-400 hover:text-white']">
          Load COCO JSON
        </button>
        <button @click="loadMode = 'images'" :class="['px-4 py-2 rounded-md text-sm font-medium transition-colors', loadMode === 'images' ? 'bg-primary text-white' : 'text-gray-400 hover:text-white']">
          Scan Images Directory
        </button>
      </div>

      <div class="card p-5 space-y-4">
        <DirectoryBrowser v-if="loadMode === 'coco'" v-model="cocoJsonPath" label="COCO JSON file" placeholder="/app/datasets/annotations.json" :show-files="true" file-pattern="*.json" path-mode="input" :restrict-to-mounts="false" />

        <!-- Images source toggle -->
        <div class="space-y-2">
          <div class="flex items-center justify-between">
            <label class="text-sm font-medium text-gray-300">Images source</label>
            <div class="flex gap-0.5 bg-background-tertiary rounded-md p-0.5 text-xs">
              <button :class="['px-2 py-1 rounded transition-colors flex items-center gap-1', imagesSource === 'path' ? 'bg-primary text-white' : 'text-gray-400 hover:text-white']" @click="imagesSource = 'path'">
                <HardDrive class="h-3 w-3" /> Server Path
              </button>
              <button :class="['px-2 py-1 rounded transition-colors flex items-center gap-1', imagesSource === 'upload' ? 'bg-primary text-white' : 'text-gray-400 hover:text-white']" @click="imagesSource = 'upload'">
                <Upload class="h-3 w-3" /> Upload from PC
              </button>
            </div>
          </div>

          <DirectoryBrowser v-if="imagesSource === 'path'" v-model="imagesDirPath" label="Images directory" placeholder="/app/datasets/images" :show-files="false" path-mode="input" :restrict-to-mounts="false" />

          <div v-else class="space-y-3">
            <div class="border-2 border-dashed border-gray-600 rounded-lg p-4 text-center hover:border-gray-500 transition-colors">
              <input type="file" multiple accept="image/*" class="hidden" id="review-image-upload" @change="onImageFilesSelected" />
              <label for="review-image-upload" class="cursor-pointer">
                <Upload class="h-8 w-8 text-gray-500 mx-auto mb-2" />
                <p class="text-sm text-gray-400">Click to select images or a folder</p>
                <p class="text-xs text-gray-500 mt-1">JPG, PNG, BMP, WebP, TIFF</p>
              </label>
            </div>
            <div v-if="imageFiles.length > 0" class="flex items-center gap-2 text-sm text-gray-300">
              <ImageIcon class="h-4 w-4" />
              <span>{{ imageFiles.length }} images selected</span>
              <button class="text-red-400 hover:text-red-300 text-xs ml-auto" @click="imageFiles = []">Clear</button>
            </div>
            <div v-if="isUploadingImages" class="space-y-1">
              <div class="h-1.5 bg-background-tertiary rounded-full overflow-hidden">
                <div class="h-full bg-primary transition-all duration-300" :style="{ width: imageUploadProgress + '%' }" />
              </div>
              <p class="text-xs text-gray-500">Uploading {{ imageUploadProgress }}%</p>
            </div>
          </div>
        </div>

        <div class="flex items-center gap-3">
          <BaseButton variant="primary" :loading="isLoadingDataset || isUploadingImages" @click="handleLoadDataset">
            <FolderOpen class="h-4 w-4" />
            {{ loadMode === 'coco' ? 'Load Dataset' : 'Scan & Start' }}
          </BaseButton>
          <span v-if="loadMode === 'images'" class="text-xs text-gray-500">Images will be scanned automatically. Add categories after loading.</span>
        </div>

        <AlertBox v-if="loadError" type="error" title="Error">{{ loadError }}</AlertBox>
      </div>

      <div v-if="isLoadingDataset" class="flex justify-center py-8"><LoadingSpinner size="lg" message="Loading..." /></div>
    </div>

    <!-- ============================================================== -->
    <!-- MAIN WORKSPACE -->
    <!-- ============================================================== -->
    <template v-else>

      <!-- TOP TOOLBAR -->
      <div class="flex-shrink-0 px-2 py-1.5 flex items-center gap-1 bg-background-secondary border-b border-gray-700/50 flex-wrap">
        <!-- Tool modes -->
        <div class="flex items-center bg-background-tertiary rounded-lg p-0.5 mr-1">
          <button @click="editor.toolMode.value = 'select'" :class="['p-1.5 rounded-md', editor.toolMode.value === 'select' ? 'bg-primary text-white' : 'text-gray-400 hover:text-white']" title="Select (V)"><MousePointer2 class="h-4 w-4" /></button>
          <button @click="editor.toolMode.value = 'draw-bbox'" :class="['p-1.5 rounded-md', editor.toolMode.value === 'draw-bbox' ? 'bg-primary text-white' : 'text-gray-400 hover:text-white']" title="Draw bbox (N)"><Square class="h-4 w-4" /></button>
          <button @click="editor.toolMode.value = 'pan'" :class="['p-1.5 rounded-md', editor.toolMode.value === 'pan' ? 'bg-primary text-white' : 'text-gray-400 hover:text-white']" title="Pan"><Hand class="h-4 w-4" /></button>
        </div>

        <!-- Draw category selector -->
        <div v-if="editor.toolMode.value === 'draw-bbox' && categories.length > 0" class="flex items-center gap-1 mr-1">
          <div class="w-3 h-3 rounded-sm" :style="{ backgroundColor: getCategoryColor(drawCategoryId) }" />
          <select v-model="drawCategoryId" class="text-xs bg-background-tertiary border border-gray-600 rounded px-1.5 py-1 text-gray-300">
            <option v-for="cat in categories" :key="cat.id" :value="cat.id">{{ cat.name }}</option>
          </select>
        </div>

        <div class="w-px h-5 bg-gray-700 mx-0.5" />

        <!-- Undo/Redo -->
        <button @click="editor.undo()" :disabled="!editor.canUndo.value" class="p-1.5 text-gray-400 hover:text-white disabled:opacity-30" title="Undo (Ctrl+Z)"><Undo2 class="h-4 w-4" /></button>
        <button @click="editor.redo()" :disabled="!editor.canRedo.value" class="p-1.5 text-gray-400 hover:text-white disabled:opacity-30" title="Redo (Ctrl+Y)"><Redo2 class="h-4 w-4" /></button>

        <div class="w-px h-5 bg-gray-700 mx-0.5" />

        <!-- Save -->
        <BaseButton variant="primary" size="sm" :loading="isSaving" :disabled="isSaving || !editor.hasUnsavedChanges.value" @click="handleSave">
          <Save class="h-3.5 w-3.5" /> Save
        </BaseButton>

        <!-- Export -->
        <BaseButton variant="outline" size="sm" :disabled="!cocoJsonPath" @click="openExport">
          <Download class="h-3.5 w-3.5" /> Export
        </BaseButton>

        <div class="w-px h-5 bg-gray-700 mx-0.5" />

        <!-- Zoom -->
        <button @click="zoomOut" :disabled="zoom <= ZOOM_MIN" class="p-1.5 text-gray-400 hover:text-white disabled:opacity-30"><ZoomOut class="h-4 w-4" /></button>
        <span class="text-xs text-gray-400 tabular-nums w-10 text-center select-none">{{ Math.round(zoom * 100) }}%</span>
        <button @click="zoomIn" :disabled="zoom >= ZOOM_MAX" class="p-1.5 text-gray-400 hover:text-white disabled:opacity-30"><ZoomIn class="h-4 w-4" /></button>
        <button @click="fitToView" class="p-1.5 text-gray-400 hover:text-white"><Maximize2 class="h-4 w-4" /></button>

        <div class="w-px h-5 bg-gray-700 mx-0.5" />

        <!-- Visibility -->
        <button @click="showBboxes = !showBboxes" :class="['p-1.5 rounded', showBboxes ? 'text-primary' : 'text-gray-500']"><component :is="showBboxes ? Eye : EyeOff" class="h-4 w-4" /></button>
        <button @click="showLabels = !showLabels" :class="['p-1.5 rounded', showLabels ? 'text-primary' : 'text-gray-500']"><Tag class="h-4 w-4" /></button>
        <button @click="showMasks = !showMasks" :class="['p-1.5 rounded', showMasks ? 'text-purple-400' : 'text-gray-500']" title="Toggle masks (M)"><Layers class="h-4 w-4" /></button>

        <!-- Right: status -->
        <div class="ml-auto flex items-center gap-3 text-xs">
          <span v-if="activeImage" class="text-gray-500">{{ activeImage.file_name }} &mdash; {{ activeImage.width }}&times;{{ activeImage.height }}px</span>
          <div v-if="editor.hasUnsavedChanges.value" class="flex items-center gap-1 text-yellow-400"><AlertCircle class="h-3.5 w-3.5" /> Unsaved</div>
          <div v-else class="flex items-center gap-1 text-green-500"><CheckCircle2 class="h-3.5 w-3.5" /> Saved</div>
        </div>
      </div>

      <!-- MAIN CONTENT -->
      <div class="flex flex-1 min-h-0">

        <!-- LEFT: Image list -->
        <div class="flex flex-col w-56 flex-shrink-0 border-r border-gray-700/50 bg-background-secondary">
          <div class="p-2 border-b border-gray-700/50">
            <div class="relative">
              <Search class="absolute left-2 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-gray-500" />
              <input v-model="searchQuery" type="text" placeholder="Search..." class="w-full pl-7 pr-2 py-1.5 text-xs bg-background-tertiary border border-gray-600 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-primary" />
            </div>
            <div class="text-xs text-gray-500 mt-1">{{ filteredImages.length }} / {{ totalImages }} images</div>
          </div>
          <div class="flex-1 overflow-y-auto">
            <div v-for="img in filteredImages" :key="img.id" @click="selectImage(img.id)"
              :class="['flex items-center gap-2 px-2 py-2 cursor-pointer transition-colors border-l-2 text-xs',
                img.id === activeImageId ? 'bg-primary/15 border-l-primary text-white' : 'border-l-transparent text-gray-400 hover:bg-gray-700/30']">
              <ImageIcon class="h-3.5 w-3.5 flex-shrink-0 opacity-50" />
              <span class="truncate flex-1">{{ img.file_name }}</span>
              <span class="text-gray-600 tabular-nums">{{ annotationCountByImage.get(img.id) ?? 0 }}</span>
            </div>
          </div>
        </div>

        <!-- CENTER: Canvas -->
        <div class="flex-1 flex flex-col min-w-0 min-h-0">
          <div ref="canvasContainer" class="flex-1 overflow-hidden bg-gray-950 relative"
            @mousedown="onCanvasMouseDown" @mousemove="onCanvasMouseMove" @mouseup="onCanvasMouseUp"
            @mouseleave="() => { isDragging = false; isResizing = false; isPanning = false; isDrawing = false }"
            @wheel="onWheel" @contextmenu.prevent
            :style="{ cursor: editor.toolMode.value === 'draw-bbox' ? 'crosshair' : editor.toolMode.value === 'pan' ? (isPanning ? 'grabbing' : 'grab') : 'default' }">
            <div v-if="!activeImage" class="flex items-center justify-center h-full text-gray-600">
              <div class="text-center"><ImageIcon class="h-16 w-16 mx-auto mb-3 opacity-20" /><p class="text-sm">Select an image</p></div>
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
                <g v-for="bbox in scaledBboxes" :key="bbox.id">
                  <rect :x="bbox.x" :y="bbox.y" :width="bbox.w" :height="bbox.h" :stroke="bbox.color" :stroke-width="bbox.selected ? 3 : 2" fill="none" :opacity="bbox.selected ? 1 : 0.8" />
                  <rect v-if="bbox.selected" :x="bbox.x" :y="bbox.y" :width="bbox.w" :height="bbox.h" :fill="bbox.color" opacity="0.1" />
                  <template v-if="showLabels">
                    <rect :x="bbox.x" :y="bbox.y - 16" :width="bbox.label.length * 7 + 8" :height="16" :fill="bbox.color" rx="2" opacity="0.9" />
                    <text :x="bbox.x + 4" :y="bbox.y - 4" fill="white" font-size="10" font-family="ui-monospace, monospace" font-weight="600">{{ bbox.label }}</text>
                  </template>
                  <template v-if="bbox.selected">
                    <rect v-for="(pos, handle) in { nw: [bbox.x-3,bbox.y-3], ne: [bbox.x+bbox.w-3,bbox.y-3], sw: [bbox.x-3,bbox.y+bbox.h-3], se: [bbox.x+bbox.w-3,bbox.y+bbox.h-3], n: [bbox.x+bbox.w/2-3,bbox.y-3], s: [bbox.x+bbox.w/2-3,bbox.y+bbox.h-3], w: [bbox.x-3,bbox.y+bbox.h/2-3], e: [bbox.x+bbox.w-3,bbox.y+bbox.h/2-3] }"
                      :key="handle" :x="pos[0]" :y="pos[1]" width="6" height="6" fill="white" :stroke="bbox.color" stroke-width="1.5" rx="1" />
                  </template>
                </g>
                <rect v-if="drawPreview" :x="drawPreview.x" :y="drawPreview.y" :width="drawPreview.w" :height="drawPreview.h" :stroke="getCategoryColor(drawCategoryId)" stroke-width="2" fill="none" stroke-dasharray="6 3" />
              </svg>
            </div>
          </div>

          <!-- Filmstrip -->
          <div class="flex-shrink-0 h-14 bg-background-secondary border-t border-gray-700/50 flex items-center px-2 gap-1">
            <button @click="navigateImage('prev')" :disabled="currentImageIndex <= 0" class="p-1 text-gray-400 hover:text-white disabled:opacity-30"><ChevronLeft class="h-4 w-4" /></button>
            <div class="flex-1 flex items-center gap-1 overflow-x-auto py-1">
              <button v-for="img in filteredImages.slice(Math.max(0, currentImageIndex - 10), currentImageIndex + 11)" :key="img.id" @click="selectImage(img.id)"
                :class="['flex-shrink-0 h-10 w-14 rounded border-2 overflow-hidden transition-all', img.id === activeImageId ? 'border-primary ring-1 ring-primary/50' : 'border-gray-700 hover:border-gray-500 opacity-60 hover:opacity-100']">
                <img :src="getImageUrl(imagesDirPath + '/' + img.file_name)" class="h-full w-full object-cover" loading="lazy" />
              </button>
            </div>
            <button @click="navigateImage('next')" :disabled="currentImageIndex >= filteredImages.length - 1" class="p-1 text-gray-400 hover:text-white disabled:opacity-30"><ChevronRight class="h-4 w-4" /></button>
            <span class="text-xs text-gray-500 ml-2 tabular-nums">{{ currentImageIndex + 1 }}/{{ filteredImages.length }}</span>
          </div>
        </div>

        <!-- RIGHT PANEL: Tabs -->
        <div class="flex flex-col w-64 flex-shrink-0 border-l border-gray-700/50 bg-background-secondary">
          <!-- Tab headers -->
          <div class="flex border-b border-gray-700/50">
            <button @click="rightTab = 'objects'" :class="['flex-1 py-2 text-xs font-medium transition-colors', rightTab === 'objects' ? 'text-primary border-b-2 border-primary' : 'text-gray-500 hover:text-gray-300']">Objects</button>
            <button @click="rightTab = 'categories'" :class="['flex-1 py-2 text-xs font-medium transition-colors', rightTab === 'categories' ? 'text-primary border-b-2 border-primary' : 'text-gray-500 hover:text-gray-300']">Categories</button>
            <button @click="rightTab = 'auto'" :class="['flex-1 py-2 text-xs font-medium transition-colors', rightTab === 'auto' ? 'text-primary border-b-2 border-primary' : 'text-gray-500 hover:text-gray-300']">
              <span class="flex items-center justify-center gap-1"><Wand2 class="h-3 w-3" /> Auto</span>
            </button>
          </div>

          <!-- TAB: Objects -->
          <div v-if="rightTab === 'objects'" class="flex flex-col flex-1 min-h-0">
            <div class="p-2 border-b border-gray-700/50">
              <select v-model="selectedCategoryFilter" class="w-full text-xs bg-background-tertiary border border-gray-600 rounded px-2 py-1 text-gray-300">
                <option :value="null">All categories ({{ activeAnnotations.length }})</option>
                <option v-for="cat in categories" :key="cat.id" :value="cat.id">{{ cat.name }}</option>
              </select>
            </div>
            <div class="flex-1 overflow-y-auto">
              <div v-if="activeAnnotations.length === 0" class="flex flex-col items-center justify-center h-32 text-gray-600">
                <Tag class="h-8 w-8 mb-2 opacity-30" /><p class="text-xs">No annotations</p>
              </div>
              <div v-for="ann in activeAnnotations" :key="ann.id" @click="editor.select(ann.id)"
                :class="['flex items-start gap-2 px-2 py-2 cursor-pointer border-l-2 transition-colors group',
                  ann.id === editor.selectedId.value ? 'bg-primary/15 border-l-primary' : 'border-l-transparent hover:bg-gray-700/20']">
                <div class="flex-shrink-0 w-2.5 h-2.5 rounded-full mt-1" :style="{ backgroundColor: getCategoryColor(ann.category_id) }" />
                <div class="flex-1 min-w-0">
                  <p class="text-xs font-medium text-gray-200 truncate">{{ getCategoryName(ann.category_id) }}</p>
                  <p class="text-xs text-gray-500 font-mono">{{ formatBbox(ann.bbox) }}</p>
                  <p v-if="ann.segmentation?.length" class="text-xs text-purple-400 flex items-center gap-0.5 mt-0.5"><Layers class="h-2.5 w-2.5" /> mask</p>
                </div>
                <button @click.stop="editor.deleteAnnotation(ann.id)" class="flex-shrink-0 p-0.5 text-gray-600 hover:text-red-400 opacity-0 group-hover:opacity-100" title="Delete"><Trash2 class="h-3.5 w-3.5" /></button>
              </div>
            </div>
            <!-- Properties -->
            <div v-if="editor.selectedAnnotation.value" class="flex-shrink-0 border-t border-gray-700/50 p-3 space-y-2">
              <p class="text-xs font-semibold text-gray-400 uppercase">Properties</p>
              <div>
                <label class="text-xs text-gray-500">Category</label>
                <select :value="editor.selectedAnnotation.value.category_id" @change="editor.updateCategory(editor.selectedId.value!, Number(($event.target as HTMLSelectElement).value))"
                  class="w-full mt-0.5 text-xs bg-background-tertiary border border-gray-600 rounded px-2 py-1.5 text-gray-300">
                  <option v-for="cat in categories" :key="cat.id" :value="cat.id">{{ cat.name }}</option>
                </select>
              </div>
              <div class="text-xs text-gray-500">Area: {{ Math.round(editor.selectedAnnotation.value.area) }}px² · ID: {{ editor.selectedAnnotation.value.id }}</div>
              <div v-if="editor.selectedAnnotation.value.segmentation?.length" class="text-xs text-purple-400 flex items-center gap-1"><Layers class="h-3 w-3" /> Mask · {{ editor.selectedAnnotation.value.segmentation.length }} polygon(s)</div>
            </div>
          </div>

          <!-- TAB: Categories -->
          <div v-if="rightTab === 'categories'" class="flex flex-col flex-1 min-h-0">
            <div class="p-3 border-b border-gray-700/50">
              <p class="text-xs font-semibold text-gray-400 uppercase mb-2">Add Category</p>
              <div class="flex gap-1">
                <input v-model="newCategoryName" @keyup.enter="addCategory" type="text" placeholder="Category name..." class="flex-1 text-xs bg-background-tertiary border border-gray-600 rounded px-2 py-1.5 text-white placeholder-gray-500 focus:outline-none focus:border-primary" />
                <button @click="addCategory" :disabled="!newCategoryName.trim()" class="p-1.5 rounded bg-primary text-white disabled:opacity-30 hover:bg-primary-hover"><Plus class="h-3.5 w-3.5" /></button>
              </div>
            </div>
            <div class="flex-1 overflow-y-auto">
              <div v-if="categories.length === 0" class="flex flex-col items-center justify-center h-32 text-gray-600">
                <Tag class="h-8 w-8 mb-2 opacity-30" /><p class="text-xs">No categories defined</p><p class="text-xs mt-1">Add categories above to start annotating</p>
              </div>
              <div v-for="cat in categories" :key="cat.id" class="flex items-center gap-2 px-3 py-2.5 border-b border-gray-700/30 group">
                <div class="w-3 h-3 rounded-sm flex-shrink-0" :style="{ backgroundColor: getCategoryColor(cat.id) }" />
                <span class="text-xs text-gray-200 flex-1">{{ cat.name }}</span>
                <span class="text-xs text-gray-600 tabular-nums">{{ editor.annotations.value.filter(a => a.category_id === cat.id).length }}</span>
                <button @click="removeCategory(cat.id)" class="p-0.5 text-gray-600 hover:text-red-400 opacity-0 group-hover:opacity-100" title="Remove category"><X class="h-3 w-3" /></button>
              </div>
            </div>
          </div>

          <!-- TAB: Auto-labeling -->
          <div v-if="rightTab === 'auto'" class="flex flex-col flex-1 min-h-0 p-3 space-y-3">
            <div>
              <p class="text-xs font-semibold text-gray-400 uppercase mb-2">SAM3 Auto-Labeling</p>
              <p class="text-xs text-gray-500 mb-3">Automatically detect and annotate objects across all images using SAM3.</p>
            </div>

            <div>
              <label class="text-xs text-gray-400 mb-1 block">Object classes (comma-separated)</label>
              <input v-model="autoLabelClasses" type="text" placeholder="fish, coral, debris..." class="w-full text-xs bg-background-tertiary border border-gray-600 rounded px-2 py-2 text-white placeholder-gray-500 focus:outline-none focus:border-primary" :disabled="autoLabelRunning" />
            </div>

            <div>
              <label class="text-xs text-gray-400 mb-1 block">Min confidence: {{ autoLabelConfidence }}</label>
              <input v-model.number="autoLabelConfidence" type="range" min="0.1" max="0.9" step="0.05" class="w-full" :disabled="autoLabelRunning" />
            </div>

            <BaseButton @click="startAutoLabeling" :loading="autoLabelRunning" :disabled="autoLabelRunning || !autoLabelClasses.trim()" variant="primary" class="w-full">
              <Wand2 class="h-4 w-4" /> {{ autoLabelRunning ? 'Running...' : 'Auto-Label All Images' }}
            </BaseButton>

            <!-- Progress -->
            <div v-if="autoLabelRunning || autoLabelStatus" class="space-y-2">
              <div v-if="autoLabelRunning" class="w-full bg-gray-700 rounded-full h-2">
                <div class="bg-primary h-2 rounded-full transition-all" :style="{ width: autoLabelProgress + '%' }"></div>
              </div>
              <p class="text-xs text-gray-400">{{ autoLabelStatus }}</p>
            </div>

            <div class="border-t border-gray-700/50 pt-3 mt-auto">
              <p class="text-xs text-gray-500">
                <strong class="text-gray-400">Workflow:</strong><br>
                1. Enter object class names to detect<br>
                2. Click Auto-Label to run SAM3<br>
                3. Review and edit results manually<br>
                4. Save the final annotations
              </p>
            </div>
          </div>
        </div>
      </div>
    </template>

    <ExportDialog
      :open="showExport"
      :coco-json-path="cocoJsonPath"
      :images-dir="imagesDirPath || undefined"
      @close="showExport = false"
    />
  </div>
</template>
