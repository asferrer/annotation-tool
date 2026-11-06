<script setup lang="ts">
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useTaskStore, type TaskLabel } from '@/stores/task'
import { useUiStore } from '@/stores/ui'
import { uploadImages, uploadCocoJson, getExportFormats, startImport, getExportJob } from '@/lib/api'
import type { ExportFormatSpec } from '@/types/api'
import BaseButton from '@/components/ui/BaseButton.vue'
import BaseInput from '@/components/ui/BaseInput.vue'
import DirectoryBrowser from '@/components/common/DirectoryBrowser.vue'
import { Plus, X, Tag, FolderOpen, Play, Upload, HardDrive, FileJson, Image as ImageIcon, Loader2 } from 'lucide-vue-next'

const router = useRouter()
const taskStore = useTaskStore()
const uiStore = useUiStore()

// Task config
const taskName = ref('')

// Data source mode: 'upload' (from PC) or 'volume' (Docker volume)
const sourceMode = ref<'upload' | 'volume' | 'import'>('upload')

// Volume mode
const imagesDirPath = ref('')
const cocoJsonPath = ref('')
const useExistingCoco = ref(false)

// Upload mode
const selectedFiles = ref<File[]>([])
const selectedCocoFile = ref<File | null>(null)
const uploadProgress = ref(0)
const isUploading = ref(false)
const uploadedDir = ref('')
const uploadedCocoPath = ref('')
const uploadStatus = ref('')

// Labels
const labels = ref<TaskLabel[]>([])
const newLabelName = ref('')

// Drag state
const isDragOver = ref(false)

// ---------------------------------------------------------------------------
// File selection
// ---------------------------------------------------------------------------
const IMAGE_RE = /\.(jpg|jpeg|png|bmp|webp|tiff|tif)$/i
const scanningFolder = ref(false)

function onFilesSelected(e: Event) {
  const input = e.target as HTMLInputElement
  if (input.files) {
    const newFiles = Array.from(input.files).filter((f) => IMAGE_RE.test(f.name))
    const json = Array.from(input.files).find((f) => f.name.endsWith('.json'))
    if (newFiles.length > 0) selectedFiles.value = [...selectedFiles.value, ...newFiles]
    if (json && !selectedCocoFile.value) selectedCocoFile.value = json
  }
  input.value = ''
}

function onCocoSelected(e: Event) {
  const input = e.target as HTMLInputElement
  if (input.files?.[0]) selectedCocoFile.value = input.files[0]
  input.value = ''
}

// Recursively traverse directory entries from drag-and-drop
async function readEntryRecursive(entry: FileSystemEntry): Promise<File[]> {
  if (entry.isFile) {
    return new Promise((resolve) => {
      (entry as FileSystemFileEntry).file((f) => resolve([f]), () => resolve([]))
    })
  }
  if (entry.isDirectory) {
    const reader = (entry as FileSystemDirectoryEntry).createReader()
    const files: File[] = []
    // readEntries may return partial results, so loop until empty
    const readBatch = (): Promise<FileSystemEntry[]> =>
      new Promise((resolve) => reader.readEntries((entries) => resolve(entries), () => resolve([])))
    let batch = await readBatch()
    while (batch.length > 0) {
      for (const child of batch) {
        files.push(...await readEntryRecursive(child))
      }
      batch = await readBatch()
    }
    return files
  }
  return []
}

async function onDrop(e: DragEvent) {
  isDragOver.value = false
  if (!e.dataTransfer) return

  const items = e.dataTransfer.items
  if (items && items.length > 0) {
    scanningFolder.value = true
    const allFiles: File[] = []

    for (let i = 0; i < items.length; i++) {
      const entry = items[i].webkitGetAsEntry?.()
      if (entry) {
        allFiles.push(...await readEntryRecursive(entry))
      }
    }

    const images = allFiles.filter((f) => IMAGE_RE.test(f.name))
    const json = allFiles.find((f) => f.name.endsWith('.json'))
    if (images.length > 0) selectedFiles.value = [...selectedFiles.value, ...images]
    if (json && !selectedCocoFile.value) selectedCocoFile.value = json
    scanningFolder.value = false
    return
  }

  // Fallback for browsers without webkitGetAsEntry
  const files = Array.from(e.dataTransfer.files)
  const images = files.filter((f) => IMAGE_RE.test(f.name))
  const json = files.find((f) => f.name.endsWith('.json'))
  if (images.length > 0) selectedFiles.value = [...selectedFiles.value, ...images]
  if (json && !selectedCocoFile.value) selectedCocoFile.value = json
}

function removeFile(idx: number) {
  selectedFiles.value = selectedFiles.value.filter((_, i) => i !== idx)
}

// ---------------------------------------------------------------------------
// Labels
// ---------------------------------------------------------------------------
function addLabel() {
  const name = newLabelName.value.trim()
  if (!name) return
  if (labels.value.some((l) => l.name.toLowerCase() === name.toLowerCase())) return
  const maxId = labels.value.reduce((max, l) => Math.max(max, l.id), 0)
  const color = taskStore.LABEL_COLORS[labels.value.length % taskStore.LABEL_COLORS.length]
  labels.value.push({ id: maxId + 1, name, color })
  newLabelName.value = ''
}

function removeLabel(id: number) {
  labels.value = labels.value.filter((l) => l.id !== id)
}

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------
const canStart = computed(() => {
  if (sourceMode.value === 'upload') return selectedFiles.value.length > 0
  if (sourceMode.value === 'import') return importPath.value.trim() !== '' && importFormat.value !== ''
  return imagesDirPath.value.trim() !== ''
})

// ---------------------------------------------------------------------------
// Import from other formats (Datumaro engine)
// ---------------------------------------------------------------------------
const importPath = ref('')
const importFormat = ref('coco')
const importFormats = ref<ExportFormatSpec[]>([])
const isImporting = ref(false)

async function loadImportFormats() {
  if (importFormats.value.length > 0) return
  try {
    importFormats.value = (await getExportFormats()).filter((f) => f.supports_import)
  } catch {
    uiStore.showError('Import', 'Could not load import formats')
  }
}

async function runImport(name: string): Promise<string | null> {
  isImporting.value = true
  try {
    const { job_id } = await startImport({
      input_path: importPath.value.trim(),
      format: importFormat.value,
      task_name: name,
    })
    // Poll until the conversion job finishes
    for (;;) {
      await new Promise((r) => setTimeout(r, 1500))
      const job = await getExportJob(job_id)
      if (job.status === 'completed') {
        const summary = (job.result_summary ?? {}) as { coco_json_path?: string }
        return summary.coco_json_path ?? job.output_path ?? null
      }
      if (['failed', 'cancelled', 'interrupted'].includes(job.status)) {
        uiStore.showError('Import failed', job.error_message || 'Conversion error')
        return null
      }
    }
  } catch (err: any) {
    uiStore.showError('Import failed', err?.response?.data?.detail ?? err?.message ?? 'Error')
    return null
  } finally {
    isImporting.value = false
  }
}

async function startAnnotation() {
  if (!canStart.value) return

  const name = taskName.value.trim() || 'Untitled Task'
  let imgDir = ''
  let cocoPath = ''

  if (sourceMode.value === 'upload') {
    // Upload files first
    isUploading.value = true
    uploadProgress.value = 0
    try {
      const result = await uploadImages(name, selectedFiles.value, (uploaded, total) => {
        uploadProgress.value = Math.round((uploaded / total) * 100)
        uploadStatus.value = `${uploaded} / ${total} images`
      })
      imgDir = result.directory
      uploadedDir.value = result.directory
      uiStore.showSuccess('Upload complete', `${result.uploaded_count} images uploaded`)

      // Upload COCO JSON if provided
      if (selectedCocoFile.value) {
        const cocoResult = await uploadCocoJson(name, selectedCocoFile.value)
        cocoPath = cocoResult.path
        uploadedCocoPath.value = cocoResult.path
      }
    } catch (err: any) {
      uiStore.showError('Upload failed', err?.message ?? 'Error uploading files')
      isUploading.value = false
      return
    } finally {
      isUploading.value = false
    }
  } else if (sourceMode.value === 'import') {
    const imported = await runImport(name)
    if (!imported) return
    cocoPath = imported
    imgDir = imported.replace(/\/annotations\/[^/]+$/, '/images')
    uiStore.showSuccess('Import complete', 'Dataset converted to COCO')
  } else {
    imgDir = imagesDirPath.value.trim()
    cocoPath = useExistingCoco.value ? cocoJsonPath.value.trim() : ''
  }

  taskStore.createTask(name, imgDir, cocoPath, [...labels.value])
  router.push('/annotate')
}
</script>

<template>
  <div class="max-w-2xl mx-auto space-y-6">
    <div>
      <h2 class="text-2xl font-bold text-white">Create Annotation Task</h2>
      <p class="mt-1 text-sm text-gray-400">Configure your dataset and labels, then start annotating.</p>
    </div>

    <!-- Task name -->
    <div class="card p-5">
      <BaseInput v-model="taskName" label="Task name" placeholder="My annotation task" />
    </div>

    <!-- Data source -->
    <div class="card p-5 space-y-4">
      <h3 class="text-sm font-semibold text-gray-300 flex items-center gap-2">
        <FolderOpen class="h-4 w-4 text-primary" /> Data Source
      </h3>

      <!-- Source mode tabs -->
      <div class="flex gap-1 bg-background-tertiary rounded-lg p-1">
        <button @click="sourceMode = 'upload'" :class="['flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-colors', sourceMode === 'upload' ? 'bg-primary text-white' : 'text-gray-400 hover:text-white']">
          <Upload class="h-4 w-4" /> My Computer
        </button>
        <button @click="sourceMode = 'volume'" :class="['flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-colors', sourceMode === 'volume' ? 'bg-primary text-white' : 'text-gray-400 hover:text-white']">
          <HardDrive class="h-4 w-4" /> Server Directory
        </button>
        <button @click="sourceMode = 'import'; loadImportFormats()" :class="['flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-colors', sourceMode === 'import' ? 'bg-primary text-white' : 'text-gray-400 hover:text-white']">
          <FileJson class="h-4 w-4" /> Import Format
        </button>
      </div>

      <!-- UPLOAD MODE -->
      <template v-if="sourceMode === 'upload'">
        <!-- Drop zone -->
        <div
          @drop.prevent="onDrop"
          @dragover.prevent="isDragOver = true"
          @dragleave="isDragOver = false"
          :class="['border-2 border-dashed rounded-xl p-8 text-center transition-colors',
            isDragOver ? 'border-primary bg-primary/5' : 'border-gray-600 hover:border-gray-500']"
        >
          <Upload :class="['h-10 w-10 mx-auto mb-3', isDragOver ? 'text-primary' : 'text-gray-500']" />
          <p class="text-sm text-gray-300 mb-1">Drag & drop images or folders here</p>
          <p class="text-xs text-gray-500 mb-3">JPG, PNG, BMP, WebP, TIFF supported. Subfolders are scanned recursively.</p>
          <div class="flex items-center justify-center gap-2">
            <button @click="($refs.fileInput as HTMLInputElement)?.click()" class="px-3 py-1.5 rounded-lg text-xs font-medium bg-background-tertiary text-gray-300 hover:text-white hover:bg-gray-600 transition-colors">
              Select Files
            </button>
            <button @click="($refs.folderInput as HTMLInputElement)?.click()" class="px-3 py-1.5 rounded-lg text-xs font-medium bg-primary/20 text-primary hover:bg-primary/30 transition-colors">
              Select Folder
            </button>
          </div>
          <div v-if="scanningFolder" class="mt-3 flex items-center justify-center gap-2 text-xs text-gray-400">
            <Loader2 class="h-3.5 w-3.5 animate-spin" /> Scanning folder...
          </div>
        </div>
        <!-- Hidden file inputs -->
        <input ref="fileInput" type="file" multiple accept="image/*,.json" class="hidden" @change="onFilesSelected" />
        <input ref="folderInput" type="file" multiple webkitdirectory class="hidden" @change="onFilesSelected" />

        <!-- Selected files count -->
        <div v-if="selectedFiles.length > 0" class="space-y-2">
          <div class="flex items-center justify-between">
            <span class="text-sm text-gray-300 flex items-center gap-2">
              <ImageIcon class="h-4 w-4 text-primary" />
              {{ selectedFiles.length }} images selected
              <span class="text-xs text-gray-500">({{ (selectedFiles.reduce((s, f) => s + f.size, 0) / 1024 / 1024).toFixed(1) }} MB)</span>
            </span>
            <button @click="selectedFiles = []" class="text-xs text-gray-500 hover:text-red-400">Clear all</button>
          </div>

          <!-- File list (scrollable, max 5 visible) -->
          <div class="max-h-32 overflow-y-auto space-y-0.5 bg-background-tertiary rounded-lg p-2">
            <div v-for="(file, idx) in selectedFiles" :key="idx" class="flex items-center gap-2 text-xs text-gray-400 group">
              <ImageIcon class="h-3 w-3 flex-shrink-0 opacity-50" />
              <span class="flex-1 truncate">{{ file.name }}</span>
              <span class="text-gray-600">{{ (file.size / 1024).toFixed(0) }}KB</span>
              <button @click="removeFile(idx)" class="p-0.5 text-gray-600 hover:text-red-400 opacity-0 group-hover:opacity-100"><X class="h-3 w-3" /></button>
            </div>
          </div>
        </div>

        <!-- COCO JSON upload (optional) -->
        <div class="border-t border-gray-700/50 pt-4">
          <div class="flex items-center justify-between mb-2">
            <span class="text-sm text-gray-400 flex items-center gap-2"><FileJson class="h-4 w-4" /> COCO JSON (optional)</span>
            <label class="text-xs text-primary hover:text-primary-hover cursor-pointer" @click="($refs.cocoInput as HTMLInputElement)?.click()">
              Browse...
            </label>
          </div>
          <input ref="cocoInput" type="file" accept=".json" class="hidden" @change="onCocoSelected" />
          <div v-if="selectedCocoFile" class="flex items-center gap-2 bg-background-tertiary rounded-lg px-3 py-2 text-sm text-gray-300">
            <FileJson class="h-4 w-4 text-green-400" />
            <span class="flex-1 truncate">{{ selectedCocoFile.name }}</span>
            <button @click="selectedCocoFile = null" class="text-gray-500 hover:text-red-400"><X class="h-4 w-4" /></button>
          </div>
          <p v-else class="text-xs text-gray-600">Upload existing annotations to review and edit.</p>
        </div>

        <!-- Upload progress -->
        <div v-if="isUploading" class="space-y-2">
          <div class="flex items-center gap-2 text-sm text-gray-300">
            <Loader2 class="h-4 w-4 animate-spin text-primary" />
            Uploading... {{ uploadStatus || uploadProgress + '%' }}
          </div>
          <div class="w-full bg-gray-700 rounded-full h-2">
            <div class="bg-primary h-2 rounded-full transition-all" :style="{ width: uploadProgress + '%' }" />
          </div>
          <p class="text-xs text-gray-500">Uploading in batches of 50 images. Do not close this page.</p>
        </div>
      </template>

      <!-- VOLUME MODE -->
      <template v-if="sourceMode === 'volume'">
        <DirectoryBrowser v-model="imagesDirPath" label="Images directory" placeholder="/app/datasets/images" :show-files="false" path-mode="input" :restrict-to-mounts="false" />

        <label class="flex items-center gap-2 cursor-pointer">
          <input type="checkbox" v-model="useExistingCoco" class="w-4 h-4 rounded border-gray-600 bg-gray-700 text-primary focus:ring-primary" />
          <span class="text-sm text-gray-300">Load existing COCO JSON</span>
        </label>

        <DirectoryBrowser v-if="useExistingCoco" v-model="cocoJsonPath" label="COCO JSON file" placeholder="/app/datasets/annotations.json" :show-files="true" file-pattern="*.json" path-mode="input" :restrict-to-mounts="false" />
      </template>

      <!-- IMPORT MODE -->
      <template v-if="sourceMode === 'import'">
        <div>
          <label class="text-sm text-gray-400 mb-1 block">Source format</label>
          <select v-model="importFormat" class="w-full bg-background-tertiary border border-gray-600 rounded-lg px-3 py-2 text-white text-sm">
            <option v-for="f in importFormats" :key="f.id" :value="f.id">{{ f.label }}</option>
          </select>
          <p class="text-xs text-gray-500 mt-1">{{ importFormats.find(f => f.id === importFormat)?.description }}</p>
        </div>
        <DirectoryBrowser v-model="importPath" label="Dataset root directory" placeholder="/app/datasets/my-yolo-dataset" :show-files="false" path-mode="input" :restrict-to-mounts="false" />
        <p class="text-xs text-gray-500">The dataset will be converted to COCO into the datasets volume before annotating.</p>
        <div v-if="isImporting" class="flex items-center gap-2 text-sm text-gray-300">
          <Loader2 class="h-4 w-4 animate-spin text-primary" /> Converting dataset...
        </div>
      </template>
    </div>

    <!-- Labels -->
    <div class="card p-5 space-y-4">
      <h3 class="text-sm font-semibold text-gray-300 flex items-center gap-2"><Tag class="h-4 w-4 text-primary" /> Labels</h3>
      <p class="text-xs text-gray-500">Define categories. You can also add them in the editor or let auto-labeling create them.</p>

      <div class="flex gap-2">
        <input v-model="newLabelName" @keyup.enter="addLabel" type="text" placeholder="Label name..." class="flex-1 text-sm bg-background-tertiary border border-gray-600 rounded-lg px-3 py-2 text-white placeholder-gray-500 focus:outline-none focus:border-primary" />
        <BaseButton @click="addLabel" :disabled="!newLabelName.trim()" variant="outline" size="sm"><Plus class="h-4 w-4" /> Add</BaseButton>
      </div>

      <div v-if="labels.length > 0" class="space-y-1">
        <div v-for="label in labels" :key="label.id" class="flex items-center gap-3 bg-background-tertiary rounded-lg px-3 py-2 group">
          <input type="color" :value="label.color" @input="(e) => { const l = labels.find(x => x.id === label.id); if (l) l.color = (e.target as HTMLInputElement).value }" class="w-6 h-6 rounded border-0 cursor-pointer bg-transparent" />
          <span class="text-sm text-gray-200 flex-1">{{ label.name }}</span>
          <button @click="removeLabel(label.id)" class="p-1 text-gray-500 hover:text-red-400 opacity-0 group-hover:opacity-100"><X class="h-4 w-4" /></button>
        </div>
      </div>
    </div>

    <!-- Start -->
    <div class="flex justify-end gap-3 pb-8">
      <BaseButton @click="router.push('/')" variant="outline">Cancel</BaseButton>
      <BaseButton @click="startAnnotation" :disabled="!canStart || isUploading || isImporting" :loading="isUploading || isImporting" variant="primary" size="lg">
        <Play class="h-4 w-4" /> Start Annotating
      </BaseButton>
    </div>
  </div>
</template>
