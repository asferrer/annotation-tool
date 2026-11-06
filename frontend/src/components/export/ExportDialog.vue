<script setup lang="ts">
import { ref, computed, watch, onUnmounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { Download, Loader2, CheckCircle2, XCircle } from 'lucide-vue-next'
import BaseButton from '@/components/ui/BaseButton.vue'
import { getExportFormats, startExport, getExportJob, getExportDownloadUrl } from '@/lib/api'
import type { ExportFormatSpec, ExportJob } from '@/types/api'
import { useUiStore } from '@/stores/ui'

const props = defineProps<{
  open: boolean
  cocoJsonPath: string
  imagesDir?: string
  videoDataset?: boolean
  stats?: { images?: number; annotations?: number; categories?: number }
}>()

const emit = defineEmits<{ (e: 'close'): void }>()

const { t } = useI18n()
const uiStore = useUiStore()

const formats = ref<ExportFormatSpec[]>([])
const selectedFormat = ref('coco')
const includeImages = ref(false)
const job = ref<ExportJob | null>(null)
const starting = ref(false)
let pollTimer: ReturnType<typeof setInterval> | null = null

const visibleFormats = computed(() =>
  formats.value.filter((f) => (props.videoDataset ? true : !f.tasks.includes('video') || f.tasks.length > 1)),
)

const isRunning = computed(() => job.value !== null && ['queued', 'running'].includes(job.value.status))
const isDone = computed(() => job.value?.status === 'completed')
const isFailed = computed(() => job.value !== null && ['failed', 'cancelled', 'interrupted'].includes(job.value.status))

watch(() => props.open, async (open) => {
  if (!open) { stopPolling(); job.value = null; return }
  if (formats.value.length === 0) {
    try {
      formats.value = await getExportFormats()
    } catch {
      uiStore.showError('Export', t('export.formatsError', 'Could not load export formats'))
    }
  }
})

function stopPolling() {
  if (pollTimer !== null) { clearInterval(pollTimer); pollTimer = null }
}

async function handleStart() {
  if (!props.cocoJsonPath) return
  starting.value = true
  job.value = null
  try {
    const { job_id } = await startExport({
      coco_json_path: props.cocoJsonPath,
      format: selectedFormat.value,
      include_images: includeImages.value,
      images_dir: props.imagesDir || null,
    })
    pollTimer = setInterval(async () => {
      try {
        job.value = await getExportJob(job_id)
        if (job.value && !['queued', 'running'].includes(job.value.status)) stopPolling()
      } catch { /* keep polling; transient errors */ }
    }, 1500)
    job.value = { id: job_id, job_type: 'export', service: 'gateway', status: 'queued' }
  } catch (err: any) {
    uiStore.showError('Export', err?.response?.data?.detail ?? err?.message ?? 'Error')
  } finally {
    starting.value = false
  }
}

function handleDownload() {
  if (job.value) window.location.href = getExportDownloadUrl(job.value.id)
}

onUnmounted(stopPolling)
</script>

<template>
  <div v-if="open" class="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm" @click.self="emit('close')">
    <div class="bg-background-secondary border border-gray-700 rounded-xl p-6 w-[26rem] space-y-4">
      <h3 class="text-lg font-semibold text-white flex items-center gap-2">
        <Download class="h-5 w-5 text-primary" /> {{ t('export.title', 'Export Dataset') }}
      </h3>

      <div>
        <label class="text-sm text-gray-400 mb-1 block">{{ t('export.format', 'Format') }}</label>
        <select v-model="selectedFormat" :disabled="isRunning" class="w-full bg-background-tertiary border border-gray-600 rounded-lg px-3 py-2 text-white text-sm">
          <option v-for="f in visibleFormats" :key="f.id" :value="f.id">{{ f.label }}</option>
        </select>
        <p class="text-xs text-gray-500 mt-1">{{ formats.find(f => f.id === selectedFormat)?.description }}</p>
      </div>

      <label class="flex items-center gap-2 text-sm text-gray-300">
        <input v-model="includeImages" type="checkbox" :disabled="isRunning" class="rounded border-gray-600 bg-background-tertiary" />
        {{ t('export.includeImages', 'Include images in the zip') }}
      </label>

      <div v-if="stats" class="text-xs text-gray-500 space-y-0.5">
        <p v-if="stats.images !== undefined"><strong>{{ t('export.images', 'Images') }}:</strong> {{ stats.images }}</p>
        <p v-if="stats.annotations !== undefined"><strong>{{ t('export.annotations', 'Annotations') }}:</strong> {{ stats.annotations }}</p>
        <p v-if="stats.categories !== undefined"><strong>{{ t('export.categories', 'Categories') }}:</strong> {{ stats.categories }}</p>
      </div>

      <div v-if="job" class="text-sm rounded-lg px-3 py-2 bg-background-tertiary border border-gray-700 flex items-center gap-2">
        <Loader2 v-if="isRunning" class="h-4 w-4 animate-spin text-primary" />
        <CheckCircle2 v-else-if="isDone" class="h-4 w-4 text-green-400" />
        <XCircle v-else-if="isFailed" class="h-4 w-4 text-red-400" />
        <span class="text-gray-300">
          {{ isRunning ? t('export.running', 'Exporting…') : isDone ? t('export.done', 'Export ready') : (job.error_message || t('export.failed', 'Export failed')) }}
        </span>
      </div>

      <div class="flex gap-2 justify-end">
        <BaseButton variant="outline" size="sm" @click="emit('close')">{{ t('common.close', 'Close') }}</BaseButton>
        <BaseButton v-if="isDone" variant="primary" size="sm" @click="handleDownload">
          <Download class="h-4 w-4" /> {{ t('export.download', 'Download zip') }}
        </BaseButton>
        <BaseButton v-else variant="primary" size="sm" :disabled="starting || isRunning || !cocoJsonPath" @click="handleStart">
          <Download class="h-4 w-4" /> {{ isRunning ? t('export.running', 'Exporting…') : t('export.start', 'Export') }}
        </BaseButton>
      </div>
    </div>
  </div>
</template>
