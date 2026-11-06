<script setup lang="ts">
import { computed, onUnmounted, ref, watch } from 'vue'
import { CheckCircle2, Copy, Loader2, Share2, XCircle } from 'lucide-vue-next'
import BaseButton from '@/components/ui/BaseButton.vue'
import DirectoryBrowser from '@/components/common/DirectoryBrowser.vue'
import { cancelPropagateJob, getPropagateJob, getPropagateStatus, startPropagation } from '@/lib/api'
import { useUiStore } from '@/stores/ui'

const props = defineProps<{
  open: boolean
  referenceImagePath: string
  referencePolygon: number[][]
  defaultLabel: string
}>()

const emit = defineEmits<{ (e: 'close'): void }>()
const uiStore = useUiStore()

const targetDirectory = ref('')
const label = ref('')
const mergeThreshold = ref(0.2)
const weightsInfo = ref<{ weights_present: boolean; instructions: string | null } | null>(null)

const jobId = ref<string | null>(null)
const jobStatus = ref<string | null>(null)
const jobProgress = ref({ processed: 0, total: 0, found: 0 })
const outputJsonPath = ref('')
let pollTimer: ReturnType<typeof setInterval> | null = null

const isRunning = computed(() => jobStatus.value !== null && ['queued', 'running'].includes(jobStatus.value))
const isDone = computed(() => jobStatus.value === 'completed')
const isFailed = computed(() => jobStatus.value !== null && ['failed', 'cancelled', 'interrupted'].includes(jobStatus.value))
const canStart = computed(() =>
  !!targetDirectory.value && !!label.value.trim() && weightsInfo.value?.weights_present && !isRunning.value,
)

watch(() => props.open, async (open) => {
  if (!open) { stopPolling(); return }
  label.value = props.defaultLabel
  jobId.value = null
  jobStatus.value = null
  try {
    const status = await getPropagateStatus()
    weightsInfo.value = status.insid3
  } catch (err: any) {
    uiStore.showError('INSID3', err?.response?.data?.detail ?? 'No se pudo consultar el estado')
  }
})

function stopPolling() {
  if (pollTimer !== null) { clearInterval(pollTimer); pollTimer = null }
}

async function handleStart() {
  if (!canStart.value) return
  const outDir = targetDirectory.value.replace(/\/$/, '')
  outputJsonPath.value = `${outDir}/insid3_${label.value.trim().replace(/\s+/g, '_')}.json`
  try {
    const res = await startPropagation({
      reference_image_path: props.referenceImagePath,
      reference_polygon: props.referencePolygon,
      target_directory: targetDirectory.value,
      label: label.value.trim(),
      merge_threshold: mergeThreshold.value,
      output_json: outputJsonPath.value,
    })
    jobId.value = res.job_id
    jobStatus.value = 'queued'
    pollTimer = setInterval(poll, 2500)
  } catch (err: any) {
    uiStore.showError('INSID3', err?.response?.data?.detail ?? err?.message ?? 'Error')
  }
}

async function poll() {
  if (!jobId.value) return
  try {
    const job = await getPropagateJob(jobId.value)
    jobStatus.value = job.status
    const details = (job as any).progress_details ?? {}
    jobProgress.value = {
      processed: job.processed_items ?? 0,
      total: job.total_items ?? 0,
      found: details.objects_found ?? 0,
    }
    if (!['queued', 'running'].includes(job.status)) stopPolling()
  } catch { /* transient */ }
}

async function handleCancel() {
  if (jobId.value) await cancelPropagateJob(jobId.value)
}

async function copyPath() {
  await navigator.clipboard.writeText(outputJsonPath.value)
  uiStore.showSuccess('Copied', 'COCO path copied to clipboard')
}

onUnmounted(stopPolling)
</script>

<template>
  <div v-if="open" class="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm" @click.self="emit('close')">
    <div class="bg-background-secondary border border-gray-700 rounded-xl p-6 w-[28rem] space-y-4 max-h-[85vh] overflow-y-auto">
      <h3 class="text-lg font-semibold text-white flex items-center gap-2">
        <Share2 class="h-5 w-5 text-primary" /> Propagate to dataset (INSID3)
      </h3>
      <p class="text-xs text-gray-500">
        Uses the selected mask as an example and segments the same concept in every image of the directory (one-shot, DINOv3, training-free).
      </p>

      <div v-if="weightsInfo && !weightsInfo.weights_present" class="text-xs text-amber-400 bg-amber-900/20 border border-amber-800/50 rounded-lg px-3 py-2">
        DINOv3 weights are missing (gated by Meta): {{ weightsInfo.instructions }}
      </div>

      <div>
        <label class="text-sm text-gray-400 mb-1 block">Label</label>
        <input v-model="label" type="text" class="w-full text-sm bg-background-tertiary border border-gray-600 rounded-lg px-3 py-2 text-white" :disabled="isRunning" />
      </div>

      <DirectoryBrowser v-model="targetDirectory" label="Target image directory" placeholder="/app/datasets/my-dataset/images" :show-files="false" path-mode="input" :restrict-to-mounts="false" />

      <div>
        <label class="text-sm text-gray-400 mb-1 block">Umbral de similitud: {{ mergeThreshold.toFixed(2) }}</label>
        <input v-model.number="mergeThreshold" type="range" min="0.05" max="0.6" step="0.05" class="w-full" :disabled="isRunning" />
        <p class="text-[11px] text-gray-600">Higher = stricter (fewer false positives, may miss instances).</p>
      </div>

      <div v-if="jobStatus" class="text-sm rounded-lg px-3 py-2 bg-background-tertiary border border-gray-700 space-y-1">
        <div class="flex items-center gap-2">
          <Loader2 v-if="isRunning" class="h-4 w-4 animate-spin text-primary" />
          <CheckCircle2 v-else-if="isDone" class="h-4 w-4 text-green-400" />
          <XCircle v-else-if="isFailed" class="h-4 w-4 text-red-400" />
          <span class="text-gray-300">
            {{ isRunning ? `Propagando… ${jobProgress.processed}/${jobProgress.total} (${jobProgress.found} encontrados)`
              : isDone ? `Completed: ${jobProgress.found} objects in ${jobProgress.total} images`
              : `Job ${jobStatus}` }}
          </span>
        </div>
        <div v-if="isDone" class="flex items-center gap-1 text-[11px] text-gray-500 break-all">
          <span class="flex-1">{{ outputJsonPath }}</span>
          <button class="p-1 text-gray-400 hover:text-white" @click="copyPath"><Copy class="h-3.5 w-3.5" /></button>
        </div>
        <p v-if="isDone" class="text-[11px] text-gray-600">Open it in Annotation Review to inspect and correct the results.</p>
      </div>

      <div class="flex gap-2 justify-end">
        <BaseButton variant="outline" size="sm" @click="emit('close')">Close</BaseButton>
        <BaseButton v-if="isRunning" variant="outline" size="sm" @click="handleCancel">Cancel job</BaseButton>
        <BaseButton v-else variant="primary" size="sm" :disabled="!canStart" @click="handleStart">
          <Share2 class="h-4 w-4" /> Propagate
        </BaseButton>
      </div>
    </div>
  </div>
</template>
