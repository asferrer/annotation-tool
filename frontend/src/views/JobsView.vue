<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import { useUiStore } from '@/stores/ui'
import { listLabelingJobs, cancelLabelingJob } from '@/lib/api'
import BaseButton from '@/components/ui/BaseButton.vue'
import AlertBox from '@/components/common/AlertBox.vue'
import {
  RefreshCw,
  StopCircle,
  CheckCircle,
  XCircle,
  Clock,
  Loader,
  Eye,
  GitCompare,
} from 'lucide-vue-next'
import type { LabelingJob } from '@/types/api'

const router = useRouter()
const uiStore = useUiStore()

const jobs = ref<LabelingJob[]>([])
const total = ref(0)
const loading = ref(false)
const error = ref<string | null>(null)
const expandedError = ref<string | null>(null)
let pollingInterval: ReturnType<typeof setInterval> | null = null

async function loadJobs() {
  loading.value = true
  error.value = null
  try {
    const res = await listLabelingJobs()
    jobs.value = res.jobs ?? []
    total.value = res.total ?? 0
  } catch (e: any) {
    error.value = e.message || 'Failed to load jobs'
  } finally {
    loading.value = false
  }
}

async function cancel(jobId: string) {
  try {
    await cancelLabelingJob(jobId)
    uiStore.showInfo('Job Cancelled', jobId.slice(0, 8))
    await loadJobs()
  } catch (e: any) {
    uiStore.showError('Cancel Failed', e.message)
  }
}

function toggleError(jobId: string) {
  expandedError.value = expandedError.value === jobId ? null : jobId
}

function reviewJob(job: LabelingJob) {
  if (!job.output_dir) return
  const cocoPath = job.output_dir.replace(/\/$/, '') + '/annotations.json'
  const imagesDir = job.image_directories?.[0] ?? ''
  router.push({
    path: '/annotation-review',
    query: { coco_json_path: cocoPath, images_dir: imagesDir },
  })
}

function compareJob(job: LabelingJob) {
  if (!job.output_dir) return
  const predPath = job.output_dir.replace(/\/$/, '') + '/annotations.json'
  router.push({ path: '/comparison', query: { pred_path: predPath } })
}

function statusIcon(status: LabelingJob['status']) {
  switch (status) {
    case 'completed': return CheckCircle
    case 'failed': return XCircle
    case 'cancelled': return XCircle
    case 'running': return Loader
    default: return Clock
  }
}

function statusColor(status: LabelingJob['status']) {
  switch (status) {
    case 'completed': return 'text-green-400'
    case 'failed': return 'text-red-400'
    case 'cancelled': return 'text-gray-400'
    case 'running': return 'text-primary animate-spin'
    default: return 'text-yellow-400'
  }
}

onMounted(() => {
  loadJobs()
  // Auto-refresh while any job is running
  pollingInterval = setInterval(async () => {
    const hasRunning = jobs.value.some(j => j.status === 'running' || j.status === 'pending')
    if (hasRunning) await loadJobs()
  }, 3000)
})

onUnmounted(() => {
  if (pollingInterval) clearInterval(pollingInterval)
})
</script>

<template>
  <div class="space-y-6">
    <div class="flex items-center justify-between">
      <div>
        <h2 class="text-2xl font-bold text-white">All Jobs</h2>
        <p class="mt-1 text-gray-400">{{ total }} total labeling jobs</p>
      </div>
      <BaseButton variant="outline" size="sm" :loading="loading" @click="loadJobs">
        <RefreshCw class="h-4 w-4" />
        Refresh
      </BaseButton>
    </div>

    <AlertBox v-if="error" type="error" :title="error" dismissible @dismiss="error = null" />

    <div v-if="jobs.length === 0 && !loading" class="card p-12 text-center">
      <Clock class="h-12 w-12 text-gray-600 mx-auto mb-3" />
      <p class="text-gray-400">No jobs found. Start one from Auto Labeling.</p>
    </div>

    <div v-else class="space-y-3">
      <div
        v-for="job in jobs"
        :key="job.job_id"
        class="card p-4"
      >
        <div class="flex items-center gap-4">
          <component
            :is="statusIcon(job.status)"
            class="h-5 w-5 flex-shrink-0"
            :class="statusColor(job.status)"
          />

          <div class="flex-1 min-w-0">
            <div class="flex items-center gap-2 flex-wrap">
              <span class="font-mono text-sm text-white">{{ job.job_id.slice(0, 8) }}</span>
              <span
                class="text-xs px-2 py-0.5 rounded-full font-medium"
                :class="{
                  'bg-green-400/20 text-green-400': job.status === 'completed',
                  'bg-red-400/20 text-red-400': job.status === 'failed',
                  'bg-gray-400/20 text-gray-400': job.status === 'cancelled',
                  'bg-primary/20 text-primary': job.status === 'running',
                  'bg-yellow-400/20 text-yellow-400': job.status === 'pending',
                }"
              >
                {{ job.status }}
              </span>
            </div>

            <p class="text-sm text-gray-400 mt-0.5">
              {{ job.processed_images }}/{{ job.total_images }} images &middot;
              {{ job.annotations_created }} annotations
              <span v-if="job.status === 'running' && job.current_image" class="block text-xs text-gray-500 truncate mt-0.5">
                Processing: {{ job.current_image }}
              </span>
            </p>
          </div>

          <div class="flex items-center gap-3">
            <span class="text-lg font-bold text-primary w-12 text-right">{{ job.progress }}%</span>

            <BaseButton
              v-if="job.status === 'completed' && job.output_dir"
              variant="outline"
              size="sm"
              @click="reviewJob(job)"
            >
              <Eye class="h-4 w-4" />
              Review
            </BaseButton>

            <BaseButton
              v-if="job.status === 'completed' && job.output_dir"
              variant="outline"
              size="sm"
              @click="compareJob(job)"
            >
              <GitCompare class="h-4 w-4" />
              Compare
            </BaseButton>

            <BaseButton
              v-if="job.status === 'failed' && job.error"
              variant="ghost"
              size="sm"
              @click="toggleError(job.job_id)"
            >
              <XCircle class="h-4 w-4 text-red-400" />
              {{ expandedError === job.job_id ? 'Hide' : 'Error' }}
            </BaseButton>

            <BaseButton
              v-if="job.status === 'running'"
              variant="ghost"
              size="sm"
              @click="cancel(job.job_id)"
            >
              <StopCircle class="h-4 w-4" />
              Cancel
            </BaseButton>
          </div>
        </div>

        <!-- Progress bar -->
        <div v-if="job.status === 'running'" class="mt-3 h-1.5 bg-background-tertiary rounded-full overflow-hidden">
          <div
            class="h-full bg-gradient-to-r from-primary to-green-400 transition-all duration-500"
            :style="{ width: `${job.progress}%` }"
          />
        </div>

        <!-- Error detail -->
        <div
          v-if="expandedError === job.job_id && job.error"
          class="mt-3 p-3 bg-red-500/10 border border-red-500/30 rounded-lg"
        >
          <p class="text-xs font-medium text-red-400 mb-1">Error</p>
          <pre class="text-xs text-red-300 whitespace-pre-wrap break-words font-mono">{{ job.error }}</pre>
        </div>
      </div>
    </div>
  </div>
</template>
