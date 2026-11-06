<script setup lang="ts">
import { computed, onUnmounted, ref } from 'vue'
import { useUiStore } from '@/stores/ui'
import {
  uploadVideo, probeVideo, extractFrames, startTracking, getVideoJob, cancelVideoJob, getImageUrl,
} from '@/lib/api'
import { useVideoTracks } from '@/composables/useVideoTracks'
import { useFrameCache } from '@/composables/useFrameCache'
import BaseButton from '@/components/ui/BaseButton.vue'
import DirectoryBrowser from '@/components/common/DirectoryBrowser.vue'
import ExportDialog from '@/components/export/ExportDialog.vue'
import VideoTimeline from '@/components/video/VideoTimeline.vue'
import TrackList from '@/components/video/TrackList.vue'
import {
  Clapperboard, Download, FolderOpen, Loader2, Play, Upload, Wand2, X,
} from 'lucide-vue-next'
import type { VideoInfo, VideoUploadResult } from '@/types/api'

const uiStore = useUiStore()

// ---------------------------------------------------------------------------
// Phase 1: source + tracking setup
// ---------------------------------------------------------------------------
type Phase = 'setup' | 'tracking' | 'review'
const phase = ref<Phase>('setup')

const sourceMode = ref<'upload' | 'server' | 'existing'>('upload')
const uploaded = ref<VideoUploadResult | null>(null)
const uploadProgress = ref(0)
const isUploading = ref(false)
const serverVideoPath = ref('')
const existingCocoPath = ref('')
const videoInfo = ref<VideoInfo | null>(null)

const promptsInput = ref('')
const minConfidence = ref(0.5)
const targetFps = ref<number | null>(null)
const isExtracting = ref(false)
const extractedInfo = ref<{ frames_dir: string; extracted_frames: number } | null>(null)

const trackingJobId = ref<string | null>(null)
const trackingProgress = ref({ processed: 0, total: 0 })
let pollTimer: ReturnType<typeof setInterval> | null = null

const videoPath = computed(() => uploaded.value?.video_path || serverVideoPath.value)

async function onFileSelected(e: Event) {
  const input = e.target as HTMLInputElement
  const file = input.files?.[0]
  if (!file) return
  isUploading.value = true
  try {
    uploaded.value = await uploadVideo(file, (p) => (uploadProgress.value = p))
    videoInfo.value = await probeVideo(uploaded.value.video_path)
    uiStore.showSuccess('Video subido', `${videoInfo.value.frame_count} frames @ ${videoInfo.value.fps} fps`)
  } catch (err: any) {
    uiStore.showError('Upload', err?.response?.data?.detail ?? err?.message ?? 'Error')
  } finally {
    isUploading.value = false
  }
}

async function probeServerVideo() {
  if (!serverVideoPath.value.trim()) return
  try {
    videoInfo.value = await probeVideo(serverVideoPath.value.trim())
  } catch (err: any) {
    uiStore.showError('Probe', err?.response?.data?.detail ?? err?.message ?? 'Error')
  }
}

async function runTracking() {
  const path = videoPath.value
  if (!path || !promptsInput.value.trim()) return
  const prompts = promptsInput.value.split(',').map((p) => p.trim()).filter(Boolean)

  isExtracting.value = true
  try {
    const extraction = await extractFrames({
      video_path: path,
      target_fps: targetFps.value ?? undefined,
    })
    extractedInfo.value = extraction
    const outputJson = extraction.frames_dir.replace(/\/frames\/?$/, '') + '/annotations.json'

    const job = await startTracking({
      frames_dir: extraction.frames_dir,
      prompts,
      min_confidence: minConfidence.value,
      output_json: outputJson,
      video_name: uploaded.value?.original_name || path.split('/').pop(),
      source_video_path: path,
      fps: extraction.effective_fps,
      frame_stride: extraction.frame_stride,
    })
    trackingJobId.value = job.job_id
    trackingProgress.value = { processed: 0, total: job.total_frames }
    phase.value = 'tracking'
    pollTimer = setInterval(pollTracking, 2500)
  } catch (err: any) {
    uiStore.showError('Tracking', err?.response?.data?.detail ?? err?.message ?? 'Error')
  } finally {
    isExtracting.value = false
  }
}

async function pollTracking() {
  if (!trackingJobId.value) return
  try {
    const job = await getVideoJob(trackingJobId.value)
    trackingProgress.value = {
      processed: job.processed_items ?? 0,
      total: job.total_items ?? trackingProgress.value.total,
    }
    if (job.status === 'completed') {
      stopPolling()
      const summary = (job.result_summary ?? {}) as { output_json?: string }
      await openReview(summary.output_json || job.output_path || '')
    } else if (['failed', 'cancelled', 'interrupted'].includes(job.status)) {
      stopPolling()
      phase.value = 'setup'
      uiStore.showError('Tracking', job.error_message || `Job ${job.status}`)
    }
  } catch { /* transient */ }
}

async function cancelTracking() {
  if (trackingJobId.value) await cancelVideoJob(trackingJobId.value)
}

function stopPolling() {
  if (pollTimer !== null) { clearInterval(pollTimer); pollTimer = null }
}

// ---------------------------------------------------------------------------
// Phase 2: review (frames + tracks)
// ---------------------------------------------------------------------------
const tracksState = useVideoTracks()
const frameCache = useFrameCache(80)
const currentFrameIndex = ref(0)
const selectedTrackId = ref<number | null>(null)
const playing = ref(false)
const showExport = ref(false)
let playTimer: ReturnType<typeof setInterval> | null = null

const currentFrame = computed(() => tracksState.frames.value[currentFrameIndex.value] ?? null)
const currentAnnotations = computed(() => {
  const frame = currentFrame.value
  if (!frame) return []
  const anns = tracksState.annotationsByFrame.value.get(frame.id) ?? []
  return selectedTrackId.value === null
    ? anns
    : anns.filter((a) => a.track_id === selectedTrackId.value)
})

async function openReview(cocoPath: string) {
  if (!cocoPath) {
    uiStore.showError('Video', 'The job did not return an annotations.json')
    return
  }
  await tracksState.load(cocoPath)
  currentFrameIndex.value = 0
  phase.value = 'review'
}

async function openExisting() {
  if (!existingCocoPath.value.trim()) return
  await openReview(existingCocoPath.value.trim())
}

function seek(frame: number) {
  currentFrameIndex.value = Math.min(Math.max(frame, 0), tracksState.frames.value.length - 1)
  frameCache.prefetch(tracksState.frames.value.map((f) => f.file_name), currentFrameIndex.value)
}

function togglePlay() {
  playing.value = !playing.value
  if (playing.value) {
    const fps = tracksState.videoMeta.value?.fps || 10
    playTimer = setInterval(() => {
      if (currentFrameIndex.value >= tracksState.frames.value.length - 1) {
        playing.value = false
        if (playTimer) { clearInterval(playTimer); playTimer = null }
        return
      }
      seek(currentFrameIndex.value + 1)
    }, 1000 / Math.min(fps, 30))
  } else if (playTimer) {
    clearInterval(playTimer)
    playTimer = null
  }
}

async function handleRename(trackId: number, label: string) {
  try { await tracksState.rename(trackId, label) } catch (err: any) {
    uiStore.showError('Track', err?.response?.data?.detail ?? 'Error')
  }
}

async function handleDelete(trackId: number) {
  try {
    await tracksState.remove(trackId)
    if (selectedTrackId.value === trackId) selectedTrackId.value = null
  } catch (err: any) {
    uiStore.showError('Track', err?.response?.data?.detail ?? 'Error')
  }
}

async function handleMerge(sourceId: number, targetId: number) {
  try { await tracksState.merge(sourceId, targetId) } catch (err: any) {
    uiStore.showError('Track', err?.response?.data?.detail ?? 'Error')
  }
}

function polygonPoints(segmentation: number[][] | undefined): string[] {
  if (!segmentation) return []
  return segmentation.map((poly) => {
    const pts: string[] = []
    for (let i = 0; i < poly.length - 1; i += 2) pts.push(`${poly[i]},${poly[i + 1]}`)
    return pts.join(' ')
  })
}

onUnmounted(() => {
  stopPolling()
  if (playTimer) clearInterval(playTimer)
})
</script>

<template>
  <div class="space-y-4">
    <div class="flex items-center justify-between">
      <div>
        <h2 class="text-2xl font-bold text-white flex items-center gap-2">
          <Clapperboard class="h-6 w-6 text-primary" /> Video Annotation
        </h2>
        <p class="mt-1 text-sm text-gray-400">Automatic tracking with SAM3 Video: upload a video, describe the objects and review the tracks.</p>
      </div>
      <div v-if="phase === 'review'" class="flex gap-2">
        <BaseButton variant="outline" size="sm" @click="showExport = true"><Download class="h-4 w-4" /> Export</BaseButton>
        <BaseButton variant="outline" size="sm" @click="phase = 'setup'">New video</BaseButton>
      </div>
    </div>

    <!-- ============ SETUP ============ -->
    <template v-if="phase === 'setup'">
      <div class="card p-5 space-y-4">
        <div class="flex gap-1 bg-background-tertiary rounded-lg p-1 max-w-xl">
          <button v-for="mode in (['upload','server','existing'] as const)" :key="mode"
            :class="['flex-1 px-3 py-2 rounded-md text-sm font-medium transition-colors', sourceMode === mode ? 'bg-primary text-white' : 'text-gray-400 hover:text-white']"
            @click="sourceMode = mode">
            {{ mode === 'upload' ? 'Upload video' : mode === 'server' ? 'Server video' : 'Existing tracking' }}
          </button>
        </div>

        <template v-if="sourceMode === 'upload'">
          <label class="block border-2 border-dashed border-gray-600 hover:border-gray-500 rounded-xl p-8 text-center cursor-pointer">
            <Upload class="h-10 w-10 mx-auto mb-3 text-gray-500" />
            <p class="text-sm text-gray-300">Select a video (mp4, avi, mov, mkv, webm)</p>
            <input type="file" accept="video/*" class="hidden" @change="onFileSelected" />
          </label>
          <div v-if="isUploading" class="flex items-center gap-2 text-sm text-gray-300">
            <Loader2 class="h-4 w-4 animate-spin text-primary" /> Subiendo... {{ uploadProgress }}%
          </div>
        </template>

        <template v-if="sourceMode === 'server'">
          <DirectoryBrowser v-model="serverVideoPath" label="Video file" placeholder="/app/datasets/videos/mi-video.mp4" :show-files="true" file-pattern="*.mp4" path-mode="input" :restrict-to-mounts="false" @select="probeServerVideo" />
          <BaseButton variant="outline" size="sm" :disabled="!serverVideoPath" @click="probeServerVideo">Check video</BaseButton>
        </template>

        <template v-if="sourceMode === 'existing'">
          <DirectoryBrowser v-model="existingCocoPath" label="annotations.json de un tracking previo" placeholder="/app/datasets/videos/<id>/annotations.json" :show-files="true" file-pattern="*.json" path-mode="input" :restrict-to-mounts="false" />
          <BaseButton variant="primary" size="sm" :disabled="!existingCocoPath" @click="openExisting"><FolderOpen class="h-4 w-4" /> Open</BaseButton>
        </template>

        <div v-if="videoInfo && sourceMode !== 'existing'" class="text-xs text-gray-400 bg-background-tertiary rounded-lg px-3 py-2">
          {{ videoInfo.width }}×{{ videoInfo.height }} · {{ videoInfo.fps }} fps · {{ videoInfo.frame_count }} frames · {{ videoInfo.duration_s }}s
        </div>
      </div>

      <div v-if="sourceMode !== 'existing'" class="card p-5 space-y-4">
        <h3 class="text-sm font-semibold text-gray-300 flex items-center gap-2"><Wand2 class="h-4 w-4 text-primary" /> Tracking SAM3</h3>
        <div>
          <label class="text-sm text-gray-400 mb-1 block">Objects to track (text, comma-separated)</label>
          <input v-model="promptsInput" type="text" placeholder="person, car, dog..." class="w-full text-sm bg-background-tertiary border border-gray-600 rounded-lg px-3 py-2 text-white placeholder-gray-500 focus:outline-none focus:border-primary" />
        </div>
        <div class="grid grid-cols-2 gap-4 max-w-lg">
          <div>
            <label class="text-sm text-gray-400 mb-1 block">Minimum confidence: {{ minConfidence.toFixed(2) }}</label>
            <input v-model.number="minConfidence" type="range" min="0.1" max="0.9" step="0.05" class="w-full" />
          </div>
          <div>
            <label class="text-sm text-gray-400 mb-1 block">Sampling FPS (empty = all)</label>
            <input v-model.number="targetFps" type="number" min="0.1" step="0.5" placeholder="p.ej. 5" class="w-full text-sm bg-background-tertiary border border-gray-600 rounded-lg px-3 py-2 text-white" />
          </div>
        </div>
        <BaseButton variant="primary" :disabled="!videoPath || !promptsInput.trim() || isExtracting" :loading="isExtracting" @click="runTracking">
          <Play class="h-4 w-4" /> Extract frames and track
        </BaseButton>
      </div>
    </template>

    <!-- ============ TRACKING ============ -->
    <div v-else-if="phase === 'tracking'" class="card p-8 text-center space-y-4">
      <Loader2 class="h-10 w-10 animate-spin text-primary mx-auto" />
      <p class="text-white font-medium">Tracking objects with SAM3 Video…</p>
      <p class="text-sm text-gray-400">{{ trackingProgress.processed }} / {{ trackingProgress.total }} frames</p>
      <div class="w-full max-w-md mx-auto bg-gray-700 rounded-full h-2">
        <div class="bg-primary h-2 rounded-full transition-all" :style="{ width: (trackingProgress.total ? (trackingProgress.processed / trackingProgress.total) * 100 : 0) + '%' }" />
      </div>
      <BaseButton variant="outline" size="sm" @click="cancelTracking"><X class="h-4 w-4" /> Cancel</BaseButton>
    </div>

    <!-- ============ REVIEW ============ -->
    <template v-else-if="phase === 'review'">
      <div class="grid grid-cols-1 lg:grid-cols-4 gap-4">
        <div class="lg:col-span-3 space-y-3">
          <div class="card p-3">
            <div class="relative bg-black rounded-lg overflow-hidden flex items-center justify-center" style="min-height: 300px">
              <img v-if="currentFrame" :src="getImageUrl(currentFrame.file_name)" class="max-w-full max-h-[60vh] object-contain" draggable="false" />
              <svg v-if="currentFrame" class="absolute inset-0 w-full h-full pointer-events-none" :viewBox="`0 0 ${currentFrame.width} ${currentFrame.height}`" preserveAspectRatio="xMidYMid meet">
                <g v-for="ann in currentAnnotations" :key="ann.id">
                  <polygon
                    v-for="(points, pi) in polygonPoints(ann.segmentation)"
                    :key="pi"
                    :points="points"
                    :fill="tracksState.trackColor(ann.track_id ?? 0) + '44'"
                    :stroke="tracksState.trackColor(ann.track_id ?? 0)"
                    stroke-width="2"
                  />
                  <rect
                    v-if="!ann.segmentation?.length"
                    :x="ann.bbox[0]" :y="ann.bbox[1]" :width="ann.bbox[2]" :height="ann.bbox[3]"
                    fill="none" :stroke="tracksState.trackColor(ann.track_id ?? 0)" stroke-width="2"
                  />
                  <text :x="ann.bbox[0]" :y="Math.max(ann.bbox[1] - 4, 12)" :fill="tracksState.trackColor(ann.track_id ?? 0)" font-size="14" font-weight="bold">
                    #{{ ann.track_id }}
                  </text>
                </g>
              </svg>
            </div>
          </div>
          <div class="card p-3">
            <VideoTimeline
              :frame-count="tracksState.frames.value.length"
              :current-frame="currentFrameIndex"
              :tracks="tracksState.tracks.value"
              :track-color="tracksState.trackColor"
              :fps="tracksState.videoMeta.value?.fps"
              :playing="playing"
              @seek="seek"
              @toggle-play="togglePlay"
            />
          </div>
        </div>

        <div class="card p-3 space-y-3 self-start">
          <h3 class="text-sm font-semibold text-gray-300">Tracks ({{ tracksState.tracks.value.length }})</h3>
          <TrackList
            :tracks="tracksState.tracks.value"
            :track-color="tracksState.trackColor"
            :selected-track-id="selectedTrackId"
            @select="selectedTrackId = $event"
            @rename="handleRename"
            @delete="handleDelete"
            @merge="handleMerge"
          />
          <div class="border-t border-gray-700/50 pt-3">
            <p class="text-[11px] text-gray-500 break-all">COCO: {{ tracksState.cocoJsonPath.value }}</p>
            <p class="text-[11px] text-gray-600 mt-1">To edit geometry frame by frame, open this JSON in Annotation Review.</p>
          </div>
        </div>
      </div>

      <ExportDialog
        :open="showExport"
        :coco-json-path="tracksState.cocoJsonPath.value"
        :video-dataset="true"
        @close="showExport = false"
      />
    </template>
  </div>
</template>
