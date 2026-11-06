<script setup lang="ts">
import { computed, ref } from 'vue'
import { Pause, Play, SkipBack, SkipForward } from 'lucide-vue-next'
import type { TrackSummary } from '@/types/api'

const props = defineProps<{
  frameCount: number
  currentFrame: number
  tracks: TrackSummary[]
  trackColor: (id: number) => string
  fps?: number
  playing: boolean
}>()

const emit = defineEmits<{
  (e: 'seek', frame: number): void
  (e: 'toggle-play'): void
}>()

const barRef = ref<HTMLDivElement | null>(null)
const scrubbing = ref(false)

const progressPct = computed(() =>
  props.frameCount > 1 ? (props.currentFrame / (props.frameCount - 1)) * 100 : 0,
)

function frameFromEvent(e: MouseEvent): number {
  const bar = barRef.value
  if (!bar || props.frameCount === 0) return 0
  const rect = bar.getBoundingClientRect()
  const ratio = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width))
  return Math.round(ratio * (props.frameCount - 1))
}

function onPointerDown(e: MouseEvent) {
  scrubbing.value = true
  emit('seek', frameFromEvent(e))
  const onMove = (ev: MouseEvent) => emit('seek', frameFromEvent(ev))
  const onUp = () => {
    scrubbing.value = false
    window.removeEventListener('mousemove', onMove)
    window.removeEventListener('mouseup', onUp)
  }
  window.addEventListener('mousemove', onMove)
  window.addEventListener('mouseup', onUp)
}

function presenceStyle(track: TrackSummary) {
  const total = Math.max(props.frameCount - 1, 1)
  const left = (track.first_frame / total) * 100
  const width = Math.max(((track.last_frame - track.first_frame) / total) * 100, 0.8)
  return {
    left: `${left}%`,
    width: `${width}%`,
    backgroundColor: props.trackColor(track.track_id),
  }
}

const timeLabel = computed(() => {
  if (!props.fps || props.fps <= 0) return `${props.currentFrame + 1} / ${props.frameCount}`
  const seconds = props.currentFrame / props.fps
  return `${seconds.toFixed(2)}s · ${props.currentFrame + 1}/${props.frameCount}`
})
</script>

<template>
  <div class="space-y-1.5 select-none">
    <div class="flex items-center gap-2">
      <button class="p-1.5 text-gray-300 hover:text-white" @click="emit('seek', Math.max(0, props.currentFrame - 1))"><SkipBack class="h-4 w-4" /></button>
      <button class="p-1.5 text-gray-300 hover:text-white" @click="emit('toggle-play')">
        <component :is="props.playing ? Pause : Play" class="h-4 w-4" />
      </button>
      <button class="p-1.5 text-gray-300 hover:text-white" @click="emit('seek', Math.min(props.frameCount - 1, props.currentFrame + 1))"><SkipForward class="h-4 w-4" /></button>
      <span class="text-xs text-gray-400 tabular-nums">{{ timeLabel }}</span>
    </div>

    <!-- Scrubber -->
    <div ref="barRef" class="relative h-3 bg-background-tertiary rounded cursor-pointer" @mousedown="onPointerDown">
      <div class="absolute inset-y-0 left-0 bg-primary/30 rounded-l" :style="{ width: progressPct + '%' }" />
      <div class="absolute top-0 bottom-0 w-0.5 bg-primary" :style="{ left: progressPct + '%' }" />
    </div>

    <!-- Presence bars per track -->
    <div class="space-y-0.5">
      <div v-for="track in props.tracks" :key="track.track_id" class="relative h-1.5 bg-background-tertiary/50 rounded">
        <div class="absolute inset-y-0 rounded opacity-80" :style="presenceStyle(track)" />
      </div>
    </div>
  </div>
</template>
