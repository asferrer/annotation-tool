<script setup lang="ts">
import { ref } from 'vue'
import { Check, GitMerge, Pencil, Trash2, X } from 'lucide-vue-next'
import type { TrackSummary } from '@/types/api'

const props = defineProps<{
  tracks: TrackSummary[]
  trackColor: (id: number) => string
  selectedTrackId: number | null
}>()

const emit = defineEmits<{
  (e: 'select', trackId: number | null): void
  (e: 'rename', trackId: number, newLabel: string): void
  (e: 'delete', trackId: number): void
  (e: 'merge', sourceId: number, targetId: number): void
}>()

const editingId = ref<number | null>(null)
const editingLabel = ref('')
const mergeSourceId = ref<number | null>(null)

function startRename(track: TrackSummary) {
  editingId.value = track.track_id
  editingLabel.value = track.label
}

function confirmRename() {
  if (editingId.value !== null && editingLabel.value.trim()) {
    emit('rename', editingId.value, editingLabel.value.trim())
  }
  editingId.value = null
}

function toggleMerge(trackId: number) {
  if (mergeSourceId.value === null) {
    mergeSourceId.value = trackId
  } else if (mergeSourceId.value === trackId) {
    mergeSourceId.value = null
  } else {
    emit('merge', mergeSourceId.value, trackId)
    mergeSourceId.value = null
  }
}
</script>

<template>
  <div class="space-y-1">
    <p v-if="mergeSourceId !== null" class="text-xs text-amber-400 px-1">
      Merge: select the target track (or click again to cancel)
    </p>
    <div
      v-for="track in props.tracks"
      :key="track.track_id"
      :class="['group flex items-center gap-2 rounded-lg px-2 py-1.5 cursor-pointer border',
        props.selectedTrackId === track.track_id ? 'border-primary bg-primary/10' : 'border-transparent hover:bg-background-tertiary',
        mergeSourceId === track.track_id ? 'ring-1 ring-amber-400' : '']"
      @click="emit('select', props.selectedTrackId === track.track_id ? null : track.track_id)"
    >
      <span class="w-3 h-3 rounded-full flex-shrink-0" :style="{ backgroundColor: props.trackColor(track.track_id) }" />

      <template v-if="editingId === track.track_id">
        <input
          v-model="editingLabel"
          class="flex-1 min-w-0 bg-background-tertiary border border-gray-600 rounded px-1.5 py-0.5 text-xs text-white"
          @keyup.enter="confirmRename"
          @keyup.escape="editingId = null"
          @click.stop
        />
        <button class="p-1 text-green-400" @click.stop="confirmRename"><Check class="h-3.5 w-3.5" /></button>
        <button class="p-1 text-gray-500" @click.stop="editingId = null"><X class="h-3.5 w-3.5" /></button>
      </template>

      <template v-else>
        <div class="flex-1 min-w-0">
          <p class="text-xs text-gray-200 truncate">#{{ track.track_id }} · {{ track.label }}</p>
          <p class="text-[10px] text-gray-500">
            frames {{ track.first_frame }}–{{ track.last_frame }} ({{ track.frame_count }}) · score {{ track.avg_score.toFixed(2) }}
          </p>
        </div>
        <div class="flex opacity-0 group-hover:opacity-100 transition-opacity">
          <button class="p-1 text-gray-400 hover:text-white" title="Rename" @click.stop="startRename(track)"><Pencil class="h-3.5 w-3.5" /></button>
          <button class="p-1 text-gray-400 hover:text-amber-400" title="Merge" @click.stop="toggleMerge(track.track_id)"><GitMerge class="h-3.5 w-3.5" /></button>
          <button class="p-1 text-gray-400 hover:text-red-400" title="Delete" @click.stop="emit('delete', track.track_id)"><Trash2 class="h-3.5 w-3.5" /></button>
        </div>
      </template>
    </div>
    <p v-if="props.tracks.length === 0" class="text-xs text-gray-500 px-1">No tracks yet.</p>
  </div>
</template>
