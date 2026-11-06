import { ref, computed } from 'vue'
import { getTracks, renameTrack, deleteTrack, mergeTracks, loadAnnotations } from '@/lib/api'
import type { CocoAnnotation, CocoCategory, CocoImage, TrackSummary, VideoMeta } from '@/types/api'

export interface VideoCocoImage extends CocoImage {
  video_id?: number
  frame_index?: number
}

export interface VideoCocoAnnotation extends CocoAnnotation {
  track_id?: number
  score?: number
}

const TRACK_COLORS = [
  '#ef4444', '#f97316', '#eab308', '#22c55e', '#06b6d4',
  '#3b82f6', '#8b5cf6', '#ec4899', '#f43f5e', '#14b8a6',
]

/**
 * State for a video-tracking COCO dataset: frames, per-frame annotations,
 * track summaries and track-level operations (rename/delete/merge).
 */
export function useVideoTracks() {
  const cocoJsonPath = ref('')
  const frames = ref<VideoCocoImage[]>([])
  const annotations = ref<VideoCocoAnnotation[]>([])
  const categories = ref<CocoCategory[]>([])
  const tracks = ref<TrackSummary[]>([])
  const videoMeta = ref<VideoMeta | null>(null)
  const loading = ref(false)

  const annotationsByFrame = computed(() => {
    const map = new Map<number, VideoCocoAnnotation[]>()
    for (const ann of annotations.value) {
      const list = map.get(ann.image_id) ?? []
      list.push(ann)
      map.set(ann.image_id, list)
    }
    return map
  })

  function trackColor(trackId: number): string {
    return TRACK_COLORS[Math.abs(trackId) % TRACK_COLORS.length]
  }

  async function load(path: string) {
    loading.value = true
    try {
      cocoJsonPath.value = path
      const [data, trackData] = await Promise.all([loadAnnotations(path), getTracks(path)])
      frames.value = [...(data.images as VideoCocoImage[])].sort(
        (a, b) => (a.frame_index ?? a.id) - (b.frame_index ?? b.id),
      )
      annotations.value = data.annotations
      categories.value = data.categories
      tracks.value = trackData.tracks
      videoMeta.value = trackData.videos[0] ?? null
    } finally {
      loading.value = false
    }
  }

  async function refresh() {
    if (cocoJsonPath.value) await load(cocoJsonPath.value)
  }

  async function rename(trackId: number, newLabel: string) {
    await renameTrack(cocoJsonPath.value, trackId, newLabel)
    await refresh()
  }

  async function remove(trackId: number) {
    await deleteTrack(cocoJsonPath.value, trackId)
    await refresh()
  }

  async function merge(sourceId: number, targetId: number) {
    await mergeTracks(cocoJsonPath.value, sourceId, targetId)
    await refresh()
  }

  return {
    cocoJsonPath, frames, annotations, categories, tracks, videoMeta, loading,
    annotationsByFrame, trackColor,
    load, refresh, rename, remove, merge,
  }
}
