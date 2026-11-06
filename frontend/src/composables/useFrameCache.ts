import { getImageUrl } from '@/lib/api'

/**
 * LRU cache of preloaded frame images for smooth timeline scrubbing.
 * Prefetches a window of frames around the current position.
 */
export function useFrameCache(maxSize = 60) {
  const cache = new Map<string, HTMLImageElement>()

  function get(path: string): HTMLImageElement {
    const hit = cache.get(path)
    if (hit) {
      // Refresh LRU position
      cache.delete(path)
      cache.set(path, hit)
      return hit
    }
    const img = new Image()
    img.src = getImageUrl(path)
    cache.set(path, img)
    while (cache.size > maxSize) {
      const oldest = cache.keys().next().value
      if (oldest === undefined) break
      cache.delete(oldest)
    }
    return img
  }

  function prefetch(paths: string[], centerIndex: number, radius = 10) {
    const start = Math.max(0, centerIndex - radius)
    const end = Math.min(paths.length - 1, centerIndex + radius)
    for (let i = start; i <= end; i++) get(paths[i])
  }

  function clear() {
    cache.clear()
  }

  return { get, prefetch, clear }
}
