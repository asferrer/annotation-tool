<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { useUiStore } from '@/stores/ui'
import { evaluateComparison } from '@/lib/api'
import BaseButton from '@/components/ui/BaseButton.vue'
import AlertBox from '@/components/common/AlertBox.vue'
import CocoJsonInput from '@/components/common/CocoJsonInput.vue'
import { BarChart3, Play } from 'lucide-vue-next'

const route = useRoute()
const uiStore = useUiStore()

const gtPath = ref('')
const predPath = ref('')
const iouThreshold = ref(0.5)
const loading = ref(false)
const error = ref<string | null>(null)
const results = ref<any>(null)

// Pre-fill paths when navigating from the Jobs view (?pred_path=...&gt_path=...)
onMounted(() => {
  const qPred = route.query.pred_path as string | undefined
  const qGt = route.query.gt_path as string | undefined
  if (qPred) predPath.value = qPred
  if (qGt) gtPath.value = qGt
})

async function runEvaluation() {
  if (!gtPath.value || !predPath.value) {
    error.value = 'Select both Ground Truth and Predictions JSON files'
    return
  }
  loading.value = true
  error.value = null
  results.value = null
  try {
    results.value = await evaluateComparison({
      ground_truth_path: gtPath.value,
      predictions_path: predPath.value,
      iou_threshold: iouThreshold.value,
    })
    uiStore.showInfo('Evaluation Complete', `${results.value.per_category.length} categories evaluated`)
  } catch (e: any) {
    error.value = e.response?.data?.detail || e.message || 'Evaluation failed'
  } finally {
    loading.value = false
  }
}

function metricColor(value: number): string {
  if (value >= 0.7) return 'text-green-400'
  if (value >= 0.4) return 'text-yellow-400'
  return 'text-red-400'
}

function pct(value: number): string {
  return (value * 100).toFixed(1) + '%'
}
</script>

<template>
  <div class="space-y-6">
    <div>
      <h2 class="text-2xl font-bold text-white">Annotation Comparison</h2>
      <p class="mt-1 text-gray-400">Compare ground truth vs auto-generated COCO annotations</p>
    </div>

    <AlertBox v-if="error" type="error" :title="error" dismissible @dismiss="error = null" />

    <!-- Config -->
    <div class="card p-6 space-y-4">
      <h3 class="text-lg font-semibold text-white">Configuration</h3>

      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <CocoJsonInput
          v-model="gtPath"
          label="Ground Truth COCO JSON"
          task-name="comparison_gt"
          placeholder="/app/datasets/annotations.json"
        />
        <CocoJsonInput
          v-model="predPath"
          label="Predictions COCO JSON"
          task-name="comparison_pred"
          placeholder="/app/datasets/output/annotations.json"
        />
      </div>

      <div class="flex items-end gap-4">
        <div class="w-48">
          <label class="block text-sm font-medium text-gray-300 mb-1">IoU Threshold</label>
          <div class="flex items-center gap-2">
            <input
              v-model.number="iouThreshold"
              type="range"
              min="0.1"
              max="1.0"
              step="0.05"
              class="flex-1"
            />
            <span class="text-sm text-white font-mono w-10">{{ iouThreshold.toFixed(2) }}</span>
          </div>
        </div>

        <BaseButton :loading="loading" @click="runEvaluation">
          <Play class="h-4 w-4" />
          Evaluate
        </BaseButton>
      </div>
    </div>

    <!-- Results -->
    <template v-if="results">
      <!-- Summary -->
      <div class="card p-6">
        <h3 class="text-lg font-semibold text-white mb-4">
          <BarChart3 class="h-5 w-5 inline mr-2" />
          Global Metrics (IoU &ge; {{ results.iou_threshold }})
        </h3>

        <div class="grid grid-cols-2 md:grid-cols-5 gap-4 mb-4">
          <div class="text-center">
            <p class="text-2xl font-bold" :class="metricColor(results.global_metrics.precision)">
              {{ pct(results.global_metrics.precision) }}
            </p>
            <p class="text-xs text-gray-400">Precision</p>
          </div>
          <div class="text-center">
            <p class="text-2xl font-bold" :class="metricColor(results.global_metrics.recall)">
              {{ pct(results.global_metrics.recall) }}
            </p>
            <p class="text-xs text-gray-400">Recall</p>
          </div>
          <div class="text-center">
            <p class="text-2xl font-bold" :class="metricColor(results.global_metrics.f1)">
              {{ pct(results.global_metrics.f1) }}
            </p>
            <p class="text-xs text-gray-400">F1 Score</p>
          </div>
          <div class="text-center">
            <p class="text-2xl font-bold text-white">{{ results.global_metrics.tp }}</p>
            <p class="text-xs text-gray-400">True Positives</p>
          </div>
          <div class="text-center">
            <p class="text-2xl font-bold text-gray-400">{{ results.matched_images }}/{{ results.total_gt_images }}</p>
            <p class="text-xs text-gray-400">Matched Images</p>
          </div>
        </div>

        <div class="flex gap-4 text-sm text-gray-400">
          <span>FP: {{ results.global_metrics.fp }}</span>
          <span>FN: {{ results.global_metrics.fn }}</span>
          <span>Pred images: {{ results.total_pred_images }}</span>
        </div>
      </div>

      <!-- Per category table -->
      <div class="card p-6">
        <h3 class="text-lg font-semibold text-white mb-4">Per-Category Metrics</h3>

        <div class="overflow-x-auto">
          <table class="w-full text-sm">
            <thead>
              <tr class="border-b border-gray-700 text-gray-400">
                <th class="text-left py-2 px-3">Category</th>
                <th class="text-right py-2 px-3">TP</th>
                <th class="text-right py-2 px-3">FP</th>
                <th class="text-right py-2 px-3">FN</th>
                <th class="text-right py-2 px-3">Precision</th>
                <th class="text-right py-2 px-3">Recall</th>
                <th class="text-right py-2 px-3">F1</th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="cat in results.per_category"
                :key="cat.category_id"
                class="border-b border-gray-700/50 hover:bg-gray-700/20"
              >
                <td class="py-2 px-3 text-white font-medium">{{ cat.category_name }}</td>
                <td class="py-2 px-3 text-right text-gray-300">{{ cat.tp }}</td>
                <td class="py-2 px-3 text-right text-gray-300">{{ cat.fp }}</td>
                <td class="py-2 px-3 text-right text-gray-300">{{ cat.fn }}</td>
                <td class="py-2 px-3 text-right font-mono" :class="metricColor(cat.precision)">{{ pct(cat.precision) }}</td>
                <td class="py-2 px-3 text-right font-mono" :class="metricColor(cat.recall)">{{ pct(cat.recall) }}</td>
                <td class="py-2 px-3 text-right font-mono" :class="metricColor(cat.f1)">{{ pct(cat.f1) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </template>
  </div>
</template>
