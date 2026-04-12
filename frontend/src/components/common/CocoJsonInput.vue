<script setup lang="ts">
import { ref } from 'vue'
import { uploadCocoJson } from '@/lib/api'
import BaseButton from '@/components/ui/BaseButton.vue'
import { FolderOpen, Upload, CheckCircle2 } from 'lucide-vue-next'

const props = defineProps<{
  modelValue: string
  label: string
  taskName: string
  placeholder?: string
}>()

const emit = defineEmits<{ 'update:modelValue': [value: string] }>()

const mode = ref<'path' | 'upload'>('path')
const fileInput = ref<HTMLInputElement | null>(null)
const uploadState = ref<'idle' | 'uploading' | 'done' | 'error'>('idle')
const uploadError = ref<string | null>(null)
const fileName = ref('')

async function handleFileChange(event: Event) {
  const file = (event.target as HTMLInputElement).files?.[0]
  if (!file) return
  fileName.value = file.name
  uploadState.value = 'uploading'
  uploadError.value = null
  try {
    const { path } = await uploadCocoJson(props.taskName, file)
    emit('update:modelValue', path)
    uploadState.value = 'done'
  } catch (e: any) {
    uploadError.value = e.response?.data?.detail ?? e.message ?? 'Upload failed'
    uploadState.value = 'error'
  }
}
</script>

<template>
  <div class="space-y-2">
    <div class="flex items-center justify-between">
      <label class="block text-sm font-medium text-gray-300">
        <FolderOpen class="h-4 w-4 inline mr-1" />
        {{ label }}
      </label>
      <div class="flex gap-0.5 bg-background-tertiary rounded-md p-0.5 text-xs">
        <button
          :class="['px-2 py-1 rounded transition-colors', mode === 'path' ? 'bg-primary text-white' : 'text-gray-400 hover:text-white']"
          @click="mode = 'path'"
        >Path</button>
        <button
          :class="['px-2 py-1 rounded transition-colors', mode === 'upload' ? 'bg-primary text-white' : 'text-gray-400 hover:text-white']"
          @click="mode = 'upload'"
        >Upload</button>
      </div>
    </div>

    <input
      v-if="mode === 'path'"
      :value="modelValue"
      @input="emit('update:modelValue', ($event.target as HTMLInputElement).value)"
      type="text"
      :placeholder="placeholder"
      class="input w-full"
    />

    <div v-else class="space-y-2">
      <input ref="fileInput" type="file" accept=".json" class="hidden" @change="handleFileChange" />
      <div class="flex items-center gap-2">
        <BaseButton variant="outline" size="sm" :loading="uploadState === 'uploading'" @click="fileInput?.click()">
          <Upload class="h-4 w-4" />
          {{ uploadState === 'uploading' ? 'Uploading...' : 'Select JSON' }}
        </BaseButton>
        <span v-if="uploadState === 'done'" class="text-xs text-green-400 flex items-center gap-1">
          <CheckCircle2 class="h-3.5 w-3.5" /> {{ fileName }}
        </span>
        <span v-if="fileName && uploadState === 'uploading'" class="text-xs text-gray-400">{{ fileName }}</span>
      </div>
      <input
        v-if="modelValue && uploadState === 'done'"
        :value="modelValue"
        type="text"
        readonly
        class="input w-full text-xs text-gray-400 bg-background-tertiary"
      />
      <p v-if="uploadState === 'error'" class="text-xs text-red-400">{{ uploadError }}</p>
    </div>
  </div>
</template>
