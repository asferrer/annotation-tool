<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { useUiStore } from '@/stores/ui'
import {
  LayoutDashboard, PenSquare, Plus, Settings,
  ChevronLeft, ChevronRight, ListChecks, Clapperboard, BarChart3,
} from 'lucide-vue-next'

const route = useRoute()
const uiStore = useUiStore()
const { t } = useI18n()

const navItems = [
  { label: 'Dashboard', path: '/', icon: LayoutDashboard },
  { label: 'New Task', path: '/create-task', icon: Plus },
  { label: 'Video', path: '/video', icon: Clapperboard },
  { label: 'Jobs', path: '/jobs', icon: ListChecks },
  { label: 'Comparison', path: '/comparison', icon: BarChart3 },
  { label: 'Settings', path: '/settings', icon: Settings },
]

const isActive = (path: string) => route.path === path

const sidebarClasses = computed(() => [
  'fixed inset-y-0 left-0 z-50 flex flex-col bg-background-secondary border-r border-gray-700/50',
  'transition-all duration-300 ease-in-out',
  uiStore.sidebarCollapsed ? 'w-20' : 'w-64',
])
</script>

<template>
  <aside :class="sidebarClasses" class="hidden lg:flex">
    <div class="flex h-16 items-center justify-between border-b border-gray-700/50 px-4">
      <router-link to="/" class="flex items-center gap-3">
        <div class="flex h-10 w-10 items-center justify-center rounded-xl bg-primary">
          <PenSquare class="h-6 w-6 text-white" />
        </div>
        <span v-if="!uiStore.sidebarCollapsed" class="font-semibold text-white">AnnTool</span>
      </router-link>
    </div>

    <nav class="flex-1 overflow-y-auto px-3 py-4 space-y-1">
      <router-link v-for="item in navItems" :key="item.path" :to="item.path"
        :class="['flex items-center gap-3 rounded-lg px-3 py-2.5 transition-colors',
          isActive(item.path) ? 'bg-primary text-white' : 'text-gray-400 hover:bg-gray-700/50 hover:text-white']">
        <component :is="item.icon" class="h-5 w-5 flex-shrink-0" />
        <span v-if="!uiStore.sidebarCollapsed">{{ item.label }}</span>
      </router-link>
    </nav>

    <div class="border-t border-gray-700/50 p-3">
      <button @click="uiStore.toggleSidebar" class="flex w-full items-center justify-center gap-2 rounded-lg px-3 py-2 text-gray-400 hover:bg-gray-700/50 hover:text-white transition-colors">
        <ChevronLeft v-if="!uiStore.sidebarCollapsed" class="h-5 w-5" />
        <ChevronRight v-else class="h-5 w-5" />
        <span v-if="!uiStore.sidebarCollapsed">Collapse</span>
      </button>
    </div>
  </aside>
</template>
