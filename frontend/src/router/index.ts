import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  {
    path: '/',
    name: 'home',
    component: () => import('@/views/HomeView.vue'),
    meta: { title: 'Dashboard' },
  },
  {
    path: '/create-task',
    name: 'create-task',
    component: () => import('@/views/CreateTaskView.vue'),
    meta: { title: 'Create Task' },
  },
  {
    path: '/annotate',
    name: 'annotate',
    component: () => import('@/views/AnnotateView.vue'),
    meta: { title: 'Annotate', fullscreen: true, hasUnsavedChanges: true },
  },
  {
    path: '/auto-labeling',
    name: 'auto-labeling',
    component: () => import('@/views/AutoLabelingView.vue'),
    meta: { title: 'Auto Labeling' },
  },
  {
    path: '/annotation-review',
    name: 'annotation-review',
    component: () => import('@/views/AnnotationReviewView.vue'),
    meta: { title: 'Annotation Review' },
  },
  {
    path: '/label-manager',
    name: 'label-manager',
    component: () => import('@/views/LabelManagerView.vue'),
    meta: { title: 'Label Manager' },
  },
  {
    path: '/sam3-tools',
    name: 'sam3-tools',
    component: () => import('@/views/Sam3ToolsView.vue'),
    meta: { title: 'SAM3 Tools' },
  },
  {
    path: '/jobs',
    name: 'jobs',
    component: () => import('@/views/JobsView.vue'),
    meta: { title: 'Jobs' },
  },
  {
    path: '/comparison',
    name: 'comparison',
    component: () => import('@/views/ComparisonView.vue'),
    meta: { title: 'Comparison' },
  },
  {
    path: '/settings',
    name: 'settings',
    component: () => import('@/views/SettingsView.vue'),
    meta: { title: 'Settings' },
  },
  {
    path: '/:pathMatch(.*)*',
    name: 'not-found',
    component: () => import('@/views/NotFoundView.vue'),
    meta: { title: 'Not Found' },
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

router.beforeEach((to, from, next) => {
  const title = to.meta.title as string | undefined
  document.title = title ? `${title} | Annotation Tool` : 'Annotation Tool'

  if (from.meta.hasUnsavedChanges) {
    const hasChanges = document.querySelector('[data-unsaved="true"]')
    if (hasChanges) {
      const confirmed = window.confirm('You have unsaved changes. Are you sure you want to leave?')
      if (!confirmed) { next(false); return }
    }
  }

  next()
})

export default router
