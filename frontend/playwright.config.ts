import { defineConfig, devices } from '@playwright/test'
import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const storage = mkdtempSync(join(process.env.ANN_LAB_TEST_TMP ?? tmpdir(), 'ann-lab-e2e-'))

export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  workers: 2,
  forbidOnly: Boolean(process.env.CI),
  retries: 0,
  expect: { timeout: 15000 },
  reporter: 'list',
  use: {
    baseURL: 'http://127.0.0.1:5174',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  projects: [{
    name: 'chrome',
    use: { ...devices['Desktop Chrome'], viewport: { width: 1440, height: 1000 }, channel: 'chrome' },
  }],
  webServer: [
    {
      command: '../backend/.venv/bin/python -m uvicorn app.main:app --app-dir ../backend --host 127.0.0.1 --port 8001',
      url: 'http://127.0.0.1:8001/api/health',
      reuseExistingServer: false,
      env: { ANN_LAB_STORAGE: storage },
      timeout: 60000,
    },
    {
      command: 'node node_modules/vite/bin/vite.js --host 127.0.0.1 --port 5174 --strictPort',
      url: 'http://127.0.0.1:5174',
      reuseExistingServer: false,
      env: { ANN_LAB_API_URL: 'http://127.0.0.1:8001' },
    },
  ],
})
