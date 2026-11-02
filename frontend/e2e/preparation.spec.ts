import { expect, test, type Page, type Response } from '@playwright/test'
import AxeBuilder from '@axe-core/playwright'
import type { PreprocessingPreview } from '../src/types'

function previewResponse(page: Page, matches: (data: Record<string, unknown>) => boolean) {
  return page.waitForResponse((response: Response) => {
    if (!response.url().endsWith('/api/preprocessing/preview') || response.request().method() !== 'POST') return false
    return matches(response.request().postDataJSON())
  })
}

test('imputation controls and comparison display actual backend calculations', async ({ page }) => {
  await page.goto('/#etapa-3')
  await expect(page.getByRole('status')).toContainText('Prévia calculada')
  const initial = await page.locator('.fitted-value').innerText()
  const responsePromise = previewResponse(page, (data) => data.missing_strategy === 'mean')
  await page.getByRole('button', { name: /^Média Sensível/ }).click()
  const response = await responsePromise
  expect(response.status()).toBe(200)
  const result: PreprocessingPreview = await response.json()
  await expect(page.locator('.fitted-value')).not.toHaveText(initial)
  await expect(page.locator('.fitted-value')).toContainText('Média aprendida no treino')
  await expect(page.getByRole('region', { name: 'Amostras antes e depois' })).toContainText(
    result.numeric_before_after[0].samples[0].imputed.toLocaleString('pt-BR', { maximumFractionDigits: 6 }),
  )
  expect(result.fitted_on).toBe('train_only')
  expect(result.test_transformed).toBe(false)
  await page.getByLabel('Estratégia para comparar').selectOption('median')
  await expect(page.getByRole('region', { name: 'Comparação entre estratégias' })).toContainText('Média')
  await expect(page.getByRole('region', { name: 'Comparação entre estratégias' })).toContainText('Mediana')
  await page.getByText('Ver o Python', { exact: true }).click()
  await expect(page.locator('pre')).toContainText("'missing_strategy': 'mean'")
  await expect(page.locator('pre')).toContainText('pipeline.fit_transform(train[FEATURES])')
})

test('KNN neighbors and row removal have observable real effects', async ({ page }) => {
  await page.goto('/#etapa-3')
  await expect(page.locator('.preparation-status')).toBeVisible()
  const knnPromise = previewResponse(page, (data) => data.missing_strategy === 'knn')
  await page.getByRole('button', { name: /^KNN Vizinhos/ }).click()
  const knn: PreprocessingPreview = await (await knnPromise).json()
  await expect(page.locator('.preparation-status')).toBeVisible()
  await expect(page.getByText(/KNN usa padronização interna/)).toBeVisible()
  const neighborsPromise = previewResponse(page, (data) => data.knn_neighbors === 3)
  await page.getByLabel('Vizinhos do KNN').selectOption('3')
  const three: PreprocessingPreview = await (await neighborsPromise).json()
  expect(three.numeric_before_after[0].samples.map((row) => row.imputed)).not.toEqual(
    knn.numeric_before_after[0].samples.map((row) => row.imputed),
  )
  const dropPromise = previewResponse(page, (data) => data.missing_strategy === 'drop')
  await page.getByRole('button', { name: /^Remover Só/ }).click()
  const drop: PreprocessingPreview = await (await dropPromise).json()
  expect(drop.rows_after_filter).toBeLessThan(10000)
  expect(drop.removed_rows + drop.rows_after_filter).toBe(10000)
  await expect(page.getByRole('region', { name: 'Cobertura da coorte' })).toContainText(
    drop.rows_after_filter.toLocaleString('pt-BR'),
  )
  await expect(page.getByRole('heading', { name: 'A população do estudo mudou' })).toBeVisible()
  await expect(page.getByRole('region', { name: 'Cobertura por classe' })).toBeVisible()
})

test('shared choices, categorical encoding, scaling and split survive navigation and reload', async ({ page }) => {
  await page.goto('/#etapa-3')
  await expect(page.locator('.preparation-status')).toBeVisible()
  const categoryPromise = previewResponse(page, (data) => data.categorical_strategy === 'constant')
  await page.getByLabel('Faltantes de environment').selectOption('constant')
  const category: PreprocessingPreview = await (await categoryPromise).json()
  expect(category.output_feature_count).toBe(14)
  await expect(page.locator('.encoded-categories')).toContainText('environment_missing')
  await page.getByRole('button', { name: 'Ajustar as escalas', exact: true }).click()
  const minmaxPromise = previewResponse(page, (data) => data.scaler === 'minmax')
  await page.getByRole('button', { name: /^MinMaxScaler/ }).click()
  const minmax: PreprocessingPreview = await (await minmaxPromise).json()
  expect(minmax.numeric_before_after[0].scaled_stats.min).toBeCloseTo(0)
  expect(minmax.numeric_before_after[0].scaled_stats.max).toBeCloseTo(1)
  await expect(page.getByRole('region', { name: 'Amostras antes e depois' }).getByRole('columnheader', { name: 'Após escala' })).toBeVisible()
  await page.getByRole('button', { name: 'Separar os dados', exact: true }).click()
  const splitPromise = previewResponse(page, (data) => data.split_strategy === '80_20')
  await page.getByRole('button', { name: /80 \/ 20 com validação interna/ }).click()
  const split: PreprocessingPreview = await (await splitPromise).json()
  expect(split.split_counts).toEqual({ train: 6400, val: 1600, test: 2000 })
  await expect(page.getByRole('region', { name: 'Contagem e classes por partição' })).toContainText('6.400')
  const seedPromise = previewResponse(page, (data) => data.seed === 43)
  await page.getByLabel('Seed reproduzível').fill('43')
  await page.getByRole('button', { name: 'Aplicar seed' }).click()
  const changed: PreprocessingPreview = await (await seedPromise).json()
  expect(changed.partition_fingerprints.train).not.toBe(split.partition_fingerprints.train)
  await page.reload()
  await expect(page.getByLabel('Seed reproduzível')).toHaveValue('43')
  await expect(page.getByRole('button', { name: /80 \/ 20 com validação interna/ })).toHaveAttribute('aria-pressed', 'true')
  await expect(page.locator('.current-preparation')).toContainText('MinMaxScaler')
  await page.getByRole('navigation').getByRole('button', { name: /Trate os faltantes/ }).click()
  await expect(page.getByLabel('Faltantes de environment')).toHaveValue('constant')
  await page.getByRole('button', { name: 'Restaurar padrões de preparação' }).click()
  await expect(page.getByLabel('Faltantes de environment')).toHaveValue('most_frequent')
  await expect(page.locator('.current-preparation')).toContainText('seed 42')
})

test('validation previews use train fit and backend errors do not retain misleading plots', async ({ page }) => {
  await page.goto('/#etapa-4')
  await expect(page.locator('.preparation-status')).toBeVisible()
  const valPromise = previewResponse(page, (data) => data.subset === 'val')
  await page.getByLabel('Visualizar subconjunto').selectOption('val')
  const val: PreprocessingPreview = await (await valPromise).json()
  expect(val.numeric_before_after[0].scaled_stats.observed).toBe(1500)
  expect(val.fitted_on).toBe('train_only')
  await expect(page.getByRole('heading', { name: 'sensor_b: antes × depois' })).toBeVisible()
  await page.route('**/api/preprocessing/preview', (route) => route.abort())
  await page.getByRole('button', { name: /^Sem escala/ }).click()
  await expect(page.getByRole('alert')).toContainText('Não foi possível carregar')
  await expect(page.getByRole('region', { name: 'Amostras antes e depois' })).toHaveCount(0)
  await page.unroute('**/api/preprocessing/preview')
  await page.getByRole('button', { name: 'Tentar novamente' }).click()
  await expect(page.getByRole('region', { name: 'Amostras antes e depois' })).toBeVisible()
  await expect(page.getByText('Sem escalonamento é um experimento didático adicional.')).toBeVisible()
})

test('preparation pages are accessible and usable on narrow screens', async ({ page }) => {
  for (const step of [3, 4, 5]) {
    await page.goto(`/#etapa-${step}`)
    await expect(page.locator('.preparation-status')).toBeVisible()
    const result = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze()
    expect(result.violations.map((violation) => ({
      id: violation.id,
      nodes: violation.nodes.map((node) => ({ target: node.target, summary: node.failureSummary })),
    }))).toEqual([])
  }
  for (const width of [390, 320]) {
    await page.setViewportSize({ width, height: 844 })
    for (const step of [3, 4, 5]) {
      await page.goto(`/#etapa-${step}`)
      await expect(page.locator('.preparation-status')).toBeVisible()
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
    }
  }
})

test('capture real preparation results', async ({ page }) => {
  for (const step of [3, 4, 5]) {
    await page.goto(`/#etapa-${step}`)
    await expect(page.locator('.preparation-status')).toBeVisible()
    await page.screenshot({ path: `test-results/etapa-${step}-desktop.png`, fullPage: true })
  }
  await page.setViewportSize({ width: 390, height: 844 })
  await page.screenshot({ path: 'test-results/divisao-mobile.png', fullPage: true })
})
