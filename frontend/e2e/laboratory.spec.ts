import { expect, test } from '@playwright/test'
import AxeBuilder from '@axe-core/playwright'

test('welcome, roles, real paginated records, Python and learning check', async ({ page }) => {
  const errors: string[] = []
  page.on('pageerror', (error) => errors.push(error.message))
  await page.goto('/')
  await expect(page.getByRole('heading', { name: /Dos dados à sua/ })).toBeVisible()
  await expect(page.getByLabel('Resumo da base real')).toContainText('10.000')
  await page.getByRole('button', { name: 'Começar a explorar' }).click()
  await page.getByLabel('Coluna do dataset').selectOption('target')
  await expect(page.getByText('Alvo · bloqueado como entrada')).toBeVisible()
  await page.getByLabel('Coluna do dataset').selectOption('sample_id')
  await expect(page.getByText('Identificador · bloqueado')).toBeVisible()
  await expect(page.getByRole('cell', { name: 'S08268', exact: true })).toBeVisible()
  await page.getByRole('button', { name: 'Próxima página', exact: true }).click()
  await expect(page.getByText('11–20 de 10.000 registros')).toBeVisible()
  await expect(page.getByRole('cell', { name: 'S08268', exact: true })).toHaveCount(0)
  await page.getByText('Ver o Python', { exact: true }).click()
  await expect(page.locator('pre')).toContainText('X = df.drop(columns=["sample_id", "target"])')
  await page.getByRole('button', { name: 'A resposta vaza para o modelo' }).click()
  await expect(page.getByText(/Isso mesmo!/)).toBeVisible()
  expect(errors).toEqual([])
})

test('EDA controls request different real columns and histogram bin counts', async ({ page }) => {
  await page.goto('/#etapa-2')
  await expect(page.getByRole('img', { name: /Distribuição de sensor_a/ })).toBeVisible()
  await page.getByLabel('Atributo', { exact: true }).selectOption('sensor_b')
  await expect(page.getByRole('img', { name: /Distribuição de sensor_b/ })).toBeVisible()
  await expect(page.getByText('1.192', { exact: true })).toBeVisible()
  const responsePromise = page.waitForResponse((response) => response.url().includes('column=sensor_b&bins=30'))
  await page.getByLabel('Intervalos (bins)').selectOption('30')
  const response = await responsePromise
  const histogram = await response.json()
  expect(histogram.histogram).toHaveLength(30)
  expect(histogram.histogram.reduce((sum: number, item: { count: number }) => sum + item.count, 0)).toBe(8808)
  await expect(page.locator('.recharts-bar-rectangle')).toHaveCount(30)
  await page.getByLabel('Atributo', { exact: true }).selectOption('measurement_i')
  await expect(page.getByRole('img', { name: /Distribuição de measurement_i/ })).toBeVisible()
  await expect(page.getByText('1.526', { exact: true })).toBeVisible()
  await expect(page.getByRole('img', { name: /Boxplot de measurement_i/ })).toBeVisible()
  await page.getByLabel('Atributo', { exact: true }).selectOption('environment')
  await expect(page.getByRole('img', { name: /Distribuição de environment/ })).toBeVisible()
  await expect(page.getByText('598', { exact: true })).toBeVisible()
  await expect(page.getByLabel('Intervalos (bins)')).toHaveCount(0)
})

test('missing values and correlation matrix are inspectable', async ({ page }) => {
  await page.goto('/#etapa-2')
  await page.getByRole('button', { name: 'Dados faltantes', exact: true }).click()
  const rows = page.getByRole('list', { name: 'Ausências por atributo' }).getByRole('listitem')
  await expect(rows.first()).toContainText('measurement_i')
  await expect(rows.first()).toContainText('15,26%')
  await page.getByRole('button', { name: 'Maior percentual primeiro' }).click()
  await expect(rows.first()).toContainText('sensor_a')
  await page.getByRole('button', { name: 'Correlações', exact: true }).click()
  await expect(page.locator('.heatmap tbody tr')).toHaveCount(10)
  await expect(page.locator('.heatmap td button')).toHaveCount(100)
  await page.getByRole('button', { name: /sensor_b e sensor_a:/ }).click()
  await expect(page.locator('.correlation-detail')).toContainText('sensor_b × sensor_a')
  await expect(page.locator('.correlation-detail')).toContainText('pares válidos')
})

test('all nine stages are implemented and navigable', async ({ page }) => {
  await page.goto('/')
  const navigation = page.getByRole('navigation', { name: 'Etapas do laboratório' })
  await expect(navigation.getByRole('button')).toHaveCount(9)
  for (let step = 1; step <= 9; step++) {
    await navigation.getByRole('button').nth(step - 1).click()
    await expect(page.getByRole('main').getByText('A implementar', { exact: true })).toHaveCount(0)
    await expect(page.getByRole('main').getByRole('heading', { level: 1 })).toBeVisible()
  }
  await expect(page.getByRole('heading', { level: 1 })).toContainText('Uma prova final')
})

test('unavailable backend gives a recoverable error, without invented data', async ({ page }) => {
  await page.route('**/api/dataset/profile', (route) => route.abort())
  await page.goto('/')
  await expect(page.getByRole('alert')).toContainText('Não foi possível carregar')
  await expect(page.getByLabel('Resumo da base real')).toHaveCount(0)
  await page.unroute('**/api/dataset/profile')
  await page.getByRole('button', { name: 'Tentar novamente' }).click()
  await expect(page.getByLabel('Resumo da base real')).toBeVisible()
})

test('mobile layout contains tables and charts without overflowing the page', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto('/')
  await expect(page.getByLabel('Resumo da base real')).toBeVisible()
  await page.getByRole('button', { name: 'Começar a explorar' }).click()
  await expect(page.getByRole('table', { name: /Registros/ })).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
  await page.getByRole('button', { name: 'Explorar os dados', exact: true }).click()
  await expect(page.getByRole('img', { name: /Distribuição de sensor_a/ })).toBeVisible()
  await page.getByRole('button', { name: 'Correlações', exact: true }).click()
  await expect(page.locator('.heatmap td button')).toHaveCount(100)
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
  await page.getByRole('button', { name: 'Entenda o porquê', exact: true }).click()
  await expect(page.getByRole('complementary', { name: 'Entenda o porquê' })).toBeVisible()
  await page.getByRole('button', { name: 'Fechar ajuda', exact: true }).click()
  await expect(page.getByRole('complementary', { name: 'Entenda o porquê' })).toBeHidden()
})

test('keyboard navigation and browser history work', async ({ page }) => {
  await page.goto('/')
  await page.keyboard.press('Tab')
  await expect(page.getByRole('link', { name: 'Pular para o conteúdo' })).toBeFocused()
  await page.getByRole('button', { name: 'Começar a explorar' }).click()
  await expect(page.getByRole('main')).toBeFocused()
  await page.getByRole('button', { name: 'Explorar os dados', exact: true }).click()
  await expect(page).toHaveURL(/#etapa-2$/)
  await page.goBack()
  await expect(page.getByRole('heading', { level: 1 })).toContainText('Entenda o problema')
})

test('capture the running application', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByLabel('Resumo da base real')).toBeVisible()
  await page.screenshot({ path: 'test-results/inicio-desktop.png', fullPage: true })
  await page.goto('/#etapa-2')
  await expect(page.getByRole('img', { name: /Distribuição de sensor_a/ })).toBeVisible()
  await page.screenshot({ path: 'test-results/eda-desktop.png', fullPage: true })
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto('/')
  await expect(page.getByLabel('Resumo da base real')).toBeVisible()
  await page.screenshot({ path: 'test-results/inicio-mobile.png', fullPage: true })
})

test('available pages meet automated WCAG AA checks', async ({ page }) => {
  for (const hash of ['', '#etapa-1', '#etapa-2']) {
    await page.goto(`/${hash}`)
    await expect(page.getByRole('heading', { level: 1 })).toBeVisible()
    if (hash === '#etapa-2') await expect(page.getByRole('img', { name: /Distribuição de sensor_a/ })).toBeVisible()
    const results = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze()
    expect(results.violations.map((violation) => ({
      id: violation.id,
      nodes: violation.nodes.map((node) => ({ target: node.target, summary: node.failureSummary })),
    }))).toEqual([])
  }
})
