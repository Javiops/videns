// Bounded end-to-end test of our local app in an isolated headless browser.
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('C:/Users/Javier/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root = path.resolve(__dirname, '..');
const output = path.join(root, 'data', 'reports');
fs.mkdirSync(output, { recursive: true });

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1280, height: 1000 } });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto('http://127.0.0.1:8011/', { waitUntil: 'domcontentloaded', timeout: 15000 });
    const links = page.locator('a[href^="/game/"]');
    const count = await links.count();
    if (!count) throw new Error('No permitted review links rendered');
    const href = await links.first().getAttribute('href');
    await links.first().click();
    await page.locator('#wp-chart svg').waitFor({ state: 'visible', timeout: 15000 });
    const before = await page.locator('#head').innerText();
    const jump = page.locator('#wp-falls button:enabled');
    const windows = await jump.count();
    if (!windows) throw new Error('No available observed-window navigation');
    await jump.first().click();
    const after = await page.locator('#head').innerText();
    if (before === after) throw new Error('Window navigation did not change the decision');
    const marker = await page.locator('#wp-current').innerText();
    if (!marker.includes('Último punto anterior')) throw new Error('No preceding-point marker after navigation');
    await page.locator('#wp-blue').click();
    const status = await page.locator('#wp-status').innerText();
    if (!status.startsWith('Equipo azul')) throw new Error('Blue perspective toggle failed');
    const screenshot = path.join(output, 'macro_review_browser.png');
    await page.locator('section.wp').screenshot({ path: screenshot });
    // A nonexistent ID tests the guard without requesting any actual sealed ID.
    const blocked = await page.request.get('http://127.0.0.1:8011/api/decisions/NOT_AN_ADMITTED_GAME/1');
    if (blocked.status() !== 404) throw new Error('Non-admitted route did not reject before a corpus request');
    if (errors.length) throw new Error('Browser errors: ' + errors.join('; '));
    const report = { status: 'passed', url: 'http://127.0.0.1:8011' + href,
      permitted_perspective_links: count, observed_window_buttons: windows,
      decision_before: before, decision_after: after, marker, status_text: status,
      non_admitted_response: blocked.status(), browser_errors: errors,
      feedback_submitted: false, screenshot };
    fs.writeFileSync(path.join(output, 'macro_review_browser.json'), JSON.stringify(report, null, 2));
    process.stdout.write(JSON.stringify(report, null, 2));
  } finally {
    await browser.close();
  }
})().catch(error => { process.stderr.write(error.stack + '\n'); process.exitCode = 1; });
