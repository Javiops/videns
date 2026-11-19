// Rasterise the V. SVGs with the installed Chrome: each size is drawn from the vector, never downscaled.
import puppeteer from "puppeteer-core";
import { readFileSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const OUT = join(dirname(fileURLToPath(import.meta.url)), "out");
const JOBS = [
  ...[16, 24, 32, 48, 64, 128, 192, 256, 512, 1024].map((n) => ["v-mark.svg", n, `v-mark-${n}.png`]),
  ["v-mark-maskable.svg", 180, "apple-touch-icon.png"],
  ["v-mark-maskable.svg", 512, "v-mark-maskable-512.png"],
  ["v-mark-glyph.svg", 512, "v-mark-glyph-512.png"],
];
const browser = await puppeteer.launch({
  executablePath: "C:/Program Files/Google/Chrome/Application/chrome.exe",
  headless: true, userDataDir: mkdtempSync(join(tmpdir(), "icons-")),
});
const page = await browser.newPage();
for (const [svg, n, png] of JOBS) {
  const data = Buffer.from(readFileSync(join(OUT, svg))).toString("base64");
  await page.setViewport({ width: n, height: n, deviceScaleFactor: 1 });
  await page.setContent(`<html><body style="margin:0;background:transparent"><img src="data:image/svg+xml;base64,${data}" width="${n}" height="${n}" style="display:block"></body></html>`);
  await page.evaluate(() => document.images[0].decode());
  await page.screenshot({ path: join(OUT, png), omitBackground: true, clip: { x: 0, y: 0, width: n, height: n } });
}
await browser.close();
console.log(`rendered ${JOBS.length} PNGs`);
