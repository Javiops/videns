# Videns brand: the V. mark

The icon is the letter V plus the dot, drawn exactly like the VIDENS. wordmark: Inter Black (weight
900, SIL OFL, the site's own font file), stretched to 118% wide and leaned left by 20.97deg, the slant
the stretch gives the V's right stroke, so that stroke is vertical. The dot is neither stretched nor
leaned, and is tucked in after the stroke.

Colours: the tile is the site's purple gradient (135deg, #dcaaff -> #b556ff at 48% -> #6a1ccf), the
letter and the dot #f6f4fb. The bare glyph keeps the wordmark's colours: letter #f6f4fb, dot #b556ff.

- `v-mark.svg`: rounded gradient tile, the master (tab favicon, app icon).
- `v-mark-maskable.svg`: full-bleed gradient square, mark inside the 80% safe circle (Android/iOS masks).
- `v-mark-glyph.svg`: the bare mark, transparent, for dark surfaces.
- `videns-app.ico`: Windows app icon, 16/24/32/48/64/128/256.
- PNGs up to 1024 for stores and installers.

`src/` regenerates everything: `make_icons.py` (fontTools + brotli) writes the SVGs from the font,
`render_icons.mjs` (puppeteer-core + Chrome) rasterises them, `pack_ico.py` builds the .ico files.
Their paths point at the session scratchpad they were written in; adjust `OUT`/`FONT` to reuse.
