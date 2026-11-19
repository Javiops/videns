// Geometría de la fila de oro, compartida por el panel y la página /faces.
// La fila (ROW.w x ROW.h px) muestra un rectángulo del splash (SPLASH.w x SPLASH.h px de arte) con la
// misma proporción que la fila. El rectángulo lo coloca y dimensiona el owner en /faces, uno por campeón
// y el mismo para las dos columnas; en RED solo cambian el lado de la cifra y el sentido del fundido.
// Se guarda en config/faces.json como {x, y, w}: centro y anchura como fracciones del splash (la altura
// sale de la proporción de la fila, así que cambiar ROW no invalida las marcas). Las marcas antiguas
// {x, y} (solo cabeza) se convierten a la franja que daban: desde la cabeza hasta el borde derecho.
// El arte se pinta como <img class="art"> posicionada dentro de la fila (no como fondo CSS): así un
// fallo de carga tiene onerror, deja ver las iniciales y se reintenta; un fondo CSS que falló una vez
// no se vuelve a pedir nunca.
const Rows = (() => {
  const SPLASH = { w: 1215, h: 717 };
  const ROW = { w: 150, h: 34 };          // debe coincidir con .prow en rows.css
  const ASPECT = ROW.w / ROW.h;
  const HEAD_MARGIN = 0.4;                // solo para convertir marcas antiguas de cabeza
  const MIN_W = 0.12, MAX_W = 1.0;        // anchura del rectángulo como fracción del splash
  const DEFAULT_RECT = { x: 0.5, y: 0.35, w: 0.5 };

  function num(v) { return v === null || v === undefined || v === "" || !Number.isFinite(+v) ? null : +v; }
  // marca guardada -> {x, y, w} normalizado (centro y anchura, fracciones del splash)
  function mark(m) {
    if (m && num(m.x) !== null && num(m.y) !== null) {
      if (num(m.w) !== null) return { x: +m.x, y: +m.y, w: Math.min(MAX_W, Math.max(MIN_W, +m.w)) };
      // marca antigua: cabeza en (x, y); franja desde la cabeza hasta el borde derecho
      const hx = m.x * SPLASH.w, hy = m.y * SPLASH.h;
      const hb = Math.max(60, (SPLASH.w - hx) / (ASPECT - HEAD_MARGIN));
      const width = hb * ASPECT, left = hx - HEAD_MARGIN * hb;
      return { x: (left + width / 2) / SPLASH.w, y: hy / SPLASH.h, w: Math.min(MAX_W, Math.max(MIN_W, width / SPLASH.w)) };
    }
    return { ...DEFAULT_RECT };
  }
  function rect(m) {                      // rectángulo del splash, en px de arte, que muestra la fila
    const c = mark(m);
    const width = c.w * SPLASH.w, height = width / ASPECT;
    return { left: c.x * SPLASH.w - width / 2, top: c.y * SPLASH.h - height / 2, width, height, scale: ROW.w / width };
  }
  function artStyle(m) {                  // estilo inline de la <img class="art">: tamaño y posición dentro de la fila
    const r = rect(m), s = r.scale;
    return `height:${(SPLASH.h * s).toFixed(2)}px;left:${(-r.left * s).toFixed(2)}px;top:${(-r.top * s).toFixed(2)}px`;
  }
  // v: versión del arte (fecha del splash propio, 0 si no lo hay); cambia la URL para que el navegador
  // no siga enseñando un arte anterior que tenía cacheado
  function artUrl(iconId, v) { return `/img/splash/${encodeURIComponent(iconId)}.jpg?v=${Number(v) || 0}`; }
  return { SPLASH, ROW, ASPECT, MIN_W, MAX_W, DEFAULT_RECT, mark, rect, artStyle, artUrl };
})();
