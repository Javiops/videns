# videns.gg

Herramientas y proyectos de investigación de League of Legends detrás de videns.gg. Cada carpeta es un proyecto independiente con su propio README, `requirements.txt` y venv.

| Carpeta | Qué es |
| --- | --- |
| `build_tracker/` | Asesor de tienda: qué compraban jugadores de high elo en estados de tienda parecidos. |
| `macro_coach/` | Macro-coaching post-partida a partir de la replay y Match-V5. |
| `live_overlay/` | Overlay en vivo con lo que el cliente del juego expone en local. |
| `winprob/` | Investigación de probabilidad de victoria retrospectiva y revisión de decisiones. |
| `draft/` | Réplica y crítica de la receta de win-probability de draft de itero.gg. |
| `brand/` | Logo e iconos (la marca V.). |
| `informes/` | Informes HTML fechados. |

La web está en su propio repo: [Javiops/videns-site](https://github.com/Javiops/videns-site).

## Lo que no está aquí

- **Datos:** corpus, bases de datos, exports, modelos y artefactos (~445 GB en local). Solo están los ficheros de `data/` que cada proyecto ya versionaba.
- **Claves:** los `.env` no se suben. Copia el `.env.example` de cada proyecto a `.env` y pon tu clave de Riot.
