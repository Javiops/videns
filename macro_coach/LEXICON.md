# LEXICON v1 — vocabulario cerrado de decisiones de jungla

Contrato del dataset: **toda decisión etiquetada es uno de estos tokens**. Un
token solo existe si es observable con las fuentes validadas (Match-V5 +
minimapa de la replay). Nada de intenciones inventadas: etiquetamos lo que el
jungla challenger **hizo**, no lo que "quería hacer".

Deriva de la taxonomía v1 auditada del prototipo (13 regiones, acciones de
`macro_v5.py`). Los tokens son estables y en mayúsculas: el modelo entrena
sobre ellos; los nombres bonitos para el revisor viven en `app/lexicon.py`.

## Tokens

| Token | Significado | Regla de detección (observable) |
| --- | --- | --- |
| `RESET` | Vuelve a base a comprar | Patrón de recall auditado (quieto ≥2 muestras y salto a base) o segmento BASE tras muerte |
| `FARMEAR_PROPIA_AZUL` | Farmea su jungla, cuadrante azul | Segmento en `JGL_PA` con rombo muerto en ventana [-5,+15] o `jungleMinionsKilled` +3 |
| `FARMEAR_PROPIA_ROJA` | Farmea su jungla, cuadrante rojo | Ídem en `JGL_PR` |
| `INVADIR_AZUL` | Entra a jungla enemiga, lado azul | Segmento en `JGL_EA` (farmee o no: pisar jungla enemiga ya es la decisión) |
| `INVADIR_ROJA` | Entra a jungla enemiga, lado rojo | Ídem en `JGL_ER` |
| `GANK_TOP` / `GANK_MID` / `GANK_BOT` | Gank a un carril | Carril con rival detectado cerca (<0.15) o kill/muerte anclada, en mitad enemiga (cx+cy ≥ 0.97); o pelea en jungla con víctima de ese carril |
| `PELEAR_TOP` / `PELEAR_MID` / `PELEAR_BOT` | Pelea en carril, lado propio | Ídem pero cx+cy < 0.97 (countergank / defensa) |
| `CUBRIR_TOP` / `CUBRIR_MID` / `CUBRIR_BOT` | Pisa un carril sin rival visto | Segmento en carril ≥8s sin rival cerca (empujar oleada, cubrir) |
| `CENTINELA_TOP` / `CENTINELA_BOT` | Va a por el centinela | Segmento en río con `jungleMinionsKilled` +1..2 en la ventana |
| `ROTAR_RIO_TOP` / `ROTAR_RIO_BOT` | Cruza/rota por el río sin CS | Segmento en río sin subida de CS de jungla |
| `OBJETIVO_DRAGON` | Juega el dragón (setup/toma/disputa) | Segmento en `PIT_DRAGON`, o `ELITE_MONSTER_KILL` DRAGON con participación |
| `OBJETIVO_LARVAS` | Juega las larvas | Segmento en `PIT_BARON` antes del min 14, o kill HORDE con participación |
| `OBJETIVO_HERALDO` | Juega el heraldo | Segmento en `PIT_BARON` min 14–20, o kill RIFTHERALD con participación |
| `OBJETIVO_BARON` | Juega el Barón | Segmento en `PIT_BARON` desde min 20, o kill BARON_NASHOR con participación |

25 tokens. Fuera de vocabulario v1 (documentado, no improvisar):

- **Visión y wards: fuera de alcance.** Ni `WARD_PLACED` (llega sin coordenadas),
  ni detección de iconos, ni features de wards. La visibilidad entra al estado
  únicamente como el **efecto neto** que muestran las pasadas con niebla.
- **Transitar no es decisión**: un segmento `cruza X` hereda la etiqueta del
  siguiente segmento real (la decisión se tomó al emprender el camino). Solo si
  el tránsito es jungla enemiga se conserva `INVADIR_*` (pisarla ya es decisión).
- Peleas en foso (`pelea en foso de Dragón/Barón`) colapsan en su `OBJETIVO_*`:
  disputar el foso es jugar el objetivo. Larvas/Heraldo/Barón se desambiguan por
  evento si lo hay y por reloj si no.
- Muerte no es decisión: un hueco por muerte no genera token; el siguiente
  segmento vivo sí.

## Punto de decisión

Un punto de decisión es **el inicio de cada segmento etiquetable** del jungla.
El snapshot del estado se toma en `t0 − 1s` (el estado con el que decidió, no el
resultado). Tokens consecutivos idénticos se colapsan (seguir farmeando el mismo
cuadrante no es una decisión nueva por campamento).

## Versionado

Cambiar un token, una regla o el colapso de tránsitos = subir versión
(`LEXICON_V` en `app/lexicon.py`) y regenerar `decisions.jsonl` de todo el
corpus. Nunca mezclar versiones en un mismo dataset de entrenamiento.
