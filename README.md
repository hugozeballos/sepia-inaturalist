# sepia-inaturalist

Procesamiento de registros de *Sepia officinalis* en iNaturalist (costa del Mar del Norte) para el paper *Barnacles rafting on floating plastic litter and on cuttlebones* (Thiel, Gutow, Ricker et al.).

El objetivo es llenar la tabla de Martin: una fila por individuo y por lado (dorsal/ventral), con estado, calidad de foto, erosión, picoteo, epibiontes y picorocos.

## Principios

- **Una sola descarga.** Los datos se bajan una vez a Google Drive; todo lo demás trabaja desde ahí.
- **Copia cruda.** Cada respuesta de la API se guarda tal como llega (JSON).
- **JSON Lines para todo lo demás** (registros, fotos, anotaciones, salidas de modelos). Las anotaciones solo se agregan; vale la última.
- **Se trabaja en JSON Lines; el Excel es solo una exportación.** Se genera cuando se pide, nunca se edita. `build_excel` produce en cualquier momento la tabla con el formato de Martin: solo valores, coordenadas numéricas sin modificar.
- **Reanudable.** Si una corrida se corta, se vuelve a correr el mismo comando.
- **Código en git, datos en Drive.** El repositorio no contiene datos, fotos ni claves.

## Estructura

```
configs/        parámetros de cada corrida (YAML, versionados) y listas de IDs
scripts/        comandos que se ejecutan (download.py, triage.py, evaluate_triage.py, build_excel.py)
src/sepia/      funciones
  config.py     Config y lectura de YAML
  io.py         carpetas, JSON Lines, logs, manifest
  inat.py       descarga de registros y fotos, tablas, versiones reducidas
  export.py     Excel de entrega
  triage.py     triaje por registro con un modelo de visión (Gemini; Qwen después)
  prompts/      prompts versionados (triage_v1.txt, triage_v2.txt)
tests/          pruebas sin internet (API simulada)
docs/           documentación
```

Datos en Drive, en `MyDrive/sepia-inaturalist/data/<corrida>/` (el `base_dir` de cada config).
El código **no** va en esa carpeta: en Colab se clona en `/content/sepia-inaturalist`.

Contenido de cada corrida:

```
raw/api_pages/     copia cruda de la API
tables/            observations.jsonl, photos.jsonl
photos/original/   fotos originales  (<observation_id>_<photo_id>.jpg)
photos/512px/      para GPT
photos/1024px/     para la interfaz de anotación
annotations/       una .jsonl por anotador
outputs/           respuestas de los modelos (triage_<proveedor>_<modelo>.jsonl)
exports/           Excel generados
logs/              registro de cada paso
manifest.json      parámetros y fecha de descarga (para el párrafo de métodos)
```

## Uso en Colab

```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/<usuario>/sepia-inaturalist.git /content/sepia-inaturalist   # la primera vez
%cd /content/sepia-inaturalist
!git pull
!pip install -q -e .

!python scripts/download.py --config configs/pilot_a1.yaml
```

Para importar `sepia` dentro del notebook (no solo con `!python`), agregar la carpeta `src`
a la ruta: `pip install -e .` no se ve en el kernel que ya está corriendo hasta reiniciarlo.

```python
import sys
sys.path.insert(0, '/content/sepia-inaturalist/src')
for m in [m for m in sys.modules if m == 'sepia' or m.startswith('sepia.')]:
    del sys.modules[m]          # después de un git pull: descartar la versión vieja en memoria
import sepia
```

`importlib.reload(sepia)` no basta: recarga solo el paquete, no sus módulos (`config`, `triage`…).

Pasos por separado (en orden: `metadata, tables, photos, resize, check`). El Excel no se genera por defecto: es solo una exportación para Martin y se pide con `build_excel.py`.

```python
!python scripts/download.py --config configs/full.yaml --count-only
!python scripts/download.py --config configs/full.yaml --steps photos,resize,check
!python scripts/build_excel.py --config configs/pilot79.yaml
```

Si el repositorio es privado, clonar con un token guardado en los secretos de Colab (`GITHUB_TOKEN`):

```python
from google.colab import userdata
token = userdata.get('GITHUB_TOKEN')
!git clone https://{token}@github.com/<usuario>/sepia-inaturalist.git /content/sepia-inaturalist
```

El código se edita en un solo lugar (fuera de Colab), se hace commit y push, y en Colab solo `git pull`.

## Triaje con Gemini

Clasifica cada registro (todas sus fotos juntas, a 512 px): estado (hueso, vivo, huevos),
número de huesos distintos y qué muestra cada foto. Proveedor, modelo, temperatura y versión
del prompt van en la sección `triage` del YAML; cada respuesta se guarda completa con el hash
del prompt y los tokens usados.

Se saltan los registros ya clasificados **con la misma versión del prompt** (cambiar de versión
vuelve a clasificar todo). `--redo` fuerza a rehacer: agrega líneas nuevas, no borra; vale la última.

```python
import os
from google.colab import userdata
os.environ['GEMINI_API_KEY'] = userdata.get('GEMINI_API_KEY')   # secreto de Colab

!python scripts/triage.py --config configs/pilot_a1.yaml --limit 5   # prueba
!python scripts/triage.py --config configs/pilot_a1.yaml             # todo (reanudable)
!python scripts/triage.py --config configs/pilot_a1.yaml --ids 262274777 --redo   # rehacer uno
```

Casos de prueba con respuesta conocida (`configs/triage_hard_cases.jsonl`): varios huesos en una
foto, el mismo hueso en varias fotos, descripciones que dicen "cientos" con una sola foto.

```python
!python scripts/triage.py --config configs/pilot_a1.yaml --ids configs/triage_hard_cases.jsonl
!python scripts/evaluate_triage.py --config configs/pilot_a1.yaml
```

Reglas fijas aplicadas sobre la respuesta del modelo (`apply_rules`, quedan registradas en
`rules_applied`): más de 5 huesos → conteo no exacto (lo cuenta una persona); conteo no exacto →
la confianza no puede ser alta.


## Lado y erosión por foto (Gemini, sin ejemplos)

Paso aparte del triaje (no cambia el conteo por registro): una consulta por foto, a 1024 px,
con `prompts/photo_traits_v2.txt` (definiciones de la guía de Martin, hechas operativas). Pide huesos visibles y,
solo si hay **exactamente un hueso**, el lado, la erosión y la calidad de la foto (regla fija: con 0 o varios huesos,
`not_applicable`). Sección `photo_traits` del YAML; salida en
`outputs/photo_traits_<proveedor>_<modelo>.jsonl`. La interfaz de anotación no muestra estos
resultados.

```python
!python scripts/photo_traits.py --config configs/pilot_a1.yaml --limit 5   # prueba
!python scripts/photo_traits.py --config configs/pilot_a1.yaml             # todas (reanudable)
!python scripts/photo_traits.py --config configs/pilot_a1.yaml --evaluate  # vs anotación manual
```

La evaluación compara huesos visibles con *Individuals in photo* y, en fotos con un único
individuo anotado, el lado y la erosión (tabla de confusión).

## Anotación manual

Interfaz en el notebook (ipywidgets, ya incluido en Colab). Usa las fotos de 1024 px
(`download.py` las genera en el paso `resize`). No muestra resultados de modelos: la anotación
es a ciegas, para poder evaluar los modelos contra ella.

```python
import sys
sys.path.insert(0, '/content/sepia-inaturalist/src')
for m in [m for m in sys.modules if m == 'sepia' or m.startswith('sepia.')]: del sys.modules[m]
from sepia import load_config, AnnotationApp

cfg = load_config('/content/sepia-inaturalist/configs/pilot_a1.yaml')
app = AnnotationApp(cfg, annotator='HZ', round_=1)   # retoma en la primera foto pendiente
app.show()
```

**Individuals in photo** (a nivel de foto, 1 por defecto): cuántos huesos *se ven* en la foto,
anotados en detalle o no; sirve para comparar con el conteo por foto del triaje. En fotos con
muchos huesos: escribir el total y **Save count only ▶** (no hace falta anotar cada uno).
El estado viene por defecto como `cuttlebone`.

Flujo por foto: elegir la letra del individuo (las ya usadas en el registro aparecen en la
lista; una letra de otra foto es el mismo hueso), llenar los campos
y **Save ▶** (guarda y pasa a la siguiente foto) o **Save + another here** (otro individuo en la misma foto). Los campos de detalle solo aparecen
para huesos; los de picorocos solo con epibiontes = yes. Otros botones: *Photo without
individuals*, *Exclude record* (con motivo), *Delete individual*, anterior/siguiente, ir a un ID.

**Recuadros (opcional):** con `jupyter-bbox-widget` instalado (`!pip install -q jupyter-bbox-widget`),
se elige la letra, se aprieta **▭ Draw box** y se arrastra un rectángulo sobre la foto; al soltar,
las coordenadas aparecen en el campo *Crop box* (editable; **✕** lo borra) y al guardar quedan en `crop_box` como
coordenadas relativas `{x0, y0, x1, y1}` (0-1) sobre la foto enderezada (EXIF aplicado), válidas
para cualquier tamaño. `sepia.annotate.crop_photo(ruta, crop_box)` recorta el individuo.

Se guarda en `annotations/manual_<anotador>.jsonl`, una línea por acción (`kind`: `individual`,
`empty_photo`, `photo_info`, `exclude_record`), con `schema_version`, `annotator`, `round` y `saved_at` (UTC).
Nada se borra: corregir es guardar de nuevo; para cada clave vale la última línea. Un anotador =
un archivo (dos personas en el mismo archivo de Drive pueden pisarse).

## Excel de entrega

```python
!python scripts/build_excel.py --config configs/pilot_a1.yaml
```

**Una fila por foto y por individuo.** Clave de una anotación: `observation_id` + `photo_id` +
`individual`. El lado (`dorsal`, `ventral`, `unclear`) es un atributo de la fila: un hueso muestra
un solo lado por foto. La letra del individuo se asigna en la primera foto donde aparece y se
mantiene en todas las fotos del registro; `number individuals` = letras distintas, no filas.
Las fotos aún no anotadas salen como una fila sin individuo.

Une, por `observation_id`: registros (`tables/`), anotaciones (`annotations/`) y triaje
(`outputs/`, modelo y versión de prompt de la sección `triage` del YAML). El triaje llena
`state` y `number individuals (cuttlebone)` **solo donde no hay anotación manual**: la de una
persona siempre gana. Las filas llenadas por el modelo quedan con `label_source = gpt`,
`model_version = <modelo>/<prompt>`, `reviewed = False` y, en `model_prob`, la confianza y si
el conteo es exacto (`count_possible`). Sin anotaciones hay una fila por foto.

## Pruebas

```bash
pip install -e ".[dev]"
pytest -q
```

## Límites de iNaturalist

El código respeta ~1 consulta por segundo a la API y menos de 4 GB de fotos por hora, e identifica las consultas con un User-Agent.
