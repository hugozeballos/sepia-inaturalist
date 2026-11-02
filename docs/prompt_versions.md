# Versiones de prompt

Registro de todas las versiones de prompt (`src/sepia/prompts/`). Los prompts nunca se editan: cada cambio es una versión nueva. El hash es el `prompt_sha` que aparece en cada línea de `outputs/` y de `outputs/metrics/` (sha256, primeros 12 caracteres).

Modelo en todas: Gemini `gemini-3.5-flash-lite`, temperatura 0. Imágenes de 512 px en el triaje y de 1024 px en los rasgos por foto.

## Triaje (por registro: estado + número de huesos)

| Versión | Hash | Qué cambia |
|---|---|---|
| `triage_v1` | `eca76c2be1ac` | Estado del registro + huesos distintos + `count_possible` + confianza. El mismo hueso en varias fotos cuenta una vez. Fallaba al contar muchos huesos. |
| `triage_v2` | `05e2c364b2f1` | Reglas de conteo explícitas: solo lo visible (ignora textos como "cientos"); el mismo hueso cuenta una vez, también un primer plano en la mano; un fragmento cuenta solo si es ≈ medio hueso o más (`fragments_present`); con más de 5 huesos, `count_possible = false` y conteo a mano; confianza `high` solo si el estado es obvio y el conteo es fácil. **En uso.** |

## Rasgos por foto (`photo_traits`: huesos visibles, lado, erosión, calidad)

Desde v1 el lado, la erosión y la calidad se describen solo si hay **exactamente 1 hueso visible**. Si no, quedan en `not_applicable` (regla fija en `apply_rules`).

| Versión | Hash | Unidad | Lado | Otros cambios |
|---|---|---|---|---|
| `photo_traits_v1` | `a18dc8eeb12a` | foto | Guía de Martin: dorsal más liso y duro, convexo, granular, sin líneas de crecimiento en capas; ventral laminado, con muchas líneas de crecimiento finas, paralelas y en arco; más blando, con picoteo. | Huesos visibles, lado y erosión 1-3 (texto de la guía). Sin calidad. |
| `photo_traits_v2` | `29b4b274691a` | foto | Igual que v1. | + `photo_quality` (high/medium/low). Erosión operativa: se juzga el contorno; 2 = falta < 10 %; 3 = falta ≥ 10 % o es un fragmento. |
| `photo_traits_v3` | `406924b00d39` | foto | Igual que v1. | Calidad: solo lo que tapa el hueso la baja; no la bajan las marcas, los epibiontes, las manchas, la textura natural ni unos pocos granos de arena. |
| `photo_traits_v4` | `abac10bd80c1` | foto | Igual que v1. | Conteo: un hueso cuenta solo si se ve al menos la mitad; los trozos en el borde no cuentan. Calidad: una capa o película de arena o algas sobre el hueso sí lo tapa. |
| `photo_traits_v5` | `1f00bb65a221` | registro | Definiciones de v4 + **regla entre fotos**: una foto dudosa se decide comparando con otra del mismo registro (mismo hueso: igual → mismo lado, distinto → opuesto; otro hueso: solo igual → mismo lado). | Todas las fotos del registro en una consulta, una línea por foto (`call_id`). Nuevo `side_basis`. |
| `photo_traits_v6` | `81ac3ead364e` | registro | **Cambio de criterio (Hugo, 30 oct):** líneas finas, rectas, a lo largo del hueso de punta a punta = **solo dorsal** (`lines_rule`); su ausencia no dice nada. Ventral sin líneas: blando, poroso, borde en U/V cerca de la espina. | — |
| `photo_traits_v7` | `a2b12fc06087` | registro | Igual que v6. | + `side_evidence`: pocas palabras con el rasgo que decidió el lado. |
| `photo_traits_v8` | `9dac5cdb136a` | registro | v7 con el dorsal descrito como **relieve**: crestas bajas, rectas y paralelas a lo largo, visibles como franjas de luz y sombra con luz lateral; no son líneas dibujadas (Hugo, 1 nov). | — |
| `photo_traits_v9` | `8cfefc974530` | registro | v8 + **bandas curvas transversales** como señal ventral (`bands_rule`). **Experimento.** En la revisión del 2 nov "vio" bandas en dorsales y usó la regla entre fotos sobre fotos ya decididas. | — |
| `photo_traits_v10` | `cda4c89c6b1b` | registro | **Literatura** (FAO, Reid, VLIZ). Rasgos en lista cerrada (`side_signs`): ventral fuerte `inner_cone_rim`, `striated_zone`; dorsal fuerte `granular_shield`; débiles `median_groove` (V), `chitinous_posterior_cap` (D); neutros `transverse_lines`, `lengthwise_lines`, `white_chalky_or_porous`. Las líneas ya no deciden. Rasgos de los dos lados o solo débiles → `unclear`. La regla entre fotos solo para fotos sin rasgo fuerte y con el mismo hueso. | + `side_signs`; `side_basis` = direct / inferred_same_bone / unclear. El código calcula `side_rule` (tabla fija). |
| `photo_traits_v11` | `b2f3ddd87f15` | registro | v10 + **guía de Martin**: `laminated_surface` (V fuerte), `bite_marks` (V débil), `smooth_hard_uniform_face` (D; cuenta solo con calidad `high` y sin ningún rasgo ventral). | Igual que v10. |

Erosión y calidad: desde v4, el texto es el mismo en todas las versiones. v5-v11 solo cambian el lado.

### `side_basis` por versión

| Valor | Significado | Versiones |
|---|---|---|
| `direct` | decidido con la foto sola | v5-v11 |
| `lines_rule` | dorsal por las líneas o crestas a lo largo | v6-v9 |
| `bands_rule` | ventral por las bandas curvas transversales | v9 |
| `inferred_same_bone` | por comparación con el mismo hueso en otra foto | v5-v11 |
| `inferred_other_bone` | por comparación con otro hueso del registro | v5-v9 |
| `unclear` / `not_applicable` | sin lado | v5-v11 |

### `side_rule` (v10-v11, calculado por el código)

`photo_traits.side_from_signs`, solo con los rasgos de esa foto:
- **ventral**: algún rasgo ventral fuerte y ningún rasgo dorsal, ni fuerte ni débil;
- **dorsal**: algún rasgo dorsal fuerte y ningún rasgo ventral, ni fuerte ni débil (`smooth_hard_uniform_face` cuenta como fuerte solo con calidad `high`);
- **unclear**: en cualquier otro caso.

## Historia del criterio de lado

1. v1-v5 siguen la guía de Martin (ventral laminado, con líneas en arco).
2. v6-v9 se basan en la observación de Hugo (líneas o crestas a lo largo = dorsal).
3. v10-v11 vuelven a la literatura y a la guía de Martin con rasgos explícitos, y las líneas dejan de decidir.

Pendiente con Martin: si las crestas a lo largo son dorsales en *S. officinalis*.
