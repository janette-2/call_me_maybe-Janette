<i>This project has been created as part of the 42 curriculum by \<janrodri\>.</i>

<br>

### Description:

**call me maybe** convierte peticiones en lenguaje natural en llamadas a funciones
estructuradas (JSON), usando un LLM pequeño (Qwen3-0.6B) combinado con
decodificación restringida. El modelo **no responde la pregunta**, sino que
**identifica qué función llamar y con qué argumentos**.

### Instrucciones del LLM:

El LLM recibe el prompt del usuario y genera una secuencia de tokens que forma
un JSON con la estructura: `{"fn_name": "...", "args": {...}}`. No se usa como
chatbot o autocompletado — se usa como un **motor de procesamiento semántico**:
dado lenguaje natural, decide qué nombre de función corresponde a la consulta que se la ha pasado y qué argumentos son
los correspondientes a los pasados a esa función.

La **decodificación restringida** fuerza al modelo a que cada token generado
forme parte del JSON esperado. En cada paso, los tokens que no cumplen la
gramática reciben `-inf` en sus logits, haciendo imposible que el modelo los
elija. Así se garantiza JSON 100% válido incluso con un modelo de 500M
parámetros.

### Instrucciones:



### Resources:

https://www.geeksforgeeks.org/python/json-loads-in-python/
https://www.geeksforgeeks.org/python/json-load-in-python/


#### Configuración de mypy para llm_sdk:
El flag `--ignore-missing-imports` del Makefile **no** cubre el error `attr-defined`
("Module llm_sdk has no attribute Small_LLM_Model"): ese flag solo ignora módulos
que no existen en el filesystem. Como `llm_sdk` sí existe pero no tiene anotaciones
de tipos, mypy no encuentra el atributo. La solución está en `pyproject.toml`:

- `mypy_path = "llm_sdk"` → apunta al paquete real (`llm_sdk/llm_sdk/`)
- `[[tool.mypy.overrides]]` con `module = "llm_sdk"`, `ignore_errors = true` y
  `follow_imports = "skip"` → ignora errores dentro del módulo y evita que mypy
  analice `torch`/`transformers` (lo que ralentizaba o colgaba el análisis).
  El SDK no se modifica.



#### LLM_SDK:
The package of the LLM (Large Language Model) used [Qwen 0.6B] contains the following key features:

**1.  encode(texto)** → Convierte el texto input a tokens (números de IDs que el modelo entiende). Se usa una vez al principio para convertir el prompt en IDs. Devuelve un listado de la conversión de cada palabra a su ID del token correspondiente.

**2.  get_logits_from_input_ids(lista de tokens)** → El modelo procesa los IDs y devuelve un vector de ~150,000 logits o elementos float. Son puntuaciones crudas de cuán probable es que la siguiente palabra se adecúe al contexto de lo anterior. Cada elemento del vector representa cada token del vocabulario en sus probabilidades de ser el siguiente elemento. Se usa en cada análisis durante el loop de generación.

**3. decode(lista de tokens)** → Convierte los tokens que se han filtrado y recopilado en la última respuesta para pasarlo de vuelta a texto legible. Se usa una vez al final, cuando termina la generación.

**4. get_path_to_vocab_file()** → Devuelve la ruta al archivo `vocab.json` del
tokenizer (se descarga de Hugging Face). Ese archivo es un diccionario JSON que
mapea **cada token (texto) a su ID numérico**. Es clave para el logit masking:
permite saber, por ejemplo, cuáles son los IDs de los dígitos `0-9`, de `true`/
`false`, o de los delimitadores (`{`, `}`, `,`, `"`, espacio), sin tener que
adivinarlos con `encode`/`decode`. Ejemplo de uso:

```python
import json
path = model.get_path_to_vocab_file()
with open(path) as f:
    vocab = json.load(f)      # dict {texto: ID}
id_true  = vocab["true"]      # 1866
id_false = vocab["false"]     # 3849
```

> **Detalle importante:** `file.read()` devuelve un **string**, no un
> diccionario. Para poder acceder por clave (`vocab["true"]`) hay que
> convertirlo: o bien usar `json.load(archivo)` (lee el archivo directamente),
> o bien `json.loads(archivo.read())`. Ambos hacen lo mismo; la diferencia
> está en si ya tienes el string o directamente el archivo.

### Algorithm:


### Design:

**Estructura de datos: `dict_functions`** (devuelto por `functions_info()`)

Diccionario cuya clave es el nombre de la función y cuyo valor es otro
diccionario con tres claves: `parameters`, `description` y `returns`.

```python
{
    "fn_add_numbers": {
        "parameters": {
            "a": {"type": "number"},
            "b": {"type": "number"},
        },
        "description": "Add two numbers together and return their sum.",
        "returns": {"type": "number"},
    },
    "fn_greet": {
        "parameters": {"name": {"type": "string"}},
        "description": "Generate a greeting message for a person by name.",
        "returns": {"type": "string"},
    },
}
```

Para acceder al tipo del parámetro `"a"` de `"fn_add_numbers"`:
`dict_functions["fn_add_numbers"]["parameters"]["a"]["type"]` → `"number"`.

> **Trampa frecuente:** al iterar con `for func in dict_functions`, la variable
> `func` es el **nombre** (string), NO el valor. Para acceder a los datos de esa
> función hay que usar `dict_functions[func]`. Intentar `func.get(...)` lanza
> `AttributeError: 'str' object has no attribute 'get'`.

**Estructura de datos: `dict_fixed`** (devuelto por `fixed_ids()`)

Diccionario que mapea cada pieza de texto del JSON a sus token IDs (lista plana
de enteros).

```python
{
    "{": [90],
    "}": [92],
    "fn_add_numbers": [8822, 2891, 32964],
    "a": [64],
    ...
}
```

**El super-prompt** (`build_super_prompt()`)

Texto que se pasa al modelo para cada prompt de usuario. Contiene: el rol del
asistente, las reglas de salida (solo JSON, estructura exacta, `fn_name` debe
ser una función disponible, `args` con los argumentos requeridos y tipos
correctos) y el listado dinámico de funciones con sus descripciones y
parámetros.

> **Aclaración importante:** el super-prompt **no es un archivo de entrada**.
> Los archivos de `data/input/` siguen siendo la única entrada del programa.
> El super-prompt es una construcción interna: tu código lee las funciones de
> `functions_definition.json` y los prompts de `function_calling_tests.json`,
> y luego monta el texto que presenta esa información al LLM. El flujo es:
>
> ```
> data/input/functions_definition.json ──┐
>                                        ▼
> data/input/function_calling_tests.json ──►  build_super_prompt() ──►  LLM
>                                             ▲
>                                         plantilla fija
>                                         (instrucciones/reglas)
> ```
>
> El prompt del usuario (p.ej. `"What is the sum of 2 and 3?"`) se inserta al
> final del super-prompt, después del listado de funciones y antes de la
> etiqueta `Output:` que indica al modelo dónde empieza su respuesta.

### Performance Analysis:


### Challenges found:

**1. Entender qué pasa dentro del LLM cuando "pasa por el modelo"**
Inicialmente no lograba visualizar qué ocurría internamente entre que entran
los tokens de entrada y salen los logits. Exploramos las tres capas internas:
Embedding (cada token → vector denso), Transformer (28 bloques de Self-Attention
+ FFN donde los tokens se contextualizan entre sí), y LM Head (el vector del
último token se proyecta al vocabulario completo generando ~150,000 logits).
Conclusión: el LLM es una máquina de predecir el siguiente token, y los logits
son puntuaciones crudas de esa predicción.

**2. Cuándo se usa decode en el flujo de generación**
Pensaba que decode se usaba paso a paso durante el loop. Descubrimos que no:
el loop entero trabaja con IDs numéricos, y decode se aplica una sola vez al
final para convertir la secuencia completa de IDs a texto legible.

**3. Propósito del LLM en el proyecto (¿por qué no regex?)**
Dudaba de por qué necesitábamos un LLM si podríamos parsear con regex.
Conclusión: el LLM aporta flexibilidad semántica para entender lenguaje natural
variado ("What is the sum of 2 and 3?" vs "Add 2 and 3" vs "Calculate 2+3")
sin hardcodear patrones. La decodificación restringida garantiza que la salida
sea JSON válido, y el LLM solo decide qué función y argumentos corresponden.

**4. Cómo hacer que el LLM conozca las funciones disponibles**
No veía claro cómo unir las definiciones de funciones con el prompt del usuario.
Llegamos a la solución del super-prompt: construir un mensaje que incluya
instrucciones del sistema + listado de funciones disponibles con sus parámetros
+ el prompt original del usuario. Esto orienta al modelo sin necesidad de
fine-tuning.

**5. Diferencia entre super-prompt y logit masking**
Confundía ambos conceptos como si fueran redundantes. Aclaramos que son
complementarios: el super-prompt orienta semánticamente al modelo (señal
débil, el modelo "intuye" lo que debe hacer), mientras que el logit masking
fuerza mecánicamente la estructura (garantía fuerte, el modelo no puede
desviarse aunque quiera).

**6. Formato de salida del LLM: semi-estructurado vs plantilla fija**
Dilema sobre cómo debería generar el LLM la respuesta: ¿en un formato
semi-estructurado tipo `fn_add_numbers(a=2, b=3)` para luego parsearlo?
¿O mejor separar la decisión del LLM de la construcción del JSON?
Conclusión: optamos por una plantilla JSON fija donde el programa fuerza
las partes invariables (`{`, `"fn_name"`, `"args"`, etc.) y el LLM solo
decide los valores variables (nombre de función y argumentos).

**7. Cómo averiguar los IDs de cada token (Token discovery)**
Para forzar tokens específicos en el loop, necesitábamos saber sus IDs
numéricos dentro del vocabulario de Qwen3 (~150,000 tokens). La solución
fue usar `encode("símbolo")` una vez al inicio para mapear cada pieza fija
a sus IDs, construyendo así un diccionario de tokens reutilizable.

**8. Detección de confianza baja del modelo**
Preocupación sobre qué hacer cuando el prompt no se corresponde con ninguna
función disponible. Exploramos la idea de analizar la distribución de
probabilidad: si el mejor nombre de función tiene una probabilidad baja
y las demás están repartidas, el modelo no tiene clara la respuesta.
Queda pendiente definir el umbral exacto y la acción a tomar.

**9. Reglas de formato JSON**
Dudas sobre si hay reglas de ordenación de claves, indentación o restricciones
adicionales. Conclusión: el orden de las claves en un objeto JSON no importa
(aunque nuestra plantilla fuerza uno concreto), y las únicas reglas estrictas
son: comillas dobles en claves y strings, sin comas finales, sin comentarios,
100% parseable por `json.loads()`.

**10. Separar decisión del LLM vs partes forzadas del programa**
Costó distinguir qué genera el LLM (solo nombre de función y valores de
argumentos) y qué fuerza el programa directamente (llaves, comillas, comas,
nombres de clave como `fn_name` y `args`). Conclusión: el programa recorre
el autómata de estados y en los estados "fijos" fuerza el token directamente;
en los estados "de decisión" deja que el LLM elija entre opciones limitadas.

**11. Manejo de prompts sin función correspondiente**
Si el modelo recibe un prompt que no encaja con ninguna función, siempre va
a generar algo (no puede "callarse"). La solución planteada es: si la
probabilidad máxima entre los nombres de función es baja, detectarlo y
manejarlo como caso especial. Aún por definir la implementación concreta.

**12. Cómo funcionan las comillas triples y la utilidad de `textwrap.dedent()`**
Al construir el super-prompt con strings multilínea (obligados a partirlos por
el límite de 79 caracteres de flake8, E501), el output salía con indentación
sobrante y líneas en blanco extra. La causa es que las comillas triples de
Python son **literales**: todo carácter escrito entre ellas —incluida la
sangría del código— forma parte del texto final. No hay "decoración visual":
cada espacio que indentamos el código dentro del string es un espacio real en
el prompt.

La solución fue `textwrap.dedent()`: calcula la sangría común mínima entre las
líneas no vacías del string y la elimina de todas por igual. No colapsa
espacios ni reordena el texto, solo resta el prefijo común. Detalle
importante: `dedent()` **no toca los saltos de línea**, así que los `\n`
iniciales y finales de las comillas triples hay que gestionarlos aparte
(`"""\` para suprimir el primer salto, o ajustando los `\n` manualmente).

Ejemplo visual con un string de 3 líneas:

```
ANTES (el string tal como está escrito en el código):

    Soy la primera linea.
        Soy la segunda, con mas sangria.
    Soy la tercera.

            ┌─ 4 espacios (común mínimo)
            │
    linea 1: 4 espacios  →  0 espacios  (resta 4)
    linea 2: 8 espacios  →  4 espacios  (resta 4)
    linea 3: 4 espacios  →  0 espacios  (resta 4)
    vacías: se ignoran

DESPUÉS de dedent():

Soy la primera linea.
    Soy la segunda, con mas sangria.
Soy la tercera.
```

La sangría común mínima es 4 (todas las líneas tienen al menos 4). `dedent()`
resta 4 a **todas por igual**: la línea de 8 se queda con 4, la de 4 con 0.
Los `\n` no cambian: quedan exactamente donde estaban.

**Solución final elegida en este proyecto:** en `template_rules` y en el
listado de funciones escribimos el contenido del string a **columna 0**
(pegado al margen izquierdo, sin indentar dentro de las comillas), igual que
las funciones. Así el string ya nace limpio y `dedent()` no tiene que hacer
nada en esos bloques. `dedent()` solo se aplica a `template_intro`, cuyo
contenido sí mantenemos indentado para que el código se lea mejor.

**Inconveniente a conocer:** mantener el texto a columna 0 dentro de las
comillas triples se ve raro en el código (el texto no queda alineado con la
indentación de Python). Es un trade-off consciente: preferimos un prompt
limpio de cara al modelo que un código "bonito". flake8 no se queja porque
no comprueba la indentación del contenido de los strings, solo la del código.
Y no hay problema con E501 porque esas líneas son cortas; si fueran largas,
habría que partirlas y volveríamos al problema inicial.

**13. Concatenar strings largos sin pasarse del límite de E501**
Al ensamblar el prompt en `build_super_prompt()`, intentamos concatenar las
cuatro partes en una sola línea:

```python
prompt = intro + rules + functions + fin
```

Esa línea supera los 79 caracteres de E501, así que flake8 la marcaría. La
solución que usamos es la **continuación implícita entre paréntesis**:
envolver la expresión en `(...)` permite partir la línea después de cada `+`
sin ningún carácter extra:

```python
prompt = (template_intro
          + template_rules
          + template_functions
          + f"\nUser: {input_call}\nOutput: ")
```

Python trata todo lo que hay entre `(` y `)` como una sola expresión
(independientemente de los saltos de línea), así que no hace falta `\` y
cada operando queda en su propia línea por debajo de los 79 caracteres.
Este patrón sirve para cualquier operación larga (sumas, concatenaciones,
argumentos de funciones), no solo para strings.

**14. Conversion de [x, y] a 'x, y'**
El -1 en Python se utiliza para referirse al último elemento de una secuencia (como una cadena de texto o una lista) contando de atrás hacia adelante.
En el truco str(mi_lista)[1:-1], estamos usando una técnica llamada slicing (rebanado) que funciona bajo la estructura [inicio:fin].
Aquí te explico exactamente por qué elimina el corchete final:
**a. El conteo inverso en Python**
Python permite indexar desde el final usando números negativos:

* -1 es el último carácter.
* -2 es el penúltimo carácter.

Al convertir una lista a texto con str([1, 2, 3]), el texto resultante es "[1, 2, 3]".

* El índice 0 es el corchete de apertura [.
* El índice -1 es el corchete de cierre ].

**b. La regla del "límite abierto"**
En Python, el índice de fin en un slicing no se incluye en el resultado (es exclusivo).
Por lo tanto, al escribir [1:-1]:

* 1: Empieza en el índice 1 (el primer número, saltándose el [ del índice 0).
* -1: Se detiene justo antes del índice -1 (saltándose el ] del final).

**Ejemplo visual con "[1, 2, 3]"**

| Carácter | [ | 1 | , | | 2 | , | | 3 | ] |
|---|---|---|---|---|---|---|---|---|---|
| Índice Positivo | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| Índice Negativo | -9 | -8 | -7 | -6 | -5 | -4 | -3 | -2 | -1 |

Al cortar desde 1 hasta -1, Python toma todo lo que está entre el fondo verde y el fondo rojo, dejando el interior intacto y eliminando ambos corchetes de un solo golpe.

**15. Atascos del Paso 4 (loop con decodificación restringida)**

* **Confundir la salida del subject con la salida del LLM.** El archivo
  `output/function_calling_results.json` del subject incluye `"prompt"`, pero
  ese campo lo escribe **tu programa** (ya posee el prompt). El LLM solo
  genera el fragmento `{"fn_name": ..., "args": ...}`; `"prompt"` no es una
  pieza que haya que forzar ni generar durante el loop.
* **`append` vs `extend` en la lista de IDs.** `list.append([90])` mete la
  sublista como un solo elemento → lista anidada `[[90], ...]`. `list.extend([90])`
  desempaqueta y añade el `90` suelto → lista plana `[90, ...]`. Para ir
  construyendo la secuencia de tokens hay que usar `extend`, no `append`.
* **No preconstruir la secuencia completa de golpe.** No se puede montar el
  JSON entero antes de tiempo porque el modelo decide en mitad del camino
  (qué función, qué valores). El loop es paso a paso: cada vuelta añade un
  token (forzado con `extend` o elegido por el modelo) y vuelve a preguntar
  con todo el contexto acumulado.
* **`list.extend()` devuelve `None`.** `final = prompt_ids.extend(ids_list)`
  guarda `None` en `final` porque `extend()` modifica la lista en el sitio y
  no devuelve nada. El error `TypeError: 'NoneType' object cannot be
  converted to 'Sequence'` aparece al hacer `model.decode(final)`. La lista
  ya contiene el resultado: hay que pasarle `prompt_ids` directamente a
  `decode()`, no una variable asignada al método.
* **Ocultar el builtin `input`.** Usar `input` como nombre de parámetro oculta
  la función nativa de Python. Se renombró a `input_call` en
  `build_super_prompt()` para que el código sea seguro y claro.

**16. Iterar sobre una lista mientras se elimina → se saltan elementos**
En el while de identificación de función, el `for ids in fn_names_tokens`
recorría la lista y hacía `fn_names_tokens.remove(ids)` cuando el candidato
no coincidía. El problema: Python avanza el iterator interno al eliminar un
elemento, así que si se eliminan dos consecutivos, el segundo se salta.
Solución: iterar sobre una **copia** (`fn_ids = fn_names_tokens.copy()`) para
que las eliminaciones no afecten al recorrido. La copia es barata (pocos
elementos) y garantiza que se examina cada candidato exactamente una vez.

**17. El prompt no se actualiza dentro del while → predicciones repetidas**
`init_prompt_ids` era la secuencia de tokens que el modelo recibía como
entrada. Si no se modificaba dentro del while, el modelo veía **siempre la
misma secuencia** y, por tanto, predecía **siempre el mismo token**. La
solución fue crear `temp_prompt = init_prompt_ids.copy()` y hacer
`temp_prompt.extend([next_id])` en cada paso, de modo que cada predicción
recibe el contexto completo incluyendo las predicciones anteriores. Se usa
una copia para que `init_prompt_ids` no acumule tokens duplicados.

**18. Condiciones invertidas al construir el JSON de argumentos**
Al cerrar las llaves del JSON, la condición `if i + 1 == len(args_fn)` se
cumplía **una posición demasiado pronto** porque `i` se incrementaba **antes**
del `if`. Con 2 argumentos funcionaba por coincidencia, pero con 3+ fallaba:
ponía `}` después del primer arg en vez de `, `. La corrección fue cambiar
la condición para que evalúe si acabamos de procesar el **último** argumento
(`i == len(args_fn)`) y actuar en consecuencia. Lección: siempre verificar
la lógica de condiciones con casos de borde (1 arg, 2 args, 3+ args).

**19. `while` infinito al generar argumentos: `or` vs `and` y comparar `int` con `list`**
El `while` que generaba tokens de argumentos colgaba `make run` sin devolver
nunca resultado. La línea defectuosa era:

```python
while next_id != dict_fixed_chars[","] or next_id != dict_fixed_chars["\""][0]:
```

Tenía **dos errores** que se combinaban:

1. **`or` en vez de `and`**: la condición `A != x or A != y` es **siempre True**
   (un valor no puede ser igual a dos cosas distintas; si no es `x`, la primera
   parte ya es verdadera). Debería ser `and` para que solo salga del buelle
   cuando `next_id` no sea ninguno de los dos delimitadores.

2. **Comparar `int` con `list`**: `dict_fixed_chars[","]` devuelve una **lista**
   `[11]`, no un entero. `next_id` es un `int`. La comparación `int != list`
   **siempre es True** en Python. Debería ser `dict_fixed_chars[","][0]` para
   acceder al elemento dentro de la lista.

Combinados, ambos errores hacían que la condición **nunca fuera falsa**,
resultando en un bucle infinito que nunca permitía al programa terminar.

Lección: al escribir condiciones de salida de `while`, verificar siempre que
(a) la operación lógica (`and`/`or`) refleje la intención real y (b) los
tipos comparados sean compatibles (no mezclar `int` con `list`).


### Tests Strategy:


### Usage examples:


