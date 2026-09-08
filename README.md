<i>This project has been created as part of the 42 curriculum by \<janrodri\>.</i>

<br>

## Description:

**call me maybe** converts natural language requests into structured function calls
(JSON), using a small LLM (Qwen3-0.6B) combined with **constrained decoding**. The
model **does not answer the question**; it **identifies which function to call and
with which arguments**. The LLM is used as a semantic-inference engine, not as a
chatbot.

The answer is a JSON object with a fixed shape:

```json
{"fn_name": "...", "args": {...}}
```

## How it works:

The LLM receives the user prompt and generates a token sequence that forms the JSON
above. It is not used for free-form generation: given a natural-language request, it
decides which function name matches the query and which arguments match the values
passed to that function.

**Constrained decoding** forces the model so that every generated token is part of the
expected JSON grammar. At each step, tokens that do not satisfy the grammar get `-inf`
in their logits, making it *impossible* for the model to choose them. In this way the
output is valid JSON 100% of the time, even with a ~500M parameter model.

The JSON is separated into two kinds of parts:

- **Fixed parts** (`{`, `"`, `fn_name`, `args`, commas, braces, key names) are forced
  directly by the program using known token ids. The model never chooses them.
- **Variable parts** (the function name and the argument values) are the only places
  where the model decides, always among a **closed set** of allowed tokens.

## How to run:

```bash
make install   # uv sync (installs the SDK, torch-cpu, numpy, pydantic)
make run       # uv run python -m src  ->  writes output/function_calling_results.json
make lint      # flake8 + mypy (warn-return-any, disallow-untyped-defs, ...)
make lint-strict
make clean     # removes __pycache__, .mypy_cache, .pytest_cache, *.pyc
```

Entry point:

```
src/__main__.py  ->  from .main import main
```

Inputs (not edited by the program):

- `data/input/functions_definition.json` — the available functions.
- `data/input/function_calling_tests.json` — the test prompts.

Output:

- `output/function_calling_results.json` — generated at runtime, **not committed**
  (see the subject: the `output/` directory is not part of the repository).

### mypy configuration for `llm_sdk`:

The `--ignore-missing-imports` flag of the Makefile **does not** cover the
`attr-defined` error ("Module llm_sdk has no attribute Small_LLM_Model"): that flag
only ignores modules that do not exist on the filesystem. Because `llm_sdk` exists but
has no type annotations, mypy cannot find the attribute. The fix lives in
`pyproject.toml`:

- `mypy_path = "llm_sdk"` → points to the real package (`llm_sdk/llm_sdk/`).
- `[[tool.mypy.overrides]]` with `module = "llm_sdk"`, `ignore_errors = true` and
  `follow_imports = "skip"` → ignores errors inside the module and prevents mypy from
  analysing `torch`/`transformers` (which slowed or hung the analysis). The SDK itself
  is never modified.

## Algorithm:

The pipeline has five stages:

```
data/input/functions_definition.json ──┐
                                       ▼
data/input/function_calling_tests.json ──► build_super_prompt() ──► loop_prompt_output() ──► decode ──► validate ► output/
```

**Stage 1 — Load the functions** (`functions_info()`): reads
`functions_definition.json` and validates it with pydantic. It distinguishes "file
missing" from "invalid JSON" (both required by the subject) and discards malformed
entries instead of crashing.

**Stage 2 — Token discovery** (`fixed_ids()`): encodes every fixed piece of the
template (punctuation, `fn_name`, `args`, every function name and parameter name)
into its token ids, once at startup. The `fn` prefix is shared by all functions, so
every name starts with the same token `[8822]`.

**Stage 3 — Super-prompt** (`build_super_prompt()`): builds the text the model sees:
assistant role, output rules (JSON only, exact structure, `fn_name` must be an
available function, `args` with the required parameters and correct types, quoted
phrases taken whole), the dynamic list of available functions, the user request, and
an `Output:` marker.

**Stage 4 — Constrained-decoding loop** (`loop_prompt_output()`): the core. It walks
the JSON state machine, forcing the fixed tokens and asking the model for the variable
ones:

1. Force the skeleton header: `{` `"` `fn_name` `"` `:` `"`.
2. **Choose the function** (`_escoge_fn()`): all names share the leading `fn` token,
   so there is a first position where their tokens differ — that position uniquely
   separates the candidates. **One** `get_logits` call scores just the tokens of that
   position and the highest logit picks the function. If nothing matches, a
   **deterministic fallback** (`resolver_falla_fn()`) scores every function by how
   many of the prompt's words appear in its description/name, so the answer is always
   one of the available functions.
3. Force `"` `,` `"` `args` `"` `:` `{`.
4. For each parameter, force `"param"` `:` and resolve the **value by type**:
   - **number** (`logit_masking_number`): the value is always one of the numbers
     written by the user. A single candidate needs 0 model calls; if several differ,
     the loop walks the token positions until they diverge and the model picks among
     the tokens of that deciding position. Each number used is removed from the
     candidate list, so two number arguments never reuse the same value.
   - **boolean** (`logit_masking_boolean`): the model may only emit `true`, `false` or
     a stop token (`,` or `}`); tokens are generated until the stop token is chosen.
   - **string**: the candidate set is the prompt itself, resolved with a
     **deterministic priority** before the model is ever asked:
     1. *regex concept*: for `regex`/`pattern` arguments, `inferir_patron_regex()`
        maps concept words (*numbers* → `\d+`, *vowels* → `[aeiouAEIOU]`).
     2. *word after "with"*: for `replacement` arguments, `extraer_palabra_tras_with()`
        grabs the word following *"with"* (`replace X with Y`).
     3. *quoted phrase*: `extraer_frases_entrecomilladas()` takes the whole phrase
        (stripping the quotes, treating `I'm` as a word), handed to
        `logit_masking_string()`.
     4. *last resort*: the model picks among the individual prompt words
        (`extraer_palabras()`), scoring all candidates with a single `get_logits`
        call and taking the best by summed token logits.
     Every chosen value is wrapped in quotes.
5. Close each argument with `,` (more args) or `}` (last arg), then the final `}`.

**Why a fixed template instead of letting the LLM emit the JSON?** If the model chose
the whole structure it could deviate (extra spaces, wrong keys, unclosed braces).
Separating decisions from structure means the model decides *what* but never *how to
format it* — the program guarantees the shape, the model only fills the values.

**Stage 5 — Output** (`main()`): for every prompt, decode the sequence back to text,
slice from `Output: `, parse with `json.loads`, validate against the pydantic schema,
coerce numeric arguments to `float` (so `2` becomes `2.0` as the schema demands), and
append `{"prompt", "fn_name", "args"}` to the results list. Missing/invalid input
files print a warning and produce an empty list instead of crashing.

The full loop works on **numeric ids** — `encode` happens once at the start and
`decode` once at the end; nothing in between touches text.

## Design:

### Data structure: `dict_functions` (returned by `functions_info()`)

A dict whose key is the function name and whose value is another dict with three keys:
`parameters`, `description` and `returns`.

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

To access the type of parameter `"a"` of `"fn_add_numbers"`:
`dict_functions["fn_add_numbers"]["parameters"]["a"]["type"]` → `"number"`.

> **Common trap:** when iterating with `for func in dict_functions`, `func` is the
> **name** (a string), NOT the value. To reach that function's data you must use
> `dict_functions[func]`. Calling `func.get(...)` raises
> `AttributeError: 'str' object has no attribute 'get'`.

### Data structure: `dict_fixed` (returned by `fixed_ids()`)

A dict mapping each fixed piece of JSON text to its flat list of token ids.

```python
{
    "{": [90],
    "}": [92],
    "fn_add_numbers": [8822, 2891, 32964],
    "a": [64],
    ...
}
```

### The super-prompt (`build_super_prompt()`)

A text built at runtime for each user prompt. It contains the assistant role, the
output rules and the dynamic function listing with descriptions and parameters.

> **Important clarification:** the super-prompt is **not an input file**. The only
> program inputs are the files in `data/input/`. The super-prompt is an internal
> construction: the code reads the functions and the prompts, then assembles the text
> that presents that information to the LLM.
>
> The user prompt (e.g. `"What is the sum of 2 and 3?"`) is inserted at the end of
> the super-prompt, after the function listing and right before the `Output:` marker
> that tells the model where its answer starts.

## Tests strategy:

The test suite is `data/input/function_calling_tests.json`, 11 prompts covering all 5
functions:

| Prompt | Expected |
| ------ | -------- |
| What is the sum of 2 and 3? | `fn_add_numbers` a=2.0 b=3.0 |
| What is the sum of 265 and 345? | `fn_add_numbers` a=265.0 b=345.0 |
| Greet shrek | `fn_greet` name=shrek |
| Greet john | `fn_greet` name=john |
| Reverse the string 'hello' | `fn_reverse_string` s=hello |
| Reverse the string 'world' | `fn_reverse_string` s=world |
| What is the square root of 16? | `fn_get_square_root` a=16.0 |
| Calculate the square root of 144 | `fn_get_square_root` a=144.0 |
| Replace all numbers in "Hello 34 I'm 233 years old" with NUMBERS | `fn_substitute_string_with_regex` source=Hello 34 I'm 233 years old regex=\d+ replacement=NUMBERS |
| Replace all vowels in 'Programming is fun' with asterisks | `fn_substitute_string_with_regex` regex=[aeiouAEIOU] replacement=asterisks |
| Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat' | `fn_substitute_string_with_regex` source=The cat sat on the mat with another cat regex=cat replacement=dog |

The suite deliberately exercises the edge cases that have historically broken the
loop:

- **Arguments count**: 1 arg (`fn_greet`, `fn_get_square_root`...), 2 args
  (`fn_add_numbers`) and 3 args (`fn_substitute_string_with_regex`). The comma/brace
  delimiter logic was unit-checked by hand for 1, 2 and 3+ args (see Challenges #18).
- **Numbers with 1 and 2 digits**, and two distinct numbers in the same prompt
  (re-using a number for the second arg would be wrong).
- **Quoted strings** with single quotes and double quotes, and a double-quoted source
  containing an apostrophe contraction (`I'm`). The `s` argument comes from the whole
  quoted phrase, not from its words.
- **Regex patterns** from concept words (*numbers* → `\d+`, *vowels* →
  `[aeiouAEIOU]`) and a literal word (*cat*).
- **Uppercase replacement values** (`NUMBERS`, `asterisks`) — the value must come from
  the prompt unchanged, including its case.
- **Backslash escaping**: `\d+` must survive `json.dump`/`json.loads`; the encoder
  doubles backslashes so the round-trip keeps the pattern intact.
- **Type coercion**: numbers are emitted as `float` (`2.0`, not `2`) to match the
  schema, validated by pydantic before writing.

Validation is double:

1. **pydantic** (`ResultadoLlamada`) — the program only writes results that satisfy
   the schema; malformed ones raise a warning and are recorded with `fn_name: null`.
2. **Deterministic fallback** — if the model produces a function name that is not in
   the definition file, `resolver_falla_fn()` picks the best match by word overlap, so
   a wrong guess never slips into the output.

## Performance analysis:

**Model calls per prompt.** The loop is extremely cheap: the super-prompt is encoded
once, and each decision point costs exactly one `get_logits` call:

- function name → 1 call (single discriminating position);
- number argument → 0 calls (single candidate) or 1 call (deciding position);
- boolean argument → 1 call per emitted token, until the stop token (~1–3);
- string argument → 1 scoring call for the whole candidate set.

Realistically **2–5 model calls per prompt**, independent of the super-prompt length,
which is negligible: the whole 11-prompt suite stays far below any reasonable budget.

**Deterministic shortcuts** push many arguments to 0 model calls: a single number in
the prompt, a recognised regex concept word, and `replacement` values following
*"with"*. The shortest string path (quoted phrase already in the prompt) still needs
only one scoring call. The model is asked only when there is genuine ambiguity.

**Correctness vs. accuracy.** The JSON *structure* is valid by construction — the
constrained decoding makes failure impossible. The *semantic value* depends on the
0.6B model with greedy decoding: simple values (single numbers, short quoted words
like `hello`, `world`, `shrek`) are reliable; the weak point is that the model does
not always know *which* slice of the prompt is the argument. Restricting the candidate
set to tokens **of the user prompt itself** reduces that ambiguity, and the
deterministic paths eliminate it for the substitution prompts.

**Verified result.** Over the current test set the program resolves **11/11** prompts
correctly (`fn_name` and `args`), as reproduced in the last run
(`output/function_calling_results.json`).

**Memory footprint.** Qwen3-0.6B ships ~1.5 GB of weights; a full run needs ~4 GB RAM
plus the tokenizer files. In this environment an 8 GB swap was added so the run never
crashes with VS Code open (see Challenges).

## Usage examples:

Run the program:

```bash
make run          # or: uv run python -m src
```

It loads the model once, processes the 11 prompts, and writes
`output/function_calling_results.json`:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "fn_name": "fn_add_numbers",
    "args": {"a": 2.0, "b": 3.0}
  },
  {
    "prompt": "Greet shrek",
    "fn_name": "fn_greet",
    "args": {"name": "shrek"}
  },
  {
    "prompt": "Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS",
    "fn_name": "fn_substitute_string_with_regex",
    "args": {
      "source_string": "Hello 34 I'm 233 years old",
      "regex": "\\d+",
      "replacement": "NUMBERS"
    }
  }
]
```

The last run over the current test set produced **11/11** correct results. The output
file is generated at runtime and must **not** be committed (it is ignored by
`.gitignore` and the subject forbids shipping it).

## Challenges found:

**1. Understanding what happens inside the LLM when it "goes through the model"**
Initially it was hard to visualise what happens between input tokens and output
logits. We explored the three internal layers: Embedding (each token → dense vector),
Transformer (28 blocks of Self-Attention + FFN where tokens contextualise each other)
and the LM Head (the last token's vector is projected into the whole vocabulary,
producing ~150,000 logits). Conclusion: the LLM is a next-token predictor and logits
are the raw scores of that prediction.

**2. When `decode` is used in the generation flow**
We thought `decode` was used step by step inside the loop. It is not: the loop works
entirely with numeric ids and `decode` is applied once at the end to turn the full id
sequence back into readable text.

**3. Purpose of the LLM in the project (why not regex?)**
We doubted the need for an LLM if the prompts could be parsed with regex. Conclusion:
the LLM gives semantic flexibility to understand varied natural language ("What is the
sum of 2 and 3?" vs "Add 2 and 3" vs "Calculate 2+3") without hardcoding patterns.
Constrained decoding guarantees valid JSON; the LLM only decides which function and
arguments match.

**4. How to make the LLM know the available functions**
We did not see how to merge the function definitions with the user prompt. We reached
the super-prompt solution: build a message with system instructions + the list of
available functions and their parameters + the original prompt. This steers the model
without fine-tuning.

**5. Difference between super-prompt and logit masking**
We confused both concepts as if they were redundant. They are complementary: the
super-prompt steers semantically (weak signal, the model "intuits"), the logit
masking enforces the structure mechanically (strong guarantee, the model cannot
deviate even if it wanted to).

**6. Output format of the LLM: semi-structured vs fixed template**
Dilemma about how the LLM should emit its answer: a semi-structured `fn_add_numbers(a=2,
b=3)` to parse later, or separate the LLM's decision from the JSON construction?
Conclusion: we chose a fixed JSON template where the program forces the invariant
parts (`{`, `"fn_name"`, `"args"`, ...) and the LLM only decides the variable values
(function name and arguments).

**7. How to find out the id of each token (token discovery)**
To force specific tokens in the loop we needed their numeric ids in the Qwen3
vocabulary (~150,000 tokens). The solution was to call `encode("symbol")` once at
startup to map every fixed piece to its ids, building a reusable token dict.

**8. Detecting low model confidence**
Concern about what to do when the prompt matches no available function. We explored
analysing the probability distribution: if the best function name has low probability
and the rest are spread out, the model is unsure. Defining the exact threshold and the
action to take is still pending.

**9. JSON formatting rules**
Doubts about key ordering, indentation or extra restrictions. Conclusion: key order
does not matter (although our template forces one), and the only strict rules are
double quotes on keys and strings, no trailing commas, no comments, 100% parseable by
`json.loads()`.

**10. Separating the LLM's decision from the program's forced parts**
It took a while to separate what the LLM generates (function name and argument
values) from what the program forces directly (braces, quotes, commas, key names such
as `fn_name` and `args`). Conclusion: the program walks the state machine and, in the
"fixed" states, forces the token; in the "decision" states, lets the LLM choose among
a limited set.

**11. Handling prompts with no matching function**
If the model receives a prompt that matches no function it will always generate
something (it cannot stay silent). The proposed solution is detecting a low maximum
probability among function names and handling it as a special case. The concrete
implementation is still undefined.

**12. Triple-quoted strings and the usefulness of `textwrap.dedent()`**
When building the super-prompt with multiline strings (forced to split by flake8's
79-char E501), the output came with leftover indentation and extra blank lines. Python
triple quotes are **literal**: every character between them — including the code's own
indentation — is part of the final text. The fix was `textwrap.dedent()`, which
computes the minimum common indentation of the non-empty lines and removes it from all
of them. It never collapses spaces or reorders text; it only subtracts the common
prefix. It does **not** touch line breaks, so the leading/trailing `\n` of the triple
quotes must be handled separately (`"""\` to suppress the first one).

**Solution chosen in this project:** in `template_rules` and the function listing the
string content is written at **column 0** (flush left, unindented inside the quotes),
just like the functions. The string is born clean and `dedent()` has nothing to do in
those blocks. `dedent()` is only applied to `template_intro`, whose content stays
indented for readability.

**Trade-off to know:** column-0 text inside triple quotes looks odd in the code (the
text does not align with the surrounding indentation). It is a conscious trade: we
prefer a clean prompt for the model over "pretty" code. flake8 does not complain
because it only checks the indentation of *code*, not of string contents.

**13. Concatenating long strings without exceeding E501**
When assembling the prompt in `build_super_prompt()`, joining the four parts in one
line exceeded 79 characters. The fix is **implicit continuation between parentheses**:
wrapping the expression in `(...)` allows breaking after each `+` with no extra
character:

```python
prompt = (template_intro
          + template_rules
          + template_functions
          + f"\nUser: {input_call}\nOutput: ")
```

**14. Converting `[x, y]` to `'x, y'`**
Python's `-1` indexes from the end of a sequence. `str([1, 2, 3])` is `"[1, 2, 3]"`,
and slicing `[1:-1]` starts at index `1` (skipping `[`) and stops before index `-1`
(skipping `]`), because the end index is exclusive. Result: the brackets disappear in
one cut, keeping the inside intact.

**15. Blockages of Step 4 (constrained-decoding loop)**

- **Confusing the subject's output with the LLM's output.** The subject's
  `output/function_calling_results.json` includes `"prompt"`, but that field is
  written by **your program** (it already holds the prompt). The LLM only generates
  `{"fn_name": ..., "args": ...}`; `"prompt"` is neither forced nor generated in the
  loop.
- **`append` vs `extend` on the id list.** `list.append([90])` inserts the sublist as
  a single element → nested `[[90], ...]`. `list.extend([90])` unpacks and adds the
  bare `90` → flat `[90, ...]`. Building the token sequence requires `extend`, never
  `append`.
- **Do not pre-build the whole sequence.** The JSON cannot be assembled in advance
  because the model decides in the middle (which function, which values). The loop is
  step-by-step: each round adds one token (forced with `extend` or chosen by the
  model) and asks again with the accumulated context.
- **`list.extend()` returns `None`.** `final = prompt_ids.extend(ids_list)` stores
  `None` in `final`. The error `TypeError: 'NoneType' object cannot be converted to
  'Sequence'` shows up in `model.decode(final)`. The list already contains the result:
  pass `prompt_ids` itself to `decode()`.
- **Hiding the builtin `input`.** Using `input` as a parameter name shadows Python's
  native function. It was renamed to `input_call` in `build_super_prompt()`.

**16. Iterating over a list while removing elements skips items**
In the function-identification loop, `for ids in fn_names_tokens` removed candidates
that did not match. Python advances the internal iterator when a removal happens, so
two consecutive removals skip the second element. Fix: iterate over a **copy**
(`fn_ids = fn_names_tokens.copy()`); the copy is cheap (few elements) and guarantees
every candidate is examined exactly once.

**17. The prompt not updating inside the loop → repeated predictions**
`init_prompt_ids` was the token sequence the model received as input; if it never
changed inside the `while`, the model saw the **same sequence** and predicted the
**same token** forever. Fix: `temp_prompt = init_prompt_ids.copy()` and
`temp_prompt.extend([next_id])` each round, so every prediction gets the full context
including previous ones. A copy avoids duplicating tokens in `init_prompt_ids`.

**18. Inverted conditions when building the argument JSON**
When closing the JSON braces, `if i + 1 == len(args_fn)` fired **one position too
early** because `i` was incremented **before** the `if`. It worked for 2 arguments by
coincidence but failed for 3+: it placed `}` after the first arg instead of `, `. Fix:
evaluate whether we just processed the **last** argument (`i == len(args_fn)`).
Lesson: always verify conditional logic with edge cases (1, 2, 3+ arguments).

**19. Infinite `while` when generating arguments: `or` vs `and`, and comparing `int` with `list`**

```python
while next_id != dict_fixed_chars[","] or next_id != dict_fixed_chars["\""][0]:
```

Two combined bugs:

1. **`or` instead of `and`**: `A != x or A != y` is **always True**. It should be
   `and`, so the loop only exits when `next_id` is neither delimiter.
2. **Comparing `int` with `list`**: `dict_fixed_chars[","]` returns a **list** `[11]`,
   not an integer. `dict_fixed_chars[","][0]` accesses the element.

Together they made the condition **never false** → infinite loop.

**20. Double layer of quotes around string values**
The quoted-phrase helper returns phrases **without** surrounding quotes, but the
string masking adds quotes around the chosen candidate. Removing the value by text
(`list.remove(model.decode(value_ids))`) failed with
`ValueError: ... x not in list` because the decoded value still carried the quotes.
Fix: strip the quotes in the decoded value before removing (decode the slice
`value_ids[1:-1]`).

**21. Infinite loop in number masking (logit_masking_number)**
The `while len(pos_i) == 1 or i == 0:` condition looped forever (the run hung ~9
minutes) whenever the filtered token list at a position was empty. The loop must walk
positions only while the candidates are not yet distinguishable, and must stop when
there is nothing to compare at a position. Fix: scan for the first position where the
candidates **differ**, exit cleanly when a position has no tokens (`len(token) > i`),
and make a single model call at that deciding position.

**22. Backslashes in regex arguments**
A pattern like `\d+` written into JSON must survive `json.dump`/`json.loads` as an
escaped `\\d+`. The string encoder doubles backslashes before encoding, so the
round-trip keeps the pattern valid and `json.loads` does not raise an invalid-escape
error.

**23. The `output/` directory is missing on a fresh clone**
`main()` writes `output/function_calling_results.json` with a plain
`open("output/...", "w")`; on a fresh checkout the directory does not exist (the
subject forbids committing it) and the write fails with `FileNotFoundError`. The
program must create the directory before writing the first file.

## Resources:

https://www.geeksforgeeks.org/python/json-loads-in-python/
https://www.geeksforgeeks.org/python/json-load-in-python/