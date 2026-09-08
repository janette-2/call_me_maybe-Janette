from typing import Any

from llm_sdk import Small_LLM_Model
import json

import numpy as np
from pydantic import BaseModel, ValidationError

# ---------------------------------------------------------------------------
# Pydantic schema models (mirror function_definitions.json / output schema)
# ---------------------------------------------------------------------------


class ParametroFuncion(BaseModel):
    """Expected shape of each entry inside 'parameters'."""
    type: str


class DefinicionFuncion(BaseModel):
    """Expected shape of each function in functions_definition.json."""
    name: str
    description: str
    parameters: dict[str, ParametroFuncion]
    returns: dict[str, str]


class ResultadoLlamada(BaseModel):
    """Exact shape each object in function_calling_results.json must have."""
    prompt: str
    fn_name: str | None
    args: dict


# Concept word -> regex pattern. Generic (concept driven), not tied to any
# concrete function, so new prompts/functions keep working.
CONCEPTO_A_REGEX = {
    "number": r"\d+", "numbers": r"\d+", "digit": r"\d+", "digits": r"\d+",
    "vowel": r"[aeiouAEIOU]", "vowels": r"[aeiouAEIOU]",
    "consonant": r"[^aeiouAEIOU\s]", "consonants": r"[^aeiouAEIOU\s]",
    "letter": r"[a-zA-Z]", "letters": r"[a-zA-Z]",
    "whitespace": r"\s+", "space": r"\s+", "spaces": r"\s+",
    "punctuation": r"[^\w\s]",
    "uppercase": r"[A-Z]", "lowercase": r"[a-z]",
    "asterisks": r"\*", "asterisk": r"\*",
}


def _encode_ids(model: Small_LLM_Model, texto: str) -> list[int]:
    """Tokenise 'texto' and return its flat list of token ids.

    Args:
        model: The loaded LLM model.
        texto: The text to tokenise.

    Returns:
        The flat list of token ids for 'texto'.
    """
    return [int(x) for x in model.encode(texto).flatten().tolist()]


def masked_argmax(llm_logits: Any, allowed_ids: list[int]) -> int:
    """Return the allowed id with the highest logit (vectorised with numpy).

    Args:
        llm_logits: Raw logits produced by 'get_logits_from_input_ids'.
        allowed_ids: The only token ids that are allowed to be chosen.

    Returns:
        The id among 'allowed_ids' whose logit is highest.
    """
    logits_array = np.asarray(llm_logits)  # Corrobora que sea un array
    mascara = np.full(logits_array.shape, -np.inf)
    mascara[allowed_ids] = logits_array[allowed_ids]
    return int(np.argmax(mascara))


def fixed_ids(model: Small_LLM_Model, dict_functions: dict) -> dict:
    """Map every fixed piece of the JSON template to its token IDs.

    Encodes the punctuation characters ('{', '}', ':', ',', ' ', '"', '\n'),
    the JSON labels ('fn_name', 'args') and the names of every function and
    parameter, so that 'loop_prompt_output' can force them directly in the
    generated token sequence without asking the model.

    Args:
        model: The loaded LLM model used to encode text into token ids.
        dict_functions: The loaded function definitions (from
            'functions_info').

    Returns:
        A dict mapping each fixed piece of text to its token id list.
    """
    dict_fixed: dict = {}
    chars_fixed = ["{", "}", ":", ",", " ", "fn_name", "args", "\"", "\n"]

    for fixed in chars_fixed:
        dict_fixed[fixed] = _encode_ids(model, fixed)

    for fun in dict_functions:
        dict_fixed[fun] = _encode_ids(model, fun)
        for parametro in dict_functions[fun].get("parameters"):
            dict_fixed[parametro] = (
                _encode_ids(model, parametro))

    return dict_fixed


def functions_info() -> dict:
    """Load and validate functions_definition.json with pydantic.

    Distinguishes "file missing" from "invalid JSON" (both required by the
    subject) and discards malformed functions instead of crashing.

    Returns:
        dict mapping every valid function name to its 'dict' information.
    """
    dict_functions: dict = {}

    try:
        with open("data/input/functions_definition.json") as f:
            f_content = json.load(f)
    except FileNotFoundError:
        print("Aviso: no se encontró data/input/functions_definition.json")
        return dict_functions
    except json.JSONDecodeError as e:
        print(f"Aviso: functions_definition.json no es un JSON válido: {e}")
        return dict_functions

    # Validacion de estructuras
    for dictionary in f_content:
        try:
            # **dictionary permite pasarle valores a todas las keys que tiene
            # como si fueran variables separadas (name = "fn_x",
            #  description = "Adds xx") no como un diccionario unificado.
            # dict = {name : "fn_x", description = "Adds xx"}
            # [despliegue de valores de keys más automática]
            funcion_validada = DefinicionFuncion(**dictionary)
        except ValidationError as e:
            print(f"Aviso: función descartada por formato inválido: {e}")
            continue

        dict_functions[funcion_validada.name] = {
            "parameters": dictionary["parameters"],
            "description": funcion_validada.description,
            "returns": dictionary["returns"],
        }

    return dict_functions


def build_super_prompt(dict_functions: dict, input_call: str) -> str:
    """Build the super-prompt that instructs the model what to output.

    Joins the assistant role, the output rules (JSON only, exact structure,
    values taken from the request, quoted phrases taken whole) and a dynamic
    list of the available functions with their descriptions and parameters,
    and ends with the user request plus an 'Output:' marker that tells the
    model where its answer starts.

    Args:
        dict_functions: The loaded function definitions (from
            'functions_info').
        input_call: The user request in natural language.

    Returns:
        The full prompt text ready to be encoded and fed to the model.
    """
    template_intro = """
You are a function-calling assistant. Your task is to
analyze a user request and respond with a valid JSON object that specifies
which available function to call and with what arguments.\n\n"""

    template_rules = """Rules:
- Output ONLY the JSON object. Do not add explanations, greetings or
any other text before or after it.
- The JSON must follow this exact structure:
{"fn_name": "<function name>", "args": {<argument name>: <value>, ...}}
- "fn_name" must be one of the available functions listed below.
- "args" must contain every required argument for that function,
and only those, with the correct type (number, string, boolean).
- Extract every argument value from the exact words or numbers written in
the User request below. Never invent values, never repeat the instruction
words, and never copy the function name or the JSON labels.
- If an argument is a quoted phrase in the request, take the whole phrase as
the value. Strip any surrounding quotes or spaces.
- When a function has more than one value to fill, give each argument the
value the User mentions that matches its labelled role, in the order the
User refers to them.\n\n"""

    template_functions = "Available functions:\n"
    for func in dict_functions:
        dict_parameters = dict_functions[func].get("parameters")
        list_parameters = [
            f"{param} ({dict_parameters[param].get('type')})"
            for param in dict_parameters
        ]
        template_functions += (
            f"- {func}: {dict_functions[func].get('description')}\n"
            f"Parameters: {', '.join(list_parameters)}\n"
        )

    return (template_intro
            + template_rules
            + template_functions
            + f"\nUser: {input_call}\nOutput: ")


def extraer_frases_entrecomilladas(texto: str) -> list[str]:
    """Extract quoted phrases (single/double quotes) in order of appearance.

    Contraction apostrophes (I'm, don't) are treated as part of the word,
    not as a quote delimiter.

    Args:
        texto: The text to scan for quoted phrases.

    Returns:
        List of the quoted phrases, in the order they appear in 'texto'.
    """
    frases: list[str] = []
    i, n = 0, len(texto)
    quote_char: str | None = None
    current: list[str] = []

    while i < n:
        char = texto[i]

        if quote_char is None:
            if char == '"':
                quote_char = '"'
                current = []
            elif char == "'":
                es_contraccion = (
                    0 < i < n - 1
                    and texto[i - 1].isalpha()
                    and texto[i + 1].isalpha())
                if not es_contraccion:
                    quote_char = "'"
                    current = []
        else:  # Ya se ha encontrado la primera comilla ->"..."
            if char == quote_char:
                # Si se encuentra la última comilla,
                # une chars de current
                frase = "".join(current).strip()
                if frase:
                    frases.append(frase)
                quote_char = None
            else:
                current.append(char)  # Va rellenando "['H', 'o', 'l', 'a']"

        i += 1

    return frases


def extraer_palabras(texto: str) -> list[str]:
    """Return the alphanumeric words of 'texto'.

    Words are delimited by any non-alphanumeric character (spaces, quotes,
    punctuation). Underscores are kept as part of the word.

    Args:
        texto: The text to split into words.

    Returns:
        The list of words, in order of appearance.
    """
    palabras: list[str] = []
    actual: list[str] = []
    for ch in texto:
        if ch.isalnum() or ch == "_":
            actual.append(ch)
        else:
            if actual:
                palabras.append("".join(actual))
                actual = []
    if actual:
        palabras.append("".join(actual))
    return palabras


def extraer_numeros(texto: str) -> list[str]:
    """Return the digit sequences found in 'texto', in order.

    Used to build the closed candidate set of numbers the model may pick
    from when generating number-typed arguments.

    Args:
        texto: The text to scan for numbers.

    Returns:
        The list of numbers (as strings), in order of appearance.
    """
    numeros: list[str] = []
    actual: list[str] = []
    for ch in texto:
        if ch.isdigit():
            actual.append(ch)
        else:
            if actual:
                numeros.append("".join(actual))
                actual = []
    if actual:
        numeros.append("".join(actual))
    return numeros


def extraer_palabra_tras_with(texto: str) -> str | None:
    """Return the word right after 'with'/'con'.

    Matches the pattern 'replace X with Y' where Y is the substitution
    target: for prompts like this it is the deterministic value of a
    'replacement' argument, so the model does not need to guess it.

    Args:
        texto: The user request text.

    Returns:
        The word following 'with'/'con', or 'None' if there is no such
        word.
    """
    palabras = extraer_palabras(texto)
    for idx, palabra in enumerate(palabras):
        if palabra.lower() in ("with", "con") and idx + 1 < len(palabras):
            return palabras[idx + 1]
    return None


def inferir_patron_regex(user_prompt: str) -> str | None:
    """Return the regex pattern for a known concept word, or 'None'.

    Looks for a recognised concept word ('numbers', 'vowels', ...) in the
    user request and maps it to a pattern using 'CONCEPTO_A_REGEX'. This is
    deterministic (0 model calls) and concept-driven, so it keeps working
    if the input files change.

    Args:
        user_prompt: The user request text.

    Returns:
        The regex pattern string, or 'None' if no concept word is found.
    """
    for palabra in extraer_palabras(user_prompt.lower()):
        if palabra in CONCEPTO_A_REGEX:
            return CONCEPTO_A_REGEX[palabra]
    return None


def codificar_valor_string(model: Small_LLM_Model, vocab: dict,
                           texto: str) -> list[int]:
    """Encode an ALREADY-DECIDED string value (without using 'get_logits...').

    Backslashes are doubled so a pattern like '\\d+' survives
    'json.loads' as backslash + d instead of an invalid JSON escape.

    Args:
        model: The loaded LLM model used to encode the value text.
        vocab: Mapping from token text to token id (from vocab.json).
        texto: The already-decided string value to encode.

    Returns:
        Token ids including the surrounding quotes.
    """
    texto_seguro = texto.replace("\\", "\\\\")
    contenido_ids = _encode_ids(model, texto_seguro)
    return [vocab["\""]] + contenido_ids + [vocab["\""]]


def logit_masking_number(vocab: dict, model: Small_LLM_Model,
                         init_prompt_ids: list[int], numbers_prompt: list[str]
                         ) -> list[int]:
    """Generate a number argument from the numbers in the user_prompt.

    The value is always one of the numbers written by the user. If there is
    a single candidate it is chosen directly (0 model calls); otherwise the
    loop walks the token positions until the candidates differ, and the
    model picks among the tokens of that deciding position with a single
    'get_logits' call.

    Args:
        vocab: Mapping from token text to token id (from vocab.json).
        model: The loaded LLM model.
        init_prompt_ids: The token sequence built so far (the context).
        numbers_prompt: The candidate numbers extracted from the prompt.

    Returns:
        The token ids of the chosen number (without surrounding quotes).
    """
    context = init_prompt_ids.copy()
    tokens_numbers = [_encode_ids(model, num) for num in numbers_prompt]
    if len(tokens_numbers) == 1:
        return tokens_numbers[0]
    else:
        i = 0
        while True:
            token_pos_i = [token[i] for token in tokens_numbers if len(
                token) > i]
            if len(token_pos_i) != 1 or not token_pos_i:
                break
            i += 1

        candidates = token_pos_i
        llm_logits = model.get_logits_from_input_ids(context)
        best_id = masked_argmax(llm_logits, candidates)

        next_id: list[int] = []
        for id in tokens_numbers:
            if len(id) > i and id[i] == best_id:
                next_id = tokens_numbers[i]

        return next_id


def logit_masking_boolean(vocab: dict, model: Small_LLM_Model,
                          init_prompt_ids: list[int]) -> list[int]:
    """Generate a boolean argument ('true'/'false') with logit masking.

    Each step the model may only choose 'true', 'false' or a stop token
    (',' or '}'). The loop keeps generating tokens until a stop token is
    chosen.

    Args:
        vocab: Mapping from token text to token id (from vocab.json).
        model: The loaded LLM model.
        init_prompt_ids: The token sequence built so far (the context).

    Returns:
        The token ids of the boolean value, without the stop token.
    """
    context = init_prompt_ids.copy()
    ids_booleans = [vocab["true"], vocab["false"]]
    ids_parada = [vocab[","], vocab["}"]]
    candidates = ids_booleans + ids_parada

    next_id: list[int] = []
    flag = True
    while flag:
        llm_logits = model.get_logits_from_input_ids(context)
        best_id = masked_argmax(llm_logits, candidates)

        if best_id in ids_parada:
            flag = False
        else:
            next_id.append(best_id)
            context.append(best_id)

    return next_id


def logit_masking_string(vocab: dict, model: Small_LLM_Model,
                         init_prompt_ids: list[int],
                         candidates_list: list[str]) -> list[int]:
    """Pick a string value from a closed candidate set with ONE model call.

    The value is guaranteed to be spelled with the exact tokens of one of
    'candidates_list' (the words of the user request). A single
    'get_logits...' call is enough: we score each candidate by the SUM of the
    logits of all its tokens and take the best one, which is both faster
    (one call instead of one-per-token) and robust to candidates that share
    a first token.

    Args:
        vocab: Mapping from token text to token id (from vocab.json).
        model: The loaded LLM model.
        init_prompt_ids: The token sequence built so far (the context).
        candidates_list: The words/phrases the value may be picked from.

    Returns:
        Token ids including the surrounding quotes.
    """
    context = init_prompt_ids.copy()
    possible_tokens: list[list[int]] = []
    for candidato in candidates_list:
        ids = _encode_ids(model, candidato)
        if ids:
            possible_tokens.append(ids)

    llm_logits = model.get_logits_from_input_ids(context)
    best_candidate = max(
        possible_tokens,
        key=lambda t: sum(np.asarray(llm_logits)[tk] for tk in t),
    )
    return [vocab["\""]] + best_candidate + [vocab["\""]]


def resolver_falla_fn(user_prompt: str, dict_functions: dict) -> str | None:
    """Pick a function deterministically when the model fails.

    Scores every available function by how many of the prompt's words appear
    in its 'description' or 'name' (lowercased). This is generic: it
    relies on the 'description' field of the schema, so it survives new
    prompts and new function sets.

    Args:
        user_prompt: The user request text.
        dict_functions: The loaded function definitions.

    Returns:
        The best-matching function name, or 'None' if no function matches.
    """
    palabras = extraer_palabras(user_prompt.lower())
    # Palabras del prompt del usuario
    vocabulario = {p for p in palabras if len(p) > 2}

    mejor_fn = None
    mejor_score = 0
    for fn, info in dict_functions.items():
        campo = (fn + " " + info.get("description", "")).lower()
        palabras_fn = set(extraer_palabras(campo))
        # Score : Recuento de palabras coincidentes entre el user prompt
        # y la descripcion y nombre de la fn
        score = len(vocabulario & palabras_fn)
        if score > mejor_score:
            mejor_score = score
            mejor_fn = fn

    return mejor_fn


def loop_prompt_output(input: str, model: Small_LLM_Model,
                       dict_fixed_chars: dict[str, list[int]],
                       dict_functions: dict,
                       user_prompt: str) -> list[int]:
    """Run constrained decoding and return the full token sequence.

    Builds the JSON token by token: the fixed parts ('{', 'fn_name',
    'args', commas, braces) are forced directly from 'dict_fixed_chars',
    while the variable parts (function name and argument values) are
    delegated to the model with logit masking, so the output is always a
    valid JSON. The function is identified by '_escoge_fn' and each
    argument is resolved by type (number/boolean/string); the string hints
    follow a priority (regex concept → quote-taken phrase → word after
    'with') before the model picks among the prompt words.

    Args:
        input: The encoded super-prompt text to start the sequence.
        model: The loaded LLM model.
        dict_fixed_chars: Mapping of fixed JSON pieces to their token ids.
        dict_functions: The loaded function definitions.
        user_prompt: The user request, used to extract argument values.

    Returns:
        The complete token id sequence of the JSON answer.
    """

    def _escoge_fn() -> list[int]:
        """Identify the function with a SINGLE 'get_logits' call.

        All function names share the leading 'fn' token, so the first
        position where their tokens differ uniquely separates them. We make
        one model call and pick, among the tokens in that deciding position,
        the one with the highest logit. This replaces the old token-by-token
        elimination (one call per token) with a single call, saving a lot of
        budget without losing precision (the deterministic fallback below
        still guarantees a correct function even if the model hesitates).

        Function defined inside of 'loop_prompt_output()' because it is
        the only function that calls it and this way it can use the local
        arguments of 'loop_prompt_output()' ('dict_fixed_chars',
        'init_prompt_ids') without having to pass them as parameters.
        """
        fn_names_tokens = [dict_fixed_chars[name] for name in dict_functions]

        longitud_max = max((len(t) for t in fn_names_tokens), default=0)
        posicion = 0
        for posicion in range(longitud_max):
            tokens_pos = {t[posicion] for t in fn_names_tokens
                          if len(t) > posicion}
            # If the functions have common prefixes, the first
            # ids sould have a set of len(tokens_pos) == 1.
            if len(tokens_pos) > 1:
                break
            # Breaks when it has arrive to the first id position
            # that is not common to all the fn_names.

        llm_logits = model.get_logits_from_input_ids(init_prompt_ids)
        allowed = [t[posicion] for t in fn_names_tokens if len(t) > posicion]
        best_id = masked_argmax(llm_logits, allowed)

        for tokens in fn_names_tokens:
            if len(tokens) > posicion and tokens[posicion] == best_id:
                return tokens

        # No function matched: fall back to a deterministic choice.
        fn = resolver_falla_fn(user_prompt, dict_functions)
        if fn is not None:
            return dict_fixed_chars[fn]
        return []  # Not expected to reach this return

    init_prompt_ids: list[int] = _encode_ids(model, input)

    init_prompt_ids.extend(dict_fixed_chars["{"])
    init_prompt_ids.extend(dict_fixed_chars["\""])
    init_prompt_ids.extend(dict_fixed_chars["fn_name"])
    init_prompt_ids.extend(dict_fixed_chars["\""])
    init_prompt_ids.extend(dict_fixed_chars[":"])
    init_prompt_ids.extend(dict_fixed_chars[" "])
    init_prompt_ids.extend(dict_fixed_chars["\""])

    fn_tokens = _escoge_fn()
    init_prompt_ids.extend(fn_tokens)

    init_prompt_ids.extend(dict_fixed_chars["\""])
    init_prompt_ids.extend(dict_fixed_chars[","])
    init_prompt_ids.extend(dict_fixed_chars[" "])
    init_prompt_ids.extend(dict_fixed_chars["\""])
    init_prompt_ids.extend(dict_fixed_chars["args"])
    init_prompt_ids.extend(dict_fixed_chars["\""])
    init_prompt_ids.extend(dict_fixed_chars[":"])
    init_prompt_ids.extend(dict_fixed_chars[" "])
    init_prompt_ids.extend(dict_fixed_chars["{"])

    fn = model.decode(fn_tokens)
    args_fn = [arg for arg in dict_functions[fn]["parameters"]]

    path_vocab = model.get_path_to_vocab_file()
    with open(path_vocab) as file:
        vocab = json.load(file)

    numeros_prompt = extraer_numeros(user_prompt)
    frases_entrecomilladas = extraer_frases_entrecomilladas(
                        user_prompt)

    n_args = len(args_fn)
    for idx, arg in enumerate(args_fn):
        arg_type = dict_functions[fn]["parameters"][arg]["type"]
        is_last = (idx == n_args - 1)  # Booleano

        init_prompt_ids.extend(dict_fixed_chars["\""])
        init_prompt_ids.extend(dict_fixed_chars[arg])
        init_prompt_ids.extend(dict_fixed_chars["\""])
        init_prompt_ids.extend(dict_fixed_chars[":"])
        init_prompt_ids.extend(dict_fixed_chars[" "])

        if arg_type == "number":
            value_ids = logit_masking_number(vocab, model,
                                             init_prompt_ids,
                                             numeros_prompt)
            numeros_prompt.remove(model.decode(value_ids))

        elif arg_type == "boolean":
            value_ids = logit_masking_boolean(vocab, model, init_prompt_ids)
        else:
            patron = None
            if "regex" in arg or "pattern" in arg:
                patron = inferir_patron_regex(user_prompt)

            if patron is not None:
                # 1. Recognised regex concept -> deterministic, 0 model calls.
                value_ids = codificar_valor_string(model, vocab, patron)

            elif (("replace" in arg or "replacement" in arg)
                  and (with_value := extraer_palabra_tras_with(user_prompt))
                  is not None):
                # The := assigns a value at the same time
                # that corroborate its condition
                value_ids = codificar_valor_string(model, vocab, with_value)

            else:
                if frases_entrecomilladas:
                    value_ids = logit_masking_string(
                                            vocab, model, init_prompt_ids,
                                            frases_entrecomilladas)
                    frases_entrecomilladas.remove(model.decode(
                        value_ids[1:-1]))
                else:
                    value_ids = logit_masking_string(
                        vocab, model, init_prompt_ids,
                        extraer_palabras(user_prompt))

        init_prompt_ids.extend(value_ids)

        if is_last:
            init_prompt_ids.extend(dict_fixed_chars["}"])
        else:
            init_prompt_ids.extend(dict_fixed_chars[","])
            init_prompt_ids.extend(dict_fixed_chars[" "])

    init_prompt_ids.extend(dict_fixed_chars["}"])
    return init_prompt_ids


def main() -> None:
    """Process every prompt in the test file and write the results.

    Orchestrates the whole pipeline: it loads the model and the function
    definitions, builds the fixed token ids and the super-prompt, runs the
    constrained-decoding loop for every prompt, validates each result with
    the pydantic schema (falling back to the deterministic description-based
    function choice when the model fails, and coercing numbers to 'float')
    and finally writes 'output/function_calling_results.json'. It prints a
    warning and exits with an empty result list if the input files are
    missing or invalid.
    """
    model = Small_LLM_Model()
    dict_functions = functions_info()

    if not dict_functions:
        print("Aviso: no hay funciones disponibles, no se puede continuar.")
        with open("output/function_calling_results.json", "w",
                  encoding="utf-8") as f:
            json.dump([], f, indent=4, ensure_ascii=False)
        return

    dict_fixed_chars = fixed_ids(model, dict_functions)

    try:
        with open("data/input/function_calling_tests.json") as file:
            tests = json.load(file)
    except FileNotFoundError:
        print("Aviso: no se encontró data/input/function_calling_tests.json")
        tests = []
    except json.JSONDecodeError as e:
        print(f"Aviso: function_calling_tests.json no es un JSON válido: {e}")
        tests = []

    results = []
    for test in tests:
        user_prompt = test.get("prompt")
        if not user_prompt:
            print(f"Aviso: entrada sin 'prompt', se omite: {test}")
            continue

        super_prompt = build_super_prompt(dict_functions, user_prompt)
        final_prompt_ids = loop_prompt_output(super_prompt, model,
                                              dict_fixed_chars,
                                              dict_functions, user_prompt)
        final_output = model.decode(final_prompt_ids)

        try:
            result_json = final_output.split("Output: ", 1)[1]
            result = json.loads(result_json)
        except (IndexError, json.JSONDecodeError):
            result = {"fn_name": None, "args": {}}

        fn_name = result.get("fn_name")
        args = result.get("args") or {}

        if fn_name not in dict_functions:
            fn_name = resolver_falla_fn(user_prompt, dict_functions)

        # Coerce argument types to the schema before validating (subject:
        # "los tipos deben coincidir con la definición", numbers -> float).
        if fn_name in dict_functions:
            parametros = dict_functions[fn_name]["parameters"]
            for nombre_arg, valor in list(args.items()):
                tipo = parametros.get(nombre_arg, {}).get("type")
                if tipo == "number" and isinstance(valor, (int, float)):
                    args[nombre_arg] = float(valor)

        try:
            resultado_validado = ResultadoLlamada(
                prompt=user_prompt, fn_name=fn_name, args=args)
            results.append(resultado_validado.model_dump())
        except ValidationError as e:
            print(f"Aviso: resultado inválido para '{user_prompt}': {e}")
            results.append({"prompt": user_prompt,
                            "fn_name": None, "args": {}})

    with open("output/function_calling_results.json", "w",
              encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
