"""
Getting JSON back out of a model's answer.

Models wrap their JSON in prose, in a fenced block, or hand back an object that
was cut mid-write. :func:`_extract_json` finds and parses the payload;
:func:`_unclosed_json` tells a cut-off answer (retry with more room) from a
merely malformed one (retry differently).
"""

import json
import re

from pysyrev.core.llm.ratelimit import TruncatedResponse


def _unclosed_json(text: str) -> bool:
    """Does `text` end inside an unfinished JSON structure?

    The tell of a cut-off response, and not the same thing as "does not end in a
    brace": a response truncated just after an array item ends in ``}`` with its
    array and object still open — which is exactly the case that used to be
    reported as a delimiter error. Strings are tracked so a brace inside a
    reasoning sentence does not count as structure.
    """
    depth = 0
    in_string = escaped = started = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "{[":
            depth += 1
            started = True
        elif char in "}]":
            depth -= 1
    return started and (depth > 0 or in_string)


def _extract_json(text: str):
    """Parse JSON from text; falls back to regex extraction if there is a preamble.

    The fallback is deliberately greedy so a fenced or prefaced object still
    parses. That same greed makes a *truncated* response fail confusingly — it
    matches up to the last complete ``}``, and the decoder then reports a
    delimiter error deep inside a response that is simply cut off — so the
    truncated case is named here rather than left to the raw decoder message.
    Providers that expose ``finish_reason`` raise :class:`TruncatedResponse`
    before reaching this point; this covers the ones that do not.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        match = re.search(r'\{.*\}|\[.*\]', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError as inner:
                if _unclosed_json(text):
                    raise TruncatedResponse() from inner
                raise ValueError(
                    f"Malformed JSON in response ({inner}): {text[:300]!r}"
                ) from inner
        if _unclosed_json(text):
            raise TruncatedResponse() from exc
        raise ValueError(f"No JSON found in response: {text[:300]!r}") from exc
