"""The joint SEGMENT+PROCEDURE system prompt, vendored for the container.

Identical to `segpro/prompts_segpro.py`, which is what the adapter was
trained with, except that `extract_answer` is inlined below -- the container cannot
import `orena_sft/prompts.py`. `check_vendored.py` asserts both halves still match
their originals.

The load-bearing difference from the procedure container's prompt is `window_line`,
which takes a START as well as an end. A segment clip is delivered trimmed but its
burned-in clock and its `<... seconds>` markers stay absolute, so the model must be
told which span the question is about: "observed from 00:47:40 to 00:52:40", not
"from 00:00:00".
"""

from __future__ import annotations

import re

from focus.foreign_objects import FO_DEFINITION, FO_DEFINITIONS_FILE, FOType

# `extract_answer` and its regexes below are copied verbatim from
# `orena_sft/prompts.py`, which the container cannot import. `check_vendored.py`
# asserts the copy still behaves identically to the original.
_ANSWER_RE = re.compile(r"[*_`\s]*ANSWER[*_`\s]*[:\-][^\S\n]*(.*)", re.IGNORECASE)
_REASONING_RE = re.compile(r"^[*_`\s]*REASONING[*_`\s]*[:\-].*$", re.IGNORECASE | re.MULTILINE)
# A dangling "Answer 00:10:12" on the fallback path -- marker without delimiter.
_LEADING_ANSWER_RE = re.compile(r"^answer\b[:\-]?\s*", re.IGNORECASE)

# Quotes/emphasis the model may wrap the value in, plus the trailing period that
# alone is enough to fail Binary.verify / Number.verify.
_STRIP_CHARS = " \t\n\"'`*_.:"

_OPEN_ENDED_MAX_LEN = 300  # focus.data.formats.OpenEnded default


def extract_answer(raw: str) -> str:
    """Reduce a structured generation to the bare value to be scored.

    Takes the text after the LAST ``ANSWER:`` marker (last, not first, so a
    model that restates the template before filling it in still parses). Falls
    back to the last non-empty line that is not the reasoning -- which is the
    right guess when the model answers without the marker, and no worse than the
    raw string when it was truncated mid-reasoning.
    """
    text = (raw or "").strip()
    if not text:
        return ""

    matches = list(_ANSWER_RE.finditer(text))
    if matches:
        answer = matches[-1].group(1)
    else:
        without_reasoning = _REASONING_RE.sub("", text)
        lines = [ln for ln in without_reasoning.splitlines() if ln.strip()]
        answer = _LEADING_ANSWER_RE.sub("", lines[-1] if lines else text)

    answer = answer.strip().strip(_STRIP_CHARS).strip()

    # An over-long answer fails OpenEnded.verify outright, i.e. is guaranteed
    # wrong; truncating leaves the judge something scoreable instead.
    if len(answer) > _OPEN_ENDED_MAX_LEN:
        answer = answer[:_OPEN_ENDED_MAX_LEN].rstrip()
    return answer

_INTRO = (
    "You are a surgical video analysis assistant. You are shown one continuous span of "
    "a laparoscopic procedure -- stated as a start time and an end time -- and asked a "
    "single question about it.\n\n"
    "The span is given as frames in chronological order, sampled across the whole "
    "observed period. Each frame carries its timestamp twice: burned into the picture "
    "as hh:mm:ss, and written before it as <SECONDS seconds> -- a frame marked "
    "<3742.0 seconds> was taken 3742.0 seconds (01:02:22) after the procedure began. "
    "Both clocks count from the START OF THE PROCEDURE, not from the start of the "
    "observed span. Read the burned-in clock when you can; it is exact.\n\n"
    "Consecutive frames can be a minute or more apart, so events happen between them. "
    "The question is about the observed span ONLY: nothing before its start and nothing "
    "after its end exists for you to reason about, even though the procedure itself "
    "continues outside it."
)

_DIRECT_SHAPE = (
    "Reply with the answer and nothing else -- no reasoning, no preamble, no\n"
    "explanation, no restating the question. A single short line."
)

_ANSWER_RULES = """\
Rules for the answer:
- Write the value only. No sentence, no explanation, no units, no trailing
  period, and never repeat the question.
- Asks yes or no -> write exactly: yes   or   no
- Asks how many / for a count -> write digits only, e.g. 0 or 3 or 12.
  Count over the WHOLE observed span, not just one frame, unless the question
  names a single time point. Distinct instances of the same class each count once.
- Asks which foreign object class(es) -> write class names exactly as spelled
  in the list above, comma-separated (e.g. Clip, Sponge), or exactly: none
  Never answer with a generic description such as "surgical instrument".
- Asks when something happens -> write hh:mm:ss, read off the clock burned into
  the frames. Those clocks count from the START OF THE PROCEDURE, not from the
  start of the observed span and not from any point mentioned in the question.
  If the event falls between two frames, give your best estimate between their
  timestamps rather than snapping to one. For several time points, separate them
  with commas: 00:09:19, 01:12:44
- Asks for a percentage or a share -> write the number only, e.g. 40
- Lists options to choose from -> copy exactly one of those options, verbatim.
- Anything else -> a short phrase, at most a few words.

If you are unsure, still commit to your single best answer in the required
form. An empty, hedged, or explanatory answer is scored as wrong.\
"""


def build_system_prompt(include_definitions: bool = False, style: str = "direct") -> str:
    """The joint system prompt. `structured` is refused as on both source tracks."""
    if style != "direct":
        raise ValueError(f"segpro supports style='direct' only, got {style!r}")

    if include_definitions:
        objects_block = FO_DEFINITIONS_FILE.read_text().strip()
    else:
        objects_block = (f"{FO_DEFINITION.strip()}\n\n"
                         f"The foreign object classes are exactly: {', '.join(FOType.names())}.")

    return f"{_INTRO}\n\n{objects_block}\n\n{_DIRECT_SHAPE}\n\n{_ANSWER_RULES}\n"


def _hhmmss(t: float) -> str:
    s = int(t)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def window_line(start_time: float, end_time: float) -> str:
    """Prefix stating the span the answer must be about.

    At start_time=0 this is byte-identical to the procedure track's own window_line, so
    procedure rows carry exactly the text the shipped model was trained on. For a
    segment row it is the only thing distinguishing an otherwise identical question from
    its procedure twin.
    """
    return f"Procedure observed from {_hhmmss(start_time)} to {_hhmmss(end_time)}."
