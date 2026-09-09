"""The joint SEGMENT+PROCEDURE system prompt.

`extract_answer` is track-agnostic and is imported from the shared `prompts` module.

The one substantive change is scope. Both tracks ask the same questions -- 703 question
strings appear verbatim in both, and shared templates cover 85% of procedure rows and
69% of segment rows -- but "in this video" means a 5-minute clip on segment and the whole
prefix on procedure. Of the 2,169 (video, question) pairs that occur in both tracks, 34%
have DIFFERENT answers for that reason. Training on the union without saying which span
is meant is training on contradictions.

So every row, on either track, is prefixed with the span it may reason about, and the
intro below describes an observed window with two stated ends instead of a procedure
observed from its beginning. For a procedure row `window_line` emits exactly the string
the shipped model was trained on, so that half of the data is unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "orena_sft"))

from focus.foreign_objects import FO_DEFINITION, FO_DEFINITIONS_FILE, FOType  # noqa: E402

from prompts import extract_answer  # noqa: E402,F401  (re-exported for the evaluator)

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
