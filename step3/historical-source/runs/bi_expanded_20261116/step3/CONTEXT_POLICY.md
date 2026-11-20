# Complete-context experiment (2026-11-16)

Based on release `a62c5a28bfbcf2824904f868ca947e1ae9d936b7`. No SFT/DPO weights.
The new options are opt-in; without them the released input policy remains active.

## Input construction

Read **full-recording** Step 2 word alignment, retaining original speaker and word
IDs. Do not build this input by joining previously cropped model payloads.

`DialogueContext` groups whole punctuated utterances, then joins consecutive
same-speaker utterances in onset order. These are retrieval blocks, not gold floor
turns. Punctuation errors remain an upstream limitation. Silence duration is not
used to split these blocks, so a hesitation cannot be assumed to be a turn boundary
before the model classifies it.

For overlaps, retrieve the complete blocks intersecting the original analysis
window and six substantive-length blocks on each side (three conversational
exchanges). Blocks of up to three words are kept without consuming that budget;
this is only a retrieval heuristic and does not assign a Backchannel label.
At recording boundaries, use every available block. Every supplied word has its
original absolute start/end timestamps. Candidate extraction uses the unchanged
local streams, preserving event IDs, direction and onset; extended context is
passed separately to Qwen and JEV.

For gaps, retain all preceding text and complete following blocks when text fits
the 12,000-character history budget. Otherwise retain the preceding/following three
exchanges as whole blocks. The budget chooses between retrieval strategies; it
never cuts a sentence midway. Always retain both target blocks and the complete
intervening partner speech. Record `context_policy` and the earliest included block.
Word timestamps can make requests considerably larger than text alone; the test
service supports 32,768 tokens. `context_budget.fit_gap_payload` queries its actual
tokenizer and capacity before generation, reserving output tokens plus a margin.
It falls back to preceding complete exchanges if the full word timeline will not
fit. If those do not fit either, return an explicit error without cutting words or
inventing a label. Auxiliary speaker views do not duplicate the word timeline.
Service errors are failed predictions, not valid binary labels.

JEV receives the extended textual history with absolute timestamps. Its mixed
audio remains the target-local clip from the stable pipeline. The prompt explicitly
distinguishes that clip from text outside its audio range. This does **not** claim
that JEV heard three full exchanges of audio. Candidate audio and original channel
synchronization are unchanged. Gap Qwen classification is textual; full-length
audio understanding is not claimed.

`--context-layout anchored` is an additional experimental layout: JEV retains
its proven local audio/transcript view, while Qwen receives both an explicit local
onset view and the complete surrounding dialogue. The full BI dialogue uses block
timestamps rather than duplicating every word's timing; candidate onsets retain
their original word-level alignment. Gap history retains word-level timing.
This layout must preserve `audio_candidate_votes`; a regression test checks that
reordering the payload cannot drop them.

## Duration and label policy

Candidate trigger: adjacent words of the same speaker separated by **more than
0.5 seconds**, as before. This interval may contain the partner's speech and is not
equivalent to continuous silence.

`--long-gap-limit 3` implements the requested engineering policy: strictly more
than 3 seconds becomes `TURN_TRANSITION`. `0` disables the cap. The output retains
`semantic_label`, `label_source`, `long_gap_policy_applied`, and
`duration_semantics_conflict`. Long semantic pauses also receive
`long_gap_semantic_pause`; this is a risk flag, not a third business label.

The optional cap is not presented as a physiological maximum for human hesitation.
The 2025 study below distinguishes long **within-turn** pauses from between-turn
lapses even when both exceed three seconds. The 2015 corpus study shows that
conversational timing depends on sequence/action structure, not duration alone.

- https://doi.org/10.1016/j.obhdp.2025.104455
- https://pmc.ncbi.nlm.nih.gov/articles/PMC4429583/
- https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2015.00731/full

Recommended evaluation: compare local-context baseline, complete context with
semantic labels, and complete context plus the three-second cap on unchanged locked
references. Five- and ten-second thresholds are exploratory sensitivity checks,
not independently selected optimal thresholds. Report regressions as well as fixes.

## Running

Append `--complete-context --long-gap-limit 0` to the existing Step 3 invocation
for semantic classification with complete context. Set `--long-gap-limit 3` for
the explicit duration-override variant. Use a Qwen endpoint with adequate context
capacity through `--qwen-endpoint`. Existing dataset manifests and output schemas
remain usable; new provenance/risk fields are additive.

Automatic gap review also keeps complete context; it no longer reduces revised
inputs back to ten words/six seconds. The original policy of preserving the first
label and recording the review proposal separately remains in force.

Do not promote a variant merely because it was requested or because multiple
model votes agree. Evaluation is locked-reference agreement, not independently
human-verified population accuracy. Publish new code only after user approval.
