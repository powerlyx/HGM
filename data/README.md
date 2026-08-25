# Data Notes

## locomo10.json (included)

The official public release of [LoCoMo](https://github.com/snap-research/locomo):
10 long-dialogue samples, 1,540 question-answer pairs across categories 1–5. The dataset
is published by Snap Research under its original license and citation requirements; it is
mirrored here unchanged, solely for evaluation reproduction.

## longmemeval_s_cleaned.json (obtain yourself; not distributed)

Official [LongMemEval](https://github.com/xiaowu0162/longmemeval) data. It is bound by the
official usage terms and cannot be redistributed. Please obtain it yourself following the
official repository instructions and place it in this directory with the exact filename
`longmemeval_s_cleaned.json` (500 questions, 6 question types, 30 of which are abstention
questions — a question is an abstention one when its `question_id` ends with `_abs`).

Fields per instance:

| Field | Description |
|---|---|
| `question_id` / `question_type` / `question` / `question_date` | Instance id, type, question text, and the date the question was asked |
| `answer` / `answer_session_ids` | Reference answer and the sessions containing the answer evidence |
| `haystack_dates` / `haystack_session_ids` / `haystack_sessions` | Three equal-length arrays that form the per-instance session haystack |
