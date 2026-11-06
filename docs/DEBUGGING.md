# Debugging drill: travel reimbursement answers go wrong after a policy update

Reproduced on purpose to match the brief's scenario: *after a "document update", users report
that questions about travel reimbursement return the expense policy's content, or "I don't know".*
Everything below was measured on this repo with the MiniLM embedder and the mock LLM.

## Timeline (git history)

| Commit | What |
|---|---|
| `0b623cd` | (prep) fix a mock-LLM stemming bug found while setting up, so it would not distort the numbers |
| `a44afa3` | **the break**: `travel.md` update (claim deadline 14 → 21 days, payout 7 → 5 business days, written-exception clause, wording aligned with the expenses policy). The index was **not** rebuilt. |
| `543bb3c` | investigation tooling: `python -m app.debug` and a per-request retrieval log line |
| `9ca8c1f`, `c028e62`, `0590d29` | **fix 1**: stale-index detection, `/health` reporting and startup rebuild, with regression tests |
| `9d32434`, `6ef6cfc` | **fix 2**: hybrid (dense + BM25) retrieval |
| `bdad7c6`, `2f3d75c` | regression question set and the top-1 document metric |

## 1. Symptoms

After the travel policy update, with the service restarted but the index untouched:

| Question | Answer given | Correct answer (new policy) |
|---|---|---|
| How long until my travel claim is reimbursed? | "…reimbursed within **7** business days." | 5 business days |
| What is the deadline to submit a travel reimbursement claim? | "…within **14** days of returning…" | 21 days |
| Who can grant an exception for a late travel claim? | the old 14-day sentence (no exception clause) | Head of Department |
| How do I get reimbursed for a business trip? | "Costs of overnight business trips… are governed by the Travel Policy…" **`[expenses-0]`** | submit in TravelDesk |

So both reported symptoms occurred: outdated travel facts, and **expense-policy content for a
travel question**. "I don't know" did not occur with MiniLM, where top scores stayed above the
0.35 refusal threshold. It appears with the weaker hash embedder, whose low scores fall under its
threshold; see the README eval section.

## 2. Investigation

**Logs.** Each `/ask` now logs the retrieved chunk ids and scores:

```
INFO helpdesk ask q='How long until my travel claim is reimbursed?'
     top=[travel-2:0.70, expenses-0:0.55, expenses-1:0.46, expenses-2:0.45]
     answer='Approved travel claims are reimbursed within 7 business days. [travel-2]'
INFO helpdesk ask q='How do I get reimbursed for a business trip?'
     top=[travel-2:0.59, expenses-0:0.59, expenses-2:0.56, expenses-1:0.51]
     answer='Costs of overnight business trips, such as flights and hotels, are governed by t…'
```

Two different patterns:

1. The right document (`travel-2`) is retrieved with a high score, but its text says "7 business
   days" while `travel.md` on disk says 5. **The chunk text does not match the file.**
2. Travel and expense chunks score almost the same (0.59 vs 0.59), and the expense chunk wins.

**Retrieved chunks.** `python -m app.debug "<question>" --full` prints the top-k chunks. It
confirmed the index text was the pre-update wording, and the index metadata showed
`built_at=…14:37`, which was before the edit. Running the same questions against a freshly
rebuilt index fixed the outdated facts, but not the expense-content answer:

```
# fresh index, dense retrieval
 # cosine   bm25  chunk_id    text
 1  0.586   3.15  expenses-0  …Costs of overnight business trips, such as flights and hotels…
 2  0.580   5.01  travel-2    …Travel claims must be submitted in TravelDesk within 21 days…
```

So there are **two independent causes**. Rebuilding alone was necessary but not sufficient.

## 3. Root causes

1. **Stale index.** The service embeds documents once at build time and never checked whether
   they changed. The doc hashes were stored in the index metadata (Phase 2) but nothing compared
   them. Every travel answer came from the old text.
2. **Near-duplicate policies competing in dense retrieval.** The expenses policy also talks about
   trips and reimbursement: its scope sentence says trip costs "are reimbursed under different
   rules". The update made this worse by aligning travel's wording with expenses ("reimbursed by
   bank transfer within N business days of final approval"). MiniLM embeds both passages almost
   identically (0.586 vs 0.580), so the ranking between them is effectively a coin toss. The
   numbers show the update's effect: top-1 document rate on the regression set was **75% on the
   old text but 62.5% after rebuilding with the new, more similar wording**.

## 4. Fixes

**Fix 1: detect and repair a stale index** (`ingest/freshness.py`, `app/main.py`)
- `check_freshness` compares each document's SHA-256 (CRLF-normalised) and the chunking
  strategy, size and overlap recorded at build time with what is on disk now.
- At startup a stale index is always logged as a warning. With `AUTO_REBUILD_INDEX=true` (the
  default) it is rebuilt using the index's own embedder, so score thresholds stay valid.
- `/health` reports `status: "stale"`, `index_stale` and `stale_documents`. It returns HTTP 200
  (still serving) so monitoring can alert without taking the service down. Edits made while the
  service runs show up there too.
- `python -m ingest.freshness` exits 1 when stale, for CI or deploy checks.

**Fix 2: hybrid retrieval** (`app/bm25.py`, `app/retriever.py`, `RETRIEVAL_MODE=hybrid`)
- A dependency-free BM25 runs alongside dense search, and the two rankings are merged with
  Reciprocal Rank Fusion. BM25 rewards the rare, distinguishing terms that embeddings blur
  ("TravelDesk", "trip", "hotel" vs "ExpenseHub"). On the example above BM25 scores travel-2
  5.01 vs expenses-0 3.15, which lifts travel to first.
- `score` stays the dense cosine, so refusal thresholds keep their meaning. The mock now gates on
  the best cosine in the result, because fused order is not cosine order.

**Rejected: section-based chunking with title/section headers.** This was my first hypothesis.
I implemented and measured it on the 17 answerable questions from both sets:

| Variant | Top-1 doc correct |
|---|---|
| word windows, dense (current at the time) | 13/17 |
| sections + title/section header, dense | 15/17 |
| word windows, **hybrid** | **17/17** |
| word windows, cross-encoder rerank | 17/17 (needs a second model download) |

Hybrid solved the problem without a new model or re-chunking, so the section chunker was
reverted rather than shipped. The chunk boundary split named in the brief was checked and did not
occur here: the reimbursement section sits inside one 120-word chunk.

## 5. Before and after

**A** = the incident (stale index, dense). **B** = after fix 1 (rebuilt, dense). **C** = after
both fixes (rebuilt, hybrid). Mock LLM, top-k 4.

| Eval set | Metric | A | B | C |
|---|---|---|---|---|
| Regression (8: 6 travel reimbursement + 2 expense controls) | top-1 document | 75% | 62.5% | **100%** |
| | answer correctness | 25% | 75% | 75% |
| Main (12) | top-1 document | 88.9% | 88.9% | **100%** |
| | answer correctness | 88.9% | 100% | 100% |
| | refusal accuracy | 100% | 100% | 100% |

The retrieval hit rate (right document anywhere in the top 4) was 100% in every column. It is
blind to this failure, which is why the top-1 document rate was added.

**Still failing: r03 and r05** ("How do I get reimbursed for a business trip?", "How do I claim
back my flight and hotel costs?"). After fix 2 the travel chunk is ranked first, but the
**mock** LLM picks sentences by keyword overlap and still prefers the expense scope sentence,
which matches more of the question's words. Telling that this sentence is only a pointer to the
Travel Policy needs language understanding, which is what the real LLM provides. I have not
verified this with a real provider (no API key in this environment). Running
`python -m eval.run_eval --questions eval/regression_questions.json` with `LLM_PROVIDER=openai`
is the check.

I chose the wording of r06 and r07 so that a correct retrieval gives a correct *mock* answer. My
first phrasings failed only because the mock is lexical ("paid" ≠ "reimbursed"), which would have
hidden the retrieval signal.

## 6. Regression protection

| Check | Catches |
|---|---|
| `tests/test_stale_index.py::test_startup_rebuilds_a_stale_index_so_answers_use_the_new_text` | a policy edit not reflected after restart (fails on the pre-fix code) |
| `test_health_reports_a_document_edited_while_running` | `/health` missing a live edit |
| `test_local_index_is_not_stale_against_the_policies` | a developer's `data/index` older than `data/policies`; skipped when no index exists |
| freshness unit tests | edits, additions, removals, chunking changes; line-ending-only changes are *not* stale |
| `tests/test_hybrid_retrieval.py` | dense ranking a near-duplicate first when the query's exact terms point elsewhere |
| `eval/regression_questions.json` + top-1 metric | the incident's questions end to end: `python -m eval.run_eval --questions eval/regression_questions.json` |
| `tests/test_eval.py` | expected facts drifting away from the policy text (both question sets) |

## Reproduce

```bash
git checkout 543bb3c                   # break + tooling, before any fix
git checkout 0b623cd -- data/policies  # old travel.md
python -m ingest.build_index           # index of the OLD text
git checkout 543bb3c -- data/policies  # the update, no rebuild
python -m app.debug "How long until my travel claim is reimbursed?"   # answers 7 days
git checkout main && python -m ingest.freshness                       # reports STALE
```
