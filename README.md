# Echo-Sight

A voice-first question-answering assistant for blind and low-vision students. Ask Northeastern University's
Code of Student Conduct a question out loud and hear a short answer grounded in the document, with the
section and page it came from.

It runs for free: on Groq's free API tier by default, or fully offline on a laptop with Ollama.

## Results

All numbers come from a held-out test split of a 186-question labelled set. Settings were chosen on a
separate dev split. See [Evaluation](#evaluation) for how each metric is computed.

<!-- RESULTS:START -->
**Retrieval**: 77 answerable test questions, no LLM involved.

| Metric | v1 | Echo-Sight |
|---|---|---|
| Top-ranked passage contains the answer (Hit@1) | 51.9% | **76.6%** |
| Answer in the top 5 passages (Hit@5) | 75.3% | **89.6%** |
| Required evidence among the 4 passages the LLM sees | 72.7% | **89.0%** |
| Mean reciprocal rank (MRR@10) | 0.619 | **0.827** |

**Answers**: 92 test questions (77 answerable, 15 out of scope).

| Metric | v1 pipeline<br>(local qwen3.5:4b) | Echo-Sight offline<br>(local qwen3.5:4b) | Echo-Sight default<br>(Groq gpt-oss-120b) |
|---|---|---|---|
| Overall accuracy (correct answer or correct refusal) | 64.1% | 73.9% | 80.4% |
| Answer accuracy (fully correct, LLM-judged) | 58.4% | 71.4% | 76.6% |
| Partly or fully correct | 85.7% | 94.8% | 92.2% |
| Faithful to the retrieved passages (LLM-judged) | — | — | 94.4% |
| Citations point to the evidence | — | 85.7% | 94.4% |
| Out-of-scope questions declined | 93.3% | 86.7% | 100.0% |
| Answerable questions wrongly declined | 10.4% | 0.0% | 6.5% |
| Required evidence in the LLM's context | 77.3% | 89.0% | 89.0% |
| Tokens per question (prompt + answer) | 997 | 815 | 867 |
| First spoken sentence, median | 6.9 s | 6.2 s | 0.9 s |
| Full answer, median | 10.2 s | 8.0 s | 0.9 s |
| Full answer, 95th percentile | 12.6 s | 10.9 s | 2.8 s |

Judge: groq/qwen/qwen3.8-27b. Citation accuracy is computed without a judge. The v1 pipeline does not cite.

**Time until a screen-reader user hears the answer** (median over test questions)

| Approach | Median | 90th percentile |
|---|---|---|
| Listen from the top of the PDF at 180 words/min | 30 min | 52 min |
| Expert heading navigation at 360 words/min | 50 s | 99 s |
| Echo-Sight offline (local qwen3.5:4b): first spoken sentence | 6.2 s | — |
| Echo-Sight default (Groq gpt-oss-120b): first spoken sentence | 0.9 s | — |
<!-- RESULTS:END -->

What the end-to-end numbers show:

- **The pipeline matters as much as the model.** With the same local 4B model, v2 answers 71.4% of questions
  fully correctly against v1's 58.4%, and stops wrongly declining answerable ones (10.4% → 0%).
- **Groq's errors are honest.** All 5 answerable questions it declined had no evidence in the 4 retrieved
  passages, and it said so rather than guessing (faithfulness 94.4%). The local model
  guesses more and gets more partial credit (94.8% partly or fully correct vs 92.2%). Retrieval on
  everyday-language "scenario" questions is the main remaining weakness: Groq gets 64.3% of them fully
  correct, against 92.6% of detail questions.
- **Voice latency.** With Groq the first sentence is spoken in under a second (p50 0.9 s), more than 50× faster
  than an expert navigating by headings.

### Example

A real answer from the offline-mode test run (local qwen3.5:4b), shown in the app's output format. Answers
vary slightly between runs because sampling temperature is 0.2. All three citations point to passages from
the Medical Amnesty subsection.

```
$ python app.py --provider ollama -q "My friend is dangerously drunk. Will I get in trouble if I call for help?"

No, you will not get in trouble for calling for medical assistance [1]. You must stay with your friend
until emergency help arrives to be protected by this policy [2]. However, the university can still take
disciplinary action if they find out you violated the rules repeatedly [4].
Sources: [1] Medical Amnesty, page 8; [2] Medical Amnesty, page 8; [4] Medical Amnesty, page 8

Spoken: "No, you will not get in trouble for calling for medical assistance. ... Source: Medical Amnesty, page 8."
```

## The problem

The Code of Student Conduct is a 16-page, 11,500-word PDF. A screen-reader user who listens from the top
reaches a typical answer after **30 minutes** at a default speech rate of 180 words per minute. Even an
expert who jumps straight from heading to heading at 360 words per minute needs about **50 seconds**,
and that assumes they recognise the right heading immediately. Echo-Sight answers the question directly.

## How it works

```
PDF ──► heading parser ──► heading-aware chunks ──┬──► BM25 (stemmed) ──────┐
        (font styles →        (≤800 chars, never  │                         ├─► RRF fusion ─► cross-encoder ─► top 4
         78-heading tree)      cross a heading,   └──► bge-base embeddings ─┘   (top 20)       reranker      passages
                               heading path kept)       (Chroma)                                                │
                                                                                                                ▼
     spoken answer ◄── sentence-by-sentence TTS ◄── answer with [n] citations ◄── LLM (Groq gpt-oss-120b, or local qwen3.5:4b)
```

1. **Parsing** ([echosight/document.py](echosight/document.py)). Headings are detected from font size and
   weight with PyMuPDF. The result is the document's real outline: 9 parts, 50 sections and 18 subsections,
   matching the PDF's own H2/H3/H4 accessibility tags. Running headers are dropped, paragraphs split by page
   breaks are rejoined, and typographic ligatures (the single character "ﬁ" in "ﬁle") become plain letters so
   keyword search can match them.
2. **Chunking** ([echosight/chunking.py](echosight/chunking.py)). Paragraphs are packed into chunks of at
   most 800 characters that never cross a heading. Each chunk is indexed with its heading path, for example
   `Standards of Conduct > DOXING`, which gives short policy definitions the context they lack.
3. **Retrieval** ([echosight/retrieval.py](echosight/retrieval.py)). BM25 and dense search
   (bge-base-en-v1.5 in Chroma) each return 20 candidates. They are fused with reciprocal rank fusion and
   reranked by a cross-encoder (ms-marco-MiniLM-L-6-v2). The top 4 passages go to the LLM.
4. **Generation** ([echosight/rag.py](echosight/rag.py), [echosight/prompts.py](echosight/prompts.py)).
   The LLM answers in two to four spoken-style sentences, cites passages as `[n]`, and replies with a fixed
   sentence when the passages don't contain the answer. Any OpenAI-compatible endpoint works: Groq, Ollama,
   Gemini or others.
5. **Voice** ([echosight/voice.py](echosight/voice.py)). The answer streams into the speech engine one
   sentence at a time, so speech starts before generation finishes. Citation markers are removed for
   speech and replaced by a spoken source line ("Source: Doxing, page 9"). `--voice` adds push-to-talk
   input with a local Whisper model (faster-whisper base.en), which transcribes a 4.5-second question in
   about 0.1 s on CPU.

### What went wrong in v1

The first version used a regex for all-caps headings. It never indexed the first 8.5% of the document
(scope, off-campus jurisdiction, student expectations), and it missed every heading containing a comma,
slash or hyphen, plus all title-case headings. As a result, one "section" held a quarter of the document:
sanctions, interim suspension, no-contact orders and record keeping were all labelled
`VIOLATION OF UNIVERSITY POLICIES`. The v1 pipeline is reproduced exactly in
[eval/v1_baseline.py](eval/v1_baseline.py) and serves as the baseline throughout.

The missing text changes answers, not just scores. Here are both pipelines with the same local LLM, on a
test question:

> **Can I get disciplined for something I did off campus?**
>
> **v1:** "Yes, you can be disciplined for actions taken off campus *if your violation involves possession or
> consumption of drugs, alcohol, or marijuana* as outlined in the Sanction Guidelines section..." This is
> wrong: the jurisdiction section was never indexed, so the model reasoned from the drug sanctions.
>
> **v2:** "Yes, you can be disciplined for behavior that occurs off campus if it negatively affects the
> university's educational mission or its relationship with the community [1]..." This is correct, and it
> cites *Applicability of the Code On and Off Campus*.

## Evaluation

### Question set

[eval/data/questions.jsonl](eval/data/questions.jsonl) holds 186 questions, split 50/50 into dev and test,
stratified by type:

| Type | Count | Example |
|---|---|---|
| Direct | 34 | "How does the Code define bullying?" |
| Detail | 55 | "How many days do I have to file an appeal?" |
| Scenario (everyday wording, no policy terms) | 57 | "My friend is dangerously drunk. Will I get in trouble if I call for help?" |
| Multi-section | 10 | "If I'm caught with alcohol a second time, what sanctions should I expect, and could my parents be notified?" |
| Out of scope (24 near-misses, 6 unrelated) | 30 | "How do I appeal a campus parking ticket?" |

Each answerable question has a reference answer and one or more **evidence quotes**, copied verbatim from
the PDF. A test checks that every quote really appears in the document. A retrieved passage counts as
relevant only if it contains an evidence quote. Because this doesn't depend on how a pipeline splits the
text, v1 and every v2 variant are scored the same way. When a long quote is cut by a chunk boundary, a
chunk holding at least half of it at the cut (the chunk ends with the quote's beginning or starts with its
end) still gets credit, which avoids penalising v1's character-based chunks.

Before the final scoring, a separate model instance independently audited every label against the PDF.
It found 14 items to fix, including 4 reference answers that overstated the Code and quotes that were too
generic, and it showed that the original partial-credit rule credited unrelated passages. All of it was
corrected and every number re-computed. The audit found all 30 out-of-scope labels correct.

The questions were written for this project with AI assistance. Every label is checked programmatically,
but the set has not been reviewed by OSCCR staff.

### Metrics

- **Hit@k / MRR@10**: whether an evidence passage ranks in the top k, and how high.
- **Evidence recall@4**: the share of required evidence present in the 4 passages the LLM actually sees.
- **Answer accuracy**: an LLM judge (a different model family from the answering model) compares each
  answer with the reference answer and grades it fully correct, partly correct or wrong. The headline
  number counts only fully correct answers.
- **Faithfulness**: the judge checks every claim in the answer against the passages the model was given.
- **Citation accuracy**: for answers that cite, whether a cited passage contains the evidence. This is
  computed deterministically, with no judge.
- **Refusals**: the share of out-of-scope questions the system declines (higher is better), and the share
  of answerable questions it wrongly declines (lower is better).
- **Latency**: time to the first complete spoken sentence and to the full answer, measured on an 8 GB
  Apple M2 laptop with retrieval models on CPU and other apps open. Local-model numbers include running the
  4B LLM on the same machine.

### Retrieval ablation

Each row adds one change to the row above it. The "alt" rows swap one component of the final
configuration instead. These are test-split numbers, computed once after the settings were fixed on dev
([dev table](eval/results/retrieval_dev.md)). Latency covers query embedding, BM25, fusion and reranking on
an M2 CPU.

<!-- ABLATION:START -->
| Configuration | Hit@1 | Hit@5 | MRR@10 | Evidence recall@4 | Latency p50 |
|---|---|---|---|---|---|
| **v1 baseline** (regex sections, 500/80 chunks, MiniLM, dense) | 51.9% | 75.3% | 0.619 | 72.7% | 8 ms |
| v2 parsing + heading-aware chunks | 59.7% | 84.4% | 0.703 | 80.5% | 6 ms |
| + heading path prepended to chunks | 64.9% | 84.4% | 0.740 | 82.5% | 7 ms |
| bge-small-en-v1.5 embeddings | 68.8% | 87.0% | 0.763 | 83.1% | 12 ms |
| bge-base-en-v1.5 embeddings | 74.0% | 85.7% | 0.796 | 84.4% | 32 ms |
| BM25 only | 62.3% | 85.7% | 0.717 | 81.8% | <1 ms |
| hybrid: bge-base + BM25 (RRF) | 71.4% | 89.6% | 0.797 | 87.0% | 37 ms |
| **+ ms-marco-MiniLM-L-6 reranker (final)** | **76.6%** | **89.6%** | **0.827** | **89.0%** | 322 ms |
| alt: bge-reranker-base instead | 79.2% | 92.2% | 0.845 | 90.3% | 1181 ms |
| alt: final with 400-char chunks | 67.5% | 87.0% | 0.761 | 85.1% | 155 ms |
| alt: final with 1200-char chunks | 75.3% | 90.9% | 0.825 | 89.0% | 311 ms |
<!-- ABLATION:END -->

What the ablation shows:

- **Consistent overall gain.** Hit@1 improves by +24.7 points on test and +19.0 on dev. Individual steps vary
  more between the splits: parsing alone gives +7.8 points on test but only +1.3 on dev. Read the per-step
  numbers as directional.
- **BM25 matters on policy text.** Students and the Code share many exact terms ("appeal", "hazing",
  "Husky Card"), so BM25 alone is competitive and costs under a millisecond. Fusing it with dense
  retrieval gives the best candidate pool.
- **Reranker choice is a latency tradeoff.** bge-reranker-base scores 2–3 points higher but takes about
  1.1 s per query on CPU. For a voice interface the 0.2–0.3 s MiniLM model is the better default; switch with
  `RetrievalConfig.reranker`.
- **Chunk size.** 400-character chunks separate definitions from the context they need. 1200 is within
  noise of 800 (−1.3 points Hit@1 on test, +3.8 on dev) but sends the LLM 50% more text.

### STAIR: table-of-contents routing

[STAIR](https://arxiv.org/abs/2609.03874) (Kumar et al., IBM, 2026) retrieves by giving an LLM the
document's table of contents with the query and generating the section that answers it. Generation is
constrained to valid ToC entries. Echo-Sight already recovers a clean 78-entry ToC, so
[echosight/toc.py](echosight/toc.py) implements the zero-shot version. Groq's gpt-oss-20b picks up to 3
section numbers, and a strict JSON schema whose enum lists every ToC number rejects anything that isn't a real
section. The paper fine-tunes Mistral-7B with LoRA, which isn't practical on an 8 GB laptop.
[eval/stair_eval.py](eval/stair_eval.py) has the evaluation.

**As a retriever, zero-shot routing loses to the existing pipeline.** Section-level Recall@k, the paper's
metric, on the test split:

<!-- STAIR:START -->
| Method | Recall@1 | Recall@3 |
|---|---|---|
| BM25 | 70.1% | 84.4% |
| Dense (bge-base) | 77.9% | 85.7% |
| Hybrid (RRF) | 76.6% | 89.6% |
| v2 (hybrid + reranker) | **83.1%** | **90.9%** |
| STAIR ToC router, zero-shot gpt-oss-20b | 59.7% | 71.4% |
<!-- STAIR:END -->

This matches the paper, where out-of-the-box LLMs also trail BM25 (Claude Haiku 45.6% Recall@1). STAIR's
82.6% comes from fine-tuning on synthetic queries. Fusing the routes into retrieval didn't help on dev
either. Adding them before reranking left results unchanged. Fusing them after reranking or restricting
reranking to routed sections cost 14–15 points of Hit@1. Using the route only as a fallback when the
reranker is unsure cost 2.5–5 points. None of these were adopted. Only 1 of 92 router outputs was rejected
as an invalid section.

**As a scope check, it works.** When the router finds no relevant section *and* the top rerank score is
below 0.05, `--stair` refuses without calling the LLM. Both conditions and the threshold were chosen on
dev. On the test split the gate:

- refused 9 of 15 out-of-scope questions before generation, and **0 of 77 answerable questions**;
- raised the offline model's correct refusals from 86.7% to 93.3% (overall accuracy 73.9% → 75.0%). Groq's
  model already declines every out-of-scope question, so with Groq the gate only saves those 9 LLM calls;
- costs nothing on confident queries, because the app waits for the router (p50 0.64 s, running alongside
  retrieval) only when the rerank score is low.

These end-to-end gate numbers are computed from the cached test answers by
[eval/stair_gate.py](eval/stair_gate.py). The gate runs before generation and changes nothing else, so the
result is exact, not an estimate.

### Reproduce

`bash eval/run_all.sh` runs everything below in order. Then `python eval/update_readme.py` rewrites the
Results section from `eval/results/`.

```bash
python eval/check_dataset.py                       # validate every evidence quote against the PDF
python eval/retrieval_eval.py --split test         # retrieval ablation (no API calls)
python eval/listening_baseline.py --split test     # screen-reader listening-time baseline
python eval/e2e_eval.py generate --system v2 --provider groq      # default mode
python eval/e2e_eval.py generate --system v2 --provider ollama    # offline mode
python eval/e2e_eval.py generate --system v1 --provider ollama    # original pipeline, same local LLM
python eval/e2e_eval.py judge --run v2-groq --faithfulness
python eval/e2e_eval.py judge --run v2-ollama
python eval/e2e_eval.py judge --run v1-ollama
python eval/stair_eval.py --split test             # STAIR ToC routing (section + chunk level)
python eval/stair_gate.py                          # end-to-end effect of the STAIR scope gate
python eval/e2e_eval.py report
python -m pytest                                   # 34 unit tests, no network needed
```

Answers and judgements are cached in `eval/results/e2e/`, so reruns cost no API calls. The evaluator
throttles itself to stay within Groq's free-tier limit of 8,000 tokens per minute.

## Getting started

```bash
git clone https://github.com/tanaymihani/echo-sight.git
cd echo-sight
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add a free Groq key from https://console.groq.com/keys
python app.py               # first run downloads the embedding and reranking models (~550 MB)
```

| Command | What it does |
|---|---|
| `python app.py` | Type questions and hear answers (Groq `openai/gpt-oss-120b`) |
| `python app.py --voice` | Push-to-talk: press Enter, speak, press Enter again |
| `python app.py --provider ollama` | Fully offline: `brew install ollama && ollama pull qwen3.5:4b` first |
| `python app.py -q "Can I appeal?"` | Answer one question and exit |
| `python app.py --mute -v` | Text only, with retrieval and generation timings |
| `python app.py --stair` | Add the STAIR table-of-contents scope gate (see [STAIR](#stair-table-of-contents-routing)) |

Inside the app, say or type "repeat" to hear the last answer again. Follow-up questions such as "what about
a second offense?" are rewritten into standalone questions using the conversation so far. `ingest.py`
rebuilds the index manually, but the app also rebuilds it automatically whenever the PDF or chunking
settings change.

## Project structure

```
echosight/
  config.py       retrieval settings and LLM provider presets (all overridable from .env)
  document.py     PDF → heading tree (font-based), text normalisation
  chunking.py     heading-aware chunks with heading paths
  index.py        sentence-transformers embeddings in a persistent Chroma collection
  retrieval.py    BM25 + dense, reciprocal rank fusion, cross-encoder reranking
  toc.py          STAIR-style table-of-contents router (schema-constrained section choice)
  llm.py          streaming client for any OpenAI-compatible API (Groq, Ollama, Gemini)
  prompts.py      grounded-answer and follow-up prompts
  rag.py          retrieve → generate → citations, refusals and spoken output
  voice.py        sentence-streaming text-to-speech, push-to-talk speech recognition
app.py            command-line assistant
ingest.py         build the index
eval/             question set, v1 reproduction, retrieval/end-to-end/listening evaluations, results
tests/            unit tests (parser, chunker, retrieval, RAG logic, streaming speech, eval metrics)
```

## Limitations

- **One document.** The parser relies on this PDF's font styles. A different handbook would need its
  heading styles mapped, or the PDF's structure tags read instead.
- **Small test set.** With 77 answerable test questions, differences of 2 to 3 points between
  configurations are within noise. The ablation shows which changes matter, not exact gains.
- **Automated judging.** Answer accuracy and faithfulness come from an LLM judge, not human graders.
- **A rerank-score gate alone doesn't work.** The cross-encoder gives very low scores to everyday-language
  questions even when the right passage ranks first. On dev, any threshold that refused meaningful
  out-of-scope traffic also refused about 1 in 16 answerable questions. Combined with the STAIR router
  (`--stair`) it refused none. That gate is opt-in because it needs a network call to Groq.
- **Scenario questions** are the weakest category (64.3% fully correct with Groq), mostly because the
  evidence isn't retrieved. Zero-shot ToC routing didn't fix this. A fine-tuned router, as in the STAIR
  paper, is the natural next step.
- Echo-Sight explains policy. It is not legal advice, and students should confirm anything important with
  OSCCR.

## License

For academic and research purposes.
