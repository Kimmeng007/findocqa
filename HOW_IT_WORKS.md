# FinDocQA — How It Actually Works, Step by Step

This traces **one single real question** all the way through the system,
start to finish. Every piece of data shown here is real, pulled from the
running code — nothing invented for illustration.

Our example question (from `data/financebench/questions.jsonl`):

> **"What is the FY2018 capital expenditure amount (in USD millions) for 3M?"**
> Correct answer: **$1577.00**

---

## The big picture, in one sentence

We can't just ask Gemini this question directly — Gemini has never read
3M's actual 2018 filing, so it would either say "I don't know" or make
something up. **RAG (Retrieval-Augmented Generation)** fixes this: we
find the *exact paragraph* from the real filing that has the answer, hand
that paragraph to Gemini along with the question, and say "answer using
only this." That's the whole idea. Everything below is the machinery to
make "find the exact paragraph" actually work.

There are two completely separate phases:
- **Phase A (done once, ahead of time)**: download filings, break them
  into searchable pieces, index them.
- **Phase B (done every time someone asks a question)**: search the
  index, hand the results to Gemini, get an answer.

---

## PHASE A — Getting ready (happens before any question is asked)

### Step 1 — Where the question list comes from

`data/financebench/questions.jsonl` is a file we downloaded once from a
public research project (FinanceBench). Each line is one test question,
already answered by human experts, in a format like:

```json
{"question": "What is the FY2018 capital expenditure amount (in USD millions) for 3M?",
 "company": "3M", "doc_name": "3M_2018_10K", "answer": "$1577.00"}
```

The `doc_name` field (`3M_2018_10K`) tells us **which filing** has the
answer. We don't have to guess — the benchmark tells us exactly where to
look. This is what you were looking at in the IDE.

### Step 2 — Turning "3M_2018_10K" into a real government document

`doc_name` isn't a downloadable file by itself — it's just a label. We
have to go find the actual filing on SEC EDGAR (the U.S. government's
public filings database). This takes a few sub-steps:

1. **Company name → CIK number.** The SEC identifies every company by a
   number (a "CIK"), not by name. We look up "3M" → CIK `0000066740`
   using a public SEC lookup file.
2. **CIK + year + form type → the exact filing.** We ask SEC's API: "show
   me every filing CIK 0000066740 has ever made," then find the one that
   is a 10-K (annual report) covering fiscal year 2018. That gives us:
   ```
   accession_number: 0001558370-19-000470
   primary_document: mmm-20181231x10k.htm
   url: https://www.sec.gov/Archives/edgar/data/66740/000155837019000470/mmm-20181231x10k.htm
   ```
3. **Download that URL.** It's a single HTML file — the entire 10-K,
   about 10 megabytes, hundreds of pages if printed.

### Step 3 — Turning one giant HTML file into organized pieces

A 10-K's raw HTML is not readable as one blob — it's hundreds of
paragraphs and tables all mashed together with formatting code. We walk
through it top to bottom and split it into an ordered list of **blocks**,
each one tagged as either:
- `"text"` — a paragraph of normal writing, or
- `"table"` — a financial table, kept as actual rows and columns (not
  flattened into one line of text)

A real table block from this exact filing looks like:
```json
["Total Company", "93,516", "91,536", "91,584", "$", "1,577", "$", "1,373", ...]
```
That `1,577` in there **is the answer** — it's just sitting inside a raw
spreadsheet-like row with no label attached yet, that's why the next
steps matter.

3M's 2018 10-K produces **466 blocks** this way.

### Step 4 — Breaking those blocks into bite-sized "chunks"

Gemini can't be handed all 466 blocks at once for every question — that's
too much text, too slow, too expensive. So we cut the blocks into small
pieces (~1,500 characters each) called **chunks**, which is the actual
unit we search over. This step is where our first real bug happened
(see `TECHNICAL_REPORT.md` §2.3): if you cut carelessly, you can slice a
table in half and separate its numbers from the label that explains what
they mean. Our current chunker (`chunk_filing_table_aware`) specifically
avoids ever slicing through the middle of one table.

### Step 5 — Turning text into numbers a computer can search ("embedding")

Computers can't search text for "meaning" directly — they need numbers.
We run every chunk through a small AI model (`sentence-transformers`)
that converts each chunk into a list of 384 numbers (a "vector") that
captures its meaning. Two chunks about similar topics end up with similar
number-lists, even if they don't share any of the same words.

A real example — the question itself, turned into its vector (truncated
to the first 10 of 384 numbers):
```
"What is the FY2018 capital expenditure amount (in USD millions) for 3M?"
→ [0.0092, -0.0600, 0.0003, -0.0019, 0.0945, 0.0103, -0.0073, 0.0059, -0.0083, 0.0212, ...]
```
This is meaningless to a human, but two vectors that are numerically
*close together* means the two texts are about similar things.

### Step 6 — Building two search indexes

We store every chunk's vector in a **FAISS index** — a structure built
for "find the closest vectors to this one, fast," even across millions of
chunks. Separately, we also build a **BM25 index**, which is old-school
keyword search (like Ctrl+F, but smarter) — good at catching exact terms
that the meaning-based search sometimes misses. We keep both because they
catch different things (see Step 9).

At this point, **Phase A is done**. The filing is downloaded, parsed,
chunked, and both indexes are built. This whole phase runs once per
filing, ahead of time, not while someone's waiting for an answer.

---

## PHASE B — Answering the question (happens live, per question)

### Step 7 — Turn the incoming question into a vector too

Same embedding model from Step 5, applied to the live question. Now the
question and every stored chunk live in the same "meaning space" and can
be compared.

### Step 8 — Search both indexes

- **FAISS** returns its top ~30 chunks whose vectors are closest to the
  question's vector (closest in meaning).
- **BM25** separately returns its own top ~30 chunks based on keyword
  overlap.

These two lists often disagree — FAISS might find a chunk that's
*topically* about capital expenditure even if it doesn't say those exact
words, while BM25 finds chunks that literally contain "capital
expenditure" even if they're not the most relevant one.

### Step 9 — Combine the two lists ("fusion")

We merge FAISS's and BM25's lists using a method called **Reciprocal Rank
Fusion** — basically, a chunk that appears near the top of *both* lists
gets boosted, while a chunk only one method liked gets a smaller boost.
This produces one combined shortlist (up to 60 candidates).

### Step 10 — Rerank with a second, more careful model

The fusion step above is fast but rough. We then run a slower, more
accurate model (a "cross-encoder") that looks at the actual question
*paired with* each candidate chunk's actual text, one at a time, and
scores how relevant it really is. This is more expensive per-chunk, which
is exactly why we only run it on the ~60 shortlisted candidates instead
of the entire corpus. The top 5 after this step are what we actually use.

*(Side note — this is the step that turned out to be genuinely uncertain
in our testing: does this careful reranking model actually help for
financial documents, or does it sometimes get confused and prefer
prose over the correct number-filled table? That's exactly what the
Week 3 evaluation, running right now, is measuring with real numbers.)*

### Step 11 — Build the actual prompt sent to Gemini

We take the top 5 chunks' text and paste them into a prompt template:

```
You are a financial analyst assistant answering questions about SEC
filings using only the provided context. If the context does not
contain enough information to answer, say so explicitly instead of
guessing. Cite the source document name when you use a figure.

Context:
[3M_2018_10K | chunk 113]
... Total Company | 93,516 | 91,536 | 91,584 | $ 1,577 | $ 1,373 ...

[... 4 more retrieved chunks, same format ...]

Question: What is the FY2018 capital expenditure amount (in USD millions) for 3M?
```

That's it — that whole block of text is one API call to Gemini.

### Step 12 — Gemini generates the answer

Gemini reads only what's in that prompt (it doesn't know anything else
about 3M beyond its own general training) and writes an answer, ideally
citing the number and which chunk it came from.

### Step 13 — Checking if the answer was actually right (evaluation)

This is the part we're running right now for Week 3. For each question we
already know the correct answer (it's in the benchmark file from Step 1),
so we can check the system automatically, three ways:
- **Faithfulness**: did Gemini's answer only use what was in the
  provided chunks, or did it add something not actually there?
- **Context precision/recall**: did Step 10 actually retrieve the right
  chunks in the first place?
- **Numerical accuracy** (our own custom check): does the number
  `1577` actually appear in Gemini's final answer text?

We run this same 13-step pipeline **three different ways** (Week 1's
simple version, Week 2's improved version with reranking, and Week 2's
improved version without reranking) across 40 real questions, so we get
an honest, measured comparison instead of a guess.

---

## One-paragraph summary

We download a real government filing, chop it into small labeled pieces,
convert those pieces into searchable number-vectors, and build two kinds
of search index over them. When someone asks a question, we search both
indexes, merge and re-score the results to find the 5 most relevant
pieces, paste those into a prompt, and ask Gemini to answer using only
that material. Because we already know the correct answers to our test
questions, we can automatically score how well each version of this
pipeline actually performs — which is what Week 3 is doing right now.
