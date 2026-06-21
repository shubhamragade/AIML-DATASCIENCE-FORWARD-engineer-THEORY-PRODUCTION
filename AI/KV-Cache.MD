# 🧠 KV Cache — Deep Dive Visual Notes

> *From novelist analogy → matrix math → GPU trade-offs. Everything in one place.*

---

## 📖 Phase 0 — The Core Analogy (Start Here)

Imagine you are a **novelist writing a story one word at a time**.

Your strict editor has one rule: before you write each new word, you must **re-read the entire story from page 1** and **rewrite all your character notes from scratch**.

```
Story length = 3 words?   → Annoying but fast
Story length = 1,000 words? → Impossibly slow
```

**The KV Cache = a sticky note pad.**

Instead of rewriting everything:
- ✅ Keep your old notes handy (the cache)
- ✅ Write one tiny new note for the newest word
- ✅ Glance at the full notepad and write the next word

That's it. Everything else below is just the math behind this idea.

---

## ⚙️ Phase 1 — What is Inference?

When you type a prompt into an AI, the AI generating a reply is called **inference**.

**Example we'll use throughout:**

```
Input Prompt → "A sunset is"
AI Goal      → Predict the next word → "extremely"
```

Computers don't understand words. So each word is converted into a
**vector** — a list of numbers.

### What is a Vector?

```
"A"      → [0.2,  0.8,  0.1,  0.5]
"sunset" → [0.9,  0.3,  0.7,  0.2]
"is"     → [0.1,  0.6,  0.4,  0.9]
```

Each word becomes a row of 4 numbers (for simplicity; real models use 768–4096+).

### What is a Matrix?

Stack all vectors → you get a **matrix** (a grid of numbers).

```
Input Matrix X  (3 tokens × 4 dimensions)

         d1    d2    d3    d4
"A"    [ 0.2   0.8   0.1   0.5 ]   ← row 1
"sun"  [ 0.9   0.3   0.7   0.2 ]   ← row 2
"is"   [ 0.1   0.6   0.4   0.9 ]   ← row 3
```

---

## 🔢 Phase 2 — The Math Recipe (Q, K, V)

The transformer takes input matrix **X** and multiplies it by three
different **weight matrices** (Wq, Wk, Wv). These are fixed numbers
the model learned during training.

### Matrix Multiplication — Plain English

Multiplying a (3×4) matrix by a (4×4) weight matrix gives a (3×4) result.
Each output row = the dot product of one input row with the weight columns.

```
X        ×   Wq    =    Q  (Queries)
(3×4)       (4×4)      (3×4)

X        ×   Wk    =    K  (Keys)
(3×4)       (4×4)      (3×4)

X        ×   Wv    =    V  (Values)
(3×4)       (4×4)      (3×4)
```

### What do Q, K, V mean?

| Matrix | Real-world meaning | Question it answers |
|--------|-------------------|---------------------|
| **Q — Query** | What a word is *looking for* | "What context do I need?" |
| **K — Key** | What a word *advertises* about itself | "What do I contain?" |
| **V — Value** | The actual *content* to pass forward | "What information do I hold?" |

**Library analogy:**
- Q = your search query typed into the catalogue
- K = the index card title of each book
- V = the actual content inside that book

---

## 📐 Phase 3 — Attention Calculation (Step by Step)

### Step 1 — Attention Scores: Q × Kᵀ

Multiply Query matrix by the **transpose** of Key matrix.

```
Scores = Q × Kᵀ

Q  (3×4)  ×  Kᵀ (4×3)  =  Scores (3×3)

Concrete numbers:

Q:              Kᵀ:              Scores:
[0.1 0.2 0.3 0.4]  ×  [0.1 0.5 0.3]  =  [score_AA  score_As  score_Ai]
[0.5 0.6 0.7 0.8]     [0.2 0.6 0.4]     [score_sA  score_ss  score_si]
[0.9 0.0 0.1 0.2]     [0.3 0.7 0.5]     [score_iA  score_is  score_ii]
                       [0.4 0.8 0.6]
```

Each score = "how much should token i pay attention to token j?"

### Step 2 — Scale the Scores

Divide every score by √(dimension size) to prevent huge numbers.

```
Scaled Score = Score / √4  =  Score / 2
```

Why? Without scaling, large numbers make Softmax output nearly 0 or 1,
destroying the gradient during training.

### Step 3 — Softmax → Attention Weights

Convert scores into **percentages that sum to 1.0** (per row).

```
Softmax formula:  e^(score_i) / Σ e^(score_j)

Example row:    [1.2,  0.8,  0.3]
After softmax:  [0.47, 0.31, 0.22]   ← must sum to 1.0 ✓
```

This gives us the **Attention Weight Matrix W** (3×3):

```
        "A"   "sun"  "is"
"A"   [ 0.47  0.31   0.22 ]   ← "A" attends to itself 47%, sunset 31%, is 22%
"sun" [ 0.15  0.62   0.23 ]
"is"  [ 0.10  0.25   0.65 ]
```

### Step 4 — Context Matrix: W × V

```
Context = W × V

W  (3×3)  ×  V (3×4)  =  Context (3×4)

Each row of Context = a weighted blend of all Value vectors
```

### Step 5 — The Golden Rule 🏆

```
┌─────────────────────────────────────────────────────┐
│  To predict the NEXT word, we only ever need the    │
│  LAST ROW of the Context matrix.                    │
│                                                     │
│  Context matrix last row → Logits → Next token      │
└─────────────────────────────────────────────────────┘
```

**Full pipeline summary:**

```
X → [×Wq] → Q ─┐
X → [×Wk] → K ─┼→ Scores = Q×Kᵀ/√d → Softmax → W → W×V → Context
X → [×Wv] → V ─┘                                        ↓
                                              Last row → Logits → "extremely"
```

---

## 🔍 Phase 4 — The Problem Without Caching

### New sequence: "A sunset is extremely" (4 tokens)

Without cache, we rebuild everything from scratch:

```
New X (4×4):
"A"        [ 0.2  0.8  0.1  0.5 ]  ← SAME as before
"sunset"   [ 0.9  0.3  0.7  0.2 ]  ← SAME as before
"is"       [ 0.1  0.6  0.4  0.9 ]  ← SAME as before
"extremely"[ 0.4  0.2  0.8  0.3 ]  ← NEW

New K (4×4):
Row 1 → K for "A"          ← ♻️ identical to Prefill K row 1
Row 2 → K for "sunset"     ← ♻️ identical to Prefill K row 2
Row 3 → K for "is"         ← ♻️ identical to Prefill K row 3
Row 4 → K for "extremely"  ← 🆕 actually new
```

**The waste:** Rows 1–3 of K and V are **100% identical** to what we
already computed. We're throwing them away and recomputing for no reason.

### Computation Cost Without Cache

```
Tokens generated = n
Work per token   = O(n)   (must process all n tokens)
Total work       = O(n²)  ← grows quadratically!

n=10   →  100  operations
n=100  →  10,000 operations
n=1000 →  1,000,000 operations  😱
```

---

## 💾 Phase 5 — KV Cache Saves the Day

### The Insight

To predict the next word, we proved:
- We only need the **Query of the newest token** (not old queries)
- We need the **full K and V matrices** (all past + new)
- But past K and V rows **never change** → cache them!

### Updated Pipeline With Cache

```
PREFILL STAGE (runs once on your full prompt):
──────────────────────────────────────────────
Input: ["A", "sunset", "is"]
↓
Compute Q, K, V for all 3 tokens
↓
Run full attention → get first output: "extremely"
↓
SAVE K and V to KV Cache:
  Cache = { K: [K_A, K_sun, K_is],
            V: [V_A, V_sun, V_is] }


DECODE STEP 1 (generate "beautiful"):
──────────────────────────────────────────────
Input: just the new token "extremely"
↓
Compute ONLY Q_ext, K_ext, V_ext   ← 3 tiny vectors, not a matrix!
↓
Fetch from cache: K_A, K_sun, K_is, V_A, V_sun, V_is
↓
Append new rows:
  Full K = [K_A, K_sun, K_is, K_ext]   (4×4)
  Full V = [V_A, V_sun, V_is, V_ext]   (4×4)
↓
Attention: Q_ext × Full_Kᵀ → scores → softmax → × Full_V
↓
Last row → "beautiful"
↓
ADD K_ext and V_ext to cache. Discard Q_ext.


DECODE STEP 2 (generate next word):
──────────────────────────────────────────────
Cache now has 4 rows. Process only "beautiful". Repeat.
```

### Why NOT cache Q (Queries)?

```
We proved: to get the last row of Context, we need:
  → Q of the NEWEST token only
  → K of ALL past tokens
  → V of ALL past tokens

Old Q vectors from "A", "sunset", "is"?
They are NEVER used again in any future step.
Caching them = wasting memory on something useless.
```

### Computation Cost WITH Cache

```
Tokens generated = n
Work per token   = O(1)   (only process 1 new token)
Total work       = O(n)   ← grows linearly!

n=10   →  10 operations
n=100  →  100 operations
n=1000 →  1,000 operations  ✅
```

---

## 📊 Phase 6 — Key Metrics and Calculations

### Metric 1 — KV Cache Memory Size

```
Formula:
  Cache size = 2 × num_layers × num_heads × head_dim × seq_len × bytes_per_param

Real example (GPT-2 small):
  num_layers   = 12
  num_heads    = 12
  head_dim     = 64
  seq_len      = 1024 tokens
  float16      = 2 bytes

  Cache = 2 × 12 × 12 × 64 × 1024 × 2
        = 2 × 12 × 12 × 64 × 2048
        = 37,748,736 bytes
        ≈ 36 MB  (for 1024 tokens, 1 sequence)

For LLaMA-2 70B at 4096 tokens:
  Cache ≈ 2 × 80 × 8192 × 4096 × 2 bytes
        ≈ ~80 GB  ← bigger than the model weights for long contexts!
```

### Metric 2 — Speedup Ratio

```
Without cache, generating token t costs:  O(t)  flops
With cache, generating token t costs:     O(1)  flops

Speedup at token t = t

At token 100  → 100×  faster per step
At token 1000 → 1000× faster per step
```

### Metric 3 — Attention Score Formula (complete)

```
                    Q × Kᵀ
Attention(Q,K,V) = Softmax(────────) × V
                     √d_k

Where:
  Q  = Query matrix   (seq_len × d_k)
  K  = Key matrix     (seq_len × d_k)
  V  = Value matrix   (seq_len × d_v)
  d_k = dimension of keys (used for scaling)
  √d_k ≈ 8 for d_k=64  (prevents score explosion)
```

### Metric 4 — Softmax Formula

```
         e^(x_i)
S(x_i) = ────────────
          Σ_j e^(x_j)

Example:
  Input scores: [2.0, 1.0, 0.5]
  e^2.0 = 7.389
  e^1.0 = 2.718
  e^0.5 = 1.649
  Sum   = 11.756

  Softmax = [7.389/11.756, 2.718/11.756, 1.649/11.756]
           = [0.628,        0.231,         0.140]
           Sums to 1.0 ✓
```

---

## 🗺️ Phase 7 — Visual Architecture Diagrams

### Diagram A — Full Attention Flow (No Cache)

```
INPUT TOKENS
┌─────────┬──────────┬──────┐
│   "A"   │ "sunset" │ "is" │
└────┬────┴────┬─────┴──┬───┘
     │         │        │
     ▼         ▼        ▼
┌─────────────────────────────┐
│   Embedding + Positional    │  Convert to vectors
│   Encoding                  │
└──────────────┬──────────────┘
               │
               │  X  (3×4 matrix)
               │
       ┌───────┼───────┐
       ▼       ▼       ▼
    ×Wq      ×Wk      ×Wv
       │       │       │
       ▼       ▼       ▼
       Q       K       V
    (3×4)   (3×4)   (3×4)
       │       │       │
       └───┬───┘       │
           │           │
           ▼           │
      Q × Kᵀ           │
      (3×3)            │
           │           │
           ▼           │
      ÷ √(d_k)         │
           │           │
           ▼           │
        Softmax        │
           │           │
           ▼           │
    Weights W          │
      (3×3)            │
           │           │
           └─────┬─────┘
                 │
                 ▼
             W  × V
             (3×4)
                 │
                 ▼
          Context Matrix
          ─────────────
          row 1: vector
          row 2: vector
          row 3: vector  ← ONLY this row used!
                 │
                 ▼
              Logits
                 │
                 ▼
           "extremely"
```

### Diagram B — KV Cache Flow (Efficient)

```
PREFILL                          DECODE (each new token)
───────────────────────────────────────────────────────

"A sunset is"                    "extremely" (just this one)
     │                                │
     ▼                                ▼
Compute Q,K,V                   Compute Q_new, K_new, V_new
for all 3 tokens                for this 1 token only
     │                                │
     ├──── K rows ──────────────────► Append to cached K
     │                                │
     ├──── V rows ──────────────────► Append to cached V
     │                                │
     ▼                                ▼
  ┌──────────────────────────────────────┐
  │           KV CACHE  💾               │
  │  ┌─────┬────────┬──────┬──────────┐  │
  │  │ K_A │ K_sun  │ K_is │ K_extrem │  │
  │  ├─────┼────────┼──────┼──────────┤  │
  │  │ V_A │ V_sun  │ V_is │ V_extrem │  │
  │  └─────┴────────┴──────┴──────────┘  │
  └──────────────────────────────────────┘
                       │
                       ▼
          Q_new × Full_Kᵀ → Softmax → × Full_V
                       │
                       ▼
                  Last row only
                       │
                       ▼
                  "beautiful"
                       │
                       ▼
          K_beautiful, V_beautiful → saved to cache
          Q_beautiful              → DISCARDED (never needed again)
```

### Diagram C — Why Q is NOT cached

```
Predicting word at position t:

Things we need:
  ✅ Q_t     (query of current/newest token)
  ✅ K_1..t  (keys of ALL tokens including current)
  ✅ V_1..t  (values of ALL tokens including current)

Things we DON'T need:
  ❌ Q_1     (old query for "A"       → never used again)
  ❌ Q_2     (old query for "sunset"  → never used again)
  ❌ Q_t-1   (old query for any past  → never used again)

Conclusion: Only cache K and V. Queries = use once, throw away.
```

### Diagram D — Complexity Comparison

```
Operations needed to generate a 10-token response:

WITHOUT KV Cache (O(n²)):
Token 1:  ■
Token 2:  ■■
Token 3:  ■■■
Token 4:  ■■■■
Token 5:  ■■■■■
Token 6:  ■■■■■■
Token 7:  ■■■■■■■
Token 8:  ■■■■■■■■
Token 9:  ■■■■■■■■■
Token 10: ■■■■■■■■■■
Total:    55 operations

WITH KV Cache (O(n)):
Token 1:  ■
Token 2:  ■
Token 3:  ■
Token 4:  ■
Token 5:  ■
Token 6:  ■
Token 7:  ■
Token 8:  ■
Token 9:  ■
Token 10: ■
Total:    10 operations

Speedup = 55/10 = 5.5×  (at 10 tokens)
At 100 tokens → ~50× speedup
At 1000 tokens → ~500× speedup
```

---

## 🔄 Phase 8 — Two-Stage Inference Summary

### Stage 1: Prefill

```
Input:   Full prompt (all tokens at once)
Output:  First generated token + populated KV Cache
Cost:    O(n²) — but only happens ONCE
```

### Stage 2: Decode (repeats until done)

```
Input:   Single new token + existing KV Cache
Output:  Next token + updated KV Cache
Cost:    O(1) per step — fast!
```

---

## ⚖️ Phase 9 — The Trade-off

```
GAIN                              COST
──────────────────                ─────────────────────────
✅ Speed: O(n) instead of O(n²)  ❌ GPU RAM: grows with seq length
✅ Chatbot stays fast in long     ❌ Large batches = huge cache
   conversations                  ❌ Cache for 100k tokens can
✅ Scales to long contexts           exceed model weight size!
   with minimal slowdown
```

### Memory optimization techniques (advanced):

| Technique | What it does |
|-----------|--------------|
| **KV Quantization** | Store cache in INT8 instead of FP16 → 2× memory saving |
| **PagedAttention** | Manage cache like OS paging (used in vLLM) |
| **Sliding Window** | Only cache last N tokens, discard old ones |
| **Multi-Query Attn** | Share K and V across heads → smaller cache |
| **Grouped-Query Attn**| Middle ground between MHA and MQA (used in LLaMA-2 70B) |

---

## 🧪 Test Your Understanding

Answer these before moving forward:

**Q1:** Why do we only compute the Query vector for the newest token during
decoding, instead of computing or caching all past queries?

> **Hint:** Think about the Golden Rule — what do we need to predict the
> next word, and where does the Query appear in that calculation?

**Q2:** In your own words, what would happen to a chatbot's speed as it
writes a very long essay if there was NO KV cache?

> **Hint:** Look at Diagram D. What does O(n²) mean for n = 5000 tokens?

---

## 📚 Quick Reference Card

```
TERM           DEFINITION
─────────────────────────────────────────────────────────────────
Token          A word or word-piece. "sunset" = 1 token.
Vector         A list of numbers representing a token. e.g. [0.2, 0.8]
Matrix         A 2D grid of numbers. (rows × columns)
X              Input matrix. Shape: (seq_len × d_model)
Wq, Wk, Wv    Learned weight matrices. Fixed after training.
Q (Query)      "What am I looking for?" matrix. X × Wq
K (Key)        "What do I advertise?" matrix.  X × Wk
V (Value)      "What do I hold?" matrix.       X × Wv
Attention      Score of how much token i attends to token j
Score
Softmax        Turns raw scores into probabilities (sum = 1)
Context        Final output of attention. Last row = next token.
Matrix
Logits         Raw unnormalized scores for each word in vocabulary
Prefill        Stage 1: process full prompt, build cache
Decode         Stage 2: generate tokens one at a time using cache
KV Cache       Storage of all past K and V vectors for reuse
O(n²)          Quadratic complexity — doubles cost when n doubles
O(n)           Linear complexity — cost grows proportionally
d_k            Dimension of keys (used in √d_k scaling)
```

---

*Built from the KV Cache video notes — every phase flows into the next.*