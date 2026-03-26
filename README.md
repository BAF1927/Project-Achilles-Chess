# Project Achilles — PyReason Chess

## How this project began

On the **night of March 16**, **Jonathan Hoster** (Associate Director for Undergraduate Admissions and Recruitment, **College of Engineering**) sent me a **LinkedIn** message about an internship with **Leibniz Lab**. He described a group dedicated to **unifying reasoning and learning** in AI: how humans both **learn from experience** and **reason with symbols** (math, code, logic)—and how hybrid approaches (often called **neurosymbolic**) try to bring **deep learning** and **logical inference** into the same pipeline. The lab works on areas like **metacognition**, **temporal logic programming**, **abductive inference**, and **deep neural networks**, with applications from industrial systems to medicine. **Applicants were strongly encouraged** to know **PyTorch** and **PyReason**.

I already had **Harvard CS50 Introduction to Artificial Intelligence**, but I had not lived in **PyReason** or in tight **game-training loops** the way this role hinted at. I wanted **research of my own** on a clock: **could I build a chess AI in about ten days** that **actually learns**, get **hands-on PyReason**, and learn the **same vocabulary** people use in **reinforcement learning** and agent papers—not just finish a course? I knew **pygame** from **tic-tac-toe**; **chess** was the step up—still visual, but deep enough that shortcuts would show.

**I started the next morning, March 17**: reading how people train game-playing agents and turning that into a real pygame + PyReason build.

---

## Why go through the challenge

I have always been drawn to **research** and **AI development**: the desire to use the **power of logic** to make tools that **reason**, not only pattern-match, and to look for ways to build something **meaningful**—systems you can **inspect** and argue about, not only black boxes. That instinct matches what **Leibniz Lab** stands for: **reasoning and learning** strengthening each other.

**My aspiration** here was simple to say and hard to do: **show up**, pick up an unfamiliar stack on a deadline, and **ship runnable work** that belongs in that world. This repository is a **sample of my commitment to learn**—read what I need, **run the code**, **iterate**, and treat the lab as a direction I want to **grow into**, not only a line on my screen.

---

## Why "Achilles"

Chess is a **war game**. I started with a **hypothesis**: train **Easy**, **Medium**, and **Hard** in **honestly different doses**. Same **one stream of self-play**, but **three memories** that **stop growing at different times**: **Easy** only absorbs the **first quarter** of lessons, **Medium** half, **Hard** the **full run** (`trainAll`). I wanted steps that felt **real**, not a fake slider.

Then I **sparred with my own AI** and the personality showed up. It **threw itself at checks** and **king pressure** like nothing could touch it. It lunged at **queen tension** I would never treat as a free lunch. Against **me**, once I **took the queen** or forced that kind of collapse, the position **died fast**: fierce early fight, **one decisive wound**. That is the myth of **Achilles** in one chess sitting. Not invincible, but **vulnerable in a specific way**.

That is when I **jumped concepts**. Not just “Hard again,” but **one champion lane**: **Achilles** as a **focused warrior**—same PyReason **weight profile** as Hard in the shipped files, but its own **`qtable_achilles.json`**, trained with a **heavier recipe** that bakes in what I saw on the board. His true **Achilles heel** in practice is **lack of training**: the symbolic brain is there; the **memory file** only gets **dangerous** when you **grind games** into it.

After I put real hours into that lane, I put **Achilles** in front of **friends** who play chess—and **he beat several of them**. I still know how to wound him; the arc I care about is **hypothesis → fight the bot → see the queen/check story → rebuild the training objective → train like it matters → let it win in the wild**.

---

### Under the hood (the science that backs the story)

#### Graded Easy / Medium / Hard

**One shared self-play stream**; **three Q-tables** that **stop updating** at **25% / 50% / 100%** of games (`trainAll`). Same policy while gathering data; **different exposure** so the files diverge.

#### No “coward” ping-pong (move filters)

Before PyReason scores what is left:

1. **No instant undo** of your previous move (`filterOutShuffleMoves` in `see.py`).
2. **No same-piece shuttle** (A–B–A–B non-capture ping-pong) (`isPieceShuttleMove` / `filterOutPieceShuttleMoves` in `training_shaping.py`, wired from `chess_ai.chooseMove`). Cuts **stalling** so learning is not about **running away** on the board.

#### Piece weight in the learning signal

Each half-move records **`0.03 × (signed material change)`** with **pawn 1 … queen 9**, king **0** in that sum, **White’s perspective** (`trainQ.py`). Q-updates **respect material**, not only tactical labels from PyReason. **SEE** in `see.py` drops **tactically losing** landings so we do not reward **SEE-bad** sacrifices.

#### “Time travel” (credit backward through the game)

When a game ends, training **walks the move list backward**:

1. **Bellman backups** (`updateQ` / `applyBellmanBackup` in `qtable.py`): blend **immediate reward** with **value bootstrapped** from the **next** position.
2. **Terminal outcome** (mate/draw from White’s view) **decays by 0.98** per step backward (`trainQ.py`) so the **ending** still **whispers** into **earlier** moves.

**Achilles-only:** `applyAchillesShaping` **replays** the game and **nudges** rewards again: **shuttle** penalties, **under-fire** terms for **queen/rook** under attack, **hindsight** that **spreads pain backward** when White **captures a Black major**—so the table learns **major losses hurt the whole sequence**, not only the capture ply.

#### Achilles in the repo

**Achilles** = **Hard-style weights** + **`qtable_achilles.json`** + **Achilles shaping** + **longer training**. Menu difficulty does not “cheat”; strength is mostly **data and the extra reward surgery** above. *(Friends’ games are **anecdote**, not a benchmark—but they matched what I saw in **table growth** and fewer **obvious** shuttle/material blunders.)*

---

## What I looked up (and what the words mean)

Research should **open doors**, not gatekeep. I read how people train game-playing agents and **mapped the usual vocabulary onto this repo** so nothing is magic. If you know chess and curiosity, you can follow along.

| What people say | In plain English | Where it lives here |
|-----------------|------------------|---------------------|
| **State / action** | The **board position** and **one move** you are scoring | A normalized position key plus the move string (UCI) |
| **Q-value** | A learned **score for "this move, from this position"** | `qtable_*.json`; optional **QValue** signal for PyReason |
| **Bellman-style update** | After you see a reward and the next board, **nudge** the stored score toward: *what you got now + credit for how good the next situation looks* | `applyBellmanBackup` in `qtable.py` |
| **Discount and step size** | **How much the future counts** vs. right now; **how fast** you move the score when new games teach you | `discountGamma`, `learningRateAlpha` in the saved table files |
| **Self-play** | The AI **plays against itself** so you get endless labeled practice | `trainQ.py`, `trainAll.py`, `trainAchilles.py` |
| **SEE (static exchange eval)** | A **fast tactical check**: "if we traded blows on this square, who comes out ahead?" Used to drop **obviously losing** moves early | `see.py` |
| **Gaussian noise / mutation** | **Small random nudges** to the weight numbers so you **try nearby strategies** and search for better ones | `train.py` when mutating `Weights` |
| **Rule-based ranking** | Each tactic (**Capture**, **Check**, **Center**, …) feeds a **single combined ranking** for that move in PyReason | `pyreason_move_selection.py` (rules like “Ranked follows from Capture,” and so on) |

**PyTorch vs. this project’s “Q”:** The lab had us set up **PyTorch** as part of the broader toolkit; I installed it like everyone else, next to **PyReason** and the rest. In **many** projects, the “Q” scores live inside a **neural network** built with PyTorch. **Here** the “Q” memory is a **simple table in JSON** updated with straight Python math, so you can **open the file and see the numbers**. Same **family of ideas** (learning move quality from experience); lighter **machinery** so a ten-day build stays readable. PyTorch is still on my machine for the lab path; this folder’s game code just does not import it for storing Q.

---

## Ten-day arc (March 17 → March 26)

Some days were reading, some were coding, some were chasing bugs. **Order is real**; hours bounced around.

- **Mar 17:** Locked scope (pygame + `python-chess`), PyReason install path, first sketch: **moves as nodes, facts on each**.
- **Mar 18:** Board on screen, pieces, clicking moves; felt the **first long PyReason startup** (why `prewarm` helps).
- **Mar 19:** Turn board patterns into **Predicates** (Capture, Check, …); wire **Ranked** so Black can pick a move through PyReason end to end.
- **Mar 20:** **Tactical filters** (SEE, no pointless take-back shuffles, no silly piece shuttles) so PyReason does not waste time on trash moves.
- **Mar 21:** **Memory table for moves** (Q-learning style): save scores to disk, update them after self-play, split files by difficulty.
- **Mar 22:** **Shared training run**: easy / medium / hard all watch the **same games**, but **stop learning** at 25% / 50% / never stop (for hard).
- **Mar 23:** **Achilles** lane: **dedicated** `qtable_achilles.json` + **`applyAchillesShaping`** (shuttle / under-fire / major-loss hindsight) on top of the shared material-based TD pipeline.
- **Mar 24:** **Evolve the feature weights** (`train.py`): let self-play **grade** small random tweaks to the numbers in `trainedWeights.json`.
- **Mar 25:** Menu, status bar, check overlays, README and comments; runthrough on a clean venv.
- **Mar 26:** Final pass, prep for GitHub.

---

## What I actually tested

No automated test suite: **I repeated these checks by hand** while building.

1. **Prewarm** finishes; the **first real move** feels smoother than a totally cold start.
2. **Play:** menu, every difficulty, play to mate or draw, **Reset** and **`ESC`** behave.
3. **Black’s turn:** only **sensible legals** reach PyReason, then facts, then reasoning, then a **picked move** (with a backup scoring path if needed).
4. **Q files grow** after training scripts; the game pulls the **right** `qtable_…json` for the difficulty.
5. **Filters:** bad SEE trades and shuffle / shuttle junk do not flood the engine.
6. **Weights:** training writes `trainedWeights.json`; the UI game **reads** per-difficulty weights from disk.

---

## Run (Python 3.9 or 3.10)

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Speed tip:** The first launch can spend time compiling helper code. Optional warm-up:

```bash
python -m src.prewarm
```

Play:

```bash
python -m src.main
```

**Controls:** White picks a piece, then a square. **`R`** or the **Reset** button; **`ESC`** quits.

**Difficulties:** **Easy**, **Medium**, **Hard** (shared training curriculum, different amounts of learning), and **Achilles** (same **weight profile** as Hard in the shipped JSON, but its own **`qtable_achilles.json`** and **heavier training recipe**—the “focused warrior” track from the story above).

---

## Train Q-tables (easy / medium / hard together)

One stream of self-play games updates **three** table files: easy stops learning after the **first quarter** of games, medium after **half**, hard keeps learning for **all** of them.

```bash
python -m src.trainAll --gameCount 10000 --playTime 120 --randomSeed 42
```

Training **adds** to existing files unless you delete `src/qtable_easy.json`, `qtable_medium.json`, `qtable_hard.json` for a full restart.

---

## Train only the Achilles table

This updates **only** `qtable_achilles.json`. It runs self-play with **`trainingDepth="achilles"`** so **`applyAchillesShaping`** in `training_shaping.py` adjusts transition rewards (shuttle, under-fire, major-loss hindsight) before the usual **backward Bellman** updates. **More games = better statistics** in the table; there is no separate “difficulty boost” besides training time and that shaping.

```bash
python -m src.trainAchilles --gameCount 50000 --playTime 120 --randomSeed 42 --checkpointEvery 500
```

`--playTime` caps how many **half-moves** one game can run before stopping.

---

## Train feature weights (optional)

```bash
python -m src.train --difficulty medium --trainingIterations 80 --evaluationGames 6 --mutationSigma 0.15 --randomSeed 42
```

---

## Stack

- **pygame**, **python-chess**, **PyReason**, **networkx** (pinned in `requirements.txt` for this repo)
- **PyTorch** on my machine **per lab setup**; this chess code stores Q in **JSON + Python**, not a neural net inside `src/`
- **Author:** Bruno Arriola Flores
- **PyReason install:** [documentation](https://pyreason.readthedocs.io/en/latest/installation.html)
