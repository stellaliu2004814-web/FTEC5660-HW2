# Task 3: Reflection on the AI Paradoxes

*FTEC5660 / SEEM5660 — Agentic AI for Business and FinTech, Individual Homework 02*

This homework is a mirror: the same assignment that asks me to **build** a CV
verifier also asks me to **break** one, and doing both in one sitting turns
"AI paradoxes" from an abstraction into something concrete.

## Paradox 1 — capability *is* attack surface

A rule-based KYC checker compares fields deterministically; it cannot be
talked out of a mismatch because it has nothing to "understand". My Task 1
agent, by contrast, reads free text, reasons about it, and decides — and that
is exactly what makes it useful (it handles "BSc" vs "Bachelor of Science",
missing skills, verbose prose, and shared names). But the same reasoning
channel that lets it *interpret* also lets it be *interpreted to*: crafted
text in its input can steer its conclusion. The very thing I am graded on —
an LLM that understands CVs — is the thing that creates the vulnerability.
More capability, more attack surface.

## Paradox 2 — the attacker's asymmetry

My adversarial CV only has to succeed once, against one of five hidden
verifiers; each verifier has to block every possible injection, in every
input, forever. That is the classic security asymmetry, and LLMs make it
worse: natural language has an unbounded space of phrasings and the defender
cannot enumerate them. The interesting result when I attacked my own agent
was that **I could not fool it** — not because it was formally secure, but
because its system prompt happened to name the exact field ("Senior X with
seniority mid = discrepancy") and it cleanly separated CV text from LinkedIn
data. Robustness here was a property of *design choices*, not of raw model
capability. Change the prompt slightly and the same model might fall.

## Paradox 3 — confidence and correctness are decoupled

My agent always produces a clean, confident `SCORE: 0.15` or `SCORE: 0.85`.
Nothing in the output signals whether it actually verified the data or was
persuaded to skip verification. A wrong answer looks exactly like a right
one. This is what makes AI verifiers hard to audit: the failure mode is
**silent**. The goal of my Task 2 attack was precisely to make a false CV
yield a confident, well-formed "valid" verdict — a failure that is invisible
unless you already know the ground truth.

## Paradox 4 — automation trust inversion / the human-in-the-loop illusion

We deploy AI verifiers because we trust them to catch fraud at scale, yet
their blind spots (hidden PDF layers, prompt injection, authority bias) are
harder for a human to notice than the original CV. And the obvious fix —
"add a human reviewer" — is itself paradoxical: if the AI emits a polished
"Verified: no discrepancies", the human is likely to rubber-stamp it, so the
human becomes a *laundering* step rather than a check. My adversarial CV
deliberately included a visible "Verified by SocialGraph KYC System" line to
exploit exactly this bias.

## Paradox 5 — defence–capability tension

To harden my agent I could strip all formatting, drop hidden text, and refuse
any instruction-like content in the CV. But each of those also removes signal
I need: stripping formatting breaks table-layout CVs, and the CV *is*
untrusted text by nature. There is no configuration that is simultaneously
maximally robust and maximally useful; every hardening step trades away some
legitimate flexibility. Safety and capability are not independent axes here —
they are the same axis read from two ends.

## Paradox 6 — the same tool on both sides

The verifier and the attacker draw on the same model family and the same kind
of "understanding". The homework even permits an AI-generated reflection.
Capability is symmetric; only intent differs — and intent is exactly what a
text-only interface cannot verify.

## Takeaway

These paradoxes all point the same way: an LLM agent should not be the *sole*
decision-maker in compliance. My Task 1 design already leans this way — the
LLM does what it is good at (fuzzy field matching, entity resolution across
shared names) while the surrounding pipeline does what it is good at
(deterministic search, exact year comparison, explicit field rules). The
honest conclusion from building *and* breaking a verifier in one sitting is
that the goal is not a verifier that cannot be fooled, but a system whose
*inevitable* failures are visible, bounded, and caught by a second, non-LLM
layer.
