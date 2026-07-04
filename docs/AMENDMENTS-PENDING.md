# Manifesto amendments pending ratification

Three places where building the system diverged from or had to fill in MANIFESTO.md
(v3). Each was flagged during the cascade (session 35) or at data freeze; none has been
applied to the manifesto yet. Ratify, amend, or reject — then this file is deleted and
the manifesto edited to match.

## A1 — Stooq acquisition method (manifesto §8 "Decided" / §10 Data)

**Manifesto says:** Stooq is the programmatic daily source ("the cleaner programmatic
daily source").
**Reality:** Stooq now login-gates all CSV/bulk export (JS proof-of-work wall on top).
The committed snapshot is a **manual export from a logged-in browser session**
(`data/raw/spy_us_d.csv`, all columns dividend/split-adjusted), canonicalized by
`scripts/freeze_snapshot.py`. CBOE VIX remains programmatic.
**Proposed amendment:** amend the data decision to record the manual-export
acquisition path and its provenance note in the manifest. The freeze-and-checksum
contract is unchanged; only the acquisition step is manual.

## A2 — The Tier 1/2/3 aux menu is now enumerated (manifesto §6 `features`)

**Manifesto says:** `features` owns "the Tier 1/2/3 aux menu" — but never enumerates
it (known manifesto gap).
**Filled in by** `specs/features/SPEC.md` §2 (pre-registered):
- **Tier 1 (target history):** lagged log-target (lags 0–21); HAR-style 5-day and
  22-day aggregates of the target through day t, in log space.
- **Tier 2 (SPY price/volume):** close-to-close log return; |return|; overnight log
  return; volume log-ratio vs trailing 22-day mean.
- **Tier 3 (VIX):** log VIX close; 1-day VIX log-change; implied daily variance
  `(VIX/100)²/252`.
**Proposed amendment:** add one sentence to §6 `features` delegating the enumerated,
pre-registered menu to the spec. Must be frozen before Stage-2 test-region contact.

## A3 — HAR is fit in log space (manifesto §7 invariant / §9 glossary)

**Manifesto says:** "ALWAYS model log-target internally" (§7); glossary marks AR(1) as
log explicitly but is **silent for HAR** (Corsi's original is level-space OLS).
**Adopted by** `specs/baselines/SPEC.md`: the §7 invariant binds *estimated* models →
AR(1) **and HAR both fit in log space** with the lognormal half-variance
back-transform; persistence/EWMA (no estimation step) and GARCH (own likelihood) stay
level-space. Pre-registered; must be settled before any test-region contact.
**Proposed amendment:** glossary HAR entry gains "fit on the log-target (log-HAR),
half-variance corrected back to level space; Corsi's level-HAR noted as the rejected
alternative."
