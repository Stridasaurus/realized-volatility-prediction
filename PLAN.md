# Presentation plan (to implement next session)

**Deadline:** Wednesday evening, Zoom, 10 minutes or less.
**Format decided:** self-contained HTML slide deck (one `.html` file, inline CSS/JS,
figures embedded as base64 data URIs — no external files, no network dependency).
Present by opening it in a browser and going full-screen during screen share; no
submission file required, but Strider wants an offline-usable copy regardless.
**Scope decided:** headline-only. No methodology deep-dive beyond one summary slide.

## Why this format

Same visual approach as the Claude.ai "Artifacts" the user liked, but produced as a
plain file in the repo so it runs with zero setup on presentation day — double-click
or open in any browser, no server, no internet. A second export to PDF (same
Edge-headless technique used for `REPORT.pdf`) gives a true offline fallback in case
Zoom screen-share wants a static file instead of a live browser tab.

## Slide-by-slide outline (~10 slides, ~1 min/slide average)

All content is a repackage of existing committed artifacts — no new computation,
consistent with the project's "everything replays" discipline. Sources noted per slide.

1. **Title** — question as the headline ("Can a Small LSTM Beat HAR?"), course/name/date.
2. **The setup** — HAR's reputation (Corsi, 2009) was earned on high-frequency RV; this
   project only has daily OHLC, so it tests range-based variance proxies instead. One
   line on why that distinction matters. *(Source: REPORT.md §1)*
3. **Method, one slide** — frozen snapshot, walk-forward test span, hyperparameters
   tuned once and frozen, DM+HLN significance, S6 seed-agreement bar. Enough to signal
   rigor without reading like methodology section. *(Source: REPORT.md §3, condensed)*
4. **Finding 1 — HAR earns its reputation here** — HAR tops the classical field; ties
   GARCH on TV, beats everything else significantly. *(Source: REPORT.md §4.1 table)*
5. **Finding 2 — architecture alone finds nothing** — RV-only LSTM is a genuine null
   (p = 0.585 / 0.667). Frame as the protocol working as designed, not a failure.
   *(Source: REPORT.md §4.2)*
6. **Finding 3 — auxiliary features reverse the null** — headline QLIKE improvements
   (17–19%), DM p-values, 100% seed agreement, all four target/horizon blocks.
   *(Source: `results/report/figures/fig1_headline_qlike.png` + REPORT.md §4.3 table)*
7. **Where the edge lives** — features not architecture (RV+aux beats RV-only
   head-to-head too); present at both horizons, both targets, every regime.
   *(Source: `fig4_cum_qlike_diff.png` + REPORT.md §4.4)*
8. **Why the frozen window mattered** — decay curve: network needs ~12.6y of history
   just to tie HAR. *(Source: `fig2_decay_curve.png`)*
9. **Limitations, one slide** — multiplicity, proxy scoping, statistical-not-economic.
   Three bullets max. *(Source: REPORT.md §5, trimmed hard)*
10. **Conclusion** — the pre-committed floor claim from REPORT.md §7, verbatim or near
    it. End on the one sentence that matters if the deck is all anyone remembers.

Cut candidates if running long: slide 8 (decay curve) folds into a bullet on slide 7;
slide 3 (method) can drop to bullet points only, no table.

## Technical build steps (next session)

1. Draft slide content/copy first as plain markdown (fast to edit, easy to review
   against the 10-minute budget before touching design).
2. Invoke the `artifact-design` skill before writing HTML — calibrate visual weight
   for a live-presented deck (large type, high contrast, minimal text per slide, one
   idea per slide) rather than a dense document.
3. Build as a single HTML file: CSS `scroll-snap` or a simple JS slide-index show/hide,
   arrow-key + spacebar navigation, no external libraries (matches the offline
   requirement even though this isn't going through the Artifact tool's CSP).
4. Embed the four PNGs from `results/report/figures/` as base64 so the file has zero
   external dependencies.
5. Rehearse with a timer once drafted; trim copy per slide until it reliably lands
   under 10 minutes — bullets are speaker-prompts, not scripts to read verbatim.
6. Export a PDF backup via the same `msedge --headless --print-to-pdf` approach used
   to rebuild `REPORT.pdf` this session, as an offline fallback file.
7. Suggested location: `presentation/slides.html` (+ `presentation/slides.pdf`).

## Open items for Strider before/during next session

- Review the slide-by-slide outline above and flag anything to cut, reorder, or
  emphasize differently before content gets written.
- Do a live timed run-through once the draft exists — 10 minutes is a hard cap.
