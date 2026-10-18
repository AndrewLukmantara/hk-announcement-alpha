# Write-up template (6–10 pages)

**Title:** Bilingual Disclosure Sentiment and Southbound Flows: Post-Announcement Drift in Hong Kong Equities

## 1. Introduction
- Question: do HKEX announcements predict tradable drift after the reaction, and is the drift stronger when mainland investors are accumulating?
- Contribution: (i) LLM scoring of bilingual filings with anonymisation; (ii) news × Southbound interaction; (iii) a cost-realistic test under HK stamp duty.

## 2. Related work
- Post-earnings announcement drift (Bernard & Thomas, 1989); text-based return prediction (Tetlock, 2007; Loughran & McDonald, 2011).
- LLMs for return prediction (Lopez-Lira & Tang, 2023) and LLM look-ahead bias.
- Stock Connect and Southbound flow literature.

## 3. Data
- Announcement counts by year, category, and session (use the timing table). Universe, liquidity filter, sample period.
- Southbound, short-selling, and price sources; survivorship caveat.

## 4. Method
- Entry-date rule. Market-model AR with est. window [-250, -30]. CAR definitions (reaction vs drift).
- Scorers: keyword baseline, category-only, LLM (EN / ZH / both). Hand-label validation results.
- Regression: drift_5 ~ s + s×flow_z + s×short_z + controls, clustered by date.
- Signal: exponentially decayed sentiment, half-life chosen in-sample. Backtest rules and costs.

## 5. Results
- Table 1: event study by sentiment bucket and event type. Figure 1: CAR paths.
- Table 2: interaction regression. Table 3: EN vs ZH horse race.
- Table 4: ablation (IS / OOS IC, gross vs net Sharpe, turnover). Figures 2–3: equity curve, IC decay.

## 6. Robustness
- Post-LLM-cutoff subsample; long-only; liquidity terciles; excluding privatisations; alternative entry lag (0 / 30 / 60 min).

## 7. Limitations and conclusion
- Survivorship, short constraints, Southbound history length, multiple testing across event types.
