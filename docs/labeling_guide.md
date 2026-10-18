# Labelling guide (step 3)

Label **400 announcements** from `data/raw/label_sample.csv`, which `hkalpha label-sample` writes. Read only the title, the category, and, if the title is unclear, the first page of the PDF. **Do not look up the stock price.** You are labelling what a reader would expect when the filing comes out, not what actually happened.

## `label_sentiment`: expected 1–5 day price impact for existing shareholders

| Score | Meaning | Typical HKEX examples |
|---:|---|---|
| **+1.0** | Strongly positive, clearly material | Privatisation or general offer at a premium; large positive profit alert |
| **+0.5** | Mildly positive | Share buy-back; special dividend; strategic cooperation or major contract; turnaround |
| **0** | Routine, or no clear direction | Monthly return; next-day disclosure; board meeting notice; AGM; proxy form; most director appointments |
| **−0.5** | Mildly negative | Director or CFO resignation; convertible bond issue; small litigation; trading halt pending an announcement |
| **−1.0** | Strongly negative, clearly material | Profit warning; placing or rights issue at a deep discount; auditor resignation; winding-up petition; default; delayed results |

Rules:

- **Use only the five values.** Coarse labels are more consistent than fine-grained ones.
- **Judge relative to expectations.** Use 0.5 instead of 1.0 when an event is widely expected, such as a routine annual buy-back mandate.
- **Placings and subscriptions** are negative because they dilute shareholders. A subscription by a strategic investor at a *premium* can be 0 or +0.5.
- **"Inside information" with a vague title:** open the PDF. If it is still unclear, label 0.
- **Label EN/ZH disagreement by the clearer of the two.** Note the case in a spare column, because these are your bilingual examples.

## `label_event_type`

Use one of: `earnings_positive, earnings_negative, dividend, buyback, placing, rights_issue, mna, privatisation, management_change, litigation, contract, trading_halt, financing, routine, other`.

## Quality check

- Label the first 50, wait a day, and relabel them blind. If you agree with yourself less than about 80% of the time on sign, tighten your rules before continuing.
- Then run `hkalpha eval-labels --labels data/raw/label_sample.csv` and report each scorer's Spearman ρ and sign accuracy. These go in the write-up's Data section.
