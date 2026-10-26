# Predicting Listing Gains in the Indian IPO Market

When a company lists on the stock exchange, the gap between its issue price and its day-one opening price is the **listing gain**. For an investor deciding whether to apply for an IPO, the question is binary: *will this list at a profit or not?*

This project frames that as binary classification and trains a neural network in TensorFlow/Keras on subscription and issue data from 319 Indian IPOs.

## Results

| Metric | Value |
| --- | --- |
| Test accuracy | **0.6094** |
| Majority-class baseline | 0.5469 |
| Lift over baseline | +0.0625 |
| ROC AUC | **0.67** |
| Precision on "Profit" | **0.6923** |

The model beats the baseline and the AUC confirms it ranks IPOs meaningfully rather than guessing. These are deliberately modest numbers, and they should be: IPO listing performance is genuinely hard to predict, and a model claiming 90% here would be a red flag for leakage.

### Confusion matrix (test set, n=64)

| | Predicted Loss | Predicted Profit |
| --- | --- | --- |
| **Actual Loss** | 21 | 8 |
| **Actual Profit** | 17 | 18 |

## Why precision matters more than accuracy here

The two errors are not equally costly:

- A **false positive** means applying for an IPO that lists at a loss — real money lost.
- A **false negative** means skipping a winner — opportunity missed, capital preserved.

For a risk-averse investor, precision on the Profit class (0.69) is the number that matters: of the IPOs this model flagged as buys, roughly 69% actually listed at a profit. The notebook includes a threshold sweep showing how raising the cut-off above 0.5 trades volume of signals for a higher hit rate.

## Approach

- **Target construction** — continuous `Listing_Gains_Percent` converted to a binary profit flag (54.5% positive, so close to balanced).
- **Data cleaning** — the raw `"Date "` column carries a trailing space; column names are stripped first.
- **Outlier handling** — subscription columns are heavily right-skewed (a few IPOs oversubscribed many times over). Capped at the 99th percentile rather than dropped, since the dataset is small and those are genuine events.
- **No leakage** — `StandardScaler` is fit on the training split only; the split is stratified on the target.
- **Architecture** — compact 32/16/8 dense network with dropout, early stopping on validation loss. With only 255 training rows, capacity is deliberately constrained.
- **Baseline first** — the majority-class rate is reported alongside every result.

## Limitations

1. **319 IPOs, 255 for training.** Small for a neural network. A regularised logistic regression or gradient boosting model is a fair competitor and might match this.
2. **No market-regime features.** IPO performance is cyclical and a bull market lifts nearly everything; the model has no sense of when an IPO listed relative to the broader market.
3. **Selection effects.** Only completed IPOs appear; withdrawn offerings do not.
4. **Not investment advice.** Predicting a positive listing gain is not the same as a profitable strategy once fees, allotment odds and holding periods are accounted for.

## Stack

Python, TensorFlow/Keras, pandas, NumPy, scikit-learn, Matplotlib, seaborn

---

Dataquest guided project, completed as part of the Data Scientist in Python path.

Marco Anthony Ayuste