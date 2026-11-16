# Project progress: Synthetic speech training results

## Work completed

The first two versions of the project were code-only. In this version, we generated a small dataset of computer-spoken keywords, trained two LSTMs, saved their models, selected confidence thresholds on validation data, and evaluated them on a separate test set.

The 12 classes were **yes, no, up, down, left, right, on, off, stop, go, unknown, and silence**. The dataset had **864 training clips**, **288 validation clips**, and **288 test clips**. Train, validation, and test used different eSpeak voice settings, but all came from the same speech-synthesis engine.

## Test results

- Standard LSTM accuracy after a full audio window: **70.1%**.
- Prefix-trained LSTM accuracy after a full audio window: **76.4%**.
- Standard LSTM: **11 of 240** positive test clips triggered an early **correct** keyword prediction (**4.6%**). On **1 of 48** negative clips it wrongly triggered a keyword (**2.1%**).
- Prefix LSTM: **12 of 240** positive test clips triggered an early **correct** keyword prediction (**5.0%**). It also produced **5 wrong early keyword decisions out of 240** positive clips (**2.1%**) and **0 of 48** negative clips triggered (**0%**).

Early-exit decisions were very conservative under the validation-selected thresholds. The prefix model was better at end-of-window accuracy, but we have **not yet proven** that it is better at reliable early recognition. With so few negative synthetic examples, a measured 0% trigger rate does not imply that the real-world false-alarm rate is zero.

## Important difference between training and finished research

The models were trained and tested on synthetic, computer-generated speech, not real people's recorded speech. The measured times are relative to the beginning of the one-second files, not the precise moment the spoken word begins. These results should be shown as a **coding/training milestone**, not research evidence of an improvement over existing keyword-spotting systems.

## Next steps

Train the same models on **Google Speech Commands v2** on a computer that can access the dataset. Then improve the early-exit decision rule and test it on realistic streams with silence, non-keyword speech, and partial words. Only after that should we compare reliable detection latency and false alarms per hour.
