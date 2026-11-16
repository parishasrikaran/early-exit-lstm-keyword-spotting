"""Clip evaluation: not a substitute for continuous-audio FA/hour."""
import numpy as np
from config import CLASSES, KEYWORDS
from early_exit import decide_from_probabilities


def score_probabilities(prob_sequences, labels, threshold, persistence=3,
                        minimum_ms=300, decision_every=2):
    positive = 0
    negative = 0
    hits = 0
    wrong_commands = 0
    negative_triggers = 0
    misses = 0
    decisions = 0
    times = []
    fixed_correct = 0
    fixed_false_triggers = 0
    for probabilities, target in zip(prob_sequences, labels):
        truth = CLASSES[int(target)]
        is_keyword = truth in KEYWORDS
        positive += int(is_keyword)
        negative += int(not is_keyword)
        full_pred = CLASSES[int(np.asarray(probabilities[-1]).argmax())]
        fixed_correct += int(full_pred == truth)
        fixed_false_triggers += int(not is_keyword and full_pred in KEYWORDS)
        decision = decide_from_probabilities(
            probabilities, threshold, persistence, minimum_ms, decision_every
        )
        if decision:
            decisions += 1
            times.append(decision["time_ms"])
            if is_keyword:
                if decision["label"] == truth:
                    hits += 1
                else:
                    wrong_commands += 1
            else:
                negative_triggers += 1
        elif is_keyword:
            misses += 1
    n = len(labels)
    return {
        "clips": n, "positive_clips": positive, "negative_clips": negative,
        "fixed_window_accuracy": fixed_correct / n if n else None,
        "fixed_negative_trigger_rate": fixed_false_triggers / negative if negative else None,
        "early_keyword_recall": hits / positive if positive else None,
        "early_wrong_keyword_rate": wrong_commands / positive if positive else None,
        "early_miss_rate": misses / positive if positive else None,
        "early_negative_trigger_rate": negative_triggers / negative if negative else None,
        "early_clip_coverage": decisions / n if n else None,
        "mean_emission_ms": float(np.mean(times)) if times else None,
        "p50_emission_ms": float(np.percentile(times, 50)) if times else None,
        "p95_emission_ms": float(np.percentile(times, 95)) if times else None,
        "note": "Emission timestamps are relative to clip start. NOT event-onset delay or false accepts/hour."
    }
