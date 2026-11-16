"""Simple transparent policy; threshold must be calibrated on validation data."""
from config import CLASSES, KEYWORDS, frame_end_ms


class EarlyExitPolicy:
    def __init__(self, threshold=0.95, persistence=3, minimum_ms=300,
                 decision_every=2):
        if not 0 < threshold <= 1:
            raise ValueError("threshold must be in (0, 1]")
        if persistence < 1 or decision_every < 1:
            raise ValueError("persistence and decision_every must be >= 1")
        self.threshold = float(threshold)
        self.persistence = int(persistence)
        self.minimum_ms = float(minimum_ms)
        self.decision_every = int(decision_every)
        self.reset()

    def reset(self):
        self.last_label = None
        self.streak = 0
        self.total_frames = 0

    def update(self, probability, frame_index=None):
        """One probability vector from a processed LSTM frame.

        A decision happens only every decision_every frames (20ms by default).
        Returns {'label', 'confidence', 'time_ms'} or None.
        """
        if frame_index is None:
            frame_index = self.total_frames
        self.total_frames = frame_index + 1
        if (frame_index + 1) % self.decision_every:
            return None
        label_id = int(probability.argmax())
        label = CLASSES[label_id]
        conf = float(probability[label_id])
        valid = label in KEYWORDS and conf >= self.threshold
        if valid:
            self.streak = self.streak + 1 if self.last_label == label_id else 1
        else:
            self.streak = 0
        self.last_label = label_id
        when = frame_end_ms(frame_index)
        if valid and self.streak >= self.persistence and when >= self.minimum_ms:
            return {"label": label, "confidence": conf, "time_ms": when}
        return None


def decide_from_probabilities(probs, threshold=0.95, persistence=3,
                              minimum_ms=300, decision_every=2):
    """Replay a frame probability array (T, 12) through the exact stop policy."""
    policy = EarlyExitPolicy(threshold, persistence, minimum_ms, decision_every)
    for idx, vector in enumerate(probs):
        prediction = policy.update(vector, idx)
        if prediction is not None:
            return prediction
    return None
