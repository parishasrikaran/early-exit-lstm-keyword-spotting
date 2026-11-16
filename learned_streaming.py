
import torch
from pathlib import Path

from config import CLASSES, KEYWORDS, frame_end_ms
from features import CausalMelStream
from model import load_checkpoint
from stop_head import StopHead


class LearnedStreamingRecognizer:
    def __init__(
        self,
        checkpoint,
        stop_head_path,
        policy,
        device="cpu"
    ):
        self.device = torch.device(device)

        self.model, _ = load_checkpoint(
            checkpoint, self.device
        )
        self.model.eval()

        self.stop_head = StopHead().to(self.device)

        saved = torch.load(
            stop_head_path,
            map_location=self.device,
            weights_only=True
        )

        self.stop_head.load_state_dict(
            saved["model_state_dict"]
        )
        self.stop_head.eval()

        self.threshold = float(policy["threshold"])
        self.persistence = int(policy["persistence"])
        self.minimum_ms = float(policy["minimum_ms"])
        self.decision_every = int(policy["decision_every"])

        self.features = CausalMelStream()
        self.reset()

    def reset(self):
        self.features.reset()
        self.state = None
        self.finished = False
        self.total_frames = 0
        self.previous = None
        self.streak = 0
        self.last_accepted = None
        self.accept_streak = 0

    @torch.inference_mode()
    def push_audio(self, samples):
        if self.finished:
            return None

        feats = self.features.push(samples)

        if feats.size(0) == 0:
            return None

        logits, self.state = self.model(
            feats.unsqueeze(0).to(self.device),
            self.state
        )

        probabilities = logits[0].softmax(-1)

        for probability in probabilities:
            frame = self.total_frames
            self.total_frames += 1

            values, indices = probability.topk(2)

            prediction = int(indices[0].item())
            confidence = float(values[0].item())
            margin = float(
                (values[0] - values[1]).item()
            )

            if prediction == self.previous:
                self.streak += 1
            else:
                self.streak = 1

            self.previous = prediction

            stability = min(self.streak, 5) / 5.0
            elapsed = min((frame + 1) * 0.01, 1.0)

            if (frame + 1) % self.decision_every != 0:
                continue

            when = frame_end_ms(frame)

            if when < self.minimum_ms:
                self.accept_streak = 0
                self.last_accepted = None
                continue

            inputs = torch.tensor(
                [[confidence, margin, stability, elapsed]],
                dtype=torch.float32,
                device=self.device
            )

            stop_probability = float(
                self.stop_head(inputs).item()
            )

            valid = (
                CLASSES[prediction] in KEYWORDS
                and stop_probability >= self.threshold
            )

            if valid:
                if prediction == self.last_accepted:
                    self.accept_streak += 1
                else:
                    self.accept_streak = 1

                self.last_accepted = prediction
            else:
                self.accept_streak = 0
                self.last_accepted = None

            if valid and self.accept_streak >= self.persistence:
                self.finished = True

                return {
                    "label": CLASSES[prediction],
                    "confidence": confidence,
                    "stop_probability": stop_probability,
                    "time_ms": when
                }

        return None
