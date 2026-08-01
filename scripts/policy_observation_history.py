#!/usr/bin/env python3
"""Adapt single-frame environment observations to policy frame histories."""

from __future__ import annotations

from collections import deque

import numpy as np


class ObservationHistoryPolicy:
    def __init__(self, policy, *, horizon: int):
        if horizon < 1:
            raise ValueError("observation history horizon must be positive")
        self.policy = policy
        self.horizon = int(horizon)
        self._history = deque(maxlen=self.horizon)

    def start_episode(self) -> None:
        self._history.clear()
        self.policy.start_episode()

    @staticmethod
    def _copy_frame(observation):
        if not isinstance(observation, dict) or not observation:
            raise ValueError("observation must be a nonempty dictionary")
        return {
            key: np.array(value, copy=True)
            for key, value in observation.items()
        }

    def __call__(self, *, ob):
        frame = self._copy_frame(ob)
        if not self._history:
            for _ in range(self.horizon):
                self._history.append(self._copy_frame(frame))
        else:
            if set(frame) != set(self._history[-1]):
                raise ValueError("observation keys changed within an episode")
            self._history.append(frame)
        stacked = {
            key: np.stack([history_frame[key] for history_frame in self._history])
            for key in frame
        }
        return self.policy(ob=stacked)
