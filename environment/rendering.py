"""Pygame GUI for ElderCare AI."""

from __future__ import annotations

import numpy as np
import pygame

from environment.custom_env import ACTION_NAMES


class ElderCareRenderer:
    def __init__(self, width: int = 960, height: int = 620):
        pygame.init()
        pygame.display.set_caption("ElderCare AI - Daily Wellbeing Support")
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        self.font = pygame.font.SysFont("Segoe UI", 20)
        self.small = pygame.font.SysFont("Segoe UI", 16)
        self.clock = pygame.time.Clock()

    def render(self, state: np.ndarray, hour: int, action: int | None,
               reward: float, step: int, rejected: bool,
               cumulative_reward: float = 0.0) -> np.ndarray:
        self.screen.fill((246, 249, 252))
        pygame.draw.rect(self.screen, (40, 112, 120), (0, 0, self.width, 72))
        title = self.font.render("ElderCare AI", True, (255, 255, 255))
        self.screen.blit(title, (24, 18))
        time_text = self.font.render(f"Simulated time: {hour:02d}:00", True, (255, 255, 255))
        self.screen.blit(time_text, (760, 18))

        # Stylized fictional older adult
        cx, cy = 178, 270
        pygame.draw.circle(self.screen, (229, 185, 145), (cx, cy - 58), 42)
        pygame.draw.rect(self.screen, (87, 145, 164), (cx - 55, cy - 5, 110, 145), border_radius=18)
        pygame.draw.circle(self.screen, (215, 215, 215), (cx - 13, cy - 61), 5)
        pygame.draw.circle(self.screen, (215, 215, 215), (cx + 13, cy - 61), 5)

        labels = ["Energy", "Social", "Physical", "Hydration", "Mood", "Cognitive", "Routine"]
        y = 180
        for i, (label, value) in enumerate(zip(labels, state[:7])):
            row_y = y + i * 50
            pygame.draw.rect(self.screen, (220, 227, 232), (300, row_y, 300, 18), border_radius=9)
            color = (74, 166, 120) if value >= 0.6 else (230, 165, 72) if value >= 0.35 else (203, 91, 91)
            pygame.draw.rect(self.screen, color, (300, row_y, int(300 * value), 18), border_radius=9)
            self.screen.blit(self.small.render(label, True, (45, 55, 65)), (300, row_y - 24))
            self.screen.blit(self.small.render(f"{value:.0%}", True, (45, 55, 65)), (610, row_y - 3))

        action_name = "None" if action is None else ACTION_NAMES[action]
        feedback = "Recommendation accepted" if action is not None and not rejected else "Recommendation declined / waiting"
        panel = pygame.Rect(650, 130, 280, 270)
        pygame.draw.rect(self.screen, (255, 255, 255), panel, border_radius=14)
        pygame.draw.rect(self.screen, (215, 222, 228), panel, 2, border_radius=14)
        self.screen.blit(self.font.render("Agent Decision", True, (45, 55, 65)), (668, 150))
        self._wrapped(action_name, 668, 195, 245)
        status_color = (203, 91, 91) if rejected else (74, 166, 120)
        status = self.small.render(feedback, True, status_color)
        self.screen.blit(status, (668, 330))

        pygame.draw.rect(self.screen, (255, 255, 255), (650, 420, 280, 150), border_radius=14)
        pygame.draw.rect(self.screen, (215, 222, 228), (650, 420, 280, 150), 2, border_radius=14)
        self.screen.blit(self.font.render(f"Reward: {reward:+.3f}", True, (45, 55, 65)), (668, 440))
        self.screen.blit(self.small.render(f"Cumulative: {cumulative_reward:+.3f}", True, (45, 55, 65)), (668, 470))
        self.screen.blit(self.small.render(f"Step: {step}/{12}", True, (45, 55, 65)), (668, 500))

        pygame.display.flip()
        return pygame.surfarray.array3d(self.screen).transpose(1, 0, 2).copy()

    def _wrapped(self, text: str, x: int, y: int, width: int) -> None:
        words = text.split()
        line = ""
        for word in words:
            test = f"{line} {word}".strip()
            if self.small.size(test)[0] <= width:
                line = test
            else:
                self.screen.blit(self.small.render(line, True, (45, 55, 65)), (x, y))
                y += 24
                line = word
        if line:
            self.screen.blit(self.small.render(line, True, (45, 55, 65)), (x, y))

    def present(self, frame: np.ndarray) -> None:
        self.clock.tick(2)

    def close(self) -> None:
        pygame.quit()
