"""Pygame GUI for ElderCare AI."""

from __future__ import annotations

import os
from time import monotonic

import numpy as np
import pygame

from environment.custom_env import ACTION_NAMES


class ElderCareRenderer:
    def __init__(self, width: int = 960, height: int = 620, *, human: bool = True):
        pygame.font.init()
        self.width = width
        self.height = height
        self.window = None
        self.renderer = None
        self.texture = None
        if human:
            from pygame._sdl2.video import Renderer, Texture, Window

            os.environ.setdefault("SDL_RENDER_DRIVER", "opengl")
            pygame.display.init()
            self.window = Window("ElderCare AI | Weekly Wellbeing Support", size=(width, height))
            self.renderer = Renderer(self.window)
            self.texture = Texture(self.renderer, (width, height), streaming=True)
        self.screen = pygame.Surface((width, height))
        self.font = pygame.font.SysFont("Segoe UI", 21, bold=True)
        self.small = pygame.font.SysFont("Segoe UI", 16)
        self.label = pygame.font.SysFont("Segoe UI", 14, bold=True)
        self.human = human
        self.step = None
        self.position = (1.55, 1.55)
        self.origin = self.position
        self.destination = self.position
        self.move_started = 0

    # World coordinates for each recommendation, in action-space order.
    STATIONS = (
        ("Home", 0, 0, "#dab77c"),
        ("Hydration", 3, 0, "#80bed4"),
        ("Movement", 6, 0, "#9db780"),
        ("Social calls", 0, 3, "#cc9fb1"),
        ("Rest", 3, 3, "#b7a9d1"),
        ("Reading", 6, 3, "#e1b286"),
        ("Community", 0, 6, "#8bb99c"),
        ("Daily routine", 3, 6, "#d7bf83"),
    )

    def _text(self, value, x, y, color="#eaf4ef", font=None, centered=False):
        surface = (font or self.small).render(value, True, color)
        self.screen.blit(surface, (x - surface.get_width() // 2 if centered else x, y))

    @staticmethod
    def _project(x, y, z=0):
        return round(304 + 31 * (x - y)), round(174 + 18 * (x + y) - z)

    def _polygon(self, points, color):
        pygame.draw.polygon(self.screen, color, [self._project(*point) for point in points])

    def _block(self, x, y, width, depth, height, color):
        edge_x, edge_y = x + width, y + depth
        top = pygame.Color(color)
        self._polygon(((x, edge_y), (edge_x, edge_y), (edge_x, edge_y, height),
                       (x, edge_y, height)), top.lerp("black", 0.27))
        self._polygon(((edge_x, y), (edge_x, edge_y), (edge_x, edge_y, height),
                       (edge_x, y, height)), top.lerp("black", 0.15))
        self._polygon(((x, y, height), (edge_x, y, height), (edge_x, edge_y, height),
                       (x, edge_y, height)), top)

    def _scene(self, step, action, rejected):
        self._block(-0.45, -0.45, 9.2, 9.2, 12, "#58776e")
        for coordinate in (2.4, 5.4):
            self._polygon(((coordinate, 0, 13), (coordinate + 0.4, 0, 13),
                           (coordinate + 0.4, 8.3, 13), (coordinate, 8.3, 13)), "#80998a")
            self._polygon(((0, coordinate, 13), (8.3, coordinate, 13),
                           (8.3, coordinate + 0.4, 13), (0, coordinate + 0.4, 13)), "#80998a")

        if step == 0 and self.step != 0:
            self.position = self.origin = self.destination = (1.55, 1.55)
        if step != self.step:
            self.step = step
            self.origin = self.position
            if action not in (None, 0) and not rejected:
                _, x, y, _ = self.STATIONS[action]
                self.destination = (x + 1.65, y + 1.65)
            else:
                self.destination = self.position
            self.move_started = monotonic()
        elapsed = monotonic() - self.move_started
        fraction = min(elapsed / 0.42, 1) if self.human else 1
        self.position = tuple(start + (end - start) * fraction
                              for start, end in zip(self.origin, self.destination))

        labels = []
        for index, (name, x, y, color) in enumerate(self.STATIONS):
            active = index == action and action != 0
            self._block(x, y, 2.3, 2.3, 16, "#fff0af" if active else "#a8b6a0")
            self._block(x + 0.2, y + 0.2, 1.15, 1.05, 39, color)
            for offset in (0.40, 0.82):
                point = self._project(x + offset, y + 1.26, 28)
                pygame.draw.circle(self.screen, "#f6ebcd", point, 3)
            labels.append((name, *self._project(x + 1.43, y + 1.74, 15), active))

        for name, x, y, active in labels:
            width = self.label.size(name)[0] + 12
            pygame.draw.rect(self.screen, "#243b34", (x - width // 2, y + 9, width, 21), border_radius=5)
            self._text(name, x, y + 10, "#fff0af" if active else "#eef4df",
                       font=self.label, centered=True)

        x, y = self.position
        px, py = self._project(x, y, 16)
        pygame.draw.ellipse(self.screen, "#273f39", (px - 10, py - 3, 20, 8))
        pygame.draw.line(self.screen, "#f9eed5", (px, py - 3), (px, py - 15), 9)
        pygame.draw.line(self.screen, "#377d79", (px, py - 4), (px, py - 14), 6)
        pygame.draw.circle(self.screen, "#f9eed5", (px, py - 21), 7)
        pygame.draw.circle(self.screen, "#d6a57c", (px, py - 20), 5)

    def render(self, state: np.ndarray, hour: int, action: int | None,
               reward: float, step: int, rejected: bool,
               cumulative_reward: float = 0.0,
               max_steps: int = 84,
               profile: dict | None = None) -> np.ndarray | None:
        self.screen.fill("#172e2c")
        pygame.draw.rect(self.screen, "#203e39", (0, 0, self.width, 64))
        self._text("ElderCare AI", 24, 18, font=self.font)
        day, days = max(1, (step + 11) // 12), (max_steps + 11) // 12
        self._text(f"Day {day}/{days}   {hour:02d}:00", 835, 22, centered=True)

        self._text("Wellbeing centre", 24, 81, font=self.font)
        if profile:
            self._text(f"Fictional participant   |   Mobility {profile['mobility_level']:.0%}"
                       f"   |   Social preference {profile['social_preference']:.0%}",
                       24, 111, "#b7cec2")
        self._scene(step, action, rejected)

        pygame.draw.rect(self.screen, "#223d38", (624, 82, 312, 478), border_radius=16)
        self._text("Agent recommendation", 644, 101, font=self.font)
        self._text("Waiting for first decision" if action is None else ACTION_NAMES[action],
                   644, 139, "#fff0af")
        if action is None:
            response, color = "Session ready", "#b7cec2"
        elif action == 0:
            response, color = "No prompt sent", "#b7cec2"
        elif rejected:
            response, color = "Declined | participant stays put", "#ffb6a7"
        else:
            response, color = "Accepted", "#a7dfb7"
        self._text(response, 644, 165, color)
        self._text(f"Step reward {reward:+.3f}     Total {cumulative_reward:+.3f}", 644, 199)
        pygame.draw.line(self.screen, "#456057", (644, 232), (916, 232))
        self._text("Wellbeing indicators", 644, 248, font=self.font)
        labels = ("Energy", "Social connection", "Physical activity", "Hydration", "Mood",
                  "Cognitive engagement", "Routine completion", "Fatigue")
        for index, (label, value) in enumerate(zip(labels, (*state[:7], state[8]))):
            y = 290 + index * 31
            self._text(label, 644, y)
            self._text(f"{value:.0%}", 916, y, centered=True)
            pygame.draw.rect(self.screen, "#456057", (644, y + 20, 272, 4), border_radius=2)
            pygame.draw.rect(self.screen, "#e4b879" if index == 7 else "#9cd5aa",
                             (644, y + 20, round(272 * value), 4), border_radius=2)

        self._text(f"Week progress   {step}/{max_steps} decisions", 24, 565, "#b7cec2")
        pygame.draw.rect(self.screen, "#456057", (24, 595, self.width - 48, 6), border_radius=3)
        pygame.draw.rect(self.screen, "#9cd5aa",
                         (24, 595, round((self.width - 48) * step / max_steps), 6), border_radius=3)
        if not self.human:
            return pygame.surfarray.array3d(self.screen).transpose(1, 0, 2).copy()
        return None

    def present(self) -> None:
        self.texture.update(self.screen)
        self.texture.draw()
        self.renderer.present()

    def close(self) -> None:
        if self.window is not None:
            self.texture = None
            self.renderer = None
            self.window.destroy()
            self.window = None
        pygame.quit()
