"""
rendering.py — Pygame-based visualization for AfricanFinanceEnv

Displays:
- Income / Savings / Debt / Investment bars
- Financial stress gauge
- Monthly timeline
- Economic shock alerts
- Action taken label
- Net worth history sparkline
"""

import pygame
import numpy as np
import sys
import os

# ── Color Palette (Pan-African inspired) ────────────────────────────────────
BG_DARK      = (12, 18, 30)
BG_CARD      = (20, 30, 50)
BG_CARD2     = (24, 38, 62)
GOLD         = (255, 193, 7)
GREEN        = (56, 196, 128)
RED          = (220, 80, 80)
BLUE         = (64, 148, 255)
ORANGE       = (255, 140, 60)
PURPLE       = (160, 100, 240)
TEAL         = (0, 200, 180)
WHITE        = (235, 240, 255)
GRAY         = (100, 115, 140)
LIGHT_GRAY   = (160, 175, 200)
DARK_CARD    = (14, 22, 40)
SHOCK_RED    = (255, 50, 50)

ACTION_NAMES = [
    "Conservative Save",
    "Balanced Allocate",
    "Aggressive Invest",
    "Debt Repayment",
    "Emergency Fund",
    "Education/Upskill",
    "Mobile Money",
    "Survival Mode",
]

ACTION_COLORS = [BLUE, GREEN, ORANGE, RED, TEAL, PURPLE, GOLD, GRAY]


class FinanceRenderer:
    WIDTH  = 1100
    HEIGHT = 720

    def __init__(self, headless=False):
        self.headless = headless
        if not pygame.get_init():
            pygame.init()

        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        else:
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("🌍 AI Micro-Investment Agent — African Finance Simulator")

        self.clock  = pygame.time.Clock()
        self._load_fonts()
        self.net_worth_history = []
        self.reward_history    = []
        self.action_history    = []
        self.frame_count       = 0

    def _load_fonts(self):
        pygame.font.init()
        try:
            self.font_title  = pygame.font.SysFont("dejavusans", 22, bold=True)
            self.font_label  = pygame.font.SysFont("dejavusans", 14)
            self.font_small  = pygame.font.SysFont("dejavusans", 12)
            self.font_large  = pygame.font.SysFont("dejavusans", 32, bold=True)
            self.font_medium = pygame.font.SysFont("dejavusans", 18, bold=True)
        except Exception:
            self.font_title  = pygame.font.Font(None, 24)
            self.font_label  = pygame.font.Font(None, 16)
            self.font_small  = pygame.font.Font(None, 13)
            self.font_large  = pygame.font.Font(None, 36)
            self.font_medium = pygame.font.Font(None, 20)

    # ── Drawing helpers ──────────────────────────────────────────────────────

    def _draw_rect(self, surface, color, rect, radius=8, alpha=255):
        s = pygame.Surface((rect[2], rect[3]), pygame.SRCALPHA)
        pygame.draw.rect(s, (*color, alpha), (0, 0, rect[2], rect[3]), border_radius=radius)
        surface.blit(s, (rect[0], rect[1]))

    def _text(self, surface, text, pos, font, color=WHITE, center=False):
        surf = font.render(str(text), True, color)
        rect = surf.get_rect()
        if center:
            rect.center = pos
        else:
            rect.topleft = pos
        surface.blit(surf, rect)

    def _bar(self, surface, x, y, w, h, ratio, color, bg=BG_DARK, label="", value_str=""):
        pygame.draw.rect(surface, bg, (x, y, w, h), border_radius=4)
        fill_w = int(np.clip(ratio, 0, 1) * w)
        if fill_w > 0:
            pygame.draw.rect(surface, color, (x, y, fill_w, h), border_radius=4)
        # glow effect
        glow = pygame.Surface((fill_w, h), pygame.SRCALPHA)
        glow.fill((*color, 40))
        surface.blit(glow, (x, y))
        if label:
            self._text(surface, label, (x, y - 18), self.font_small, LIGHT_GRAY)
        if value_str:
            self._text(surface, value_str, (x + w + 6, y + h // 2 - 6), self.font_small, color)

    def _sparkline(self, surface, history, x, y, w, h, color):
        if len(history) < 2:
            return
        mn = min(history)
        mx = max(history)
        rng = max(mx - mn, 1)
        pts = []
        for i, v in enumerate(history[-w:]):
            px = x + int(i * w / len(history[-w:]))
            py = y + h - int((v - mn) / rng * h)
            pts.append((px, py))
        if len(pts) >= 2:
            pygame.draw.lines(surface, color, False, pts, 2)

    # ── Main render ──────────────────────────────────────────────────────────

    def render(self, info: dict, step: int, action_allocations: dict, action: int = None, reward: float = None):
        self.frame_count += 1
        net_worth = info.get("net_worth", 0)
        self.net_worth_history.append(net_worth)
        if reward is not None:
            self.reward_history.append(reward)
        if action is not None:
            self.action_history.append(action)

        self.screen.fill(BG_DARK)
        self._draw_frame(info, step, action, reward)

        if not self.headless:
            pygame.display.flip()
            self.clock.tick(10)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    def _draw_frame(self, info, step, action, reward):
        W, H = self.WIDTH, self.HEIGHT
        screen = self.screen

        # ── Header ───────────────────────────────────────────────────────
        self._draw_rect(screen, BG_CARD, (0, 0, W, 70), radius=0)
        self._text(screen, "🌍  AI Micro-Investment Agent — African Financial Simulator",
                   (20, 12), self.font_title, GOLD)
        self._text(screen, f"Month {step} / 60  |  Net Worth: ${info.get('net_worth', 0):,.0f}",
                   (20, 42), self.font_label, LIGHT_GRAY)
        if info.get("financial_stress", 0) > 0.7:
            self._text(screen, "⚠ HIGH FINANCIAL STRESS", (W - 250, 25), self.font_medium, RED)

        # Shock alert
        if info.get("economic_shock", False):
            self._draw_rect(screen, SHOCK_RED, (W // 2 - 150, 10, 300, 50), radius=8, alpha=200)
            self._text(screen, "⚡ ECONOMIC SHOCK ACTIVE", (W // 2, 35), self.font_medium, WHITE, center=True)

        # ── Left panel: Financial bars ────────────────────────────────────
        px, py = 20, 90
        pw, ph = 340, 580
        self._draw_rect(screen, BG_CARD, (px, py, pw, ph), radius=12)
        self._text(screen, "FINANCIAL STATUS", (px + 14, py + 14), self.font_medium, GOLD)

        income   = info.get("income", 0)
        savings  = info.get("savings", 0)
        debt     = info.get("debt", 0)
        invest   = info.get("investment_value", 0)
        emerg    = info.get("emergency_fund", 0)
        stress   = info.get("financial_stress", 0)
        inv_tgt  = 10000.0

        bars = [
            ("Income",          income / 2000,     GREEN,  f"${income:,.0f}"),
            ("Savings",         savings / 5000,    BLUE,   f"${savings:,.0f}"),
            ("Investment",      invest / inv_tgt,  ORANGE, f"${invest:,.0f}"),
            ("Emergency Fund",  emerg / 2000,      TEAL,   f"${emerg:,.0f}"),
            ("Debt",            debt / 10000,      RED,    f"${debt:,.0f}"),
        ]

        for i, (lbl, ratio, col, val) in enumerate(bars):
            bx, by = px + 16, py + 54 + i * 72
            self._text(screen, lbl, (bx, by), self.font_small, LIGHT_GRAY)
            self._text(screen, val, (bx + 220, by), self.font_small, col)
            self._bar(screen, bx, by + 18, pw - 40, 22, ratio, col)

        # Stress gauge
        sy = py + 430
        self._text(screen, "Financial Stress Index", (px + 16, sy), self.font_small, LIGHT_GRAY)
        stress_color = GREEN if stress < 0.4 else ORANGE if stress < 0.7 else RED
        self._bar(screen, px + 16, sy + 20, pw - 40, 28, stress, stress_color)
        self._text(screen, f"{stress:.0%}", (px + pw // 2, sy + 20), self.font_small, WHITE, center=True)

        # Inflation indicator
        iy = sy + 70
        infl = info.get("inflation_rate", 0.08)
        self._text(screen, f"Inflation Rate: {infl:.1%}/yr", (px + 16, iy), self.font_label, ORANGE)

        # Net worth label
        ny = iy + 32
        nw = info.get("net_worth", 0)
        nw_col = GREEN if nw >= 0 else RED
        self._text(screen, f"Net Worth: ${nw:,.0f}", (px + 16, ny), self.font_medium, nw_col)

        # ── Center: Action panel ──────────────────────────────────────────
        cx, cy = 375, 90
        cw, ch = 360, 280
        self._draw_rect(screen, BG_CARD, (cx, cy, cw, ch), radius=12)
        self._text(screen, "CURRENT ACTION", (cx + 14, cy + 14), self.font_medium, GOLD)

        if action is not None:
            ac = ACTION_COLORS[action]
            self._draw_rect(screen, ac, (cx + 14, cy + 46, cw - 28, 48), radius=8, alpha=60)
            pygame.draw.rect(screen, ac, (cx + 14, cy + 46, cw - 28, 48), width=2, border_radius=8)
            self._text(screen, f"Action {action}: {ACTION_NAMES[action]}",
                       (cx + cw // 2, cy + 70), self.font_medium, ac, center=True)

            # Allocation breakdown
            alloc = action_allocations.get(action, {})
            ay = cy + 110
            cols = [(BLUE, "savings"), (ORANGE, "investments"), (RED, "debt"),
                    (TEAL, "emergency"), (PURPLE, "education"), (GREEN, "expenses")]
            for col, key in cols:
                val = alloc.get(key, 0)
                if val > 0:
                    self._bar(screen, cx + 14, ay, (cw - 28) * val, 14, 1.0, col)
                    self._text(screen, f"{key.title()} {val:.0%}", (cx + 14 + (cw - 28) * val + 4, ay),
                               self.font_small, col)
                    ay += 22

        if reward is not None:
            r_col = GREEN if reward >= 0 else RED
            self._text(screen, f"Reward: {reward:+.2f}", (cx + 14, cy + 240), self.font_medium, r_col)

        # ── Center bottom: Action legend ──────────────────────────────────
        lx, ly = 375, 385
        lw, lh = 360, 285
        self._draw_rect(screen, BG_CARD, (lx, ly, lw, lh), radius=12)
        self._text(screen, "ACTION LEGEND", (lx + 14, ly + 14), self.font_medium, GOLD)
        for i, name in enumerate(ACTION_NAMES):
            row = i % 4
            col = i // 4
            bx2 = lx + 14 + col * 175
            by2 = ly + 44 + row * 56
            ac  = ACTION_COLORS[i]
            if action == i:
                self._draw_rect(screen, ac, (bx2, by2, 158, 44), radius=6, alpha=60)
                pygame.draw.rect(screen, ac, (bx2, by2, 158, 44), width=2, border_radius=6)
            else:
                self._draw_rect(screen, BG_DARK, (bx2, by2, 158, 44), radius=6)
            self._text(screen, f"{i}: {name}", (bx2 + 8, by2 + 4), self.font_small, ac)
            self._text(screen, ACTION_NAMES[i], (bx2 + 8, by2 + 22), self.font_small, GRAY)

        # ── Right panel: Sparklines ───────────────────────────────────────
        rx, ry = 750, 90
        rw, rh = 335, 580
        self._draw_rect(screen, BG_CARD, (rx, ry, rw, rh), radius=12)
        self._text(screen, "PERFORMANCE HISTORY", (rx + 14, ry + 14), self.font_medium, GOLD)

        # Net worth sparkline
        self._text(screen, "Net Worth Over Time", (rx + 14, ry + 50), self.font_small, LIGHT_GRAY)
        self._sparkline(screen, self.net_worth_history, rx + 14, ry + 70, rw - 28, 100, GREEN)
        pygame.draw.rect(screen, GRAY, (rx + 14, ry + 70, rw - 28, 100), width=1, border_radius=4)

        # Reward sparkline
        self._text(screen, "Reward Signal", (rx + 14, ry + 195), self.font_small, LIGHT_GRAY)
        self._sparkline(screen, self.reward_history, rx + 14, ry + 215, rw - 28, 80, BLUE)
        pygame.draw.rect(screen, GRAY, (rx + 14, ry + 215, rw - 28, 80), width=1, border_radius=4)

        # Action distribution bar chart
        self._text(screen, "Action Frequency", (rx + 14, ry + 320), self.font_small, LIGHT_GRAY)
        if self.action_history:
            from collections import Counter
            counts = Counter(self.action_history)
            max_c  = max(counts.values())
            bar_w  = (rw - 28) // 8
            for i in range(8):
                c    = counts.get(i, 0)
                bh   = int(c / max(max_c, 1) * 80)
                bx2  = rx + 14 + i * bar_w
                by2  = ry + 420 - bh
                pygame.draw.rect(screen, ACTION_COLORS[i], (bx2, by2, bar_w - 4, bh), border_radius=3)
                self._text(screen, str(i), (bx2 + bar_w // 2 - 4, ry + 424), self.font_small, GRAY)

        # Stats summary
        self._text(screen, "── SUMMARY ──", (rx + 14, ry + 455), self.font_small, GOLD)
        stats = [
            ("Months Simulated", f"{step}"),
            ("Total Net Worth",  f"${info.get('net_worth', 0):,.0f}"),
            ("Investment Value", f"${info.get('investment_value', 0):,.0f}"),
            ("Emergency Fund",   f"${info.get('emergency_fund', 0):,.0f}"),
        ]
        for i, (k, v) in enumerate(stats):
            self._text(screen, k, (rx + 14, ry + 475 + i * 24), self.font_small, LIGHT_GRAY)
            self._text(screen, v, (rx + rw - 16, ry + 475 + i * 24), self.font_small, WHITE)

        # ── Footer ───────────────────────────────────────────────────────
        self._draw_rect(screen, BG_CARD, (0, H - 30, W, 30), radius=0)
        self._text(screen, "AI Micro-Investment RL Agent | African Financial Stability Simulator | Press Q to quit",
                   (W // 2, H - 15), self.font_small, GRAY, center=True)

    def get_rgb_array(self, info, step):
        self._draw_frame(info, step, None, None)
        return pygame.surfarray.array3d(self.screen).transpose(1, 0, 2)

    def close(self):
        pygame.quit()
