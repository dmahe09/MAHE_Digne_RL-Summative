"""
rendering.py — Rich Pygame Visualization v2
Rwanda Entrepreneurship Journey Simulator
Robot character + stage scenes + journey track
"""
import math
import os

import numpy as np
import pygame

# ── Colours ──────────────────────────────────────────────────────────────────
BG      = (12,  18,  38)
PANEL   = (22,  32,  58)
PANEL2  = (30,  42,  72)
ACCENT  = (0,  190, 140)
ACCENT2 = (255, 185,  50)
WHITE   = (225, 238, 255)
GREY    = (90,  108, 138)
RED     = (220,  65,  70)
GREEN   = (55,  205, 100)
BLUE    = (55,  145, 225)
YELLOW  = (255, 215,  60)
PURPLE  = (165, 105, 225)
ORANGE  = (255, 145,  50)
TEAL    = (0,   200, 180)
PINK    = (220,  80, 140)

STAGE_COLORS = {
    "Training":   (55,  145, 225),
    "Mentorship": (165, 105, 225),
    "Funding":    (255, 185,  50),
    "Planning":   (255, 145,  50),
    "Launch":     (55,  205, 100),
    "Investing":  (0,   200, 180),
    "Hiring":     (220,  80, 140),
    "Idle":       (90,  108, 138),
}
STAGES = ["Training", "Mentorship", "Funding", "Planning", "Launch", "Investing", "Hiring"]

WIDTH, HEIGHT   = 1100, 700
SCENE_W, SCENE_H = 400, 310


def _clamp01(v):
    return max(0.0, min(1.0, float(v)))

def _lerp(c1, c2, t):
    t = _clamp01(t)
    return tuple(int(c1[i] + (c2[i]-c1[i])*t) for i in range(3))

def rrect(surf, color, rect, r=10, bw=0, bc=None):
    pygame.draw.rect(surf, color, rect, border_radius=r)
    if bw and bc:
        pygame.draw.rect(surf, bc, rect, bw, border_radius=r)

def bar(surf, x, y, w, h, val, col, lbl, font):
    pygame.draw.rect(surf, (35, 46, 72), (x, y, w, h), border_radius=5)
    fw = int(w * _clamp01(val))
    if fw > 2:
        pygame.draw.rect(surf, col, (x, y, fw, h), border_radius=5)
    pygame.draw.rect(surf, GREY, (x, y, w, h), 1, border_radius=5)
    surf.blit(font.render(lbl, True, WHITE), (x, y - 17))
    surf.blit(font.render(f"{int(_clamp01(val)*100)}%", True, col), (x + w + 5, y - 1))


# ── Robot ─────────────────────────────────────────────────────────────────────

def robot(surf, cx, cy, s, stage, tick):
    col  = STAGE_COLORS.get(stage, GREY)
    bob  = int(math.sin(tick * 0.15) * 3 * s)
    swing = int(math.sin(tick * 0.22) * 9 * s)

    # torso
    bw, bh = int(28*s), int(36*s)
    bx, by = cx - bw//2, cy - bh//2 + bob
    rrect(surf, col, (bx, by, bw, bh), r=7)

    # head
    hw = int(30*s)
    hx, hy = cx - hw//2, by - int(26*s)
    rrect(surf, _lerp(col, WHITE, 0.35), (hx, hy, hw, int(24*s)), r=7)
    # antenna
    pygame.draw.line(surf, col, (cx, hy), (cx, hy-int(11*s)), max(1,int(2*s)))
    pygame.draw.circle(surf, ACCENT2, (cx, hy-int(13*s)), max(2,int(4*s)))
    # eyes
    for ex in [cx-int(7*s), cx+int(7*s)]:
        pygame.draw.circle(surf, BG, (ex, hy+int(10*s)), max(2,int(5*s)//2))
        pygame.draw.circle(surf, WHITE, (ex+max(1,int(1*s)), hy+int(10*s)), max(1,int(2*s)))
    # smile
    pygame.draw.arc(surf, BG, (cx-int(7*s), hy+int(15*s), int(14*s), int(7*s)),
                    math.pi, 2*math.pi, max(1,int(2*s)))

    # legs
    ly = by + bh
    pygame.draw.line(surf, col, (cx-int(7*s), ly), (cx-int(7*s)+swing, ly+int(26*s)), max(2,int(3*s)))
    pygame.draw.line(surf, col, (cx+int(7*s), ly), (cx+int(7*s)-swing, ly+int(26*s)), max(2,int(3*s)))
    pygame.draw.circle(surf, _lerp(col,BG,0.3), (cx-int(7*s)+swing, ly+int(26*s)), max(3,int(5*s)))
    pygame.draw.circle(surf, _lerp(col,BG,0.3), (cx+int(7*s)-swing, ly+int(26*s)), max(3,int(5*s)))

    # arms — pose by stage
    ay = by + int(11*s)
    if stage == "Training":
        pygame.draw.line(surf, col, (bx, ay), (bx-int(20*s), ay-int(18*s)), max(2,int(3*s)))
        rrect(surf, BLUE, (bx-int(38*s), ay-int(28*s), int(16*s), int(22*s)), r=3)
    elif stage == "Mentorship":
        pygame.draw.line(surf, col, (bx, ay), (bx-int(22*s), ay+swing), max(2,int(3*s)))
        pygame.draw.line(surf, col, (bx+bw, ay), (bx+bw+int(22*s), ay-swing), max(2,int(3*s)))
    elif stage == "Funding":
        pygame.draw.line(surf, col, (bx+bw, ay), (bx+bw+int(20*s), ay-int(18*s)), max(2,int(3*s)))
        pygame.draw.circle(surf, YELLOW, (bx+bw+int(26*s), ay-int(24*s)), max(4,int(10*s)))
    elif stage == "Planning":
        pygame.draw.line(surf, col, (bx, ay), (bx-int(16*s), ay+int(10*s)), max(2,int(3*s)))
        pygame.draw.line(surf, col, (bx+bw, ay), (bx+bw+int(16*s), ay+int(10*s)), max(2,int(3*s)))
    elif stage == "Launch":
        pygame.draw.line(surf, col, (bx, ay), (bx-int(18*s), ay-int(20*s)+swing), max(2,int(3*s)))
        pygame.draw.line(surf, col, (bx+bw, ay), (bx+bw+int(18*s), ay-int(20*s)-swing), max(2,int(3*s)))
    elif stage == "Hiring":
        pygame.draw.line(surf, col, (bx+bw, ay), (bx+bw+int(28*s), ay+int(4*s)), max(2,int(3*s)))
        pygame.draw.circle(surf, PINK, (bx+bw+int(34*s), ay+int(4*s)), max(3,int(7*s)))
    else:
        pygame.draw.line(surf, col, (bx, ay), (bx-int(15*s)+swing, ay+int(15*s)), max(2,int(3*s)))
        pygame.draw.line(surf, col, (bx+bw, ay), (bx+bw+int(15*s)-swing, ay+int(15*s)), max(2,int(3*s)))


# ── Stage Scenes ──────────────────────────────────────────────────────────────

def scene(surf, rx, ry, rw, rh, stage, tick):
    col = STAGE_COLORS.get(stage, GREY)
    rrect(surf, PANEL2, (rx, ry, rw, rh), r=12, bw=2, bc=col)

    if stage == "Training":
        # chalkboard
        rrect(surf, (28, 76, 48), (rx+18, ry+18, rw-36, rh//2), r=6, bw=2, bc=(55,160,80))
        try:
            f = pygame.font.SysFont("segoeui", 15, bold=True)
            surf.blit(f.render("  ENTREPRENEURSHIP 101", True, (160,240,160)), (rx+28, ry+30))
            surf.blit(f.render("  Skills + Knowledge", True, (120,200,120)), (rx+28, ry+52))
        except Exception:
            pass
        pygame.draw.line(surf, GREY, (rx+10, ry+rh-28), (rx+rw-10, ry+rh-28), 2)

    elif stage == "Mentorship":
        # two chairs
        for xo in [rx+50, rx+rw-90]:
            rrect(surf, GREY, (xo, ry+rh-66, 38, 38), r=4, bw=2, bc=col)
        # speech bubble
        rrect(surf, col, (rx+rw//2-48, ry+22, 96, 44), r=10)
        try:
            f = pygame.font.SysFont("segoeui", 13)
            surf.blit(f.render("Mentorship", True, BG), (rx+rw//2-33, ry+32))
            surf.blit(f.render("session", True, BG), (rx+rw//2-22, ry+48))
        except Exception:
            pass
        pygame.draw.line(surf, GREY, (rx+10, ry+rh-28), (rx+rw-10, ry+rh-28), 2)

    elif stage == "Funding":
        # bank
        rrect(surf, (38,52,88), (rx+rw//2-48, ry+rh-96, 96, 66), r=4, bw=2, bc=YELLOW)
        for ci in range(3):
            pygame.draw.rect(surf, (58,78,118), (rx+rw//2-34+ci*32, ry+rh-88, 10, 50))
        pts = [(rx+rw//2-58, ry+rh-96), (rx+rw//2, ry+rh-112), (rx+rw//2+58, ry+rh-96)]
        pygame.draw.polygon(surf, YELLOW, pts)
        # floating coins
        for i in range(4):
            a = tick*0.06 + i*1.57
            cx2 = int(rx+50+i*68+math.sin(a)*7)
            cy2 = int(ry+40+math.cos(a+i)*14)
            pygame.draw.circle(surf, YELLOW, (cx2,cy2), 9)
            try:
                f = pygame.font.SysFont("segoeui", 10, bold=True)
                surf.blit(f.render("$", True, BG), (cx2-4, cy2-6))
            except Exception:
                pass
        pygame.draw.line(surf, GREY, (rx+10, ry+rh-28), (rx+rw-10, ry+rh-28), 2)

    elif stage == "Planning":
        rrect(surf, (48,62,98), (rx+28, ry+rh-78, rw-56, 48), r=6)
        rrect(surf, WHITE, (rx+rw//2-48, ry+rh-118, 96, 66), r=4, bw=2, bc=ORANGE)
        try:
            f = pygame.font.SysFont("segoeui", 12, bold=True)
            surf.blit(f.render("BUSINESS PLAN", True, (50,50,80)), (rx+rw//2-42, ry+rh-112))
            f2 = pygame.font.SysFont("segoeui", 10)
            for li,ln in enumerate(["Market analysis","Revenue model","Growth plan"]):
                surf.blit(f2.render(ln, True, (80,80,120)), (rx+rw//2-40, ry+rh-94+li*14))
        except Exception:
            pass

    elif stage == "Launch":
        rrect(surf, (38,88,48), (rx+rw//2-66, ry+rh-118, 132, 88), r=6, bw=2, bc=GREEN)
        rrect(surf, GREEN, (rx+rw//2-76, ry+rh-128, 152, 18), r=4)
        try:
            f = pygame.font.SysFont("segoeui", 16, bold=True)
            surf.blit(f.render("OPEN!", True, YELLOW), (rx+rw//2-24, ry+rh-110))
        except Exception:
            pass
        for i in range(10):
            a = tick*0.08+i*0.63
            cx3 = int(rx+rw//2+math.cos(a)*(28+i*7))
            cy3 = int(ry+rh//2+math.sin(a*0.7)*24)
            pygame.draw.circle(surf, [RED,YELLOW,GREEN,BLUE,PURPLE][i%5], (cx3,cy3), 5)
        pygame.draw.line(surf, GREY, (rx+10, ry+rh-28), (rx+rw-10, ry+rh-28), 2)

    elif stage == "Investing":
        pts = [(rx+28,ry+rh-48),(rx+78,ry+rh-88),(rx+138,ry+rh-108),
               (rx+208,ry+rh-78),(rx+268,ry+rh-128),(rx+rw-28,ry+rh-158)]
        pygame.draw.lines(surf, TEAL, False, pts, 3)
        for p in pts:
            pygame.draw.circle(surf, TEAL, p, 5)
        pygame.draw.line(surf, GREY, (rx+24,ry+24), (rx+24,ry+rh-44), 2)
        pygame.draw.line(surf, GREY, (rx+24,ry+rh-44), (rx+rw-18,ry+rh-44), 2)
        try:
            f = pygame.font.SysFont("segoeui", 13)
            surf.blit(f.render("Business Growth", True, TEAL), (rx+rw//2-46, ry+10))
        except Exception:
            pass

    elif stage == "Hiring":
        for i in range(3):
            ex = rx+50+i*108
            ey = ry+rh-76
            ec = [PINK, BLUE, GREEN][i]
            pygame.draw.circle(surf, ec, (ex, ey-24), 17)
            rrect(surf, ec, (ex-13, ey-6, 26, 38), r=6)
            pygame.draw.line(surf, GREEN, (ex-7, ey+38), (ex-2, ey+45), 3)
            pygame.draw.line(surf, GREEN, (ex-2, ey+45), (ex+7, ey+31), 3)
        pygame.draw.line(surf, GREY, (rx+10, ry+rh-28), (rx+rw-10, ry+rh-28), 2)

    else:
        # idle — hills + sun
        pygame.draw.ellipse(surf, (28,68,34), (rx-18,ry+rh-76,190,96))
        pygame.draw.ellipse(surf, (38,88,44), (rx+rw-148,ry+rh-66,190,96))
        pygame.draw.circle(surf, YELLOW, (rx+rw-56,ry+48), 26)


# ── Journey Track ─────────────────────────────────────────────────────────────

def journey_track(surf, x, y, w, cur, font):
    n = len(STAGES)
    sw = w // n
    cur_idx = STAGES.index(cur) if cur in STAGES else 0
    for i, stage in enumerate(STAGES):
        sx = x + i*sw + sw//2
        col = STAGE_COLORS[stage]
        if i < n-1:
            done = cur_idx > i
            pygame.draw.line(surf, col if done else GREY, (sx+15,y+13), (sx+sw-15,y+13), 3)
        is_cur  = (i == cur_idx)
        is_done = (cur_idx > i)
        if is_done:
            pygame.draw.circle(surf, col, (sx,y+13), 15)
            pygame.draw.circle(surf, WHITE, (sx,y+13), 5)
        elif is_cur:
            pygame.draw.circle(surf, col, (sx,y+13), 17)
            pygame.draw.circle(surf, BG, (sx,y+13), 7)
            pygame.draw.circle(surf, col, (sx,y+13), 4)
        else:
            pygame.draw.circle(surf, PANEL2, (sx,y+13), 15)
            pygame.draw.circle(surf, GREY, (sx,y+13), 15, 2)
        lbl = font.render(stage[:7], True, col if (is_cur or is_done) else GREY)
        surf.blit(lbl, (sx - lbl.get_width()//2, y+30))


# ── Main Renderer ─────────────────────────────────────────────────────────────
# ── Main Renderer ─────────────────────────────────────────────────────────────

RWF_PER_UNIT = 5.0   # 1.0 capital unit ~ 5 million RWF


class EntrepreneurshipRenderer:
    """Pygame dashboard. `headless=True` renders off-screen (SDL dummy driver) for GIFs/screenshots."""

    def __init__(self, headless=False, fps=6):
        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        pygame.font.init()
        self.headless = headless
        self.fps = fps
        if headless:
            self.screen = pygame.Surface((WIDTH, HEIGHT))
        else:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Rwanda Entrepreneurship Journey Simulator")
        mk = lambda sz, b=False: pygame.font.SysFont("segoeui,dejavusans,arial", sz, bold=b)
        self.font_xl, self.font_lg = mk(24, True), mk(18, True)
        self.font_md, self.font_sm, self.font_xs = mk(15), mk(12), mk(11)
        self.clock = pygame.time.Clock()
        self._tick = 0
        self._flash = 0
        self.hist_capital, self.hist_demand, self.hist_cash = [], [], []
        self._last_step = -1

    # -- stage used to pick the scene/pose
    def _stage(self, info):
        a = info.get("last_action", "")
        if info["business_launched"]:
            if "Invest" in a:
                return "Investing"
            if "Hire" in a or info["employees"] >= 1:
                return "Hiring"
            return "Launch"
        if "Training" in a:
            return "Training"
        if "Mentorship" in a:
            return "Mentorship"
        if "loan" in a.lower() or "Repay" in a:
            return "Funding"
        if "Plan" in a:
            return "Planning"
        if "Launch" in a:
            return "Launch"
        if info["readiness"] > 0.1:
            return "Planning"
        if info["network"] > 0.05:
            return "Mentorship"
        if info["skill"] > 0.05:
            return "Training"
        return "Idle"

    def _history(self, info):
        if info["step"] <= 1 or info["step"] < self._last_step:
            self.hist_capital, self.hist_demand, self.hist_cash = [], [], []
        self._last_step = info["step"]
        self.hist_capital.append(info["capital"])
        self.hist_demand.append(info["demand"])
        self.hist_cash.append(info["cashflow"])

    def _chart(self, x, y, w, h, col):
        rrect(self.screen, PANEL, (x, y, w, h), r=8, bw=2, bc=col)
        self.screen.blit(self.font_md.render("Capital (yellow)  vs  Market demand (blue)", True, col), (x + 10, y + 6))
        cx, cy, cw, ch = x + 12, y + 34, w - 24, h - 48
        pygame.draw.rect(self.screen, (35, 46, 72), (cx, cy, cw, ch), border_radius=4)
        for series, c in ((self.hist_demand, BLUE), (self.hist_capital, YELLOW)):
            if len(series) > 1:
                n = len(series)
                pts = [(cx + int(i * cw / max(n - 1, 1)), cy + ch - int(_clamp01(v) * (ch - 4)) - 2)
                       for i, v in enumerate(series)]
                pygame.draw.lines(self.screen, c, False, pts, 2)

    def draw(self, obs, info, human=True):
        self._tick += 1
        self._history(info)
        self.screen.fill(BG)
        stg = self._stage(info)
        col = STAGE_COLORS.get(stg, GREY)
        reward = info["last_reward"]

        if reward > 5:
            self._flash = 8
        if self._flash > 0:
            fs = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            fs.fill((*col, self._flash * 7))
            self.screen.blit(fs, (0, 0))
            self._flash -= 1

        # header
        pygame.draw.rect(self.screen, PANEL, (0, 0, WIDTH, 54))
        pygame.draw.line(self.screen, col, (0, 54), (WIDTH, 54), 2)
        self.screen.blit(self.font_xl.render("Rwanda Entrepreneurship Journey Simulator", True, col), (14, 12))
        et = self.font_sm.render(f"Month {info['step']}/{info['max_steps']}   Reward {reward:+.2f}   "
                                 f"Return {info['total_reward']:.1f}", True, ACCENT2)
        self.screen.blit(et, (WIDTH - et.get_width() - 14, 18))
        journey_track(self.screen, 8, 62, WIDTH - 16, stg, self.font_xs)

        # scene + robot
        sx, sy = 12, 116
        scene(self.screen, sx, sy, SCENE_W, SCENE_H, stg, self._tick)
        robot(self.screen, sx + SCENE_W // 2, sy + SCENE_H - 82, 1.45, stg, self._tick)
        slbl = self.font_lg.render(stg.upper(), True, BG)
        pygame.draw.rect(self.screen, col, (sx, sy + SCENE_H - 28, SCENE_W, 28))
        self.screen.blit(slbl, (sx + SCENE_W // 2 - slbl.get_width() // 2, sy + SCENE_H - 24))

        # agent stats
        px, py, pw, ph = sx + SCENE_W + 12, 116, 296, SCENE_H
        rrect(self.screen, PANEL, (px, py, pw, ph), r=10, bw=2, bc=col)
        self.screen.blit(self.font_lg.render("Agent Stats", True, col), (px + 10, py + 6))
        rows = [(info["skill"], BLUE, "Skill"), (info["capital"], YELLOW, "Capital"),
                (info["network"], PURPLE, "Network"), (info["readiness"], ORANGE, "Readiness"),
                (info["debt"] / 0.5, RED, "Debt (of limit)"), (info["health"], ACCENT, "Business health"),
                (info["demand"], TEAL, "Market demand")]
        for i, (v, c, l) in enumerate(rows):
            bar(self.screen, px + 12, py + 52 + i * 36, pw - 62, 13, v, c, l, self.font_sm)

        # business panel
        bx, by = px + pw + 12, 116
        bw = WIDTH - bx - 8
        rrect(self.screen, PANEL, (bx, by, bw, SCENE_H), r=10, bw=2, bc=col)
        self.screen.blit(self.font_lg.render("Business Status", True, col), (bx + 10, by + 6))
        launched = info["business_launched"]
        self.screen.blit(self.font_md.render("LAUNCHED" if launched else "Not launched", True,
                                             GREEN if launched else RED), (bx + 12, by + 34))
        self.screen.blit(self.font_sm.render(f"Market: {info['market']}", True, BLUE if launched else GREY),
                         (bx + 12, by + 58))
        emp = info["employees"]
        self.screen.blit(self.font_lg.render(f"Employees: {emp}  (goal 2)", True, GREEN if emp >= 2 else WHITE),
                         (bx + 12, by + 80))
        for j in range(3):
            ec = GREEN if j < emp else (40, 55, 80)
            ex, ey = bx + 28 + j * 52, by + 124
            pygame.draw.circle(self.screen, ec, (ex, ey), 19)
            pygame.draw.circle(self.screen, BG, (ex, ey - 6), 6)
            pygame.draw.rect(self.screen, BG, (ex - 7, ey + 2, 14, 12), border_radius=3)
        cf = info["cashflow"]
        self.screen.blit(self.font_md.render(f"Cash flow: {cf*RWF_PER_UNIT*1000:+.0f}k RWF / month", True,
                                             GREEN if cf >= 0 else RED), (bx + 12, by + 160))
        self.screen.blit(self.font_sm.render(f"Cash: {info['capital']*RWF_PER_UNIT:.2f}M RWF   "
                                             f"Debt: {info['debt']*RWF_PER_UNIT:.2f}M RWF", True, YELLOW),
                         (bx + 12, by + 184))
        bar(self.screen, bx + 12, by + 224, bw - 62, 12, info["sustain_months"] / 9, GREEN,
            f"Sustained months: {info['sustain_months']} / 9", self.font_sm)
        ev = info["event"]
        if ev != "-":
            evc = RED if ("downturn" in ev or "rejected" in ev) else GREEN
            rrect(self.screen, PANEL2, (bx + 8, by + 256, bw - 16, 38), r=8, bw=2, bc=evc)
            self.screen.blit(self.font_sm.render(ev, True, evc), (bx + 16, by + 267))

        # bottom row
        bot_y = sy + SCENE_H + 10
        h = HEIGHT - bot_y - 8
        aw = WIDTH // 2 - 12
        rrect(self.screen, PANEL, (10, bot_y, aw, h), r=8, bw=2, bc=ACCENT2)
        self.screen.blit(self.font_md.render("Last action (agent decision this month)", True, ACCENT2), (20, bot_y + 8))
        self.screen.blit(self.font_lg.render(info["last_action"], True, col), (20, bot_y + 34))
        self.screen.blit(self.font_md.render(f"Reward: {reward:+.2f}", True, GREEN if reward > 0 else RED),
                         (20, bot_y + 66))
        self.screen.blit(self.font_sm.render("Goal: launch, hire 2+ people, keep cash flow >= 0 for 9 months, debt <= 20%.",
                                             True, WHITE), (20, bot_y + 100))
        self.screen.blit(self.font_sm.render("Failure: cash reaches 0 (bankrupt) or 120 months pass.", True, GREY),
                         (20, bot_y + 120))
        self.screen.blit(self.font_sm.render(f"Status: {info['status']}", True,
                                             GREEN if info["status"] == "SUCCESS" else
                                             RED if info["status"] == "BANKRUPT" else ACCENT),
                         (20, bot_y + 150))
        self._chart(aw + 20, bot_y, WIDTH - aw - 30, h, ACCENT)

        if info["status"] == "SUCCESS":
            ws = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            ws.fill((55, 205, 100, 28 + int(abs(math.sin(self._tick * 0.1)) * 35)))
            self.screen.blit(ws, (0, 0))
            wt = self.font_xl.render("SUCCESS - Sustainable Business Created!", True, GREEN)
            wx, wy = WIDTH // 2 - wt.get_width() // 2, HEIGHT // 2 - 20
            rrect(self.screen, PANEL, (wx - 18, wy - 10, wt.get_width() + 36, 48), r=10, bw=3, bc=GREEN)
            self.screen.blit(wt, (wx, wy))
        elif info["status"] == "BANKRUPT":
            wt = self.font_xl.render("BANKRUPT - ran out of capital", True, RED)
            wx, wy = WIDTH // 2 - wt.get_width() // 2, HEIGHT // 2 - 20
            rrect(self.screen, PANEL, (wx - 18, wy - 10, wt.get_width() + 36, 48), r=10, bw=3, bc=RED)
            self.screen.blit(wt, (wx, wy))

        if human and not self.headless:
            pygame.display.flip()
            self.clock.tick(self.fps)
        return np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2))

    def save(self, path):
        pygame.image.save(self.screen, path)

    def close(self):
        pygame.quit()
