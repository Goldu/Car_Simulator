"""
CARLA-style top-down driving simulator (pygame).

Classes: Config, MathUtil, Camera, CityMap, TrafficLight, TrafficLightManager,
         Autopilot, Vehicle, EgoVehicle, NPCVehicle, Pedestrian, Sensor,
         LidarSensor, CollisionSensor, WeatherSystem, World, HUD, Simulator

Run:  pip install pygame && python carla_style_sim.py

Controls
  W/Up accelerate | S/Down brake/reverse | A/D or Left/Right steer | SPACE handbrake
  P autopilot | L lidar | R weather | H help | O pause | BACKSPACE reset
  N spawn NPC | Mouse wheel or +/- zoom | ESC quit
"""
import math
import random
import pygame
from pygame.math import Vector2 as V2


# --------------------------------------------------------------------------- #
class Config:
    WIDTH, HEIGHT = 1280, 760
    FPS = 60
    GRID = 5                 # intersections per side
    SPACING = 600            # px between intersections (10 px = 1 m)
    ROAD_W = 90
    LANE_OFF = 22
    NPC_COUNT = 16
    PED_COUNT = 26
    LIDAR_RAYS = 90
    LIDAR_RANGE = 260
    GRASS = (58, 98, 52)
    ASPHALT = (62, 64, 68)
    SIDEWALK = (150, 150, 145)
    CAR_COLORS = [(200, 50, 50), (50, 110, 200), (230, 230, 230), (40, 40, 45),
                  (230, 170, 40), (60, 160, 90), (150, 80, 170), (200, 200, 60)]


# --------------------------------------------------------------------------- #
class MathUtil:
    @staticmethod
    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    @staticmethod
    def wrap(a):
        return (a + math.pi) % (2 * math.pi) - math.pi

    @staticmethod
    def ray_circle(o, d, c, r):
        f = o - c
        b = f.dot(d)
        disc = b * b - (f.dot(f) - r * r)
        if disc < 0:
            return None
        t = -b - math.sqrt(disc)
        return t if t >= 0 else None

    @staticmethod
    def ray_rect(o, d, rect):
        tmin, tmax = 0.0, 1e9
        for oo, dd, lo, hi in ((o.x, d.x, rect.left, rect.right),
                               (o.y, d.y, rect.top, rect.bottom)):
            if abs(dd) < 1e-9:
                if oo < lo or oo > hi:
                    return None
            else:
                t1, t2 = (lo - oo) / dd, (hi - oo) / dd
                if t1 > t2:
                    t1, t2 = t2, t1
                tmin, tmax = max(tmin, t1), min(tmax, t2)
                if tmin > tmax:
                    return None
        return tmin


# --------------------------------------------------------------------------- #
class Camera:
    def __init__(self, size):
        self.size = V2(size)
        self.pos = V2(0, 0)
        self.zoom = 1.0
        self.target_zoom = 1.0

    def follow(self, target, dt):
        self.pos += (target - self.pos) * min(1.0, 6 * dt)
        self.zoom += (self.target_zoom - self.zoom) * min(1.0, 8 * dt)

    def zoom_by(self, f):
        self.target_zoom = MathUtil.clamp(self.target_zoom * f, 0.3, 2.5)

    def to_screen(self, p):
        return (p - self.pos) * self.zoom + self.size / 2

    def rect(self, r):
        tl = self.to_screen(V2(r.topleft))
        return pygame.Rect(int(tl.x), int(tl.y), int(r.w * self.zoom) + 1, int(r.h * self.zoom) + 1)

    def visible(self, p, margin=100):
        s = self.to_screen(p)
        return -margin <= s.x <= self.size.x + margin and -margin <= s.y <= self.size.y + margin

    def view_rect(self):
        w, h = self.size.x / self.zoom, self.size.y / self.zoom
        return pygame.Rect(int(self.pos.x - w / 2), int(self.pos.y - h / 2), int(w), int(h))


# --------------------------------------------------------------------------- #
class CityMap:
    """Grid city: roads, sidewalks, buildings and a node graph for routing."""

    def __init__(self):
        c = Config
        self.n, S, hw = c.GRID, c.SPACING, c.ROAD_W // 2
        self.extent = (self.n - 1) * S
        self.nodes = {(i, j): V2(i * S, j * S) for i in range(self.n) for j in range(self.n)}
        self.roads, self.blocks, self.buildings, self.building_colors = [], [], [], []
        for k in range(self.n):
            self.roads.append(pygame.Rect(-hw, k * S - hw, self.extent + 2 * hw, c.ROAD_W))
            self.roads.append(pygame.Rect(k * S - hw, -hw, c.ROAD_W, self.extent + 2 * hw))
        rng = random.Random(7)
        for i in range(self.n - 1):
            for j in range(self.n - 1):
                b = pygame.Rect(i * S + hw, j * S + hw, S - c.ROAD_W, S - c.ROAD_W)
                self.blocks.append(b)
                inner = b.inflate(-30, -30)
                w = (inner.w - 14) // 2
                for sx in (0, 1):
                    for sy in (0, 1):
                        self.buildings.append(pygame.Rect(inner.x + sx * (w + 14),
                                                          inner.y + sy * (w + 14), w, w))
                        g = rng.randint(95, 165)
                        self.building_colors.append((g, g - rng.randint(0, 20), g - rng.randint(0, 35)))

    def neighbors(self, key):
        i, j = key
        return [(a, b) for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1))
                if 0 <= a < self.n and 0 <= b < self.n]

    def edges(self):
        return [(a, b) for a in self.nodes for b in self.neighbors(a)]

    def sidewalk_paths(self):
        paths = []
        for b in self.blocks:
            r = b.inflate(-14, -14)
            c = [V2(r.topleft), V2(r.topright), V2(r.bottomright), V2(r.bottomleft)]
            paths += [(c[k], c[(k + 1) % 4]) for k in range(4)]
        return paths

    def collides(self, pos, radius):
        e = Config.SPACING * 0 + self.extent
        if pos.x < -150 or pos.y < -150 or pos.x > e + 150 or pos.y > e + 150:
            return True
        for b in self.buildings:
            if abs(b.centerx - pos.x) > 135 or abs(b.centery - pos.y) > 135:
                continue
            cx = MathUtil.clamp(pos.x, b.left, b.right)
            cy = MathUtil.clamp(pos.y, b.top, b.bottom)
            if (pos.x - cx) ** 2 + (pos.y - cy) ** 2 < radius * radius:
                return True
        return False

    def draw(self, surf, cam):
        c = Config
        surf.fill(c.GRASS)
        view = cam.view_rect().inflate(120, 120)
        for b in self.blocks:
            if view.colliderect(b):
                pygame.draw.rect(surf, c.SIDEWALK, cam.rect(b))
        for b, col in zip(self.buildings, self.building_colors):
            if view.colliderect(b):
                pygame.draw.rect(surf, col, cam.rect(b))
                pygame.draw.rect(surf, tuple(max(0, v - 35) for v in col), cam.rect(b.inflate(-24, -24)))
        for r in self.roads:
            if view.colliderect(r):
                pygame.draw.rect(surf, c.ASPHALT, cam.rect(r))
        self._draw_markings(surf, cam, view)

    def _draw_markings(self, surf, cam, view):
        S, hw = Config.SPACING, Config.ROAD_W // 2
        for k in range(self.n):
            for i in range(self.n - 1):
                a, b = i * S + hw, (i + 1) * S - hw
                for horizontal in (True, False):
                    line = k * S
                    for t in range(a, b, 44):
                        p0 = V2(t, line) if horizontal else V2(line, t)
                        p1 = V2(t + 22, line) if horizontal else V2(line, t + 22)
                        if view.collidepoint(p0):
                            pygame.draw.line(surf, (225, 190, 50), cam.to_screen(p0),
                                             cam.to_screen(p1), max(1, int(2 * cam.zoom)))
                    for off in (-hw + 3, hw - 3):
                        p0 = V2(a, line + off) if horizontal else V2(line + off, a)
                        p1 = V2(b, line + off) if horizontal else V2(line + off, b)
                        if view.collidepoint(p0) or view.collidepoint(p1) or view.collidepoint((p0 + p1) / 2):
                            pygame.draw.line(surf, (210, 210, 210), cam.to_screen(p0),
                                             cam.to_screen(p1), max(1, int(2 * cam.zoom)))


# --------------------------------------------------------------------------- #
class TrafficLight:
    CYCLE = 20.0

    def __init__(self, key, pos, offset):
        self.key, self.pos, self.offset = key, pos, offset

    def state(self, axis, t):
        c = (t + self.offset) % self.CYCLE
        if axis == 'h':
            return 'green' if c < 8 else 'yellow' if c < 10 else 'red'
        return 'red' if c < 10 else 'green' if c < 18 else 'yellow'


class TrafficLightManager:
    COLORS = {'green': (60, 230, 90), 'yellow': (250, 210, 40), 'red': (240, 50, 50)}

    def __init__(self, city):
        self.t = 0.0
        self.lights = {k: TrafficLight(k, p, (k[0] + k[1]) * 5.0) for k, p in city.nodes.items()}

    def update(self, dt):
        self.t += dt

    def state(self, key, axis):
        return self.lights[key].state(axis, self.t)

    def draw(self, surf, cam):
        d = Config.ROAD_W // 2 + 8
        corners = {(-1, 1): 'h', (1, -1): 'h', (1, 1): 'v', (-1, -1): 'v'}
        for lt in self.lights.values():
            if not cam.visible(lt.pos, 120):
                continue
            for (sx, sy), axis in corners.items():
                p = cam.to_screen(lt.pos + V2(sx * d, sy * d))
                pygame.draw.circle(surf, (20, 20, 20), p, max(3, int(7 * cam.zoom)))
                pygame.draw.circle(surf, self.COLORS[lt.state(axis, self.t)], p, max(2, int(5 * cam.zoom)))


# --------------------------------------------------------------------------- #
class Autopilot:
    """Lane following + traffic-light, vehicle and pedestrian awareness."""
    DECEL = 160.0

    def __init__(self, city, edge=None, cruise=None):
        self.city = city
        self.cruise = cruise or random.uniform(100, 140)
        self.edge, self.next, self.stuck = edge, None, 0.0
        if edge:
            self._pick_next()

    def _pick_next(self):
        a, b = self.edge
        opts = [n for n in self.city.neighbors(b) if n != a] or [a]
        self.next = (b, random.choice(opts))

    def _geom(self, edge):
        A, B = self.city.nodes[edge[0]], self.city.nodes[edge[1]]
        d = (B - A).normalize()
        return A, B, d, V2(-d.y, d.x)

    def snap(self, v):
        best = None
        for e in self.city.edges():
            A, B, d, r = self._geom(e)
            lane = A + r * Config.LANE_OFF
            t = (v.pos - lane).dot(d)
            if 0 <= t <= (B - A).length() and d.dot(v.forward()) > 0.3:
                lat = abs((v.pos - lane).dot(r))
                if best is None or lat < best[0]:
                    best = (lat, e)
        if best is None:
            n = min(self.city.nodes, key=lambda k: self.city.nodes[k].distance_to(v.pos))
            e = max(((n, nb) for nb in self.city.neighbors(n)),
                    key=lambda e: self._geom(e)[2].dot(v.forward()))
        else:
            e = best[1]
        self.edge = e
        self._pick_next()

    def control(self, v, world):
        c, nodes = Config, self.city.nodes
        if self.edge is None:
            self.snap(v)
        A, B, d, r = self._geom(self.edge)
        dist_b = (B - v.pos).dot(d)
        if dist_b < 45:
            self.edge = self.next
            self._pick_next()
            A, B, d, r = self._geom(self.edge)
            dist_b = (B - v.pos).dot(d)

        # steering: pure pursuit on the right-hand lane
        lane = A + r * c.LANE_OFF
        t = max((v.pos - lane).dot(d), 0)
        target = lane + d * (t + 35 + v.speed * 0.3)
        err = MathUtil.wrap(math.atan2(target.y - v.pos.y, target.x - v.pos.x) - v.heading)
        steer = MathUtil.clamp(err * 1.8, -1, 1)

        # speed planning
        desired = self.cruise * (1 - min(0.55, abs(err) * 0.8))
        dn = (nodes[self.next[1]] - nodes[self.next[0]]).normalize()
        if abs(d.cross(dn)) > 0.5 and dist_b < 200:
            desired = min(desired, 65)

        axis = 'h' if abs(d.x) > 0.5 else 'v'
        state = world.lights.state(self.edge[1], axis)
        gap = dist_b - (c.ROAD_W / 2 + 12 + v.LENGTH / 2)
        if state != 'green' and gap > -6:
            if state == 'red' or v.speed ** 2 / (2 * self.DECEL) <= gap:
                desired = min(desired, math.sqrt(2 * self.DECEL * max(gap, 0)))

        if self.stuck < 14:
            fwd = v.forward()
            rt = V2(-fwd.y, fwd.x)
            for o in world.vehicles:
                if o is v:
                    continue
                rel = o.pos - v.pos
                ahead = rel.dot(fwd)
                if 0 < ahead < 170 and abs(rel.dot(rt)) < 26:
                    desired = min(desired, math.sqrt(2 * self.DECEL * max(ahead - v.LENGTH - 14, 0)))
            for p in world.pedestrians:
                rel = p.pos - v.pos
                ahead = rel.dot(fwd)
                if 0 < ahead < 130 and abs(rel.dot(rt)) < 24:
                    desired = min(desired, math.sqrt(2 * self.DECEL * max(ahead - v.LENGTH / 2 - 12, 0)))

        if v.speed < desired:
            throttle, brake = MathUtil.clamp((desired - v.speed) * 0.08, 0, 0.8), 0.0
        else:
            throttle, brake = 0.0, MathUtil.clamp((v.speed - desired) * 0.04, 0, 1)
        if desired <= 0.5 and v.speed < 3:
            throttle, brake = 0.0, 1.0
        self.stuck = self.stuck + 1 / Config.FPS if abs(v.speed) < 3 else 0.0
        return throttle, brake, steer


# --------------------------------------------------------------------------- #
class Vehicle:
    LENGTH, WIDTH, WHEELBASE = 46, 22, 30
    MAX_ACC, MAX_BRAKE, DRAG = 190.0, 420.0, 0.6
    MAX_SPEED, MAX_REVERSE, MAX_STEER = 320.0, 80.0, 0.62

    def __init__(self, pos, heading, color, name='vehicle'):
        self.pos, self.heading, self.color, self.name = V2(pos), heading, color, name
        self.speed = 0.0
        self.throttle = self.brake = self.steer_cmd = self.steer = 0.0

    def forward(self):
        return V2(math.cos(self.heading), math.sin(self.heading))

    def circles(self):
        f = self.forward()
        r = self.WIDTH * 0.55
        return [(self.pos + f * self.LENGTH * 0.25, r), (self.pos - f * self.LENGTH * 0.25, r)]

    def apply_control(self, throttle, brake, steer):
        self.throttle, self.brake, self.steer_cmd = throttle, brake, steer

    def on_collision(self, other, intensity):
        pass

    def update(self, dt, world):
        fr = world.weather.cur['friction']
        sgn = 1 if self.speed >= 0 else -1
        a = self.throttle * self.MAX_ACC - self.DRAG * self.speed
        if self.brake > 0:
            a -= sgn * self.brake * self.MAX_BRAKE * fr
        elif self.throttle == 0:
            a -= sgn * 15
        new = self.speed + a * dt
        if (self.brake > 0 or self.throttle == 0) and self.speed * new < 0:
            new = 0.0
        self.speed = MathUtil.clamp(new, -self.MAX_REVERSE, self.MAX_SPEED)

        limit = self.MAX_STEER / (1 + abs(self.speed) / 180) * (0.6 + 0.4 * fr)
        target = self.steer_cmd * limit
        self.steer += MathUtil.clamp(target - self.steer, -3.5 * dt, 3.5 * dt)

        old_pos, old_h = V2(self.pos), self.heading
        self.heading += self.speed / self.WHEELBASE * math.tan(self.steer) * dt
        self.pos += self.forward() * self.speed * dt
        if any(world.city.collides(c, r) for c, r in self.circles()):
            impact = abs(self.speed) * 0.1
            self.pos, self.heading = old_pos, old_h
            self.speed *= -0.25
            self.on_collision('building', impact)

    def draw(self, surf, cam, highlight=False):
        if not cam.visible(self.pos, 80):
            return
        f = self.forward()
        r = V2(-f.y, f.x)
        L, W = self.LENGTH / 2, self.WIDTH / 2

        def P(lx, ly):
            return cam.to_screen(self.pos + f * lx + r * ly)

        pygame.draw.polygon(surf, self.color, [P(-L, -W), P(L, -W), P(L, W), P(-L, W)])
        pygame.draw.polygon(surf, (25, 30, 40), [P(L * .1, -W * .8), P(L * .45, -W * .7),
                                                 P(L * .45, W * .7), P(L * .1, W * .8)])
        pygame.draw.polygon(surf, (25, 30, 40), [P(-L * .55, -W * .7), P(-L * .8, -W * .6),
                                                 P(-L * .8, W * .6), P(-L * .55, W * .7)])
        rad = max(2, int(3 * cam.zoom))
        for s in (-1, 1):
            pygame.draw.circle(surf, (255, 250, 200), P(L, s * W * .7), rad)
            pygame.draw.circle(surf, (255, 40, 40) if self.brake > 0.1 else (130, 20, 20),
                               P(-L, s * W * .7), rad)
        pygame.draw.polygon(surf, (255, 215, 0) if highlight else (10, 10, 10),
                            [P(-L, -W), P(L, -W), P(L, W), P(-L, W)], 2 if highlight else 1)


class NPCVehicle(Vehicle):
    def __init__(self, pos, heading, color, autopilot):
        super().__init__(pos, heading, color, 'npc')
        self.autopilot = autopilot

    def update(self, dt, world):
        self.apply_control(*self.autopilot.control(self, world))
        super().update(dt, world)


class EgoVehicle(Vehicle):
    def __init__(self, pos, heading, city):
        super().__init__(pos, heading, (235, 120, 30), 'ego')
        self.autopilot = Autopilot(city, None, cruise=130)
        self.autopilot_on = False
        self.lidar = LidarSensor(self)
        self.collision = CollisionSensor(self)
        self.sensors = [self.lidar, self.collision]

    def toggle_autopilot(self):
        self.autopilot_on = not self.autopilot_on
        if self.autopilot_on:
            self.autopilot.snap(self)
        return self.autopilot_on

    def on_collision(self, other, intensity):
        self.collision.on_event(other, intensity)

    def read_keys(self, keys, world):
        if self.autopilot_on:
            self.apply_control(*self.autopilot.control(self, world))
            return
        k = pygame
        th = br = st = 0.0
        if keys[k.K_w] or keys[k.K_UP]:
            if self.speed < -5:
                br = 1.0
            else:
                th = 1.0
        if keys[k.K_s] or keys[k.K_DOWN]:
            if self.speed > 8:
                br = 1.0
            else:
                th = -0.6
        if keys[k.K_SPACE]:
            th, br = 0.0, 1.0
        if keys[k.K_d] or keys[k.K_RIGHT]:
            st += 1
        if keys[k.K_a] or keys[k.K_LEFT]:
            st -= 1
        self.apply_control(th, br, st)

    def update(self, dt, world):
        super().update(dt, world)

    def draw(self, surf, cam, highlight=True):
        super().draw(surf, cam, True)


# --------------------------------------------------------------------------- #
class Pedestrian:
    RADIUS = 4

    def __init__(self, a, b):
        self.a, self.b = V2(a), V2(b)
        self.pos = self.a.lerp(self.b, random.random())
        self.dir = random.choice((-1, 1))
        self.speed = random.uniform(12, 18)
        self.hit_timer = 0.0
        self.color = random.choice([(240, 200, 60), (80, 170, 240), (230, 90, 160), (240, 240, 240)])

    def hit(self):
        self.hit_timer = 6.0

    def update(self, dt):
        if self.hit_timer > 0:
            self.hit_timer -= dt
            return
        tgt = self.b if self.dir > 0 else self.a
        to = tgt - self.pos
        step = self.speed * dt
        if to.length() <= step:
            self.pos = V2(tgt)
            self.dir *= -1
        else:
            self.pos += to.normalize() * step

    def draw(self, surf, cam):
        if cam.visible(self.pos, 20):
            col = (220, 30, 30) if self.hit_timer > 0 else self.color
            pygame.draw.circle(surf, col, cam.to_screen(self.pos), max(2, int(self.RADIUS * cam.zoom)))


# --------------------------------------------------------------------------- #
class Sensor:
    def __init__(self, owner):
        self.owner = owner

    def update(self, world, dt):
        pass

    def draw(self, surf, cam):
        pass


class LidarSensor(Sensor):
    def __init__(self, owner):
        super().__init__(owner)
        self.enabled = True
        self.timer = 0.0
        self.origin = V2(owner.pos)
        self.points = []

    def update(self, world, dt):
        self.timer += dt
        if not self.enabled or self.timer < 0.05:
            return
        self.timer = 0.0
        rng, n, o = Config.LIDAR_RANGE, Config.LIDAR_RAYS, self.owner
        origin = o.pos + o.forward() * 8
        rects = [b for b in world.city.buildings
                 if abs(b.centerx - origin.x) < rng + 130 and abs(b.centery - origin.y) < rng + 130]
        circles = []
        for v in world.vehicles:
            if v is not o and v.pos.distance_to(origin) < rng + 40:
                circles += v.circles()
        circles += [(p.pos, p.RADIUS) for p in world.pedestrians if p.pos.distance_to(origin) < rng]
        pts = []
        for k in range(n):
            ang = o.heading + k * 2 * math.pi / n
            d = V2(math.cos(ang), math.sin(ang))
            best, hit = rng, False
            for rc in rects:
                t = MathUtil.ray_rect(origin, d, rc)
                if t is not None and t < best:
                    best, hit = t, True
            for c, rad in circles:
                t = MathUtil.ray_circle(origin, d, c, rad)
                if t is not None and t < best:
                    best, hit = t, True
            pts.append((origin + d * best, hit))
        self.origin, self.points = origin, pts

    def draw(self, surf, cam):
        if not self.enabled:
            return
        o = cam.to_screen(self.origin)
        for p, hit in self.points:
            s = cam.to_screen(p)
            pygame.draw.line(surf, (35, 150, 80), o, s, 1)
            if hit:
                pygame.draw.circle(surf, (255, 70, 70), s, 2)


class CollisionSensor(Sensor):
    def __init__(self, owner):
        super().__init__(owner)
        self.total, self.last = 0, None
        self.cooldown = 0.0

    def update(self, world, dt):
        self.cooldown = max(0.0, self.cooldown - dt)

    def on_event(self, other, intensity):
        if self.cooldown > 0:
            return
        self.cooldown = 0.6
        self.total += 1
        self.last = (other, intensity)


# --------------------------------------------------------------------------- #
class WeatherSystem:
    PRESETS = [
        dict(name='Clear Noon', dark=0, fog=0, rain=False, friction=1.0),
        dict(name='Clear Night', dark=170, fog=0, rain=False, friction=1.0),
        dict(name='Soft Rain', dark=35, fog=35, rain=True, friction=0.75),
        dict(name='Hard Rain Night', dark=155, fog=45, rain=True, friction=0.6),
        dict(name='Foggy Dawn', dark=20, fog=125, rain=False, friction=0.9),
    ]

    def __init__(self, size):
        self.idx = 0
        self.size = size
        self.layer = pygame.Surface(size, pygame.SRCALPHA)
        self.drops = [[random.uniform(0, size[0]), random.uniform(0, size[1])] for _ in range(240)]

    @property
    def cur(self):
        return self.PRESETS[self.idx]

    def cycle(self):
        self.idx = (self.idx + 1) % len(self.PRESETS)
        return self.cur['name']

    def update(self, dt):
        if self.cur['rain']:
            w, h = self.size
            for d in self.drops:
                d[0] = (d[0] - 160 * dt) % w
                d[1] = (d[1] + 900 * dt) % h

    def draw(self, surf, world, cam):
        w = self.cur
        if w['dark']:
            self.layer.fill((5, 8, 25, w['dark']))
            for v in world.vehicles:
                if not cam.visible(v.pos, 150):
                    continue
                for rad, k in ((95, .7), (68, .45), (42, .2)):
                    c = cam.to_screen(v.pos + v.forward() * 60)
                    pygame.draw.circle(self.layer, (5, 8, 25, int(w['dark'] * k)), c, int(rad * cam.zoom))
            surf.blit(self.layer, (0, 0))
        if w['fog']:
            self.layer.fill((190, 195, 200, w['fog']))
            surf.blit(self.layer, (0, 0))
        if w['rain']:
            for x, y in self.drops:
                pygame.draw.line(surf, (170, 190, 225), (x, y), (x + 3, y - 12), 1)


# --------------------------------------------------------------------------- #
class World:
    def __init__(self):
        size = (Config.WIDTH, Config.HEIGHT)
        self.city = CityMap()
        self.lights = TrafficLightManager(self.city)
        self.weather = WeatherSystem(size)
        self.vehicles, self.pedestrians, self.ego = [], [], None
        self.reset()

    def reset(self):
        S = Config.SPACING
        self.vehicles, self.pedestrians = [], []
        self.ego = EgoVehicle(V2(S - 260, S + Config.LANE_OFF), 0.0, self.city)
        self.vehicles.append(self.ego)
        for _ in range(Config.NPC_COUNT):
            self.spawn_npc()
        paths = self.city.sidewalk_paths()
        for _ in range(Config.PED_COUNT):
            self.pedestrians.append(Pedestrian(*random.choice(paths)))

    def spawn_npc(self):
        for _ in range(40):
            a, b = random.choice(self.city.edges())
            A, B = self.city.nodes[a], self.city.nodes[b]
            d = (B - A).normalize()
            r = V2(-d.y, d.x)
            length = (B - A).length()
            pos = A + d * random.uniform(110, length - 110) + r * Config.LANE_OFF
            if all(v.pos.distance_to(pos) > 80 for v in self.vehicles):
                ap = Autopilot(self.city, (a, b))
                npc = NPCVehicle(pos, math.atan2(d.y, d.x), random.choice(Config.CAR_COLORS), ap)
                npc.speed = ap.cruise * 0.7
                self.vehicles.append(npc)
                return npc
        return None

    def update(self, dt, keys):
        self.lights.update(dt)
        self.weather.update(dt)
        self.ego.read_keys(keys, self)
        for v in self.vehicles:
            v.update(dt, self)
        for p in self.pedestrians:
            p.update(dt)
        self._resolve_vehicles()
        self._resolve_pedestrians()
        for s in self.ego.sensors:
            s.update(self, dt)

    def _resolve_vehicles(self):
        vs = self.vehicles
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                a, b = vs[i], vs[j]
                if a.pos.distance_squared_to(b.pos) > 3600:
                    continue
                for ca, ra in a.circles():
                    for cb, rb in b.circles():
                        d = ca - cb
                        dist = d.length()
                        if dist < ra + rb:
                            n = d / dist if dist > 1e-3 else V2(1, 0)
                            push = (ra + rb - dist) / 2
                            a.pos += n * push
                            b.pos -= n * push
                            rel = abs(a.speed - b.speed) * 0.1
                            a.speed *= 0.7
                            b.speed *= 0.7
                            a.on_collision(b.name, rel)
                            b.on_collision(a.name, rel)

    def _resolve_pedestrians(self):
        for p in self.pedestrians:
            if p.hit_timer > 0:
                continue
            for v in self.vehicles:
                if p.pos.distance_squared_to(v.pos) > 2500:
                    continue
                if any(p.pos.distance_to(c) < r + p.RADIUS for c, r in v.circles()) and abs(v.speed) > 25:
                    p.hit()
                    v.on_collision('pedestrian', abs(v.speed) * 0.1)

    def draw(self, surf, cam):
        self.city.draw(surf, cam)
        self.lights.draw(surf, cam)
        for p in self.pedestrians:
            p.draw(surf, cam)
        for v in self.vehicles:
            if v is not self.ego:
                v.draw(surf, cam)
        self.ego.draw(surf, cam)
        self.ego.lidar.draw(surf, cam)
        self.weather.draw(surf, self, cam)


# --------------------------------------------------------------------------- #
class HUD:
    MM = 190

    def __init__(self, size, city):
        self.size = size
        self.font = pygame.font.SysFont('consolas,dejavusansmono,monospace', 15)
        self.big = pygame.font.SysFont('consolas,dejavusansmono,monospace', 38, bold=True)
        self.show_help = True
        self.notes = []
        self.scale = self.MM / (city.extent + 300)
        self.mm_static = pygame.Surface((self.MM, self.MM))
        self.mm_static.fill((40, 70, 40))
        for r in city.roads:
            pygame.draw.rect(self.mm_static, (110, 110, 115), pygame.Rect(
                (r.x + 150) * self.scale, (r.y + 150) * self.scale,
                max(1, r.w * self.scale), max(1, r.h * self.scale)))

    def notify(self, text, dur=2.5):
        self.notes.append([text, dur])

    def update(self, dt):
        for n in self.notes:
            n[1] -= dt
        self.notes = [n for n in self.notes if n[1] > 0]

    def _text(self, surf, text, pos, color=(235, 235, 235), font=None):
        surf.blit((font or self.font).render(text, True, color), pos)

    def _bar(self, surf, label, val, x, y, color):
        self._text(surf, label, (x, y))
        pygame.draw.rect(surf, (70, 70, 70), (x + 90, y + 3, 120, 10))
        pygame.draw.rect(surf, color, (x + 90, y + 3, int(120 * MathUtil.clamp(val, 0, 1)), 10))

    def draw(self, surf, world, cam, clock, paused):
        ego = world.ego
        panel = pygame.Surface((250, 250), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 150))
        surf.blit(panel, (10, 10))
        kmh = ego.speed * 0.36
        lines = [f"FPS: {clock.get_fps():5.1f}", "Map: Town-Grid5",
                 f"Weather: {world.weather.cur['name']}",
                 f"Autopilot: {'ON' if ego.autopilot_on else 'OFF'}",
                 f"Lidar: {'ON' if ego.lidar.enabled else 'OFF'}",
                 f"Vehicles: {len(world.vehicles)}  Peds: {len(world.pedestrians)}",
                 f"Collisions: {ego.collision.total}"]
        for i, t in enumerate(lines):
            self._text(surf, t, (20, 18 + i * 20))
        if ego.collision.last:
            o, inten = ego.collision.last
            self._text(surf, f"Last: {o} ({inten:.1f} m/s)", (20, 18 + 7 * 20), (255, 150, 100))
        self._bar(surf, "Throttle", max(ego.throttle, 0), 20, 180, (80, 220, 100))
        self._bar(surf, "Brake", ego.brake, 20, 200, (240, 80, 80))
        self._bar(surf, "Steer", abs(ego.steer) / ego.MAX_STEER, 20, 220, (90, 160, 255))

        gear = 'R' if ego.speed < -1 else 'D'
        self._text(surf, f"{abs(kmh):3.0f}", (self.size[0] // 2 - 50, self.size[1] - 70), font=self.big)
        self._text(surf, f"km/h  {gear}", (self.size[0] // 2 + 25, self.size[1] - 52))

        self._draw_minimap(surf, world, cam)
        if self.show_help:
            help_lines = ["W/S throttle-brake  A/D steer  SPACE handbrake", "P autopilot  L lidar  R weather",
                          "N spawn NPC  O pause  BACKSPACE reset  H help"]
            for i, t in enumerate(help_lines):
                self._text(surf, t, (14, self.size[1] - 70 + i * 20), (200, 200, 120))
        for i, (t, _) in enumerate(self.notes):
            self._text(surf, t, (self.size[0] // 2 - 120, 24 + i * 22), (255, 230, 120))
        if paused:
            self._text(surf, "PAUSED", (self.size[0] // 2 - 55, self.size[1] // 2), (255, 255, 255), self.big)

    def _draw_minimap(self, surf, world, cam):
        x0, y0 = self.size[0] - self.MM - 10, 10
        mm = self.mm_static.copy()
        s = self.scale

        def M(p):
            return int((p.x + 150) * s), int((p.y + 150) * s)

        for v in world.vehicles:
            pygame.draw.circle(mm, (255, 255, 255), M(v.pos), 1)
        for lt in world.lights.lights.values():
            pygame.draw.circle(mm, (200, 200, 60), M(lt.pos), 1)
        pygame.draw.circle(mm, (255, 150, 30), M(world.ego.pos), 4)
        vr = cam.view_rect()
        pygame.draw.rect(mm, (255, 255, 255), pygame.Rect(M(V2(vr.topleft)), (int(vr.w * s), int(vr.h * s))), 1)
        surf.blit(mm, (x0, y0))
        pygame.draw.rect(surf, (230, 230, 230), (x0, y0, self.MM, self.MM), 2)


# --------------------------------------------------------------------------- #
class Simulator:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("CARLA-style Simulator (pygame)")
        self.screen = pygame.display.set_mode((Config.WIDTH, Config.HEIGHT))
        self.clock = pygame.time.Clock()
        self.world = World()
        self.camera = Camera((Config.WIDTH, Config.HEIGHT))
        self.camera.pos = V2(self.world.ego.pos)
        self.hud = HUD((Config.WIDTH, Config.HEIGHT), self.world.city)
        self.running, self.paused = True, False
        self._last_collisions = 0

    def handle_events(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.running = False
            elif e.type == pygame.MOUSEWHEEL:
                self.camera.zoom_by(1.1 ** e.y)
            elif e.type == pygame.KEYDOWN:
                k, w = e.key, self.world
                if k == pygame.K_ESCAPE:
                    self.running = False
                elif k == pygame.K_p:
                    self.hud.notify(f"Autopilot {'ON' if w.ego.toggle_autopilot() else 'OFF'}")
                elif k == pygame.K_l:
                    w.ego.lidar.enabled = not w.ego.lidar.enabled
                elif k == pygame.K_r:
                    self.hud.notify(f"Weather: {w.weather.cycle()}")
                elif k == pygame.K_h:
                    self.hud.show_help = not self.hud.show_help
                elif k == pygame.K_o:
                    self.paused = not self.paused
                elif k == pygame.K_n:
                    self.hud.notify("NPC spawned" if w.spawn_npc() else "No free spawn point")
                elif k in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    self.camera.zoom_by(1.15)
                elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self.camera.zoom_by(1 / 1.15)
                elif k == pygame.K_BACKSPACE:
                    w.reset()
                    self.camera.pos = V2(w.ego.pos)
                    self._last_collisions = 0
                    self.hud.notify("World reset")

    def update(self, dt):
        if not self.paused:
            self.world.update(dt, pygame.key.get_pressed())
        ego = self.world.ego
        self.camera.follow(ego.pos + ego.forward() * ego.speed * 0.25, dt)
        self.hud.update(dt)
        if ego.collision.total != self._last_collisions:
            self._last_collisions = ego.collision.total
            self.hud.notify(f"Collision with {ego.collision.last[0]}!")

    def render(self):
        self.world.draw(self.screen, self.camera)
        self.hud.draw(self.screen, self.world, self.camera, self.clock, self.paused)
        pygame.display.flip()

    def run(self):
        while self.running:
            dt = min(self.clock.tick(Config.FPS) / 1000.0, 0.05)
            self.handle_events()
            self.update(dt)
            self.render()
        pygame.quit()


if __name__ == "__main__":
    Simulator().run()
