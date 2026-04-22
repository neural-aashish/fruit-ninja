import cv2
import mediapipe as mp
import numpy as np
import random
import math
import time

# =============================
# SETUP
# =============================
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

# =============================
# VARIABLES
# =============================
trail = []
MAX_TRAIL = 20
score = 0
fruits = []
spawn_time = time.time()

# =============================
# HAND DRAW
# =============================
def draw_hand(frame, hand_landmarks, w, h):
    pts = []
    for i in range(21):
        x = int(hand_landmarks.landmark[i].x * w)
        y = int(hand_landmarks.landmark[i].y * h)
        pts.append((x,y))

    connections = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20)
    ]

    for c in connections:
        cv2.line(frame, pts[c[0]], pts[c[1]], (0,200,255), 2)

    for (x,y) in pts:
        cv2.circle(frame, (x,y), 4, (255,255,255), -1)

    return pts

# =============================
# FRUIT CLASS
# =============================
class Fruit:
    def __init__(self, w, h):
        self.x = random.randint(100, w-100)
        self.y = h + 50
        self.vx = random.uniform(-3,3)
        self.vy = random.uniform(-22, -28)
        self.r = random.randint(25,40)
        self.color = random.choice([
            (0,255,0),(0,0,255),(255,0,0),(0,255,255)
        ])
        self.sliced = False

    def update(self):
        self.vy += 0.45
        self.x += self.vx
        self.y += self.vy

    def draw(self, frame):
        if not self.sliced:
            cv2.circle(frame, (int(self.x), int(self.y)), self.r, self.color, -1)

# =============================
# COLLISION
# =============================
def line_circle(px, py, x1, y1, x2, y2):
    line_mag = math.hypot(x2-x1, y2-y1)
    if line_mag < 1:
        return 999
    u = ((px-x1)*(x2-x1)+(py-y1)*(y2-y1))/(line_mag**2)
    u = max(min(u,1),0)
    ix = x1 + u*(x2-x1)
    iy = y1 + u*(y2-y1)
    return math.hypot(px-ix, py-iy)

# =============================
# MAIN LOOP
# =============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame,1)
    h, w = frame.shape[:2]

    # LIGHT UI STYLE
    overlay = np.full_like(frame, (220,220,220))
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

    # HAND DETECTION
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    finger = None

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            pts = draw_hand(frame, handLms, w, h)
            x, y = pts[8]

            # smoothing
            if trail:
                px, py = trail[-1]
                x = int(px*0.6 + x*0.4)
                y = int(py*0.6 + y*0.4)

            finger = (x,y)

            # green glow
            for r in range(15,5,-3):
                cv2.circle(frame, (x,y), r, (0,255,0), -1)

    # TRAIL
    if finger:
        trail.append(finger)
    if len(trail) > MAX_TRAIL:
        trail.pop(0)

    # DRAW SLASH
    for i in range(1,len(trail)):
        cv2.line(frame, trail[i-1], trail[i], (0,0,255), 6)
        cv2.line(frame, trail[i-1], trail[i], (0,150,255), 2)

    # SPAWN FRUITS
    if time.time() - spawn_time > 1:
        fruits.append(Fruit(w,h))
        spawn_time = time.time()

    # UPDATE FRUITS
    for fruit in fruits[:]:
        fruit.update()
        fruit.draw(frame)

        # SLICE DETECTION
        for i in range(1,len(trail)):
            x1,y1 = trail[i-1]
            x2,y2 = trail[i]

            speed = math.hypot(x2-x1, y2-y1)

            if speed > 15:
                dist = line_circle(fruit.x, fruit.y, x1,y1,x2,y2)
                if dist < fruit.r:
                    fruit.sliced = True
                    score += 1
                    fruits.remove(fruit)
                    break

        # REMOVE IF FALLS DOWN
        if fruit.y > h + 50:
            fruits.remove(fruit)

    # UI
    cv2.putText(frame, f"Score: {score}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.imshow("AI Fruit Slice", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()