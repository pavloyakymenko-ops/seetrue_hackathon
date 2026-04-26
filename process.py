# process.py

import cv2
import collections
import time
import threading
import urllib.parse
import urllib.request
import json

from ultralytics import YOLO
from person_database import PersonDatabase
from voice_assistant import VoiceAssistant, draw_voice_status

VGA_W, VGA_H = 640, 480

# ── Configuration ─────────────────────────────────────────────────────────────
SERPAPI_KEY = "2e046fcc720d02544597c5d60eb664fdfea7bd8e0b1f3718d2683fc24d62b7fe"

HIGHLIGHT_FADE = 8.0   # seconds the box + text stay visible
DWELL_SECONDS  = 1.5   # seconds gaze must dwell before trigger
DWELL_RADIUS   = 60    # pixel radius gaze must stay within

PUPIL_CHANGE_THRESHOLD = 0.08   # 8% pupil diameter change triggers search

# ── Cognitive Load Tracker config ─────────────────────────────────────────────
CLT_WINDOW_SEC       = 30.0    # rolling window length (seconds)
CLT_PUPIL_BASELINE_SEC = 10.0  # first N seconds used to learn personal baseline
CLT_UPDATE_INTERVAL  = 0.25    # recalc score every 250 ms (not every frame)


# Person-overlay colours  (BGR)
PERSON_BOX_COLOR  = (0, 200, 255)   # amber/gold
UNKNOWN_BOX_COLOR = (120, 120, 120) # grey for unrecognised faces


# Category → colour mapping
CATEGORY_COLORS = {
    "phone":       (255, 100, 50),   # blue
    "tablet":      (200, 150, 50),   # teal
    "laptop":      (50, 200, 255),   # orange
    "desktop":     (50, 150, 200),   # brown-orange
    "peripheral":  (180, 120, 255),  # pink
    "audio":       (100, 255, 100),  # green
    "camera":      (0, 200, 200),    # yellow
    "wearable":    (255, 50, 150),   # purple
    "gaming":      (0, 100, 255),    # red
    "networking":  (200, 200, 50),   # cyan
    "storage":     (100, 180, 220),  # sand
    "smart_home":  (50, 255, 200),   # mint
    "power":       (80, 80, 220),    # dark red
    "maker":       (0, 180, 180),    # olive
    "other":       (180, 180, 180),  # grey
}

# Class name → category
CLASS_CATEGORY = {}
_cat_map = {
    "phone":      ["iPhone", "Samsung Galaxy phone", "Google Pixel phone", "foldable smartphone"],
    "tablet":     ["iPad Pro", "iPad mini", "Android tablet", "Kindle e-reader", "Microsoft Surface tablet"],
    "laptop":     ["MacBook laptop", "Dell XPS laptop", "ThinkPad laptop", "gaming laptop", "Chromebook"],
    "desktop":    ["computer monitor", "curved monitor", "desktop computer tower", "iMac desktop", "Mac mini"],
    "peripheral": ["computer mouse", "trackball mouse", "mechanical keyboard", "Apple Magic Keyboard",
                   "laptop trackpad", "webcam", "drawing tablet", "stylus pen", "Apple Pencil", "stream deck"],
    "audio":      ["AirPods earbuds", "wireless earbuds", "over-ear headphones", "gaming headset",
                   "Bluetooth speaker", "studio monitor speaker", "podcast microphone", "lavalier microphone"],
    "camera":     ["digital camera", "DSLR camera", "mirrorless camera", "GoPro action camera", "camera lens", "ring light"],
    "wearable":   ["Apple Watch", "Garmin smartwatch", "fitness tracker", "VR headset", "Meta Quest"],
    "gaming":     ["PlayStation controller", "Xbox controller", "Nintendo Switch", "Steam Deck console",
                   "gaming console", "vr controllers"],
    "networking": ["wifi router", "network switch", "NAS server"],
    "storage":    ["USB flash drive", "external hard drive", "portable SSD", "USB-C hub", "SD card", "SD card reader"],
    "smart_home": ["smart speaker", "Amazon Echo", "Google Nest Hub", "smart thermostat", "smart bulb",
                   "security camera", "video doorbell"],
    "power":      ["power bank", "wireless charging pad", "MagSafe charger", "laptop power brick",
                   "power strip", "surge protector", "USB-C cable", "HDMI cable"],
    "maker":      ["Raspberry Pi", "Arduino board", "soldering iron", "multimeter", "3D printer",
                   "drone", "calculator", "laser pointer"],
}

for cat, classes in _cat_map.items():
    for cls in classes:
        CLASS_CATEGORY[cls] = cat


def _category_color(class_name: str):
    cat = CLASS_CATEGORY.get(class_name, "other")
    return CATEGORY_COLORS[cat]


# Classes we should NOT search for on Google Shopping
IGNORE_SEARCH_CLASSES = {
    "person", "cat", "dog", "bird", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe",
}

# ── YOLO colour palette ───────────────────────────────────────────────────────
_PALETTE = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
    (49, 210, 207), (10, 249, 72),  (23, 204, 146), (134, 219, 61),
    (52, 147, 26),  (187, 212, 0),  (168, 153, 44), (255, 194, 0),
    (147, 69, 52),  (255, 115, 100),(236, 24, 0),   (255, 56, 132),
    (133, 0, 82),   (255, 56, 203), (200, 149, 255),(199, 55, 255),
]

def _class_color(class_id: int):
    return _PALETTE[class_id % len(_PALETTE)]


# ── Product search helpers ────────────────────────────────────────────────────

def _search_product(query: str) -> dict:
    """Return {'price': str} via SerpAPI Google Shopping."""
    if SERPAPI_KEY:
        params = urllib.parse.urlencode({
            "engine": "google_shopping",
            "q": query,
            "api_key": SERPAPI_KEY,
            "num": 1,
        })
        api_url = f"https://serpapi.com/search?{params}"
        try:
            req = urllib.request.Request(api_url,
                                         headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode())
            items = data.get("shopping_results", [])
            if items:
                price = items[0].get("price", "Price N/A")
                return {"price": price}
        except Exception as exc:
            print(f"[ProductSearch] SerpAPI error: {exc}")

    return {"price": "Price N/A"}


# ── OpenCV face detector (Haar – no extra deps) ───────────────────────────────

def _make_face_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        print("[FaceDetect] WARNING: Haar cascade not found – face detection disabled.")
        return None
    return detector


# ─────────────────────────────────────────────────────────────────────────────
# ██████████████████   CONFIGURE YOUR PEOPLE HERE   ██████████████████████████
# ─────────────────────────────────────────────────────────────────────────────
#
#  Add one entry per person.  photo_path can be absolute or relative to the
#  working directory where you launch main.py.
#
#  Example:
#       {"name": "Alice Smith",  "age": 29, "role": "Engineer",
#        "photo_path": "photos/alice.jpg"},
#
PEOPLE_REGISTRY: list[dict] = [

    {"name": "Faisal", "age": 22, "role": "Student",
         "photo_path": r"C:\\Users\\Pavel\\Pictures\\faisalphoto.jpg"},
    {"name": "Sai Kiran", "age": 29, "role": "Job Seeker", 
            "photo_path": r"C:\\Users\\Pavel\\Pictures\\saikiranphoto.jpg"},
    # ← ADD YOUR ENTRIES HERE ↓
    # {"name": "Alice Smith",   "age": 29, "role": "Engineer",    "photo_path": "photos/alice.jpg"},
    # {"name": "Bob Johnson",   "age": 45, "role": "Businessman", "photo_path": "photos/bob.jpg"},
    # {"name": "Carol White",   "age": 22, "role": "Student",     "photo_path": "photos/carol.jpg"},
    # ← ADD YOUR ENTRIES HERE ↑
]
# ─────────────────────────────────────────────────────────────────────────────

# ── Cognitive Load Tracker ────────────────────────────────────────────────────
#
#  READ-ONLY observer of eye signals.  Never writes to shared_data["eyeEvent"]
#  so it won't collide with double-blink detection or any other event consumer.
#
 
class CognitiveLoadTracker:
    """
    Produces a 0–100 cognitive load score from three signals:
      • pupil dilation vs personal baseline   (weight 0.45)
      • blink rate  (low = focused/overloaded) (weight 0.25)
      • fixation duration  (long = deep processing) (weight 0.30)
 
    All signals are observed passively — nothing is consumed or modified.
    """
 
    def __init__(self):
        self._pupil_history   = collections.deque()   # (timestamp, avg_pupil)
        self._blink_times     = collections.deque()   # timestamps of detected blinks
        self._fixation_durs   = collections.deque()   # (timestamp, duration_estimate)
 
        self._baseline_pupil  = None
        self._baseline_locked = False
        self._start_time      = None
 
        # Track event transitions (read-only) to count blinks
        self._last_event      = ""
        self._fixation_start  = None
 
        # Output
        self.score            = 0       # 0 = relaxed, 100 = overloaded
        self.level            = "calm"  # "calm" | "focused" | "high"
        self._last_calc       = 0
 
    def update(self, pupil_avg: float, eye_event: str, now: float):
        """Call every frame. Reads pupil + event, never writes to shared_data."""
        if self._start_time is None:
            self._start_time = now
 
        # ── Record pupil ──────────────────────────────────────────────────────
        if pupil_avg > 0:
            self._pupil_history.append((now, pupil_avg))
 
        # ── Detect blink transitions (READ-ONLY on eye_event) ─────────────────
        #    "BB" = blink begin.  We count the transition INTO BB, not BB itself,
        #    so repeated BB frames don't double-count.
        if eye_event == "BB" and self._last_event != "BB":
            self._blink_times.append(now)
        self._last_event = eye_event
 
        # ── Fixation duration tracking ────────────────────────────────────────
        if eye_event == "FB":
            if self._fixation_start is None:
                self._fixation_start = now
        else:
            if self._fixation_start is not None:
                dur = now - self._fixation_start
                if dur > 0.05:  # ignore micro-fixations
                    self._fixation_durs.append((now, dur))
                self._fixation_start = None
 
        # ── Trim old data outside the rolling window ──────────────────────────
        cutoff = now - CLT_WINDOW_SEC
        while self._pupil_history and self._pupil_history[0][0] < cutoff:
            self._pupil_history.popleft()
        while self._blink_times and self._blink_times[0] < cutoff:
            self._blink_times.popleft()
        while self._fixation_durs and self._fixation_durs[0][0] < cutoff:
            self._fixation_durs.popleft()
 
        # ── Learn baseline pupil from first N seconds ─────────────────────────
        if not self._baseline_locked:
            elapsed = now - self._start_time
            if elapsed >= CLT_PUPIL_BASELINE_SEC and len(self._pupil_history) > 10:
                vals = [v for _, v in self._pupil_history]
                self._baseline_pupil = sum(vals) / len(vals)
                self._baseline_locked = True
                print(f"[CogLoad] Baseline pupil locked: {self._baseline_pupil:.2f}")
 
        # ── Recalculate score (throttled) ─────────────────────────────────────
        if now - self._last_calc < CLT_UPDATE_INTERVAL:
            return
        self._last_calc = now
        self._calc_score(now)
 
    def _calc_score(self, now: float):
        # --- Pupil component (0–100) ---
        pupil_score = 50  # neutral until baseline is ready
        if self._baseline_pupil and self._baseline_pupil > 0 and self._pupil_history:
            recent = [v for _, v in self._pupil_history]
            avg    = sum(recent) / len(recent)
            # +20% dilation → score 100,  −10% constriction → score 0
            ratio  = (avg - self._baseline_pupil) / self._baseline_pupil
            pupil_score = max(0, min(100, 50 + ratio * 250))
 
        # --- Blink rate component (0–100) ---
        blink_score = 50
        window_len  = min(CLT_WINDOW_SEC, (now - self._start_time) if self._start_time else 1)
        if window_len > 3:
            blinks_per_min = len(self._blink_times) / window_len * 60
            # Normal: ~15–20 blinks/min.  <8 = deep focus/overload, >25 = fatigue
            if blinks_per_min < 8:
                blink_score = 80 + min(20, (8 - blinks_per_min) * 3)
            elif blinks_per_min > 25:
                blink_score = 70 + min(30, (blinks_per_min - 25) * 2)
            else:
                blink_score = max(0, 50 - (blinks_per_min - 8) * 3)
 
        # --- Fixation duration component (0–100) ---
        fix_score = 50
        if self._fixation_durs:
            durations = [d for _, d in self._fixation_durs]
            avg_dur   = sum(durations) / len(durations)
            # >600 ms avg fixation = heavy processing, <200 ms = scanning
            fix_score = max(0, min(100, (avg_dur - 0.2) / 0.5 * 100))
 
        # --- Weighted blend ---
        self.score = int(
            pupil_score * 0.45 +
            blink_score * 0.25 +
            fix_score   * 0.30
        )
        self.score = max(0, min(100, self.score))
 
        if self.score < 35:
            self.level = "calm"
        elif self.score < 65:
            self.level = "focused"
        else:
            self.level = "high"
 
 
def _draw_cognitive_bar(frame, tracker: CognitiveLoadTracker):
    """Draw a small cognitive load meter in the top-right corner."""
    score = tracker.score
    level = tracker.level
 
    bar_w, bar_h = 120, 16
    margin = 10
    x1 = frame.shape[1] - bar_w - margin
    y1 = margin
    x2 = x1 + bar_w
    y2 = y1 + bar_h
 
    # Background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1 - 2, y1 - 20), (x2 + 2, y2 + 2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
 
    # Colour: green → yellow → red
    if score < 35:
        color = (80, 200, 80)     # green
    elif score < 65:
        color = (0, 200, 255)     # yellow
    else:
        color = (50, 50, 230)     # red
 
    # Filled portion
    fill_w = int(bar_w * score / 100)
    cv2.rectangle(frame, (x1, y1), (x1 + fill_w, y2), color, -1)
    # Border
    cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)
 
    # Label
    label = f"Load: {score}%  ({level})"
    cv2.putText(frame, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
 

# ── Main class ────────────────────────────────────────────────────────────────

class process:
    def __init__(self, shared_data, image_buffer_scene):
        print("enter main")
        self.shared_data        = shared_data
        self.image_buffer_scene = image_buffer_scene

        # Gaze smoothing
        self.gazeX_history = collections.deque(maxlen=10)
        self.gazeY_history = collections.deque(maxlen=10)

        # Eye event state
        self.current_event = "NA"

        # Dwell & Pupil state
        self.dwell_start_time = None
        self.dwell_anchor     = None
        self.baseline_pupil   = None

        # Active highlight overlays
        self.active_highlights = []
        self._lock             = threading.Lock()

         # ── Cognitive load tracker (read-only observer) ───────────────────────
        self.cog_tracker = CognitiveLoadTracker()

        # ── Voice assistant ───────────────────────────────────────────────────
        self.voice = VoiceAssistant()

        # ── Person database ───────────────────────────────────────────────────
        print("Loading person database…")
        self.person_db     = PersonDatabase()
        self.face_detector = _make_face_detector()

        if PEOPLE_REGISTRY:
            self.person_db.register_many(PEOPLE_REGISTRY)
            print(f"Person database ready: {self.person_db.size} person(s) registered.")
        else:
            print("Person database is empty – add entries to PEOPLE_REGISTRY in process.py.")

        # ── YOLO gadget model ─────────────────────────────────────────────────
        print("Loading Open-Vocabulary YOLO model…")
        self.yolo = YOLO("yolov8s-world.pt")

        custom_classes = [
            "iPhone", "Samsung Galaxy phone", "Google Pixel phone", "foldable smartphone",
            "iPad Pro", "iPad mini", "Android tablet", "Kindle e-reader", "Microsoft Surface tablet",
            "MacBook laptop", "Dell XPS laptop", "ThinkPad laptop", "gaming laptop", "Chromebook",
            "computer monitor", "curved monitor", "desktop computer tower", "iMac desktop", "Mac mini",
            "computer mouse", "trackball mouse", "mechanical keyboard", "Apple Magic Keyboard",
            "laptop trackpad", "webcam", "drawing tablet", "stylus pen", "Apple Pencil", "stream deck",
            "AirPods earbuds", "wireless earbuds", "over-ear headphones", "gaming headset",
            "Bluetooth speaker", "studio monitor speaker", "podcast microphone", "lavalier microphone",
            "digital camera", "DSLR camera", "mirrorless camera", "GoPro action camera", "camera lens", "ring light",
            "Apple Watch", "Garmin smartwatch", "fitness tracker", "VR headset", "Meta Quest",
            "PlayStation controller", "Xbox controller", "Nintendo Switch", "Steam Deck console", "gaming console",
            "wifi router", "network switch", "NAS server", "USB flash drive", "external hard drive",
            "portable SSD", "USB-C hub", "SD card", "SD card reader",
            "smart speaker", "Amazon Echo", "Google Nest Hub", "smart thermostat", "smart bulb",
            "security camera", "video doorbell", "power bank", "wireless charging pad", "MagSafe charger",
            "laptop power brick", "power strip", "surge protector", "USB-C cable", "HDMI cable",
            "Raspberry Pi", "Arduino board", "soldering iron", "multimeter", "3D printer",
            "drone", "calculator", "laser pointer", "vr controllers",
        ]
        self.yolo.set_classes(custom_classes)
        print(f"YOLO-World model loaded with {len(custom_classes)} custom gadget classes.")

    # ── Gaze helpers ─────────────────────────────────────────────────────────

    def get_filtered_gaze(self):
        rawX = self.shared_data["GazeX"].value * VGA_W
        rawY = self.shared_data["GazeY"].value * VGA_H
        if rawX != 0 or rawY != 0:
            self.gazeX_history.append(rawX)
            self.gazeY_history.append(rawY)
        if not self.gazeX_history:
            return 0, 0
        return (int(sum(self.gazeX_history) / len(self.gazeX_history)),
                int(sum(self.gazeY_history) / len(self.gazeY_history)))

    @staticmethod
    def _dist(p1, p2):
        return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) ** 0.5

    # ── Face detection & identification ───────────────────────────────────────

    def _find_face_near_gaze(self, frame: "np.ndarray", gaze_x: int, gaze_y: int):
        """
        Run Haar face detection and return the face closest to the gaze point.
        Uses stricter settings to avoid partial-face false detections.
        Returns (x1, y1, x2, y2) bounding box, or None.
        """
        if self.face_detector is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # CLAHE equalization helps Haar work better in varied lighting
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray  = clahe.apply(gray)

        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=8,        # higher = fewer false partial-face detections
            minSize=(80, 80),      # ignore tiny detections (partial features)
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        if len(faces) == 0:
            return None

        best, best_dist = None, float("inf")
        for (fx, fy, fw, fh) in faces:
            cx, cy = fx + fw // 2, fy + fh // 2
            d = self._dist((gaze_x, gaze_y), (cx, cy))
            if d < best_dist:
                best_dist = d
                best = (fx, fy, fx + fw, fy + fh)
        return best

    def _identify_person(self, frame: "np.ndarray", box) -> dict | None:
        """
        Crop the face region from frame and run recognition.
        Uses a generous padding so the recogniser always sees the full face.
        Returns person dict or None.
        """
        x1, y1, x2, y2 = box
        face_w = x2 - x1
        face_h = y2 - y1
        # Dynamic padding: 25% of face size so forehead/chin are included
        pad_x = int(face_w * 0.25)
        pad_y = int(face_h * 0.25)
        fh, fw = frame.shape[:2]
        x1c = max(0, x1 - pad_x)
        y1c = max(0, y1 - pad_y)
        x2c = min(fw, x2 + pad_x)
        y2c = min(fh, y2 + pad_y)
        face_crop = frame[y1c:y2c, x1c:x2c]
        return self.person_db.identify(face_crop)

    # ── YOLO gadget detection ────────────────────────────────────────────────

    def _run_yolo(self, frame, gaze_x, gaze_y):
        results = self.yolo(frame, verbose=False)[0]
        if results.boxes is None or len(results.boxes) == 0:
            return None

        best, best_dist = None, float("inf")
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx, cy = (x1+x2)//2, (y1+y2)//2
            d = self._dist((gaze_x, gaze_y), (cx, cy))
            if d < best_dist:
                best_dist = d
                cls_id   = int(box.cls[0])
                conf     = float(box.conf[0])
                cls_name = self.yolo.names[cls_id]
                best = dict(
                    box           = (x1, y1, x2, y2),
                    label         = f"{cls_name} {conf:.0%}",
                    class_name    = cls_name,
                    color         = _class_color(cls_name),
                    expires_at    = time.time() + HIGHLIGHT_FADE,
                    link_status   = "searching",
                    product_price = "",
                    kind          = "gadget",          # ← tag so draw knows type
                )
        return best

    # ── Face-only detection (triggered by dwell alone) ──────────────────────

    def _run_face_only(self, frame, gaze_x: int, gaze_y: int):
        """
        Runs only face detection + identification.
        Called on every dwell, no pupil change required.
        Returns a highlight dict or None.
        """
        try:
            face_box = self._find_face_near_gaze(frame, gaze_x, gaze_y)
            if face_box is None:
                print("[FaceDetect] No face found near gaze point.")
                return None
            print(f"[FaceDetect] Face box: {face_box}")
            person = self._identify_person(frame, face_box)
            if person:
                color = PERSON_BOX_COLOR
                label = person["name"]
            else:
                color = UNKNOWN_BOX_COLOR
                label = "Unknown person"
                person = {"name": "Unknown", "age": "?", "role": "?"}
            return dict(
                box           = face_box,
                label         = label,
                class_name    = "person",
                color         = color,
                expires_at    = time.time() + HIGHLIGHT_FADE,
                kind          = "person",
                person        = person,
                link_status   = "ignored",
                product_price = "",
            )
        except Exception as e:
            print(f"[FaceDetect] ERROR: {e}")
            import traceback; traceback.print_exc()
            return None

    # ── Gaze-triggered detection: face AND gadget run in parallel ───────────

    def _run_detection(self, frame, gaze_x: int, gaze_y: int) -> dict | None:
        """
        Run face detection and YOLO gadget detection concurrently in threads.
        Both results are scored by distance to gaze; the closer one wins.
        This ensures looking at a person always triggers person ID even when
        a gadget is also present in the scene.
        """
        face_result   = [None]
        gadget_result = [None]

        def _detect_face():
            try:
                face_box = self._find_face_near_gaze(frame, gaze_x, gaze_y)
                if face_box is None:
                    print("[FaceDetect] No face found near gaze point.")
                    return
                print(f"[FaceDetect] Face box found: {face_box}")
                person = self._identify_person(frame, face_box)
                if person:
                    color = PERSON_BOX_COLOR
                    label = person["name"]
                else:
                    color = UNKNOWN_BOX_COLOR
                    label = "Unknown person"
                    person = {"name": "Unknown", "age": "?", "role": "?"}
                face_result[0] = dict(
                    box           = face_box,
                    label         = label,
                    class_name    = "person",
                    color         = _category_color(cls_name),
                    expires_at    = time.time() + HIGHLIGHT_FADE,
                    kind          = "person",
                    person        = person,
                    link_status   = "ignored",
                    product_price = "",
                )
            except Exception as e:
                print(f"[FaceDetect] ERROR in face thread: {e}")
                import traceback; traceback.print_exc()

        def _detect_gadget():
            try:
                gadget_result[0] = self._run_yolo(frame, gaze_x, gaze_y)
            except Exception as e:
                print(f"[FaceDetect] ERROR in gadget thread: {e}")

        t_face   = threading.Thread(target=_detect_face,   daemon=True)
        t_gadget = threading.Thread(target=_detect_gadget, daemon=True)
        t_face.start();   t_gadget.start()
        t_face.join();    t_gadget.join()

        f = face_result[0]
        g = gadget_result[0]

        if f is None and g is None:
            return None
        if f is None:
            return g
        if g is None:
            return f

        # Both found: pick whichever centre is closest to the gaze point
        def _centre_dist(h):
            x1, y1, x2, y2 = h["box"]
            return self._dist((gaze_x, gaze_y), ((x1+x2)//2, (y1+y2)//2))

        return f if _centre_dist(f) <= _centre_dist(g) else g

    # ── Background product fetch ──────────────────────────────────────────────

    def _fetch_link_async(self, highlight: dict):
        def _worker():
            result = _search_product(highlight["class_name"])
            with self._lock:
                highlight["product_price"] = result["price"]
                highlight["link_status"]   = "ready"
            print(f"[ProductSearch] '{highlight['class_name']}' → {result['price']}")
        threading.Thread(target=_worker, daemon=True).start()

    # ── Drawing ──────────────────────────────────────────────────────────────

    def _draw_highlight(self, frame, h: dict):
        """Unified draw for both person and gadget highlights."""
        if h.get("kind") == "person":
            self._draw_person_box(frame, h)
        else:
            self._draw_gadget_box(frame, h)

    def _draw_person_box(self, frame, h: dict):
        """
        Draws the AR-style person identification overlay:

            ┌──────────────────┐
            │ Alice Smith      │   ← name  (large, white)
            │ Age: 29          │   ← age   (medium, light)
            │ Role: Engineer   │   ← role  (medium, light)
            └──────────────────┘
        """
        x1, y1, x2, y2 = h["box"]
        color = h["color"]
        lw    = max(2, int((frame.shape[0]+frame.shape[1]) / 2 * 0.003))

        # ── Corner-bracket style box ──────────────────────────────────────────
        arm = min((x2-x1), (y2-y1)) // 4   # bracket arm length
        pts = [
            # top-left
            [(x1, y1+arm), (x1, y1), (x1+arm, y1)],
            # top-right
            [(x2-arm, y1), (x2, y1), (x2, y1+arm)],
            # bottom-left
            [(x1, y2-arm), (x1, y2), (x1+arm, y2)],
            # bottom-right
            [(x2-arm, y2), (x2, y2), (x2, y2-arm)],
        ]
        for bracket in pts:
            for i in range(len(bracket)-1):
                cv2.line(frame, bracket[i], bracket[i+1], color, lw+1, cv2.LINE_AA)

        # ── Info badge ────────────────────────────────────────────────────────
        person = h["person"]
        name   = person["name"]
        age    = person["age"]
        role   = person["role"]

        line1 = name
        line2 = f"Age: {age}"
        line3 = f"Role: {role}"

        font   = cv2.FONT_HERSHEY_SIMPLEX
        fs1, fs2 = 0.60, 0.48
        thick = 1

        (w1, h1), _ = cv2.getTextSize(line1, font, fs1, thick)
        (w2, h2), _ = cv2.getTextSize(line2, font, fs2, thick)
        (w3, h3), _ = cv2.getTextSize(line3, font, fs2, thick)

        pad     = 6
        badge_w = max(w1, w2, w3) + pad * 2
        badge_h = h1 + h2 + h3 + pad * 4

        # Place badge above the box; clamp to frame edges
        bx1 = x1
        bx2 = min(bx1 + badge_w, frame.shape[1])
        by2 = y1
        by1 = max(0, y1 - badge_h)

        # Semi-transparent filled rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), color, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Text colours
        brightness = 0.299*color[2] + 0.587*color[1] + 0.114*color[0]
        tc      = (0, 0, 0)       if brightness > 160 else (255, 255, 255)
        tc_dim  = (30, 30, 30)    if brightness > 160 else (210, 210, 210)

        y_cursor = by1 + h1 + pad
        cv2.putText(frame, line1, (bx1+pad, y_cursor),
                    font, fs1, tc, thick, cv2.LINE_AA)
        y_cursor += h2 + pad
        cv2.putText(frame, line2, (bx1+pad, y_cursor),
                    font, fs2, tc_dim, thick, cv2.LINE_AA)
        y_cursor += h3 + pad
        cv2.putText(frame, line3, (bx1+pad, y_cursor),
                    font, fs2, tc_dim, thick, cv2.LINE_AA)

    def _draw_gadget_box(self, frame, h: dict):
        """Original product / gadget box drawing (unchanged logic)."""
        x1, y1, x2, y2 = h["box"]
        color = h["color"]
        lw    = max(2, int((frame.shape[0]+frame.shape[1]) / 2 * 0.003))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, lw)

        font = cv2.FONT_HERSHEY_SIMPLEX
        fs1, fs2 = 0.58, 0.48
        thick = 1

        line1 = h["label"]
        if h["link_status"] == "searching":
            line2 = "fetching price…"
        elif h["link_status"] == "ignored":
            line2 = ""
        else:
            line2 = h.get("product_price", "")

        (w1, h1), bl1 = cv2.getTextSize(line1, font, fs1, thick)
        (w2, h2), bl2 = cv2.getTextSize(line2, font, fs2, thick) if line2 else ((0,0), 0)

        pad     = 5
        badge_w = max(w1, w2) + pad * 2
        badge_h = h1 + (h2 if line2 else 0) + bl1 + (bl2 if line2 else 0) + pad * (3 if line2 else 2)

        by2 = y1
        by1 = max(0, y1 - badge_h)
        bx2 = min(x1 + badge_w, frame.shape[1])

        cv2.rectangle(frame, (x1, by1), (bx2, by2), color, -1)

        brightness = 0.299*color[2] + 0.587*color[1] + 0.114*color[0]
        tc     = (0, 0, 0)     if brightness > 160 else (255, 255, 255)
        tc_dim = tuple(max(0, int(c * 0.6)) for c in tc)

        y_pos = by1 + h1 + pad
        cv2.putText(frame, line1, (x1+pad, y_pos), font, fs1, tc, thick, cv2.LINE_AA)
        if line2:
            y_pos += h2 + bl1 + pad
            cv2.putText(frame, line2, (x1+pad, y_pos), font, fs2, tc_dim, thick, cv2.LINE_AA)

    # ── Voice context builder ────────────────────────────────────────────────
 
    def _build_voice_context(self, highlights: list) -> dict:
        """Extract current gaze scene info for the voice assistant."""
        objects = []
        person  = None
        for h in highlights:
            if h.get("kind") == "person":
                info = h.get("person_info", {})
                name = info.get("name", h.get("label", "Unknown"))
                age  = info.get("age", "?")
                role = info.get("role", "?")
                person = f"{name} ({role}, {age}y)"
            else:
                label = h.get("label", "object")
                price = h.get("product_price", "")
                entry = label
                if price:
                    entry += f" — {price}"
                objects.append(entry)
 
        return {
            "objects":  objects,
            "person":   person,
            "cog_load": f"{self.cog_tracker.level} ({self.cog_tracker.score}%)",
        }

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        print("enter run")
        cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Video", VGA_W*2, VGA_H*2)

        while not self.shared_data["stop"].value:
            GazeX, GazeY  = self.get_filtered_gaze()
            now            = time.time()
            current_pupil  = (self.shared_data["PupilSizeLeft"].value +
                               self.shared_data["PupilSizeRight"].value) / 2.0

            # ── Eye event ────────────────────────────────────────────────────
            ev = self.shared_data["eyeEvent"].value
            if ev and ev != self.current_event:
                self.current_event = ev

            gaze_color = (0, 0, 255)
            if self.current_event == "FB":
                gaze_color = (0, 255, 0)
            elif self.current_event == "BB":
                gaze_color = (255, 0, 0)


            # ── Cognitive load update (read-only, won't touch eyeEvent) ───────
            self.cog_tracker.update(current_pupil, self.current_event, now)

            # ── Dwell + pupil trigger ─────────────────────────────────────────
            if GazeX == 0 and GazeY == 0:
                self.dwell_anchor     = None
                self.dwell_start_time = None
                self.baseline_pupil   = None
            else:
                if self.dwell_anchor is None:
                    self.dwell_anchor     = (GazeX, GazeY)
                    self.dwell_start_time = now
                    self.baseline_pupil   = current_pupil
                elif self._dist((GazeX, GazeY), self.dwell_anchor) > DWELL_RADIUS:
                    self.dwell_anchor     = (GazeX, GazeY)
                    self.dwell_start_time = now
                    self.baseline_pupil   = current_pupil
                elif now - self.dwell_start_time >= DWELL_SECONDS:
                    pupil_change_ratio = 0.0
                    if self.baseline_pupil and self.baseline_pupil > 0:
                        pupil_change_ratio = abs(
                            current_pupil - self.baseline_pupil
                        ) / self.baseline_pupil

                    # ── Two-tier trigger ──────────────────────────────────────
                    # PERSON:  dwell alone is enough (no pupil change needed)
                    # GADGET:  requires dwell + pupil change (original behaviour)
                    #
                    # We always run face detection on dwell.
                    # We only run YOLO + price search when pupil also changed.
                    snap = self.image_buffer_scene.copy()

                    # Always attempt face detection on dwell
                    face_highlight = self._run_face_only(snap, GazeX, GazeY)

                    # Only attempt gadget detection if pupil changed enough
                    gadget_highlight = None
                    if pupil_change_ratio >= PUPIL_CHANGE_THRESHOLD:
                        gadget_highlight = self._run_yolo(snap, GazeX, GazeY)

                    # Pick best result: face wins if closer to gaze, else gadget
                    highlight = None
                    if face_highlight and gadget_highlight:
                        def _cd(h):
                            x1,y1,x2,y2 = h["box"]
                            return self._dist((GazeX,GazeY),((x1+x2)//2,(y1+y2)//2))
                        highlight = face_highlight if _cd(face_highlight) <= _cd(gadget_highlight) else gadget_highlight
                    elif face_highlight:
                        highlight = face_highlight
                    elif gadget_highlight:
                        highlight = gadget_highlight

                    if highlight:
                            hcx = (highlight["box"][0]+highlight["box"][2])//2
                            hcy = (highlight["box"][1]+highlight["box"][3])//2

                            # Start price fetch only for searchable gadgets
                            if (highlight.get("kind") == "gadget" and
                                    highlight["class_name"] not in IGNORE_SEARCH_CLASSES):
                                self._fetch_link_async(highlight)
                            else:
                                highlight["link_status"] = "ignored"

                            with self._lock:
                                # Remove any nearby existing highlight
                                self.active_highlights = [
                                    h for h in self.active_highlights
                                    if self._dist(
                                        ((h["box"][0]+h["box"][2])//2,
                                         (h["box"][1]+h["box"][3])//2),
                                        (hcx, hcy)
                                    ) > DWELL_RADIUS
                                ]
                                self.active_highlights.append(highlight)

                    self.dwell_anchor     = None
                    self.dwell_start_time = None
                    self.baseline_pupil   = None

            # Expire old highlights
            with self._lock:
                self.active_highlights = [
                    h for h in self.active_highlights if h["expires_at"] > now
                ]
                snapshot = list(self.active_highlights)

            # ── Draw ──────────────────────────────────────────────────────────
            frame = self.image_buffer_scene.copy()

            for h in snapshot:
                self._draw_highlight(frame, h)

            r = max(int(current_pupil), 4)
            cv2.circle(frame, (GazeX, GazeY), r, gaze_color, 2)

             # ── Cognitive load indicator ──────────────────────────────────────
            _draw_cognitive_bar(frame, self.cog_tracker)

            # ── Voice assistant overlay ───────────────────────────────────────
            draw_voice_status(frame, self.voice)

            # ── "Press V to talk" hint (only when idle) ───────────────────────
            if self.voice.status == "idle":
                cv2.putText(frame, "Press V to talk", (10, frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1, cv2.LINE_AA)


            cv2.imshow("Video", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.shared_data["stop"].value = True
                print("Q key pressed. Stopping all processes…")
                break

        cv2.destroyAllWindows()
        print("Process stopped.")