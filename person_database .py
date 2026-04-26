# person_database.py
"""
Person database for gaze-based identification.

HOW TO ADD PEOPLE
─────────────────
Fill in PEOPLE_REGISTRY in process.py:

    PEOPLE_REGISTRY = [
        {"name": "Faisal", "age": 22, "role": "Student",
         "photo_path": r"C:\\Users\\Pavel\\Pictures\\faisalphoto.jpg"},
    ]

One clear frontal photo per person is enough – the system auto-augments it.
"""

import cv2
import numpy as np

try:
    import face_recognition
    _BACKEND = "face_recognition"
except ImportError:
    _BACKEND = "opencv_lbph"


# ─────────────────────────────────────────────────────────────────────────────
# Shared image helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_bgr(path: str):
    img = cv2.imread(path)
    if img is None:
        print(f"[PersonDB] WARNING: could not load '{path}'")
    return img


def _equalize(gray: np.ndarray) -> np.ndarray:
    """CLAHE equalization – makes LBPH robust to lighting differences."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _preprocess(bgr: np.ndarray, size=(100, 100)) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, size)
    return _equalize(gray)


def _augment(bgr: np.ndarray) -> list:
    """
    Generate ~12 training variants from one photo so LBPH handles
    lighting changes, slight angles, and blur at runtime.
    """
    h, w = bgr.shape[:2]
    variants = [bgr.copy(), cv2.flip(bgr, 1)]

    # Brightness / gamma variants
    for gamma in (0.6, 0.8, 1.3, 1.6):
        lut = np.array([
            min(255, int((i / 255.0) ** (1.0 / gamma) * 255))
            for i in range(256)
        ], dtype=np.uint8)
        variants.append(cv2.LUT(bgr, lut))

    # Small rotations
    cx, cy = w // 2, h // 2
    for angle in (-10, -5, 5, 10):
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        variants.append(cv2.warpAffine(bgr, M, (w, h)))

    # Slight crop (simulate partial face crop from detector)
    margin = max(1, int(min(h, w) * 0.05))
    variants.append(bgr[margin:h - margin, margin:w - margin])

    # Gaussian blur (simulate defocus / low-res frame)
    variants.append(cv2.GaussianBlur(bgr, (3, 3), 0))

    return [_preprocess(v) for v in variants]


# ─────────────────────────────────────────────────────────────────────────────
# face_recognition backend  (dlib – preferred when available)
# ─────────────────────────────────────────────────────────────────────────────

class _FaceRecBackend:
    def __init__(self, tolerance: float = 0.55):
        self._tolerance = tolerance
        self._encodings = []
        self._meta = []

    def add_person(self, name, age, role, bgr_img) -> bool:
        import face_recognition as fr
        rgb  = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        encs = fr.face_encodings(rgb)
        if not encs:
            print(f"[PersonDB] No face found in photo for '{name}' – skipped.")
            return False
        self._encodings.append(encs[0])
        self._meta.append({"name": name, "age": age, "role": role})
        print(f"[PersonDB] Registered '{name}' ({role}, {age}y) via face_recognition.")
        return True

    def identify(self, bgr_face):
        if not self._encodings:
            return None
        import face_recognition as fr
        rgb  = cv2.cvtColor(bgr_face, cv2.COLOR_BGR2RGB)
        encs = fr.face_encodings(rgb)
        if not encs:
            return None
        distances = fr.face_distance(self._encodings, encs[0])
        best      = int(np.argmin(distances))
        if distances[best] <= self._tolerance:
            return dict(self._meta[best])
        return None


# ─────────────────────────────────────────────────────────────────────────────
# OpenCV LBPH backend  (fallback – augmented so one photo is enough)
# ─────────────────────────────────────────────────────────────────────────────

class _LBPHBackend:
    """
    LBPH confidence is a distance (lower = better).
    With augmented training data, threshold=120 works well for a single
    registered person.  If still showing Unknown, raise it toward 150.
    The confidence value is printed each trigger so you can tune it.
    """

    def __init__(self, confidence_threshold: float = 200.0):  # raised – tune down once confidence is known
        self._threshold  = confidence_threshold
        # Finer grid and more neighbours → more discriminative histogram
        self._recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=2, neighbors=16, grid_x=8, grid_y=8
        )
        self._meta         = []
        self._train_images = []
        self._train_labels = []
        self._trained      = False

    def add_person(self, name: str, age: int, role: str, bgr_img: np.ndarray) -> bool:
        label     = len(self._meta)
        augmented = _augment(bgr_img)

        self._meta.append({"name": name, "age": age, "role": role})
        self._train_images.extend(augmented)
        self._train_labels.extend([label] * len(augmented))

        self._recognizer.train(
            self._train_images, np.array(self._train_labels)
        )
        self._trained = True
        print(
            f"[PersonDB] Registered '{name}' ({role}, {age}y) via OpenCV LBPH "
            f"({len(augmented)} augmented samples)."
        )
        return True

    def identify(self, bgr_face: np.ndarray):
        if not self._trained or not self._meta:
            print("[PersonDB] identify called but not trained yet – skipping.")
            return None
        if bgr_face is None or bgr_face.size == 0:
            print("[PersonDB] identify called with empty crop – skipping.")
            return None

        # DEBUG: save the crop so you can inspect what LBPH actually sees
        try:
            import tempfile, os, time as _t
            debug_path = os.path.join(tempfile.gettempdir(),
                                      f"lbph_crop_{int(_t.time()*1000)}.jpg")
            cv2.imwrite(debug_path, bgr_face)
            print(f"[PersonDB] DEBUG crop saved → {debug_path}")
        except Exception as _e:
            print(f"[PersonDB] DEBUG save failed: {_e}")

        try:
            proc              = _preprocess(bgr_face)
            label, confidence = self._recognizer.predict(proc)
            print(f"[PersonDB] LBPH confidence={confidence:.1f}  (threshold≤{self._threshold})  label={label}")
            if confidence <= self._threshold:
                return dict(self._meta[label])
            else:
                print(f"[PersonDB] No match – raise threshold above {confidence:.0f} to force-match.")
            return None
        except Exception as e:
            print(f"[PersonDB] LBPH predict error: {e}")
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────

class PersonDatabase:
    """High-level registry – automatically picks the best available backend."""

    def __init__(self):
        if _BACKEND == "face_recognition":
            self._backend = _FaceRecBackend()
            print("[PersonDB] Using face_recognition backend (dlib).")
        else:
            self._backend = _LBPHBackend()
            print("[PersonDB] Using OpenCV LBPH backend (with augmentation).")
        self._count = 0

    def register(self, name: str, age: int, role: str, photo_path: str) -> bool:
        """Register one person from a photo file."""
        img = _load_bgr(photo_path)
        if img is None:
            return False
        ok = self._backend.add_person(name, age, role, img)
        if ok:
            self._count += 1
        return ok

    def register_many(self, people: list) -> None:
        """Register a list of dicts with keys: name, age, role, photo_path."""
        for p in people:
            self.register(p["name"], p["age"], p["role"], p["photo_path"])

    def identify(self, face_bgr: np.ndarray):
        """Match a cropped face. Returns {name, age, role} or None."""
        if face_bgr is None or face_bgr.size == 0:
            return None
        return self._backend.identify(face_bgr)

    @property
    def size(self) -> int:
        return self._count