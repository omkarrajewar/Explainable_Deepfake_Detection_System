# testing.py

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess

import mediapipe as mp

# =========================
# Config
# =========================
IMG_SIZE = (299, 299)
MODEL_PATH = "new_deepfake_detector.h5"   # hardcoded model path
TEST_IMAGE = r"E:\TECNOBIJ 2025-2026\deepppp\datsset\Celeb-DF-v2\Celeb-synthesis\id50_id54_0002\frame_0007_03s000ms.jpg"
# Focus only on strong CAM (red/orange) regions
# Text explanation should consider only strong CAM (red/orange)
FOCUS_TOP_PERCENT = 85    # keep top 15% activations
ABSOLUTE_MIN = 0.60       # floor for strong activations (0..1)
MORPH_OPEN_K = 3          # clean small speckles; 0 = off
           # denoise mask (3x3), set 0 to disable

# IMPORTANT: pick ONE to match your pipeline
# - 'xception'  -> matches your Flask app/predict.py
# - 'rescale'   -> matches your train.py ImageDataGenerator(rescale=1/255)
PREPROCESS_MODE = "xception"   # <<< set to 'xception' to match Flask, 'rescale' to match training

# Toggle alignment/CLAHE (turn OFF to match Flask exactly)
FACE_ALIGN = False             # Flask does no face-crop
USE_CLAHE  = False             # Flask does no CLAHE

# Label semantics for your sigmoid head
POSITIVE_LABEL_IS_REAL = True  # sigmoid=1 means REAL

# Grad-CAM params
GRADCAM_THRESHOLD = 0.30
HEATMAP_MIN_VALID = 0.05
LAST_CONV_CANDIDATES = [
    "block14_sepconv2_act", "block14_sepconv2",
    "block14_sepconv1_act", "block13_sepconv2_act", "block13_sepconv2"
]

# =========================
# MediaPipe helpers (for explanations)
# =========================
mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE_IDX  = [33, 133, 160, 158, 159, 144, 145, 153, 154, 155, 246]
RIGHT_EYE_IDX = [362, 263, 387, 385, 386, 373, 374, 380, 381, 382, 466]
OUTER_LIPS_IDX = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
                  308, 324, 318, 402, 317, 14, 87]

# --- Context gating (outside-face -> REAL) ---
OUTSIDE_DOMINANCE_FACTOR = 2.0   # outside must be > 2x inside to trigger the gate
FORCE_REAL_WHEN_OUTSIDE_DOMINATES = True  # set False if you only want to annotate, not override
MIN_FORCED_REAL_PROB = 0.75      # when forcing REAL, ensure p_real is at least this


def _landmarks_to_xy(img_w, img_h, landmarks, idx_list):
    pts = []
    for i in idx_list:
        lm = landmarks[i]
        pts.append([int(lm.x*img_w), int(lm.y*img_h)])
    return np.array(pts, dtype=np.int32)

def _region_masks_from_facemesh(img_bgr, landmarks):
    h, w = img_bgr.shape[:2]
    masks = {}
    # Eyes
    left_eye_pts  = _landmarks_to_xy(w, h, landmarks, LEFT_EYE_IDX)
    right_eye_pts = _landmarks_to_xy(w, h, landmarks, RIGHT_EYE_IDX)
    left_eye_mask  = np.zeros((h, w), dtype=np.uint8); cv2.fillPoly(left_eye_mask,  [left_eye_pts], 1)
    right_eye_mask = np.zeros((h, w), dtype=np.uint8); cv2.fillPoly(right_eye_mask, [right_eye_pts], 1)
    masks["eyes"] = (left_eye_mask | right_eye_mask)
    # Mouth
    mouth_pts = _landmarks_to_xy(w, h, landmarks, OUTER_LIPS_IDX)
    mouth_mask = np.zeros((h, w), dtype=np.uint8); cv2.fillPoly(mouth_mask, [mouth_pts], 1)
    masks["mouth"] = mouth_mask
    # Nose/cheeks (ellipse around the nose tip)
    nose_cx = int(landmarks[1].x * w); nose_cy = int(landmarks[1].y * h)
    rx, ry = int(0.12 * w), int(0.18 * h)
    nose_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(nose_mask, (nose_cx, nose_cy), (rx, ry), 0, 0, 360, 1, -1)
    masks["nose/cheeks"] = nose_mask
    # Forehead & jawline via convex hull
    all_pts = np.array([[int(lm.x*w), int(lm.y*h)] for lm in landmarks], dtype=np.int32)
    hull = cv2.convexHull(all_pts)
    hull_mask = np.zeros((h, w), dtype=np.uint8); cv2.fillConvexPoly(hull_mask, hull, 1)
    band_jaw = np.zeros((h, w), dtype=np.uint8);  band_jaw[int(0.65*h):, :] = 1
    masks["jawline"] = (hull_mask & band_jaw)
    band_fore = np.zeros((h, w), dtype=np.uint8); band_fore[:int(0.35*h), :] = 1
    masks["forehead"] = (hull_mask & band_fore)
    return masks

def map_gradcam_regions_mediapipe(img_rgb_uint8, gradcam_mask_uint8, overlap_thresh=100):
    try:
        with mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True, max_num_faces=1) as fm:
            res = fm.process(img_rgb_uint8)
        if not res.multi_face_landmarks:
            return ["no face landmarks detected"]
        landmarks = res.multi_face_landmarks[0].landmark
        region_masks = _region_masks_from_facemesh(cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2BGR), landmarks)
        regions_hit = []
        gmask = gradcam_mask_uint8.astype(bool)
        for name, mask in region_masks.items():
            overlap = np.sum(np.logical_and(mask.astype(bool), gmask))
            if overlap > overlap_thresh:
                regions_hit.append(name)
        if not regions_hit:
            return ["no specific region"]
        preferred = ["eyes", "mouth", "nose/cheeks", "forehead", "jawline"]
        ordered = [r for r in preferred if r in regions_hit]
        return ordered if ordered else regions_hit
    except Exception:
        return ["mediapipe_error"]


def face_hull_mask(img_rgb):
    """Return uint8 face convex-hull mask (0/255). None if landmarks fail."""
    h, w = img_rgb.shape[:2]
    with mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True, max_num_faces=1) as fm:
        res = fm.process(img_rgb)
    if not res.multi_face_landmarks:
        return None
    pts = np.array([[int(l.x*w), int(l.y*h)] for l in res.multi_face_landmarks[0].landmark], dtype=np.int32)
    hull = cv2.convexHull(pts)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    return mask

def largest_face_component(mask_uint8, face_mask_uint8=None, min_area_ratio=0.002):
    """Keep largest connected component (centroid inside face); drop tiny noise."""
    m = (mask_uint8 > 0).astype(np.uint8)
    if m.max() == 0:
        return mask_uint8
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(m, connectivity=8)
    H, W = m.shape[:2]
    best_idx, best_area = -1, -1
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area_ratio * H * W:
            continue
        cx, cy = centroids[i]
        if face_mask_uint8 is not None and face_mask_uint8[int(cy), int(cx)] == 0:
            continue
        if area > best_area:
            best_area = area
            best_idx = i
    out = np.zeros_like(m)
    if best_idx > 0:
        out[labels == best_idx] = 1
    else:
        out = m
    return (out * 255).astype(np.uint8)


def humanize_regions(regions):
    if not regions or "no specific region" in regions:
        return "Manipulation detected, but not localized to a clear facial feature."
    if "no face landmarks detected" in regions:
        return "Manipulation detected, but facial landmarks were not detected reliably."
    if "mediapipe_error" in regions:
        return "Manipulation detected, but face analysis encountered an error."
    nice = {
        "eyes": "around the eyes (blink edges and eyelid contours look inconsistent)",
        "mouth": "near the lips and smile lines (texture boundaries look irregular)",
        "nose/cheeks": "across the mid-face (nose and cheek textures don't blend naturally)",
        "forehead": "on the forehead (skin tone and shading look uneven)",
        "jawline": "along the jawline (edge transitions appear unnaturally sharp)"
    }
    parts = [nice.get(r, r) for r in regions]
    return "Manipulation signs are visible " + "; ".join(parts) + "."

# =========================
# Preprocessing / alignment
# =========================
def crop_face_mediapipe(img_rgb):
    with mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True, max_num_faces=1) as fm:
        res = fm.process(img_rgb)
    if not res.multi_face_landmarks:
        return None
    h, w = img_rgb.shape[:2]
    pts = np.array([[int(l.x*w), int(l.y*h)] for l in res.multi_face_landmarks[0].landmark])
    x1, y1 = pts.min(axis=0); x2, y2 = pts.max(axis=0)
    mx = int(0.15*(x2-x1)); my = int(0.20*(y2-y1))
    x1 = max(0, x1-mx); y1 = max(0, y1-my); x2 = min(w, x2+mx); y2 = min(h, y2+my)
    if x2 <= x1 or y2 <= y1: return None
    return img_rgb[y1:y2, x1:x2]

def apply_clahe(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    L2 = clahe.apply(L)
    return cv2.cvtColor(cv2.merge([L2, A, B]), cv2.COLOR_LAB2RGB)

def preprocess_for_model(img_rgb):
    """Return (batch_tensor_for_model, display_img_rgb) following PREPROCESS_MODE & toggles."""
    # (optional) align
    if FACE_ALIGN:
        face = crop_face_mediapipe(img_rgb) or img_rgb
    else:
        face = img_rgb
    # (optional) CLAHE
    if USE_CLAHE:
        face = apply_clahe(face)
    # resize
    face = cv2.resize(face, IMG_SIZE, interpolation=cv2.INTER_AREA)
    disp = face.copy()

    # to float32
    arr = face.astype(np.float32)
    if PREPROCESS_MODE == "rescale":
        arr = arr / 255.0
    elif PREPROCESS_MODE == "xception":
        # expects RGB in [0..255]
        arr = xception_preprocess(arr.copy())
    else:
        raise ValueError("PREPROCESS_MODE must be 'rescale' or 'xception'")

    bat = np.expand_dims(arr, 0)
    return bat, disp

def load_rgb(path):
    img = Image.open(path).convert("RGB")
    return np.array(img)

# =========================
# Grad-CAM for binary sigmoid (use logits)
# =========================
def pick_last_conv_layer(model):
    for lname in LAST_CONV_CANDIDATES:
        try:
            _ = model.get_layer(lname); return lname
        except Exception:
            continue
    for layer in reversed(model.layers):
        if "conv" in layer.name:
            return layer.name
    raise ValueError("No suitable conv layer found")

def gradcam_binary(img_batch, model, last_conv, force_class=None):
    """
    Compute Grad-CAM for a binary sigmoid model.
    Handles scalar or batched outputs to avoid index errors.
    """
    conv_layer = model.get_layer(last_conv)
    grad_model = tf.keras.Model(model.inputs, [conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_out, prob = grad_model(img_batch, training=False)  # prob in [0,1] for positive class
        
        # Ensure prob is always indexable (batch_size, 1)
        prob = tf.convert_to_tensor(prob)
        if len(prob.shape) == 0:        # scalar
            prob = tf.reshape(prob, (1, 1))
        elif len(prob.shape) == 1:      # vector
            prob = tf.expand_dims(prob, 1)

        p = prob[:, 0]

        # Compute logit (more stable than probability for gradients)
        z = tf.math.log(p / (1.0 - p + 1e-8) + 1e-8)

        # Determine target for Grad-CAM
        if force_class is None:
            target = (z if p >= 0.5 else -z) if POSITIVE_LABEL_IS_REAL else (-z if p >= 0.5 else z)
        elif force_class.upper() == "REAL":
            target = z if POSITIVE_LABEL_IS_REAL else -z
        else:  # FAKE
            target = -z if POSITIVE_LABEL_IS_REAL else z

    # Compute gradients
    grads = tape.gradient(target, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.tensordot(conv_out[0], weights, axes=(2, 0))
    cam = tf.nn.relu(cam)
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    
    return cam.numpy()


def strong_cam_mask(hm, top_percent=85, absolute_min=0.60, morph_k=3):
    """Return uint8 mask (0/255) keeping only strongest activations."""
    thr_p = np.percentile(hm, top_percent)
    thr = max(thr_p, absolute_min)
    m = (hm >= thr).astype(np.uint8) * 255
    if morph_k and morph_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    return m



# =========================
# Explain + visualize
# =========================
def explain_image(img_path, model, force_class=None):
    # ---- local defaults in case the new config flags aren't defined globally ----
    ODF = globals().get('OUTSIDE_DOMINANCE_FACTOR', 2.0)            # outside dominance factor
    FORCE_REAL = globals().get('FORCE_REAL_WHEN_OUTSIDE_DOMINATES', True)
    MIN_FORCED_REAL_PROB = float(globals().get('MIN_FORCED_REAL_PROB', 0.75))

    img_rgb = load_rgb(img_path)
    x, display_face = preprocess_for_model(img_rgb)

    # prediction
    prob = float(model.predict(x, verbose=0)[0][0])  # sigmoid for positive class
    p_real = prob if POSITIVE_LABEL_IS_REAL else (1.0 - prob)
    pred_label = "REAL" if p_real >= 0.5 else "FAKE"
    conf = p_real if pred_label == "REAL" else (1.0 - p_real)

    # Grad-CAM
    last_conv = pick_last_conv_layer(model)
    heatmap = gradcam_binary(x, model, last_conv, force_class=force_class)

    if heatmap is None or np.max(heatmap) < HEATMAP_MIN_VALID:
        msg = ("No manipulation found — features look natural."
               if pred_label == "REAL" else
               "Manipulation detected but regions unclear.")
        plt.imshow(display_face); plt.title(f"{pred_label} ({conf:.2%})\n{msg}")
        plt.axis("off"); plt.show()
        return pred_label, p_real, msg

    # Resize CAM to image size
    H, W = display_face.shape[:2]
    hm = cv2.resize(heatmap, (W, H))

    # --- keep the Grad-CAM view as before (no change) ---
    jet = cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(display_face[..., ::-1], 0.6, jet, 0.4, 0)[..., ::-1]

    # ===== make TEXT follow the PICTURE (use global strong CAM, not face-renorm) =====
    thr_global = max(np.percentile(hm, FOCUS_TOP_PERCENT), ABSOLUTE_MIN)
    strong_global = (hm >= thr_global).astype(np.uint8) * 255
    if MORPH_OPEN_K and MORPH_OPEN_K > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_OPEN_K, MORPH_OPEN_K))
        strong_global = cv2.morphologyEx(strong_global, cv2.MORPH_OPEN, k)

    # Split strong mask into inside-face vs outside-face
    fmask = face_hull_mask(display_face)  # 0/255 or None
    override_reason = None  # <-- declare early to avoid editor warning

    if fmask is not None:
        strong_face    = cv2.bitwise_and(strong_global, fmask)
        strong_outside = cv2.bitwise_and(strong_global, cv2.bitwise_not(fmask))
        # keep only the most plausible in-face blob
        strong_face = largest_face_component(strong_face, face_mask_uint8=fmask, min_area_ratio=0.002)
        cnt_face    = int(np.count_nonzero(strong_face))
        cnt_outside = int(np.count_nonzero(strong_outside))
        face_area   = np.count_nonzero(fmask)
        strong_ratio = (cnt_face / (face_area + 1e-8)) if face_area > 0 else 0.0

        # === Context gate: if Grad-CAM is dominated by outside (neck/background), force REAL ===
        if FORCE_REAL and (cnt_outside > ODF * max(cnt_face, 1)):
            if pred_label != "REAL":
                override_reason = "Outside-face Grad-CAM dominates (neck/background). Forcing REAL."
            pred_label = "REAL"
            p_real = max(p_real, MIN_FORCED_REAL_PROB)
            conf = p_real
    else:
        strong_face    = strong_global
        cnt_face       = int(np.count_nonzero(strong_face))
        cnt_outside    = 0
        strong_ratio   = float(cnt_face) / (H * W)
        # No face mask -> cannot apply outside/neck dominance logic

    # ---------------- Explanation text ----------------
    if pred_label == "REAL" and force_class is None and not override_reason:
        explanation = "No manipulation found — facial features appear natural and consistent."
    else:
        if fmask is not None and cnt_outside > 2 * max(cnt_face, 1):
            base_expl = ("Model’s strongest evidence lies outside the face region "
                         "(e.g., neck/background). This can reflect context/dataset bias; "
                         "interpret with caution.")
        else:
            regions = map_gradcam_regions_mediapipe(display_face, strong_face)
            base_expl = humanize_regions(regions) + f" High-confidence face area ≈ {strong_ratio:.1%}."

        # Append override reason if applied
        if override_reason:
            explanation = base_expl + f" Override applied: {override_reason}"
        else:
            explanation = base_expl

    # === RENDER: images on top row, full-width text at bottom ===
    import textwrap
    from matplotlib.gridspec import GridSpec

    wrapped_expl = "\n".join(textwrap.wrap(explanation, width=110))

    fig = plt.figure(figsize=(14, 9))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[3.0, 1.1])

    ax_img1 = fig.add_subplot(gs[0, 0])
    ax_img2 = fig.add_subplot(gs[0, 1])
    ax_text = fig.add_subplot(gs[1, :])

    ax_img1.imshow(display_face)
    ax_img1.set_title("Aligned Face" if FACE_ALIGN else "Input Face")
    ax_img1.axis("off")

    ax_img2.imshow(overlay)
    ax_img2.set_title("Grad-CAM (manipulation areas)")
    ax_img2.axis("off")

    ax_text.axis("off")
    ax_text.text(
        0.5, 0.5,
        f"Pred: {pred_label} ({conf:.2%})  •  Preproc={PREPROCESS_MODE}  •  Align={FACE_ALIGN}\n"
        f"{wrapped_expl}\n(CAM layer: {last_conv})",
        ha="center", va="center", fontsize=11, wrap=True
    )

    fig.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.06, hspace=0.25, wspace=0.08)
    plt.show()

    return pred_label, p_real, explanation

# =========================
# Main
# =========================
if __name__=="__main__":
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    if not os.path.exists(TEST_IMAGE):
        raise FileNotFoundError(f"Test image not found at {TEST_IMAGE}")

    model = load_model(MODEL_PATH, compile=False)
    label, prob, explanation = explain_image(TEST_IMAGE, model)
    print("Final Result:", label, prob, explanation)
