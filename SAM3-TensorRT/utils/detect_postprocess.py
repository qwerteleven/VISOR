import cv2
import numpy as np
import random

def process_sam3_results(
    outputs,
    img_h,
    img_w,
    score_thr=0.4,
    mask_thr=0.5,
    max_inst=30,
    boxes_normalized=True,
):
    pred_masks, pred_boxes, pred_logits = outputs

    if pred_masks.ndim == 4:
        pred_masks = pred_masks[0]      # [N, Hm, Wm]
    if pred_boxes.ndim == 3:
        pred_boxes = pred_boxes[0]      # [N, 4]
    if pred_logits.ndim == 2:
        pred_logits = pred_logits[0]    # [N]
    
    """
    pred_masks  = pred_masks.astype(np.float32)
    pred_boxes  = pred_boxes.astype(np.float32)
    pred_logits = pred_logits.astype(np.float32)
    """
    
    N, Hm, Wm = pred_masks.shape

    # logits -> scores
    scores = 1.0 / (1.0 + np.exp(-pred_logits))  # [N]

    indices = list(range(N))
    indices = sorted(indices, key=lambda i: float(scores[i]), reverse=True)
    indices = indices[:max_inst]
    indices = [i for i in indices if float(scores[i]) >= score_thr]

    results = []
    for i in indices:
        mask  = pred_masks[i]          # [Hm, Wm]
        score = float(scores[i])
        box   = pred_boxes[i]          # [4]

        mask_resized = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        m = (mask_resized > mask_thr).astype(np.uint8)  # [H, W]

        if m.sum() == 0:
            continue

        x1, y1, x2, y2 = box

        if boxes_normalized:
            x1 = int(x1 * img_w)
            x2 = int(x2 * img_w)
            y1 = int(y1 * img_h)
            y2 = int(y2 * img_h)
        else:
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

        x1 = max(0, min(x1, img_w - 1))
        x2 = max(0, min(x2, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        y2 = max(0, min(y2, img_h - 1))

        results.append({
            "mask": m,          # [H, W] uint8
            "box": [x1, y1, x2, y2],
            "score": score,
        })
        
    return results




colors = [random.randint(0, 255) for _ in range(300)]

def draw_sam3_results(img, results):
    if not results:
        return img.copy()

    vis_img = img.astype(np.float32)
    n = len(results)

    # ── 1. Stack all masks & colors ──────────────────────────────────────
    masks = np.stack([item["mask"].astype(np.uint8) for item in results], axis=0)  # [N, H, W]
    cols  = np.array([colors[i] for i in range(n)], dtype=np.float32)             # [N, 3]

    # ── 2. Build per-pixel color index ───────────────────────────────────
    # weighted[y,x] = last mask index (1-based) covering that pixel
    weights  = np.arange(1, n + 1, dtype=np.uint8)                 # [N]
    mask_idx = (masks * weights[:, None, None]).max(axis=0)        # [H, W], 1-based
    any_mask = mask_idx > 0                                         # [H, W] bool

    mask_idx_safe = np.clip(mask_idx - 1, 0, n - 1)               # [H, W], 0-based
    color_layer   = cols[mask_idx_safe][..., np.newaxis]                             # [H, W, 3]  ✓

    # ── 3. Blend only where a mask exists ────────────────────────────────
    any_mask3 = any_mask[:, :, None]                                # [H, W, 1]
    vis_img   = np.where(any_mask3, vis_img * 0.5 + color_layer * 0.5, vis_img)

    # ── 4. Boxes & labels ────────────────────────────────────────────────
    vis_img = np.clip(vis_img, 0, 255).astype(np.uint8)

    for i, item in enumerate(results):
        x1, y1, x2, y2 = item["box"]
        color = colors[i]
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)
        cv2.putText(
            vis_img, f"{item['score']:.2f}",
            (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )

    return vis_img