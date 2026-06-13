import numpy as np
import torch


def _prepare_uint8_image_from_tensor(img_tensor):
    channels, height, width = img_tensor.shape
    if channels >= 3:
        img_3ch = img_tensor[:3, :, :]
    else:
        img_3ch = img_tensor[0:1, :, :].repeat(3, 1, 1)

    img = img_3ch.permute(1, 2, 0).cpu().numpy()
    img_min = float(img.min())
    img_max = float(img.max())
    if img_max > img_min:
        img = ((img - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)
    else:
        img = np.zeros_like(img, dtype=np.uint8)
    return img, height, width


def build_point_grid(height, width, grid_size):
    grid_size = max(1, int(grid_size))
    ys = np.linspace(height / (grid_size + 1), grid_size * height / (grid_size + 1), grid_size)
    xs = np.linspace(width / (grid_size + 1), grid_size * width / (grid_size + 1), grid_size)
    points = []
    for y in ys:
        for x in xs:

            points.append([float(x), float(y)])
    return np.asarray(points, dtype=np.float32)


def mask_iou(mask_a, mask_b):
    mask_a = mask_a.astype(bool)
    mask_b = mask_b.astype(bool)
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def select_multi_region_masks(
    candidate_masks,
    candidate_scores,
    max_regions=6,
    min_mask_area_ratio=0.01,
    max_mask_area_ratio=0.90,
    mask_iou_dedup_thresh=0.85,
):
    if len(candidate_masks) == 0:
        return []

    max_regions = max(1, int(max_regions))
    min_mask_area_ratio = float(max(min_mask_area_ratio, 0.0))
    max_mask_area_ratio = float(min(max_mask_area_ratio, 1.0))
    mask_iou_dedup_thresh = float(max(min(mask_iou_dedup_thresh, 1.0), 0.0))

    mask_area = candidate_masks[0].shape[0] * candidate_masks[0].shape[1]
    filtered = []
    for mask, score in zip(candidate_masks, candidate_scores):
        area_ratio = float(mask.astype(np.float32).mean())
        if area_ratio < min_mask_area_ratio:
            continue
        if area_ratio > max_mask_area_ratio:
            continue
        filtered.append((float(score), mask))

    if len(filtered) == 0:
        return []

    filtered.sort(key=lambda item: item[0], reverse=True)
    selected = []
    for score, mask in filtered:
        duplicate = False
        for _, existing_mask in selected:
            if mask_iou(mask, existing_mask) >= mask_iou_dedup_thresh:
                duplicate = True
                break
        if duplicate:
            continue
        selected.append((score, mask))
        if len(selected) >= max_regions:
            break

    return [mask for _, mask in selected]


def extract_multi_region_raw_sam_outputs(
    extractor,
    x,
    point_grid_size=4,
    max_regions=6,
    min_mask_area_ratio=0.01,
    max_mask_area_ratio=0.90,
    mask_iou_dedup_thresh=0.85,
):
    if extractor.sam_model is None:
        return None, None

    try:
        batch_size, channels, _, _ = x.shape
        if batch_size != 1:
            raise ValueError(
                f"extract_multi_region_raw_sam_outputs expects batch size 1, got {batch_size}"
            )

        img_tensor = x[0].detach()
        img, height, width = _prepare_uint8_image_from_tensor(img_tensor)
        predictor = extractor.predictor
        predictor.set_image(img)

        features = predictor.features
        point_grid = build_point_grid(height, width, point_grid_size)

        candidate_masks = []
        candidate_scores = []
        for point in point_grid:
            masks, scores, _ = predictor.predict(
                point_coords=point[None, :],
                point_labels=np.ones(1, dtype=np.int32),
                multimask_output=True,
            )
            if masks is None or scores is None:
                continue
            for mask, score in zip(masks, scores):
                candidate_masks.append(np.asarray(mask, dtype=np.float32))
                candidate_scores.append(float(score))

        selected_masks = select_multi_region_masks(
            candidate_masks,
            candidate_scores,
            max_regions=max_regions,
            min_mask_area_ratio=min_mask_area_ratio,
            max_mask_area_ratio=max_mask_area_ratio,
            mask_iou_dedup_thresh=mask_iou_dedup_thresh,
        )

        if len(selected_masks) == 0:
            raw_features, raw_masks = extractor.extract_raw_sam_outputs(x)
            if raw_masks is not None and raw_masks.dim() == 3:
                raw_masks = raw_masks.unsqueeze(1)
            return raw_features, raw_masks

        if isinstance(features, np.ndarray):
            features_tensor = torch.from_numpy(features).float().to(extractor.device)
        else:
            features_tensor = features.float().to(extractor.device)

        masks_tensor = torch.from_numpy(np.stack(selected_masks, axis=0)).float().to(extractor.device)
        return features_tensor, masks_tensor.unsqueeze(0)

    except Exception as exc:
        print(f"[SAM-CACHE] Warning: multi-region extraction failed, fallback to single-best mode: {exc}")
        raw_features, raw_masks = extractor.extract_raw_sam_outputs(x)
        if raw_masks is not None and raw_masks.dim() == 3:
            raw_masks = raw_masks.unsqueeze(1)
        return raw_features, raw_masks
