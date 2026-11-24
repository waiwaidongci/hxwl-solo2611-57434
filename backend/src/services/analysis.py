"""Fit a batch-level sorting risk model and replay alerts on later work dates."""

import math
from collections.abc import Mapping
from datetime import date


FEATURES = (
    ("mixed_appearance_pct", "入线抽样外观混杂率"),
    ("unreadable_label_pct", "入线抽样标签难辨率"),
    ("belt_load_pct", "入线时输送带负荷"),
    ("moisture_pct", "来料含水率"),
)
L2_STRENGTH = 0.05
MIN_TRAIN_DAYS = 3
MIN_TRAIN_ROWS = 12
MIN_TRAIN_PER_CLASS = 2


def _integer(value, label, minimum, maximum):
    try:
        if isinstance(value, bool) or (
            isinstance(value, float) and not value.is_integer()
        ):
            raise ValueError
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{label}必须是 {minimum} 至 {maximum} 之间的整数") from None
    if not minimum <= number <= maximum:
        raise ValueError(f"{label}必须是 {minimum} 至 {maximum} 之间的整数")
    return number


def _number(value, label, maximum):
    try:
        if isinstance(value, bool):
            raise ValueError
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{label}必须是 0 至 {maximum} 之间的有限数字") from None
    if not math.isfinite(number) or not 0 <= number <= maximum:
        raise ValueError(f"{label}必须是 0 至 {maximum} 之间的有限数字")
    return number


def _parse_rows(rows):
    parsed = []
    seen_batches = set()
    for line, row in enumerate(rows, start=2):
        if not isinstance(row, Mapping):
            raise ValueError(f"第 {line} 行必须是一条来料批次记录")
        batch = row.get("batch_id")
        if not isinstance(batch, str) or not batch.strip():
            raise ValueError(f"第 {line} 行缺少来料批次号")
        batch = batch.strip()
        if batch in seen_batches:
            raise ValueError(f"第 {line} 行来料批次号重复（{batch}）")
        seen_batches.add(batch)

        row_label = f"第 {line} 行（{batch}）"
        raw_date = row.get("sort_date")
        try:
            work_date = date.fromisoformat(raw_date)
            if work_date.isoformat() != raw_date:
                raise ValueError
        except (TypeError, ValueError):
            raise ValueError(f"{row_label}的分拣作业日须为有效日期 YYYY-MM-DD") from None

        indicators = tuple(
            _number(row.get(key), f"{row_label}的{label}", 100)
            for key, label in FEATURES
        )
        actual = _integer(row.get("mis_sorted"), f"{row_label}的抽检错投标记", 0, 1)
        parsed.append((work_date, batch, indicators, actual))
    return sorted(parsed, key=lambda row: (row[0], row[1]))


def _sigmoid(log_odds):
    if log_odds >= 0:
        return 1 / (1 + math.exp(-log_odds))
    exp_log_odds = math.exp(log_odds)
    return exp_log_odds / (1 + exp_log_odds)


def _design_row(indicators, means, scales):
    return (1.0,) + tuple(
        (value - mean) / scale if scale else 0.0
        for value, mean, scale in zip(indicators, means, scales)
    )


def _fit(train, positives):
    row_count = len(train)
    columns = list(zip(*(row[2] for row in train)))
    means = [math.fsum(column) / row_count for column in columns]
    scales = [
        math.sqrt(math.fsum((value - mean) ** 2 for value in column) / row_count)
        if min(column) != max(column) else 0.0
        for column, mean in zip(columns, means)
    ]
    design = [_design_row(row[2], means, scales) for row in train]
    labels = [row[3] for row in train]
    weights = [math.log(positives / (row_count - positives))] + [0.0] * len(FEATURES)

    # The trace of XᵀX / (4n) bounds logistic curvature, including the intercept.
    squared_norm = math.fsum(value * value for row in design for value in row)
    curvature_bound = squared_norm / (4 * row_count)
    step = 1 / (curvature_bound + L2_STRENGTH)
    for _ in range(1000):
        gradient = [0.0] * len(weights)
        for row, actual in zip(design, labels):
            residual = _sigmoid(math.fsum(w * x for w, x in zip(weights, row))) - actual
            for index, value in enumerate(row):
                gradient[index] += residual * value
        gradient[0] /= row_count
        for index in range(1, len(weights)):
            gradient[index] = gradient[index] / row_count + L2_STRENGTH * weights[index]
        if max(map(abs, gradient)) < 1e-8:
            break
        weights = [weight - step * change for weight, change in zip(weights, gradient)]
    return weights, means, scales


def _percent(numerator, denominator):
    return round(100 * numerator / denominator, 1) if denominator else "不适用"


def analyze(rows, params):
    """Analyze completed historical batches using a strict, whole-date holdout."""
    holdout_days = _integer(params.get("holdout_days", 4), "留出作业日数", 1, 14)
    threshold = _number(params.get("threshold", 0.55), "预警概率阈值", 1)
    history = _parse_rows(rows)
    dates = sorted({row[0] for row in history})
    if len(dates) < holdout_days + MIN_TRAIN_DAYS:
        raise ValueError(
            f"当前仅有 {len(dates)} 个作业日。留出 {holdout_days} 个作业日后，"
            f"训练区间至少需要 {MIN_TRAIN_DAYS} 个完整作业日"
        )
    cutoff = dates[-holdout_days]
    train = [row for row in history if row[0] < cutoff]
    holdout = [row for row in history if row[0] >= cutoff]
    if len(train) < MIN_TRAIN_ROWS:
        raise ValueError(f"训练期仅有 {len(train)} 批，至少需要 {MIN_TRAIN_ROWS} 批")
    positives = sum(row[3] for row in train)
    if min(positives, len(train) - positives) < MIN_TRAIN_PER_CLASS:
        raise ValueError(
            f"训练期有 {positives} 批错投及 {len(train) - positives} 批未发现错投，"
            f"每类至少需要 {MIN_TRAIN_PER_CLASS} 批"
        )
    weights, means, scales = _fit(train, positives)

    # Compare log odds to avoid saturating extreme scores to probability 0 or 1.
    if threshold == 0:
        threshold_log_odds = -math.inf
    elif threshold == 1:
        threshold_log_odds = math.inf
    else:
        threshold_log_odds = math.log(threshold) - math.log1p(-threshold)

    counts = {(1, 1): 0, (0, 1): 0, (1, 0): 0, (0, 0): 0}
    outcomes = {(1, 1): "命中", (0, 1): "误报", (1, 0): "漏报", (0, 0): "未报且未检出"}
    scored = []
    for work_date, batch, indicators, actual in holdout:
        features = _design_row(indicators, means, scales)
        log_odds = math.fsum(weight * value for weight, value in zip(weights, features))
        alerted = int(log_odds >= threshold_log_odds)
        counts[(actual, alerted)] += 1
        scored.append((log_odds, work_date, batch, actual, alerted))
    scored.sort(key=lambda row: (-row[0], row[1], row[2]))

    tp, fp = counts[(1, 1)], counts[(0, 1)]
    fn, tn = counts[(1, 0)], counts[(0, 0)]
    alerts = tp + fp
    return {
        "summary": (
            f"留出期 {cutoff:%Y-%m-%d} 至 {dates[-1]:%Y-%m-%d} 共 {len(holdout)} 批。"
            f"阈值 {threshold:.8g} 下建议复核 {alerts} 批，其中 {tp} 批命中、{fp} 批误报；"
            f"另有 {fn} 批错投未触发预警。班组可用误报量估算额外复核工时，"
            "按漏报名单回查原单并决定是否加抽。"
        ),
        "metrics": [
            {"label": "训练批次", "value": len(train), "detail": f"{train[0][0]} 至 {train[-1][0]}，错投 {positives} 批"},
            {"label": "留出批次", "value": len(holdout), "detail": f"{holdout_days} 个作业日，实际错投 {tp + fn} 批"},
            {"label": "建议复核", "value": alerts, "detail": f"命中 {tp} 批，误报 {fp} 批"},
            {"label": "漏报批次", "value": fn, "detail": "抽检发现错投且未触发预警"},
            {"label": "预警命中率（%）", "value": _percent(tp, alerts), "detail": "命中批次 ÷ 预警批次。无预警时不适用。"},
            {"label": "错投检出率（%）", "value": _percent(tp, tp + fn), "detail": "命中批次 ÷ 实际错投批次。留出期无错投时不适用。"},
        ],
        "tables": [
            {
                "title": "留出期预警与抽检对照",
                "columns": ["抽检结果", "触发预警（批）", "未触发预警（批）"],
                "rows": [["发现错投", tp, fn], ["未发现错投", fp, tn]],
            },
            {
                "title": "留出期批次复核清单",
                "columns": ["作业日", "来料批次号", "错投概率（0–1）", "触发预警", "抽检发现错投", "复盘结果"],
                "rows": [
                    [
                        work_date.isoformat(), batch, round(_sigmoid(score), 6),
                        "是" if alerted else "否", "是" if actual else "否",
                        outcomes[(actual, alerted)],
                    ]
                    for score, work_date, batch, actual, alerted in scored
                ],
            },
            {
                "title": "训练期拟合系数",
                "columns": ["入线指标", "训练均值（%）", "训练标准差（百分点）", "标准化系数"],
                "rows": [
                    [label, round(mean, 2), round(scale, 2), round(weight, 4)]
                    for (_, label), mean, scale, weight in zip(FEATURES, means, scales, weights[1:])
                ],
            },
        ],
        "notes": [
            "整日留出最近的作业日。均值、标准差和拟合权重仅从训练期取得。训练期无变化的指标不计入风险分。",
            "拟合目标为平均二元交叉熵加 L2 正则项，正则强度为 0.05，截距不受惩罚。标准化系数描述模型内关联。",
            "阈值比较采用未舍入的模型分数。表中概率保留六位小数。误报表示触发预警且抽检未发现错投，仍以抽检覆盖范围为准。",
            "降低阈值会扩大人工复核范围，可能减少漏报。确定阈值后应使用后续独立作业日验证。",
        ],
        "series": [{"label": label, "value": counts[key]} for key, label in outcomes.items()],
    }
