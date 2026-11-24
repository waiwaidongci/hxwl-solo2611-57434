"""Score the next shift's arrivals from history closed at publication.

Publishing a review roster freezes three things:

* the training snapshot: only inspections closed at or before ``publish_at``
  may influence the model;
* the participating batches: next-shift arrivals registered at or before
  ``publish_at``;
* the selected batches: the highest-risk participants that fit the review quota,
  ties broken by batch id.

Later arrivals and later inspection versions can be replayed against the stored
roster, but they never change participation or selection.  Inspections may be
reported out of order: the highest version wins, and conflicting conclusions
inside that version leave the batch pending (待核).  Batches without an
inspection are listed separately and never counted as false alarms or misses.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta, timezone


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
LOCAL_TZ = timezone(timedelta(hours=8))
LEGACY_ARRIVAL_TIME = time(7, 0)
LEGACY_CLOSE_TIME = time(20, 0)
DEFAULT_PUBLISH_AT = "2026-11-21T08:00:00+08:00"
DEFAULT_QUOTA = 3
OUTCOMES = {
    (1, 1): "命中",
    (0, 1): "误报",
    (1, 0): "漏报",
    (0, 0): "未报且未检出",
}


@dataclass(frozen=True)
class Inspection:
    version: int
    result: int
    closed_at: datetime
    line: int


@dataclass
class Arrival:
    work_date: date
    batch_id: str
    indicators: tuple
    received_at: datetime | None
    line: int


@dataclass
class Records:
    arrivals: dict = field(default_factory=dict)
    inspections: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Resolution:
    status: str
    version: int | None = None
    result: int | None = None
    results: tuple = ()
    closed_at: datetime | None = None


def _text(row, key):
    value = row.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return value


def _integer(value, label, minimum, maximum):
    try:
        if value is None or isinstance(value, bool) or (
            isinstance(value, float) and not value.is_integer()
        ):
            raise ValueError
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{label}必须是 {minimum} 至 {maximum} 之间的整数") from None
    if not minimum <= number <= maximum:
        raise ValueError(f"{label}必须是 {minimum} 至 {maximum} 之间的整数")
    return number


def _number(value, label):
    try:
        if value is None or isinstance(value, bool):
            raise ValueError
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{label}必须是 0 至 100 之间的有限数字") from None
    if not math.isfinite(number) or not 0 <= number <= 100:
        raise ValueError(f"{label}必须是 0 至 100 之间的有限数字")
    return number


def _work_date(value, label):
    try:
        if not isinstance(value, str):
            raise ValueError
        parsed = date.fromisoformat(value)
        if parsed.isoformat() != value:
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError(f"{label}须为有效日期 YYYY-MM-DD") from None
    return parsed


def _timestamp(value, label):
    try:
        if isinstance(value, datetime):
            parsed = value
        elif isinstance(value, str):
            parsed = datetime.fromisoformat(value.replace(" ", "T"))
        else:
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError(f"{label}须为 ISO 8601 时刻，例如 2026-11-21T08:00:00+08:00") from None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=LOCAL_TZ)
    return parsed.astimezone(UTC).replace(tzinfo=None)


def _local_default(work_date, clock):
    return datetime.combine(work_date, clock, tzinfo=LOCAL_TZ).astimezone(UTC).replace(tzinfo=None)


def _iso(stamp):
    return stamp.replace(tzinfo=UTC).isoformat(timespec="seconds")


def _local_text(stamp):
    return stamp.replace(tzinfo=UTC).astimezone(LOCAL_TZ).strftime("%Y-%m-%d %H:%M")


def _local_of_iso(text):
    return _local_text(_timestamp(text, "发布时刻"))


def _merge_arrival(records, batch, work_date, indicators, received_at, line):
    known = records.arrivals.get(batch)
    if known is None:
        records.arrivals[batch] = Arrival(work_date, batch, indicators, received_at, line)
        return
    if known.work_date and work_date and known.work_date != work_date:
        raise ValueError(f"第 {line} 行（{batch}）的作业日与第 {known.line} 行不一致")
    if indicators:
        if known.indicators and known.indicators != indicators:
            raise ValueError(f"第 {line} 行（{batch}）的入线抽样指标与第 {known.line} 行不一致")
        if not known.indicators:
            known.indicators = indicators
    if not known.work_date and work_date:
        known.work_date = work_date
    if received_at is not None:
        known.received_at = received_at if known.received_at is None else min(known.received_at, received_at)


def _parse_records(rows):
    records = Records()
    for line, row in enumerate(rows, start=2):
        if not isinstance(row, Mapping):
            raise ValueError(f"第 {line} 行必须是一条来料批次记录")
        raw_batch = row.get("batch_id")
        if not isinstance(raw_batch, str) or not raw_batch.strip():
            raise ValueError(f"第 {line} 行缺少来料批次号")
        batch = raw_batch.strip()
        row_label = f"第 {line} 行（{batch}）"

        raw_date = _text(row, "sort_date")
        work_date = _work_date(raw_date, f"{row_label}的分拣作业日") if raw_date else None

        present = [row.get(key) not in (None, "") for key, _ in FEATURES]
        has_arrival = any(present)
        if has_arrival and not all(present):
            missing = "、".join(label for (key, label), seen in zip(FEATURES, present) if not seen)
            raise ValueError(f"{row_label}的入线抽样指标不完整，缺少{missing}")
        if has_arrival:
            if work_date is None:
                raise ValueError(f"{row_label}缺少分拣作业日")
            indicators = tuple(_number(row.get(key), f"{row_label}的{label}")
                               for key, label in FEATURES)
            raw_received = _text(row, "received_at")
            received_at = (_timestamp(raw_received, f"{row_label}的入线登记时刻")
                           if raw_received is not None
                           else _local_default(work_date, LEGACY_ARRIVAL_TIME))
            _merge_arrival(records, batch, work_date, indicators, received_at, line)

        raw_result = _text(row, "mis_sorted")
        version_value = _text(row, "inspection_version")
        raw_closed = _text(row, "inspection_closed_at")
        has_inspection = raw_result is not None or version_value is not None or raw_closed is not None
        if has_inspection:
            if raw_result is None:
                raise ValueError(f"{row_label}的班末抽检记录缺少抽检结论")
            result = _integer(raw_result, f"{row_label}的抽检错投标记", 0, 1)
            version = (_integer(version_value, f"{row_label}的抽检记录版本", 1, 1000000)
                       if version_value is not None else 1)
            if raw_closed is not None:
                closed_at = _timestamp(raw_closed, f"{row_label}的抽检结案时刻")
            elif work_date is not None:
                closed_at = _local_default(work_date, LEGACY_CLOSE_TIME)
            elif batch in records.arrivals and records.arrivals[batch].work_date:
                closed_at = _local_default(records.arrivals[batch].work_date, LEGACY_CLOSE_TIME)
            else:
                raise ValueError(f"{row_label}缺少分拣作业日或抽检结案时刻")
            if batch not in records.arrivals:
                derived = work_date or closed_at.replace(tzinfo=UTC).astimezone(LOCAL_TZ).date()
                _merge_arrival(records, batch, derived, (), None, line)
            elif (work_date is not None and records.arrivals[batch].work_date
                  and work_date != records.arrivals[batch].work_date):
                raise ValueError(f"{row_label}的作业日与入线抽样记录不一致")
            records.inspections.setdefault(batch, set()).add(
                Inspection(version, result, closed_at, line)
            )

        if not has_arrival and not has_inspection:
            raise ValueError(f"{row_label}既没有入线抽样指标，也没有班末抽检结论")
    return records


def validate_rows(rows):
    """Reject malformed rows early; conflicting versions become 待核 at analysis."""
    _parse_records(rows)


def _resolve(records, cutoff=None):
    """Use the newest version, ignoring records that were not closed by cutoff."""
    picked = sorted((item for item in records if cutoff is None or item.closed_at <= cutoff),
                    key=lambda item: (item.version, item.closed_at, item.line))
    if not picked:
        return Resolution("none")
    version = picked[-1].version
    current = [item for item in picked if item.version == version]
    results = tuple(sorted({item.result for item in current}))
    closed_at = max(item.closed_at for item in current)
    if len(results) > 1:
        return Resolution("conflict", version=version, results=results, closed_at=closed_at)
    return Resolution("ok", version=version, result=results[0], results=results, closed_at=closed_at)


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


def _log_odds(indicators, weights, means, scales):
    return math.fsum(weight * value
                     for weight, value in zip(weights, _design_row(indicators, means, scales)))


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


def _snapshot(records, publish_at, shift_date):
    """Training rows closed at publication, plus history still unresolved then.

    Batches on the scored shift are kept out of training even if an early
    inspection closed in time, so the roster never trains on its own targets.
    """
    train, pending, incomplete = [], [], []
    for batch, arrival in records.arrivals.items():
        if arrival.work_date >= shift_date:
            continue
        resolution = _resolve(records.inspections.get(batch, ()), publish_at)
        if resolution.status != "ok":
            pending.append(batch)
        elif not arrival.indicators:
            incomplete.append(batch)
        elif arrival.received_at is not None and arrival.received_at > publish_at:
            pending.append(batch)
        else:
            train.append((arrival.work_date, batch, arrival.indicators, resolution.result))
    train.sort(key=lambda row: (row[0], row[1]))
    return train, set(pending), set(incomplete)


def _next_shift(records, publish_at):
    """The shift being scored: newest work date registered by publication.

    Only arrivals registered at or before ``publish_at`` can set the date, so a
    batch that first appears after the roster was sent cannot move the shift.
    """
    registered = [item for item in records.arrivals.values()
                  if item.indicators and item.received_at is not None
                  and item.received_at <= publish_at]
    if not registered:
        raise ValueError("数据中没有可打分的来料批次，请先导出入线抽样记录")
    shift_date = max(item.work_date for item in registered)
    participants = sorted((item for item in registered if item.work_date == shift_date),
                          key=lambda item: item.batch_id)
    unscored = sorted(item.batch_id for item in records.arrivals.values()
                      if item.work_date == shift_date and not item.indicators)
    late = sorted(item.batch_id for item in records.arrivals.values()
                  if item.work_date == shift_date and item.indicators
                  and (item.received_at is None or item.received_at > publish_at))
    return shift_date, participants, unscored, late


def _ranked(entries, quota):
    """Sort by raw score (ties by batch id) and mark the quota as selected."""
    ordered = sorted(entries, key=lambda entry: (-entry["_score"], entry["batch_id"]))
    cleaned = []
    for rank, entry in enumerate(ordered, start=1):
        entry.pop("_score")
        entry["rank"] = rank
        entry["selected"] = rank <= quota
        cleaned.append(entry)
    return cleaned


def _roster(records, params):
    publish_at = _timestamp(params.get("publish_at", DEFAULT_PUBLISH_AT), "名单发布时刻")
    quota = _integer(params.get("review_quota", DEFAULT_QUOTA), "复核名额", 1, 200)
    shift_date, participants, unscored, late = _next_shift(records, publish_at)
    train, unresolved, incomplete = _snapshot(records, publish_at, shift_date)
    dates = sorted({row[0] for row in train})
    if len(dates) < MIN_TRAIN_DAYS:
        raise ValueError(
            f"截至发布时刻仅有 {len(dates)} 个作业日有可用结案抽检，"
            f"至少需要 {MIN_TRAIN_DAYS} 个完整作业日"
        )
    if len(train) < MIN_TRAIN_ROWS:
        raise ValueError(f"截至发布时刻仅有 {len(train)} 批可用结案抽检，至少需要 {MIN_TRAIN_ROWS} 批")
    positives = sum(row[3] for row in train)
    if min(positives, len(train) - positives) < MIN_TRAIN_PER_CLASS:
        raise ValueError(
            f"训练期有 {positives} 批错投及 {len(train) - positives} 批未发现错投，"
            f"每类至少需要 {MIN_TRAIN_PER_CLASS} 批"
        )
    weights, means, scales = _fit(train, positives)

    entries = []
    for arrival in participants:
        resolution = _resolve(records.inspections.get(arrival.batch_id, ()), publish_at)
        score = _log_odds(arrival.indicators, weights, means, scales)
        entries.append({
            "batch_id": arrival.batch_id,
            "work_date": arrival.work_date.isoformat(),
            "probability": round(_sigmoid(score), 6),
            "publish_version": resolution.version,
            "publish_result": resolution.result,
            "publish_status": resolution.status,
            "_score": score,
        })
    entries = _ranked(entries, quota)

    history_pending = sorted(batch for batch in unresolved
                             if records.arrivals[batch].indicators)
    return {
        "publish_at": _iso(publish_at),
        "shift_date": shift_date.isoformat(),
        "quota": quota,
        "training": {
            "batches": len(train), "days": len(dates), "positives": positives,
            "start": dates[0].isoformat(), "end": dates[-1].isoformat(),
            "means": means, "scales": scales, "weights": weights,
        },
        "entries": entries,
        "history_pending": history_pending,
        "history_incomplete": sorted(incomplete),
        "unscored": unscored,
        "late_arrivals": late,
    }


def _late(roster, records):
    """Batches on the roster's shift date registered after publication."""
    shift_date = date.fromisoformat(roster["shift_date"])
    publish_at = _timestamp(roster["publish_at"], "发布时刻")
    listed = {entry["batch_id"] for entry in roster["entries"]}
    return sorted(
        item.batch_id for item in records.arrivals.values()
        if item.work_date == shift_date and item.batch_id not in listed
        and item.indicators
        and (item.received_at is None or item.received_at > publish_at)
    )


def _current(roster, records):
    """Replay the frozen roster against the newest inspection versions."""
    counts = {"hit": 0, "false_positive": 0, "miss": 0, "true_negative": 0}
    rows, pending, conflicts = [], [], []
    for entry in roster["entries"]:
        resolution = _resolve(records.inspections.get(entry["batch_id"], ()))
        prefix = [
            entry["rank"], entry["batch_id"], entry["work_date"], entry["probability"],
            "是" if entry["selected"] else "否",
            entry["publish_version"] if entry["publish_version"] is not None else "—",
        ]
        if resolution.status == "none":
            pending.append(prefix + ["—", "未抽检", "未抽检"])
            rows.append(prefix + ["—", "未抽检", "未抽检"])
        elif resolution.status == "conflict":
            detail = f"同版结论 {'/'.join(map(str, resolution.results))} 冲突"
            conflicts.append(prefix + [resolution.version, detail])
            rows.append(prefix + [resolution.version, detail, "待核"])
        else:
            actual, alerted = resolution.result, int(entry["selected"])
            key = {(1, 1): "hit", (0, 1): "false_positive",
                   (1, 0): "miss", (0, 0): "true_negative"}[(actual, alerted)]
            counts[key] += 1
            rows.append(prefix + [
                resolution.version,
                "抽检发现错投" if actual else "抽检未发现错投",
                OUTCOMES[(actual, alerted)],
            ])
    rows.sort(key=lambda row: row[0])
    return counts, rows, pending, conflicts


def _tables(roster, current, coefficients=None):
    counts, rows, pending, conflicts = current
    tables = [
        {
            "title": "下一班复核名单与当前复盘",
            "columns": ["排名", "来料批次号", "作业日", "错投概率（0–1）", "入选复核",
                        "发布时抽检版本", "当前抽检版本", "当前结论", "复盘结果"],
            "rows": rows,
        },
        {
            "title": "未抽检批次（单列，不计误报漏报）",
            "columns": ["排名", "来料批次号", "作业日", "错投概率（0–1）", "入选复核", "发布时抽检版本"],
            "rows": [[row[0], row[1], row[2], row[3], row[4], row[5]] for row in pending],
        },
        {
            "title": "待核批次（同版结论冲突）",
            "columns": ["排名", "来料批次号", "作业日", "错投概率（0–1）", "入选复核", "当前抽检版本", "冲突结论"],
            "rows": [[row[0], row[1], row[2], row[3], row[4], row[6], row[7]] for row in conflicts],
        },
        {
            "title": "当前复盘汇总",
            "columns": ["抽检结果", "入选复核（批）", "未入选（批）"],
            "rows": [
                ["抽检发现错投", counts["hit"], counts["miss"]],
                ["抽检未发现错投", counts["false_positive"], counts["true_negative"]],
            ],
        },
    ]
    if coefficients:
        tables.append({
            "title": "训练期拟合系数",
            "columns": ["入线指标", "训练均值（%）", "训练标准差（百分点）", "标准化系数"],
            "rows": coefficients,
        })
    return tables


def _series(counts, pending, conflicts):
    return [
        {"label": "命中", "value": counts["hit"]},
        {"label": "误报", "value": counts["false_positive"]},
        {"label": "漏报", "value": counts["miss"]},
        {"label": "未报且未检出", "value": counts["true_negative"]},
        {"label": "待核", "value": len(conflicts)},
        {"label": "未抽检", "value": len(pending)},
    ]


def _metrics(roster, counts, pending, conflicts, training=False):
    metrics = []
    if training:
        info = roster["training"]
        metrics.append({
            "label": "训练批次", "value": info["batches"],
            "detail": f"{info['start']} 至 {info['end']}，错投 {info['positives']} 批",
        })
        metrics.append({"label": "训练作业日", "value": info["days"], "detail": "截至发布时刻已结案"})
    metrics.extend([
        {"label": "参评批次", "value": len(roster["entries"]),
         "detail": f"{roster['shift_date']} 已登记，发布后补传不计入"},
        {"label": "入选复核", "value": sum(1 for entry in roster["entries"] if entry["selected"]),
         "detail": f"复核名额 {roster['quota']} 批，同分按批次号升序"},
        {"label": "当前命中", "value": counts["hit"], "detail": "入选且抽检发现错投"},
        {"label": "当前误报", "value": counts["false_positive"], "detail": "入选但抽检未发现错投"},
        {"label": "当前漏报", "value": counts["miss"], "detail": "未入选但抽检发现错投"},
        {"label": "待核", "value": len(conflicts), "detail": "同一最高版本结论冲突"},
        {"label": "未抽检", "value": len(pending), "detail": "单列，不计入误报或漏报"},
    ])
    return metrics


def _notes(roster):
    notes = [
        "名单在发布时刻冻结：发布后补传的来料不计入参评批次，历史更正只更新当前复盘，不改变参评批次和入选结果。",
        "班末抽检按版本取最新；同一最高版本结论不一致的批次列为待核，不计入误报、漏报和训练。",
        "每批只计一次。重复导入、重复补报或重跑同一名单不会重复计数。",
        "当前复盘只使用名单内批次；未抽检批次单列，不按误报或漏报处理。",
        "排名按未舍入的模型分数，同分按批次号升序；表中概率保留六位小数，入选结果以本名单为准。",
    ]
    if roster.get("history_pending"):
        notes.append(f"训练历史中有 {len(roster['history_pending'])} 批待核或缺少发布时结论，未参与拟合。")
    if roster.get("history_incomplete"):
        notes.append(f"另有 {len(roster['history_incomplete'])} 批只有抽检结论、缺少入线指标，未参与拟合。")
    if roster.get("unscored"):
        notes.append(f"下一班有 {len(roster['unscored'])} 批缺少入线指标，未参评。")
    if roster.get("late_arrivals"):
        notes.append(f"发布后登记 {len(roster['late_arrivals'])} 批来料，未纳入本次名单。")
    return notes


def _summary(roster, counts, pending, conflicts):
    selected = sum(1 for entry in roster["entries"] if entry["selected"])
    return (
        f"名单发布于 {_local_of_iso(roster['publish_at'])}，"
        f"下一班 {roster['shift_date']} 参评 {len(roster['entries'])} 批，"
        f"按名额 {roster['quota']} 批选出入选 {selected} 批。当前复盘："
        f"命中 {counts['hit']} 批、误报 {counts['false_positive']} 批、漏报 {counts['miss']} 批；"
        f"待核 {len(conflicts)} 批，未抽检 {len(pending)} 批。"
    )


def _coefficients(info):
    return [
        [label, round(mean, 2), round(scale, 2), round(weight, 4)]
        for (_, label), mean, scale, weight
        in zip(FEATURES, info["means"], info["scales"], info["weights"][1:])
    ]


def analyze(rows, params):
    records = _parse_records(rows)
    roster = _roster(records, params)
    current = _current(roster, records)
    info = roster["training"]
    summary = (
        f"截至 {_local_of_iso(roster['publish_at'])} "
        f"已结案 {info['batches']} 批（{info['days']} 个作业日、错投 {info['positives']} 批）用于拟合。"
        + _summary(roster, current[0], current[2], current[3]) + "名单发出后补传来料或更正历史都不改变参评批次和入选结果。"
    )
    return {
        "summary": summary,
        "metrics": _metrics(roster, current[0], current[2], current[3], training=True),
        "tables": _tables(roster, current, _coefficients(info)),
        "notes": _notes(roster),
        "series": _series(current[0], current[2], current[3]),
        "roster": roster,
    }


def replay(roster, rows):
    """Replay a published roster against newer data without re-selecting batches."""
    records = _parse_records(rows)
    current = _current(roster, records)
    view = {**roster, "late_arrivals": _late(roster, records)}
    return {
        "summary": _summary(view, current[0], current[2], current[3]),
        "metrics": _metrics(view, current[0], current[2], current[3]),
        "tables": _tables(view, current),
        "notes": _notes(view),
        "series": _series(current[0], current[2], current[3]),
        "roster": view,
    }
