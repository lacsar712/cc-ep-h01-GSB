"""回归：RunCompleted 后投影状态/版本/指标必须与事件流一致。

Bug 背景：CompleteProjectionBypass 曾让完成事件不翻转 status（停留 running），
并带有清空指标的钩子。本文件锁定修复后的行为：
- 完成事件写入 status=completed、result_summary、finished_at
- 完成不得清空/改动真实指标
- 投影 version 与最后一条事件一致
- 按事件重放（rebuild）得到的投影与库存投影一致
"""

import hashlib

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.cqrs import (
    ConflictError,
    attach_artifact,
    complete_run,
    list_events,
    rebuild_projection_from_events,
    record_metric,
    start_run,
)
from app.database import Base
from app.models import RunProjection


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


@pytest.fixture()
def db():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    from sqlalchemy.dialects.postgresql import JSONB
    from sqlalchemy.ext.compiler import compiles

    @compiles(JSONB, "sqlite")
    def _compile_jsonb_sqlite(_type, compiler, **kw):
        return "JSON"

    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()


def _start_run_with_metrics(db):
    run = start_run(
        db,
        actor="researcher",
        project="p1",
        name="n1",
        dataset_content_sha256=sha("ds"),
        code_commit_sha="abc1234",
        description="d",
    )
    run = record_metric(
        db,
        run_id=run.id,
        actor="researcher",
        name="acc",
        value=0.9,
        step=1,
        expected_version=run.version,
    )
    run = record_metric(
        db,
        run_id=run.id,
        actor="researcher",
        name="loss",
        value=0.12,
        step=2,
        expected_version=run.version,
    )
    return run


def test_complete_writes_terminal_state_summary_and_finished_at(db):
    run = _start_run_with_metrics(db)
    run = complete_run(
        db,
        run_id=run.id,
        actor="researcher",
        result_summary="训练收敛",
        expected_version=run.version,
    )

    assert run.status == "completed"
    assert run.result_summary == "训练收敛"
    assert run.finished_at is not None

    # 详情/列表走的是投影表：重新读取必须立即是已完成
    stored = db.get(RunProjection, run.id)
    assert stored.status == "completed"
    assert stored.result_summary == "训练收敛"
    assert stored.finished_at is not None


def test_projection_version_matches_last_event_after_complete(db):
    run = _start_run_with_metrics(db)
    run = complete_run(
        db,
        run_id=run.id,
        actor="researcher",
        result_summary="ok",
        expected_version=run.version,
    )

    events = list_events(db, run.id)
    last = events[-1]
    assert last.event_type == "RunCompleted"
    assert run.version == last.version
    assert run.finished_at == last.occurred_at

    stored = db.get(RunProjection, run.id)
    assert stored.version == last.version
    assert stored.status == "completed"


def test_complete_does_not_clear_or_mutate_metrics(db):
    run = _start_run_with_metrics(db)
    before = [dict(m) for m in run.metrics_json]
    assert len(before) == 2

    run = complete_run(
        db,
        run_id=run.id,
        actor="researcher",
        result_summary="ok",
        expected_version=run.version,
    )

    assert len(run.metrics_json) == len(before)
    for expected, actual in zip(before, run.metrics_json):
        assert actual["name"] == expected["name"]
        assert actual["value"] == expected["value"]
        assert actual["step"] == expected["step"]

    stored = db.get(RunProjection, run.id)
    assert [m["name"] for m in stored.metrics_json] == ["acc", "loss"]


def test_event_replay_after_complete_matches_projection(db):
    run = _start_run_with_metrics(db)
    run = attach_artifact(
        db,
        run_id=run.id,
        actor="researcher",
        name="model.bin",
        uri="file:///tmp/model.bin",
        content_sha256=sha("model"),
        media_type="application/octet-stream",
        expected_version=run.version,
    )
    run = complete_run(
        db,
        run_id=run.id,
        actor="researcher",
        result_summary="replay-check",
        expected_version=run.version,
    )

    rebuilt = rebuild_projection_from_events(db, run.id)
    stored = db.get(RunProjection, run.id)

    assert rebuilt is not None
    assert rebuilt.status == "completed" == stored.status
    assert rebuilt.version == stored.version
    assert rebuilt.result_summary == stored.result_summary
    assert rebuilt.finished_at == stored.finished_at
    # 指标按事件重放仍正确（内容逐项一致）
    assert len(rebuilt.metrics_json) == len(stored.metrics_json) == 2
    for r, s in zip(rebuilt.metrics_json, stored.metrics_json):
        assert r["name"] == s["name"]
        assert r["value"] == s["value"]
        assert r["step"] == s["step"]
    assert len(rebuilt.artifacts_json) == len(stored.artifacts_json) == 1


def test_completed_run_rejects_further_commands(db):
    run = _start_run_with_metrics(db)
    run = complete_run(
        db,
        run_id=run.id,
        actor="researcher",
        result_summary="done",
        expected_version=run.version,
    )
    assert run.status == "completed"

    with pytest.raises(ConflictError):
        record_metric(
            db,
            run_id=run.id,
            actor="researcher",
            name="acc",
            value=0.95,
            step=3,
            expected_version=run.version,
        )
    with pytest.raises(ConflictError):
        complete_run(
            db,
            run_id=run.id,
            actor="researcher",
            result_summary="again",
            expected_version=run.version,
        )

    # 拒绝命令后投影与事件流仍未被污染
    events = list_events(db, run.id)
    assert events[-1].event_type == "RunCompleted"
    stored = db.get(RunProjection, run.id)
    assert stored.status == "completed"
    assert stored.version == events[-1].version
