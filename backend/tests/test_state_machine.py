import hashlib
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.cqrs import (
    ConflictError,
    DomainError,
    abort_run,
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
    # JSONB not available on SQLite — remap via create_all with JSON
    from sqlalchemy import JSON
    from sqlalchemy.dialects.postgresql import JSONB

    # For SQLite tests, compile JSONB as JSON
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


def test_start_and_complete_happy_path(db):
    run = start_run(
        db,
        actor="researcher",
        project="p1",
        name="n1",
        dataset_content_sha256=sha("ds"),
        code_commit_sha="abc1234",
        description="d",
    )
    assert run.status == "running"
    assert run.version == 1

    run = record_metric(
        db,
        run_id=run.id,
        actor="researcher",
        name="acc",
        value=0.9,
        step=1,
        expected_version=1,
    )
    assert run.version == 2
    assert len(run.metrics_json) == 1

    run = complete_run(
        db,
        run_id=run.id,
        actor="researcher",
        result_summary="done",
        expected_version=2,
    )
    assert run.status == "completed"
    assert run.version == 3

    with pytest.raises(ConflictError):
        record_metric(
            db,
            run_id=run.id,
            actor="researcher",
            name="acc",
            value=0.95,
            step=2,
            expected_version=3,
        )


def test_optimistic_lock_conflict(db):
    run = start_run(
        db,
        actor="researcher",
        project="p1",
        name="n1",
        dataset_content_sha256=sha("ds2"),
        code_commit_sha="abc1234",
        description=None,
    )
    with pytest.raises(ConflictError):
        record_metric(
            db,
            run_id=run.id,
            actor="researcher",
            name="loss",
            value=1.0,
            step=1,
            expected_version=0,
        )


def test_abort_terminal(db):
    run = start_run(
        db,
        actor="researcher",
        project="p1",
        name="n1",
        dataset_content_sha256=sha("ds3"),
        code_commit_sha="abc1234",
        description=None,
    )
    run = abort_run(
        db,
        run_id=run.id,
        actor="researcher",
        reason="OOM",
        expected_version=1,
    )
    assert run.status == "aborted"
    with pytest.raises(ConflictError):
        complete_run(
            db,
            run_id=run.id,
            actor="researcher",
            result_summary="nope",
            expected_version=2,
        )


def test_projection_matches_event_replay(db):
    run = start_run(
        db,
        actor="researcher",
        project="p1",
        name="n1",
        dataset_content_sha256=sha("ds4"),
        code_commit_sha="deadbeef",
        description="x",
        run_id=uuid4(),
    )
    run = record_metric(
        db,
        run_id=run.id,
        actor="researcher",
        name="f1",
        value=1.5,
        step=0,
        expected_version=run.version,
    )
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
        result_summary="ok",
        expected_version=run.version,
    )

    events = list_events(db, run.id)
    assert [e.event_type for e in events] == [
        "RunStarted",
        "MetricRecorded",
        "ArtifactAttached",
        "RunCompleted",
    ]

    rebuilt = rebuild_projection_from_events(db, run.id)
    stored = db.get(RunProjection, run.id)
    assert rebuilt is not None and stored is not None
    assert rebuilt.status == stored.status
    assert rebuilt.version == stored.version
    assert rebuilt.dataset_content_sha256 == stored.dataset_content_sha256
    assert rebuilt.code_commit_sha == stored.code_commit_sha
    assert len(rebuilt.metrics_json) == len(stored.metrics_json)
    assert len(rebuilt.artifacts_json) == len(stored.artifacts_json)


def test_cannot_command_before_start(db):
    missing = uuid4()
    with pytest.raises(DomainError):
        record_metric(
            db,
            run_id=missing,
            actor="researcher",
            name="x",
            value=1,
            step=0,
            expected_version=0,
        )


def test_completed_projection_matches_last_event_and_keeps_metrics(db):
    """回归：RunCompleted 后投影必须立即为终态，版本/摘要/结束时间与最后事件一致，
    且完成动作不得清空真实指标。"""
    run = start_run(
        db,
        actor="researcher",
        project="p1",
        name="n1",
        dataset_content_sha256=sha("ds-complete"),
        code_commit_sha="abc1234",
        description=None,
        run_id=uuid4(),
    )
    run = record_metric(
        db,
        run_id=run.id,
        actor="researcher",
        name="loss",
        value=0.42,
        step=1,
        expected_version=run.version,
    )
    run = record_metric(
        db,
        run_id=run.id,
        actor="researcher",
        name="acc",
        value=0.91,
        step=2,
        expected_version=run.version,
    )
    run = complete_run(
        db,
        run_id=run.id,
        actor="researcher",
        result_summary="all green",
        expected_version=run.version,
    )

    # 重新从库里读投影，避免拿内存对象
    db.expire_all()
    stored = db.get(RunProjection, run.id)
    events = list_events(db, run.id)
    last_event = events[-1]

    assert last_event.event_type == "RunCompleted"
    # 状态立即为已完成 —— 详情/列表读到的就是终态
    assert stored.status == "completed"
    # 投影版本与最后事件版本一致
    assert stored.version == last_event.version == len(events)
    # 结果摘要与结束时间已写入投影
    assert stored.result_summary == "all green"
    assert stored.finished_at is not None
    assert stored.finished_at == last_event.occurred_at
    # 完成不得清空真实指标
    assert len(stored.metrics_json) == 2
    assert [(m["name"], m["value"], m["step"]) for m in stored.metrics_json] == [
        ("loss", 0.42, 1),
        ("acc", 0.91, 2),
    ]


def test_replay_after_complete_rebuilds_identical_projection(db):
    """回归：按事件流重放得到的投影与线上投影在状态、版本、摘要、指标上一致。"""
    run = start_run(
        db,
        actor="researcher",
        project="p1",
        name="n1",
        dataset_content_sha256=sha("ds-replay"),
        code_commit_sha="deadbeef",
        description="x",
        run_id=uuid4(),
    )
    run = record_metric(
        db,
        run_id=run.id,
        actor="researcher",
        name="f1",
        value=1.5,
        step=0,
        expected_version=run.version,
    )
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
        result_summary="ok",
        expected_version=run.version,
    )

    rebuilt = rebuild_projection_from_events(db, run.id)
    db.expire_all()
    stored = db.get(RunProjection, run.id)
    events = list_events(db, run.id)

    assert rebuilt is not None
    assert rebuilt.status == "completed" == stored.status
    assert rebuilt.version == stored.version == events[-1].version
    assert rebuilt.result_summary == stored.result_summary == "ok"
    assert rebuilt.finished_at == stored.finished_at
    # 指标按事件重放仍然完整（完成事件没有抹掉它们）
    assert len(rebuilt.metrics_json) == len(stored.metrics_json) == 1
    assert [m["name"] for m in rebuilt.metrics_json] == [
        m["name"] for m in stored.metrics_json
    ]
    assert [(m["name"], m["value"], m["step"]) for m in rebuilt.metrics_json] == [
        ("f1", 1.5, 0)
    ]
    assert len(rebuilt.artifacts_json) == len(stored.artifacts_json) == 1
    assert rebuilt.artifacts_json[0]["name"] == stored.artifacts_json[0]["name"] == "model.bin"
