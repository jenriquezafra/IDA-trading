from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research_app.artifacts import read_markdown, read_parquet_preview, resolve_existing_path
from src.research_app.decisions import create_decision_log, list_decision_logs
from src.research_app.registry import DEFAULT_DB_PATH, index_workspace, list_artifacts, list_candidates, list_reports, list_runs
from src.research_app.service import filter_frame, metric_value


st.set_page_config(page_title="IDA Research Lab", layout="wide")


def db_path_from_sidebar() -> Path:
    return Path(st.sidebar.text_input("Registry DB", str(DEFAULT_DB_PATH)))


@st.cache_data(show_spinner=False)
def cached_runs(db_path: str) -> pd.DataFrame:
    return list_runs(db_path)


@st.cache_data(show_spinner=False)
def cached_candidates(db_path: str, run_id: str | None) -> pd.DataFrame:
    return list_candidates(db_path, run_id=run_id if run_id and run_id != "ALL" else None)


@st.cache_data(show_spinner=False)
def cached_reports(db_path: str, run_id: str | None) -> pd.DataFrame:
    return list_reports(db_path, run_id=run_id if run_id and run_id != "ALL" else None)


@st.cache_data(show_spinner=False)
def cached_artifacts(db_path: str, run_id: str | None) -> pd.DataFrame:
    return list_artifacts(db_path, run_id=run_id if run_id and run_id != "ALL" else None)


def show_decision_board(candidates: pd.DataFrame, runs: pd.DataFrame) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Runs indexados", len(runs))
    c2.metric("Candidatos fuente", len(candidates))
    c3.metric("Targets", metric_value(runs, "instrument"))
    c4.metric("Timeframes", metric_value(runs, "timeframe"))

    st.info(
        "Default: esta vista es read-only. Ordena por estado/evidencia, no por mejor Sharpe aislado. "
        "Los runs legacy aparecen con warning hasta que tengan manifest completo."
    )

    if candidates.empty:
        st.warning("No hay candidatos indexados. Ejecuta el indexador desde la sidebar.")
        return

    preferred = [
        "target_symbol",
        "timeframe",
        "candidate_id",
        "source_file",
        "status",
        "validation_status",
        "decision",
        "test_net_primary",
        "test_sharpe_primary",
        "test_profit_factor_primary",
        "test_avg_trade_net_primary",
        "test_trades_primary",
        "test_net_conservative",
        "test_net_stress",
        "net_return",
        "daily_sharpe",
        "avg_trade_net",
        "trades",
        "run_id",
    ]
    visible = [col for col in preferred if col in candidates.columns]
    st.dataframe(candidates[visible].head(500), width="stretch", hide_index=True)


def show_run_browser(runs: pd.DataFrame, artifacts: pd.DataFrame, reports: pd.DataFrame) -> None:
    st.subheader("Runs")
    st.dataframe(runs, width="stretch", hide_index=True)
    st.subheader("Artifacts")
    artifact_cols = [col for col in ["run_id", "logical_name", "artifact_type", "rows", "size_bytes", "path"] if col in artifacts]
    st.dataframe(artifacts[artifact_cols].head(500), width="stretch", hide_index=True)
    st.subheader("Reports")
    report_cols = [col for col in ["run_id", "report_type", "path"] if col in reports]
    st.dataframe(reports[report_cols].head(500), width="stretch", hide_index=True)


def show_candidate_explorer(candidates: pd.DataFrame, artifacts: pd.DataFrame) -> None:
    if candidates.empty:
        st.warning("No hay candidatos indexados.")
        return
    candidate_id = st.selectbox("Candidate", sorted(candidates["candidate_id"].dropna().astype(str).unique()))
    selected = candidates[candidates["candidate_id"].astype(str) == candidate_id]
    st.dataframe(selected, width="stretch", hide_index=True)

    source_paths = sorted(selected["source_path"].dropna().astype(str).unique()) if "source_path" in selected else []
    if source_paths:
        source = st.selectbox("Fuente parquet", source_paths)
        try:
            preview = read_parquet_preview(resolve_existing_path(source, ROOT), limit=200)
            st.dataframe(preview, width="stretch", hide_index=True)
        except Exception as exc:
            st.error(f"No se pudo leer {source}: {exc}")

    candidate_artifacts = artifacts[artifacts["path"].astype(str).str.contains(candidate_id, regex=False, na=False)] if not artifacts.empty else pd.DataFrame()
    if not candidate_artifacts.empty:
        st.subheader("Artifacts con candidate_id en path")
        st.dataframe(candidate_artifacts, width="stretch", hide_index=True)


def show_reports(reports: pd.DataFrame) -> None:
    if reports.empty:
        st.warning("No hay reports indexados.")
        return
    md_reports = reports[reports["path"].astype(str).str.endswith(".md")]
    if md_reports.empty:
        st.warning("No hay markdown reports para el filtro actual.")
        return
    path = st.selectbox("Report", md_reports["path"].astype(str).tolist())
    try:
        st.markdown(read_markdown(resolve_existing_path(path, ROOT)))
    except Exception as exc:
        st.error(f"No se pudo leer {path}: {exc}")


def show_decisions(db_path: Path, runs: pd.DataFrame, candidates: pd.DataFrame) -> None:
    st.subheader("Decision log")
    existing = list_decision_logs(db_path)
    if not existing.empty:
        st.dataframe(existing, width="stretch", hide_index=True)

    with st.form("decision_form"):
        run_options = [""] + sorted(runs["run_id"].dropna().astype(str).unique().tolist()) if not runs.empty else [""]
        candidate_options = [""] + sorted(candidates["candidate_id"].dropna().astype(str).unique().tolist()) if not candidates.empty else [""]
        run_id = st.selectbox("Run", run_options)
        candidate_id = st.selectbox("Candidate", candidate_options)
        decision_type = st.selectbox("Tipo", ["reject", "keep_in_research", "freeze_draft", "paper_candidate", "note"])
        decision = st.text_input("Decision")
        rationale = st.text_area("Rationale")
        evidence_path = st.text_input("Evidence path", placeholder="results/.../candidate_decisions.parquet")
        next_action = st.text_input("Next action")
        submitted = st.form_submit_button("Registrar decision")
    if submitted:
        if not decision or not evidence_path:
            st.error("Decision y evidence path son obligatorios.")
        else:
            evidence = [{"path": evidence_path, "run_id": run_id or None, "candidate_id": candidate_id or None}]
            log = create_decision_log(
                db_path=db_path,
                decision_type=decision_type,
                decision=decision,
                evidence=evidence,
                run_id=run_id or None,
                candidate_id=candidate_id or None,
                rationale=rationale or None,
                next_action=next_action or None,
            )
            st.success(f"Decision registrada: {log.decision_id}")
            st.cache_data.clear()


st.title("IDA Research Lab")
st.caption("Research dashboard local, read-only y artifact-driven para IDA Trading.")

db_path = db_path_from_sidebar()
if st.sidebar.button("Indexar artifacts", type="primary"):
    with st.spinner("Indexando results/ y reports/..."):
        summary = index_workspace(db_path=db_path, reset=True)
    st.sidebar.success(
        f"runs={summary.runs} artifacts={summary.artifacts} "
        f"reports={summary.reports} candidates={summary.candidates}"
    )
    st.cache_data.clear()

if not db_path.exists():
    st.warning("No existe registry SQLite. Pulsa 'Indexar artifacts' para crear la primera version.")

runs_df = cached_runs(str(db_path)) if db_path.exists() else pd.DataFrame()
run_options = ["ALL"] + sorted(runs_df["run_id"].dropna().astype(str).unique().tolist()) if not runs_df.empty else ["ALL"]
run_filter = st.sidebar.selectbox("Run", run_options)

candidate_df = cached_candidates(str(db_path), run_filter) if db_path.exists() else pd.DataFrame()
reports_df = cached_reports(str(db_path), run_filter) if db_path.exists() else pd.DataFrame()
artifacts_df = cached_artifacts(str(db_path), run_filter) if db_path.exists() else pd.DataFrame()

if not candidate_df.empty:
    targets = sorted(candidate_df["target_symbol"].fillna("UNKNOWN").astype(str).unique()) if "target_symbol" in candidate_df else []
    timeframes = sorted(candidate_df["timeframe"].fillna("UNKNOWN").astype(str).unique()) if "timeframe" in candidate_df else []
    selected_targets = st.sidebar.multiselect("Target", targets, default=targets)
    selected_timeframes = st.sidebar.multiselect("Timeframe", timeframes, default=timeframes)
    candidate_df = filter_frame(candidate_df, "target_symbol", selected_targets)
    candidate_df = filter_frame(candidate_df, "timeframe", selected_timeframes)

tabs = st.tabs(["Decision Board", "Run Browser", "Candidate Explorer", "Reports", "Decisions"])
with tabs[0]:
    show_decision_board(candidate_df, runs_df)
with tabs[1]:
    show_run_browser(runs_df, artifacts_df, reports_df)
with tabs[2]:
    show_candidate_explorer(candidate_df, artifacts_df)
with tabs[3]:
    show_reports(reports_df)
with tabs[4]:
    show_decisions(db_path, runs_df, candidate_df)
