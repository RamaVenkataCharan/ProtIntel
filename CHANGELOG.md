# Changelog

All notable changes to this project are recorded in this file.

## 2026-11-21 — Service lifecycle, launcher stability & UI telemetry

- **Service Lifecycle & Launcher**:
  - Rewrote `start_protintel.bat` and `stop_protintel.bat` in clean ASCII encoding to eliminate Windows `cmd.exe` multi-byte parsing errors.
  - Implemented an automated health polling loop in `start_protintel.bat` to verify FastAPI backend readiness before launching the browser.
  - Bound exact virtual environment Python interpreter (`.venv\Scripts\python.exe`) for deterministic execution.
  - Enhanced `stop_protintel.bat` to reliably terminate lingering background processes on ports 8000 and 5173.
- **Frontend & UI Telemetry**:
  - Optimized backend health status check in `frontend/src/components/Layout.tsx` with dynamic polling (3s during startup, 30s once online).
  - Updated DSSP Q8 evaluation metrics in `frontend/src/config/modelMetrics.ts`, `frontend/src/data/benchmarkData.ts`, and `frontend/src/pages/Evaluation.tsx`.
- **System Integration**:
  - Verified desktop shortcut generation with custom brand icon asset (`protintel.ico`).

## 2026-10-06 — Documentation refresh

- Polished and modernized README.md (hero, badges, TOC, quick-start, architecture, API, XAI, evaluation)
- Added CONTRIBUTING.md with development setup, branching workflow, testing, and PR guidelines
- Updated docs/README.md to align with the new README style and to provide a documentation overview

These updates focus on improving discoverability, onboarding, and professional presentation for both researchers and contributors.
