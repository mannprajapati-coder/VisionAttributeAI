---
name: plan
description: >-
  Architecture research and implementation planning skill. Use when the user
  types /plan or requests a technical design, dependency analysis, and implementation roadmap prior to making code edits.
---

# Plan Skill - Research & Implementation Planning

Use this skill when the user initiates work with `/plan` or asks to analyze the codebase and design an implementation plan before writing code.

## Objective
Thoroughly inspect the repository, trace execution flows, analyze dependencies, and document a structured implementation plan without making premature code modifications.

---

## Workflow

### 1. Read-Only Research Phase
* Search and view relevant source files, configuration schemas, models, controllers, and services.
* Do **NOT** modify source code or execute destructive write operations during research.
* Trace dependencies and identify potential breaking changes or API contract adjustments.

### 2. Formulate Implementation Plan
* Create or update the `implementation_plan.md` artifact in the conversation artifact directory.
* Structurally outline:
  1. **Goal Description**: Clear summary of problem and target state.
  2. **User Review Required**: Highlight critical design choices, breaking changes, or risks using GitHub alert callouts (`> [!IMPORTANT]`, `> [!WARNING]`).
  3. **Open Questions**: Clarifying points if any minor trade-offs remain.
  4. **Proposed Changes**: Grouped by component with exact file references:
     - `#### [MODIFY] [file basename](file:///path/to/file)`
     - `#### [NEW] [file basename](file:///path/to/file)`
     - `#### [DELETE] [file basename](file:///path/to/file)`
  5. **Verification Plan**: Commands for building (`dotnet build`), running unit tests, and manual verification steps.

### 3. Pause for User Approval
* Present the generated plan to the user clearly.
* **Stop and wait** for the user's explicit approval before proceeding to `/implement`.
