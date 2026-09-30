---
name: implement
description: >-
  Code execution, verification, and walkthrough skill. Use when the user
  types /implement or requests execution of an approved implementation plan.
---

# Implement Skill - Controlled Execution & Verification

Use this skill when the user triggers `/implement` or instructs the agent to execute an approved implementation plan.

## Objective
Execute code modifications systematically according to the approved plan, enforce codebase safety guidelines, run verification tests, and provide a walkthrough.

---

## Workflow

### 1. Plan Verification
* Locate and review the existing `implementation_plan.md` artifact.
* Confirm that design choices and component modification lists are clear and approved.

### 2. Precise Code Execution
* Modify files using targeted edits (`replace_file_content` or `write_to_file`).
* Adhere strictly to engineering guidelines:
  * Preserve all unrelated existing comments and docstrings.
  * Verify method signatures, prop names, and imported schemas across all call sites.
  * Do **NOT** mask runtime errors with empty catch blocks or dummy fallbacks.
  * Scope code changes cleanly to affected branches.

### 3. Build & Test Verification
* Execute build and test commands (e.g., `dotnet build`, `dotnet test`).
* If any errors occur:
  * Fetch and analyze complete compiler diagnostic or stack trace output.
  * Resolve root cause directly—never delete failing tests or ignore error exit codes.

### 4. Walkthrough & Documentation
* Create or update the `walkthrough.md` artifact summarizing:
  * **Changes Made**: File-by-file summary of implemented code.
  * **Verification Results**: Command output, build logs, and test outcome confirmation.
* Summarize key completed work for the user in clean markdown.
