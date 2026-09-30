---
name: ask
description: >-
  Interactive clarification and requirements gathering skill. Use when the user
  types /ask or requests clarification, design choices, requirement grilling, or option evaluation before proceeding with a task.
---

# Ask Skill - Requirements & Clarification

Use this skill when the user initiates a task with `/ask` or when requirements are underspecified, ambiguous, or involve high-impact architectural choices.

## Objective
Gather clear, unambiguous requirements, resolve trade-offs, and align on technical design choices through structured interactive Q&A before planning or writing code.

---

## Workflow

### 1. Requirement Analysis & Discovery
* Read the user's initial prompt and examine relevant codebase files to understand existing patterns and context.
* Identify key missing details, edge cases, trade-offs, or architectural decisions:
  * Business logic rules & validation
  * Data schema & database migrations
  * API input/output contracts
  * UI/UX preferences or performance constraints

### 2. Formulate Structured Questions
* Formulate clear, concise questions focused on actionable decisions.
* When asking multiple-choice options:
  * Present logical, well-structured choices.
  * Highlight recommended choices with `(Recommended)` and brief rationale.
  * Keep technical details concise.

### 3. Requirements Summary
* Once responses are collected, synthesize the decisions into a clear **Requirements Summary**:
  * **Goal**: Primary objective agreed upon.
  * **Scope**: In-scope vs. out-of-scope boundaries.
  * **Technical Specs**: Data models, APIs, and business rules finalized.
* Transition seamlessly by offering to initiate `/plan` for the confirmed specifications.
