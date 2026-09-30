---
name: endreport
description: >-
  Standard End-of-Day (EOD) Log generator matching Amit Sir's reporting format (8:45 total hours). Generates clean, professional, and natural human-written daily logs.
---

# End Report Skill — Professional Human-Written EOD Log

Use this skill when the user initiates `/endreport` or requests an End-of-Day (EOD) update.

## Guidelines for Tone & Style
* **Authentic Human Developer Tone**: Write tasks clearly and naturally as a professional software engineer would report them. Avoid overly robotic, academic, or AI-sounding buzzwords (avoid phrases like "orchestrated paradigm", "transparency engine", "multi-faceted audit", "rectified anomalies").
* **Clear Action Verbs**: Use direct, practical language: *Researched*, *Implemented*, *Configured*, *Fixed issue where...*, *Refactored*, *Tested*, *Explored*, *Integrated*.
* **Strict Time Accounting**: The sum of all task hours must always total exactly **8 Hrs 45 mins** (Total Available: 0:0 Hr).
* **Realistic Breakdown**: Split the day's work into 3 to 5 clear, meaningful tasks.
* **Direct Output**: Always provide the full report directly in the chat in raw text/markdown so the user can copy-paste it immediately.

---

## Standard EOD Template Format

```markdown
EOD Update – <DD Mon YYYY>

Hello Amit Sir,
Please find below my EOD update for today:

| Task | Project | Description | Status | ETA | Hours |
| :---: | :---: | :--- | :---: | :---: | :---: |
| 1 | <Project> | <Natural, professional task description> | Completed / In Progress | - | <X.XX> Hrs |
| 2 | <Project> | <Natural, professional task description> | Completed / In Progress | - | <X.XX> Hrs |
| 3 | <Project> | <Natural, professional task description> | Completed / In Progress | - | <X.XX> Hrs |
| 4 | <Project> | <Natural, professional task description> | Completed / In Progress | - | <X.XX> Hrs |

Total Hours: 8 Hrs 45 mins
Total Available: 0:0 Hr

Blockers/Dependencies:
<Concise explanation of actual blockers or 'None'>

Plan for Tomorrow:
- <Concrete action item 1>
- <Concrete action item 2>
```
