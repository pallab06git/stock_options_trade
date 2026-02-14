# Initial Prompt for Claude Code

## Project: SPY Options Intelligence Platform

Copy and paste this prompt when starting Claude Code:

---

```
I have a SPY Options Intelligence data ingestion platform project.

CRITICAL - Architecture First:
Before implementing any code, you MUST:
1. Read CLAUDE.md for development constraints and workflow
2. Read SCOPE.md for Phase 1 requirements
3. Read SECURITY.md for credential handling requirements
4. Review the current project structure

Then:
5. Generate a comprehensive Technical Architecture Proposal using the template in STEP_0_ARCHITECTURE_PROPOSAL.md
6. Include:
   - System architecture diagram
   - Technology stack for each layer (with justification)
   - Component specifications (library choices, error handling, performance)
   - Data flow diagrams
   - Trade-offs and decisions
   - Risk assessment
   - Phased development plan
   - Testing strategy

7. STOP and wait for architecture approval

DO NOT implement any code until architecture is approved.

Project location: [your local path]/spy-options-intelligence/
```

---

## Expected Workflow

### Step 0: Architecture Proposal
**Claude Code will:**
1. Analyze requirements from CLAUDE.md, SCOPE.md, SECURITY.md
2. Review existing project structure
3. Generate comprehensive architecture proposal
4. Propose technology choices for each module
5. Present phased development plan
6. STOP for review

**You will:**
1. Review architecture proposal
2. Ask questions or request changes
3. Approve architecture (or request revisions)

### Step 1+: Implementation
**Claude Code will:**
1. Implement Step 1 (Configuration System) using approved architecture
2. Write unit tests
3. Generate Task Completion Report
4. STOP for review

**You will:**
1. Review implementation
2. Run tests
3. Approve and say "continue with Step 2" (or request changes)

### Iterative Process
Repeat for Steps 2-20, one at a time.

---

## What Makes This Different

**Old approach:** 
- Jump straight to coding
- Risk of architectural mistakes
- Harder to refactor later

**New approach:**
- Architecture approved upfront
- Clear technology choices
- Phased implementation
- Lower risk, higher quality

---

## Key Documents

1. **CLAUDE.md** - Development rules (architecture-first, stop after each step)
2. **SCOPE.md** - Requirements (APIs, data sources, failure handling)
3. **SECURITY.md** - Security rules (credential redaction, no secrets in logs)
4. **STEP_0_ARCHITECTURE_PROPOSAL.md** - Template for architecture proposal

---

## After Architecture Approval

Claude Code will follow the 20-step implementation plan:
- Step 1: Configuration System
- Step 2: Logging Infrastructure
- Step 3: Retry & Connection Management
- ... (continue through Step 20)

Each step includes:
- Implementation
- Testing
- Task Completion Report
- STOP for review

---

**Start with the initial prompt above. Claude Code will handle the rest.**
