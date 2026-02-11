---
name: cursor-plan-mode
description: Create, manage, and update multi-step execution plans for complex tasks in Cursor. Use when the user asks for a plan/roadmap/phases/checklist, or when work has multiple dependent steps, higher risk, or requires coordination. Skip for straightforward tasks. Integrates with Cursor's todo_write tool for plan tracking.
---

# Cursor Plan Mode

## Overview

Provide a concise workflow for producing actionable plans and keeping them updated during execution in Cursor IDE. This skill integrates with Cursor's native todo management system for seamless plan tracking.

## Decide Whether to Plan

- Use plan mode when the task is multi-step, risky, cross-cutting, or explicitly requests a plan.
- Skip planning for trivial or single-action tasks.
- Consider plan mode for tasks requiring:
  - Multiple file edits
  - Complex refactoring
  - Feature development with dependencies
  - Integration work across components
  - Testing and validation workflows

## Create the Plan

### Plan Structure

- Identify the smallest set of steps that cover the full task end-to-end.
- Use 3-7 steps; avoid single-step plans.
- Write outcome-based steps (verb + object), not low-level implementation details.
- Include a verification step when testing or validation is expected.
- Initialize all steps as pending; set only one step to in_progress at a time.

### Using todo_write Tool

When creating a plan, use the `todo_write` tool to track progress:

```python
# Example plan creation
todos = [
    {"id": "1", "content": "Analyze current codebase structure", "status": "pending"},
    {"id": "2", "content": "Design new feature architecture", "status": "pending"},
    {"id": "3", "content": "Implement core functionality", "status": "pending"},
    {"id": "4", "content": "Add unit tests", "status": "pending"},
    {"id": "5", "content": "Update documentation", "status": "pending"}
]

todo_write(merge=False, todos=todos)
```

### Plan Best Practices

1. **Clear Step Names**: Use specific, actionable step names
   - ✅ Good: "Refactor authentication module to use JWT"
   - ❌ Bad: "Fix auth stuff"

2. **Logical Ordering**: Arrange steps in dependency order
   - Dependencies should be completed before dependent tasks
   - Parallel tasks can be marked as such

3. **Verification Steps**: Always include validation
   - Testing steps
   - Code review checkpoints
   - Integration verification

4. **Resource Awareness**: Consider available resources
   - File system constraints
   - Tool limitations
   - Time estimates

## Execute and Update

### Progress Tracking

- Move the current step to completed when done.
- Set the next step to in_progress before starting it.
- Update todos using `todo_write` with `merge=True`:

```python
# Mark step as completed and start next
todo_write(merge=True, todos=[
    {"id": "1", "status": "completed"},
    {"id": "2", "status": "in_progress"}
])
```

### Plan Revision

- Revise the plan if scope changes; keep the explanation brief and focused.
- Avoid re-planning unless the change is material.
- Document significant changes in plan comments.

### Error Handling

- If a step fails, mark it appropriately
- Create recovery steps if needed
- Update plan to reflect new understanding

## Output Style

- Keep the plan concise and scannable.
- Prefer short, specific step names over long sentences.
- Do not include unrelated commentary inside the plan.
- Use markdown formatting for clarity:
  - **Bold** for step names
  - `Code blocks` for technical details
  - Lists for sub-tasks

## Integration with Cursor Features

### File Management

- Plan file edits in logical groups
- Consider file dependencies
- Batch related changes together

### Terminal Integration

- Plan command execution steps
- Consider command dependencies
- Include verification commands

### Code Analysis

- Plan refactoring with codebase search
- Use semantic search for understanding
- Plan incremental changes

## Examples of Triggers

### Explicit Plan Requests
- "Make a phased plan to merge branches and keep custom features."
- "Give me a roadmap to refactor the orchestration system."
- "Create a checklist for a release and verification run."

### Implicit Plan Needs
- "Add authentication to the entire app" (multi-step, cross-cutting)
- "Refactor the database layer" (risky, requires coordination)
- "Implement a new feature with tests" (multiple dependent steps)

## Example Plans

### Feature Development Plan

```python
# Create feature development plan
todos = [
    {
        "id": "feature-1",
        "content": "Design feature architecture and API contracts",
        "status": "pending"
    },
    {
        "id": "feature-2",
        "content": "Implement core business logic",
        "status": "pending"
    },
    {
        "id": "feature-3",
        "content": "Create database migrations",
        "status": "pending"
    },
    {
        "id": "feature-4",
        "content": "Add API endpoints",
        "status": "pending"
    },
    {
        "id": "feature-5",
        "content": "Write unit and integration tests",
        "status": "pending"
    },
    {
        "id": "feature-6",
        "content": "Update API documentation",
        "status": "pending"
    }
]
```

### Refactoring Plan

```python
# Create refactoring plan
todos = [
    {
        "id": "refactor-1",
        "content": "Analyze current code structure and identify refactoring targets",
        "status": "pending"
    },
    {
        "id": "refactor-2",
        "content": "Create new abstraction layer",
        "status": "pending"
    },
    {
        "id": "refactor-3",
        "content": "Migrate existing code to use new abstraction",
        "status": "pending"
    },
    {
        "id": "refactor-4",
        "content": "Run test suite to verify no regressions",
        "status": "pending"
    },
    {
        "id": "refactor-5",
        "content": "Remove deprecated code",
        "status": "pending"
    }
]
```

## Advanced Features

### Parallel Execution

When steps can run in parallel, mark them appropriately:

```python
# Parallel steps example
todos = [
    {"id": "1", "content": "Task A (can run in parallel)", "status": "pending"},
    {"id": "2", "content": "Task B (can run in parallel)", "status": "pending"},
    {"id": "3", "content": "Task C (depends on A and B)", "status": "pending"}
]
```

### Conditional Steps

Document conditional logic in plan:

```python
# Conditional steps
todos = [
    {
        "id": "1",
        "content": "If feature flag enabled: implement new logic, else: keep existing",
        "status": "pending"
    }
]
```

### Checkpoint Management

For long-running plans, create checkpoints:

```python
# Checkpoint example
todos = [
    {"id": "checkpoint-1", "content": "Phase 1 complete - verify before proceeding", "status": "pending"}
]
```

## Best Practices Summary

1. ✅ **Start with analysis**: Understand before planning
2. ✅ **Keep it simple**: 3-7 steps maximum
3. ✅ **Be specific**: Clear, actionable step names
4. ✅ **Track progress**: Update todos regularly
5. ✅ **Verify results**: Include validation steps
6. ✅ **Adapt flexibly**: Revise when needed, but avoid over-planning
7. ✅ **Document changes**: Explain significant plan modifications

## Anti-Patterns to Avoid

- ❌ Creating plans for trivial tasks
- ❌ Over-detailed implementation steps
- ❌ Ignoring plan updates during execution
- ❌ Creating plans without clear outcomes
- ❌ Forgetting verification steps

---

*This plan mode skill is optimized for Cursor IDE's workflow and integrates seamlessly with Cursor's native todo management system.*
