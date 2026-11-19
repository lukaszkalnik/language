# Tasks

Concepts and specifications for upcoming work, one file per task. The order in which they are
done is kept in [ROADMAP.md](../../ROADMAP.md), not here.

## Naming

`NNNN-short-title.md`: a four-digit number with leading zeros, then a few lowercase words joined
by hyphens, e.g. `0007-breton-support.md`.

- Numbers are given in order of creation and never reused or changed, so links stay valid.
  Priority lives in the roadmap.
- The title can be sharpened later; the number is the task's identity (refer to it as "task 0007").

## Contents

A task starts as a concept (a few lines on the problem and idea) and grows into a
specification before it is built. Use this outline and drop sections that do not apply yet:

```markdown
# NNNN Short title

Status: concept | specified | in progress | done | dropped

## Problem
What is missing or wrong, and for whom.

## Proposal
What to build; the user-visible behaviour.

## Details
Tools, models (with sizes and licences), files touched, open questions.

## Done when
Checks that show it works (tests, a real run, docs updated).
```

When a task is done, set its status to `done`, link the commits or release, and move its
roadmap entry to the Done list.
