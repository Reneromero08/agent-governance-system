# Subagent Custody Addendum

**Scope:** autonomous Small Wall work in this directory and descendants.

This addendum sharpens the subagent rules in `AGENTS.md` after a read-only design
subagent edited files and executed a live transaction. The retained result is not hidden,
but the behavior must not recur.

## Sole executor

The persistent main agent is the only agent allowed to:

- mutate the shared checkout;
- contact or administer the lab device;
- transfer files to the lab device;
- compile or execute code on the lab device;
- start stimulus or measurement processes;
- perform live-run sequencing;
- change CPU-frequency controls;
- clean remote run roots;
- commit or push;
- promote or adjudicate a scientific claim.

Live authorization in the main `/goal` belongs only to the persistent main agent. It is
never inherited by a subagent.

## Mandatory subagent prompt boundary

Every subagent prompt must include the routing header and this explicit boundary:

```text
CUSTODY: READ_ONLY
NO_CHECKOUT_MUTATION: true
NO_LAB_DEVICE_CONTACT: true
NO_GIT_WRITE: true
RETURN: analysis, equations, audit findings, experiment design, or a proposed patch as text only
```

A subagent may inspect repository content and retained evidence, search primary sources,
derive mechanisms, audit code, design experiments, or return a textual patch. It must
not use shell or connector actions that mutate files, Git state, processes, or the lab
device.

## Delegated implementation exception

The main agent may delegate implementation only when all of the following hold:

1. the scientific contract is already frozen;
2. the module is local, offline, and nonoverlapping with every other active edit;
3. the subagent has no live-device authority;
4. the subagent does not commit or push;
5. the main agent reviews and integrates the change before any execution.

Prefer textual patches over direct edits even under this exception.

## Violation response

If a subagent mutates files or performs a live action outside its assignment:

1. stop further delegation to that subagent;
2. inspect Git status, processes, device state, restoration, and raw outputs;
3. preserve the evidence and document the custody violation plainly;
4. do not promote a material live result until the main agent has audited the raw data
   and, when scientifically necessary, reproduced the result under proper custody;
5. repair the delegation prompt before continuing.

A custody violation is not permission to delete valid evidence or rewrite history.

## Parallelism law

Parallel subagents may investigate independent theory, source, or audit questions. They
must not contend for the same files or coordinate live experiments. Sol Ultra may
coordinate independent read-only investigations, but it does not receive checkout or
lab-device custody.
