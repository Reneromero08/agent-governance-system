const { execSync } = require("child_process");

/* 
  AGS CANON GOVERNANCE CHECK (V1)
  
  Purpose: Enforce that meaningful system changes are accompanied by proper documentation.
  
  Categories:
  1. BEHAVIOR CHANGE → Code, tools, primitives → Requires CANON/CHANGELOG
  2. RULE CHANGE → Canon specs, contracts, invariants → Requires CANON/CHANGELOG
  3. DECISION CHANGE → ADRs, governance decisions → Requires CANON/CHANGELOG
  4. AGENTS.md SYNC → Canon changes should update AGENTS.md too → Warning only
  5. PLANNING CHANGE → Proposals/strategies → Informational only
  
  Usage:
    node check-canon-governance.js          # Normal check
    node check-canon-governance.js --verbose # Show all detected files
*/

const VERBOSE = process.argv.includes("--verbose") || process.argv.includes("-v");

function log(...args) {
    if (VERBOSE) console.log(...args);
}

function normalizePath(p) {
    return String(p || "").replace(/\\/g, "/");
}

function gitPaths(args) {
    const output = execSync(`git ${args}`, { encoding: "buffer" });
    return [...new Set(output
        .toString("utf8")
        .split("\0")
        .map((line) => normalizePath(line))
        .filter(Boolean))];
}

function getChangedPaths() {
    if (process.env.CANON_GOVERNANCE_CHANGED_PATHS) {
        return process.env.CANON_GOVERNANCE_CHANGED_PATHS
            .split(/[\r\n,]+/)
            .map((line) => normalizePath(line.trim()))
            .filter(Boolean);
    }
    try {
        if (process.env.CANON_GOVERNANCE_SCOPE === "staged") {
            return gitPaths("diff --cached --name-only -z --");
        }
        if (process.env.CANON_GOVERNANCE_DIFF_RANGE) {
            const range = process.env.CANON_GOVERNANCE_DIFF_RANGE;
            if (!/^[0-9a-fA-F.^~]+$/.test(range)) {
                throw new Error("unsafe CANON_GOVERNANCE_DIFF_RANGE");
            }
            return gitPaths(`diff --name-only -z ${range} --`);
        }
        // Get staged and unstaged changes
        const staged = execSync("git diff --cached --name-only", { encoding: "utf8" }).trim();
        const unstaged = execSync("git diff --name-only", { encoding: "utf8" }).trim();
        const combined = `${staged}\n${unstaged}`.trim();
        if (!combined) return [];
        return [...new Set(combined
            .split(/\r?\n/)
            .map((line) => normalizePath(line.trim()))
            .filter(Boolean))];
    } catch (e) {
        log("⚠️  Git diff failed, skipping checks.");
        return [];
    }
}

// === CATEGORIZATION FUNCTIONS ===

function isBehaviorChange(p) {
    if (!p) return false;

    // EXCLUDE: THOUGHT/LAB is experimental - doesn't require main changelog
    if (p.startsWith("THOUGHT/LAB/")) return false;

    // TOOLS folder (runtime behavior)
    if (p.startsWith("CAPABILITY/TOOLS/") && !p.endsWith(".md")) return true;
    if (p.startsWith("CAPABILITY/PRIMITIVES/") && !p.endsWith(".md")) return true;
    if (p.startsWith("CAPABILITY/PIPELINES/") && !p.endsWith(".md")) return true;
    if (p.startsWith("CAPABILITY/SKILLS/") && !p.endsWith(".md")) return true;

    // CI workflows (enforcement behavior)
    if (p.startsWith(".github/workflows/")) return true;

    return false;
}

function isRuleChange(p) {
    if (!p) return false;

    // Core canon documents
    if (p.startsWith("LAW/CANON/") && p.endsWith(".md")) {
        // Exclude planning and archive
        if (p.startsWith("LAW/CANON/planning/")) return false;
        if (p.startsWith("LAW/CANON/archive/")) return false;
        return true;
    }

    // SPECTRUM specifications (frozen law)
    if (p.startsWith("LAW/CONTEXT/SPECTRUM/") && p.endsWith(".md")) return true;

    // Schemas (contract definitions)
    if (p.startsWith("LAW/SCHEMAS/") && p.endsWith(".json")) return true;
    if (p.startsWith("SCHEMAS/") && p.endsWith(".json")) return true;

    return false;
}

function isDecisionChange(p) {
    if (!p) return false;

    // ADRs (Architecture Decision Records)
    if (p.startsWith("LAW/CONTEXT/decisions/") && p.endsWith(".md")) return true;

    return false;
}

function isPlanningChange(p) {
    if (!p) return false;

    // Planning docs are informational only
    if (p.startsWith("LAW/CANON/planning/")) return true;
    if (p.startsWith("LAW/CONTEXT/planning/")) return true;

    // Roadmaps (tracked separately)
    if (p.includes("ROADMAP") && p.endsWith(".md")) return true;

    return false;
}

function isChangelogUpdate(p) {
    return p === "CHANGELOG.md";
}

function isAgentsUpdate(p) {
    return p === "AGENTS.md";
}

function isCatDptChangelogUpdate(p) {
    return p === "CHANGELOG.md";
}

// === MAIN LOGIC ===

function run() {
    console.log("\n🔍 AGS Canon Governance Check (V1)\n");

    let changed;
    try {
        changed = getChangedPaths();
    } catch (error) {
        console.warn("⚠️  Canon governance check skipped (git diff failed).");
        return;
    }

    if (!changed.length) {
        console.log("✅ No uncommitted changes detected.\n");
        return;
    }

    log(`📁 Detected ${changed.length} changed file(s):\n`);
    if (VERBOSE) {
        changed.forEach(p => console.log(`   ${p}`));
        console.log("");
    }

    // Categorize changes
    const behaviorChanges = changed.filter(isBehaviorChange);
    const ruleChanges = changed.filter(isRuleChange);
    const decisionChanges = changed.filter(isDecisionChange);
    const planningChanges = changed.filter(isPlanningChange);

    const changelogUpdated = changed.some(isChangelogUpdate);
    const agentsUpdated = changed.some(isAgentsUpdate);
    const catDptChangelogUpdated = changed.some(isCatDptChangelogUpdate);

    // Determine which changelog is required
    const hasCatDptChanges = changed.some(p => p.startsWith("CATALYTIC-DPT/"));
    const hasAgsChanges = behaviorChanges.length > 0 || ruleChanges.length > 0 || decisionChanges.length > 0;

    // Summary
    const hasSignificantChanges = hasAgsChanges || hasCatDptChanges;

    // === CHECKS ===

    let hasError = false;
    let hasWarning = false;

    // Check 1: AGS changes require CHANGELOG.md
    if (hasAgsChanges && !changelogUpdated) {
        hasError = true;
        console.error("❌ CHANGELOG.md REQUIRED\n");

        if (behaviorChanges.length > 0) {
            console.error("   System behavior changes:");
            behaviorChanges.slice(0, 5).forEach(p => console.error(`     • ${p}`));
            if (behaviorChanges.length > 5) console.error(`     ... and ${behaviorChanges.length - 5} more`);
        }

        if (ruleChanges.length > 0) {
            console.error("\n   Canon rule changes:");
            ruleChanges.forEach(p => console.error(`     • ${p}`));
        }

        if (decisionChanges.length > 0) {
            console.error("\n   Decision changes:");
            decisionChanges.forEach(p => console.error(`     • ${p}`));
        }

        console.error("\n   → Update CHANGELOG.md before committing.\n");
    }

    // Check 2: CAT-DPT changes require CATALYTIC-DPT/CHANGELOG
    if (hasCatDptChanges && !catDptChangelogUpdated) {
        hasWarning = true;
        console.warn("⚠️  CATALYTIC-DPT/CHANGELOG.md WARNING\n");
        console.warn("   CAT-DPT files changed but CATALYTIC-DPT/CHANGELOG.md wasn't updated.");
        console.warn("   Consider updating if this is a significant CAT-DPT change.\n");
    }

    // Check 3: Rule changes should update AGENTS.md for navigation sync
    if (ruleChanges.length > 0 && !agentsUpdated) {
        hasWarning = true;
        console.warn("⚠️  AGENTS.MD SYNC WARNING\n");
        console.warn("   Canon docs changed but AGENTS.md wasn't updated.");
        console.warn("   Consider updating AGENTS.md if navigation/paths changed.\n");
    }

    // Info: Planning changes (no action required)
    if (planningChanges.length > 0) {
        console.log("ℹ️  Planning docs modified (no changelog required):");
        planningChanges.forEach(p => console.log(`     • ${p}`));
        console.log("");
    }

    // === RESULT ===

    if (hasError) {
        console.log("━".repeat(50));
        console.log("❌ GOVERNANCE CHECK FAILED");
        console.log("   Run with --verbose for full file list.");
        console.log("━".repeat(50) + "\n");
        process.exit(1);
    }

    if (hasWarning) {
        console.log("━".repeat(50));
        console.log("⚠️  GOVERNANCE CHECK PASSED (with warnings)");
        console.log("━".repeat(50) + "\n");
        return;
    }

    if (hasSignificantChanges && (changelogUpdated || catDptChangelogUpdated)) {
        console.log("━".repeat(50));
        console.log("✅ GOVERNANCE CHECK PASSED");
        console.log("   Changelog updated - good documentation hygiene!");
        console.log("━".repeat(50) + "\n");
        return;
    }

    console.log("✅ No governance-tracked changes detected.\n");
}

run();
