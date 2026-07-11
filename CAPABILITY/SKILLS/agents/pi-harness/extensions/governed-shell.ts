import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { extname, isAbsolute, resolve } from "node:path";
import { Type } from "typebox";

const MAX_ARGS = 64;
const MAX_ARG_CHARS = 4096;
const MAX_TOTAL_ARG_CHARS = 32768;
const MAX_OUTPUT_CHARS = 50000;
const MAX_OUTPUT_LINES = 2000;

type ProgramMap = Record<string, string>;

function requiredJson<T>(name: string): T {
  const raw = process.env[name];
  if (!raw) throw new Error(`${name} is required`);
  return JSON.parse(raw) as T;
}

function requiredString(name: string): string {
  const value = process.env[name];
  if (!value) throw new Error(`${name} is required`);
  return value;
}

function inside(root: string, candidate: string): boolean {
  const normalizedRoot = resolve(root);
  const normalizedCandidate = resolve(candidate);
  const prefix = normalizedRoot.endsWith("\\") || normalizedRoot.endsWith("/")
    ? normalizedRoot
    : `${normalizedRoot}${process.platform === "win32" ? "\\" : "/"}`;
  const fold = (value: string) => process.platform === "win32" ? value.toLowerCase() : value;
  return fold(normalizedCandidate) === fold(normalizedRoot) || fold(normalizedCandidate).startsWith(fold(prefix));
}

function truncate(value: string): { text: string; truncated: boolean } {
  const lines = value.split("\n");
  let text = lines.slice(-MAX_OUTPUT_LINES).join("\n");
  let truncated = lines.length > MAX_OUTPUT_LINES;
  if (text.length > MAX_OUTPUT_CHARS) {
    text = text.slice(-MAX_OUTPUT_CHARS);
    truncated = true;
  }
  return { text, truncated };
}

function safeEnvironment(): Record<string, string> {
  const allowed = [
    "PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "COMSPEC",
    "TEMP", "TMP", "HOME", "USERPROFILE", "LANG", "LC_ALL",
  ];
  const env: Record<string, string> = { CI: "1", PYTHONDONTWRITEBYTECODE: "1" };
  for (const key of allowed) {
    const value = process.env[key];
    if (value !== undefined) env[key] = value;
  }
  return env;
}

export default function (pi: ExtensionAPI) {
  const workspace = resolve(requiredString("PI_HARNESS_WORKSPACE"));
  const writeRoots = requiredJson<string[]>("PI_HARNESS_WRITE_ROOTS").map((value) => resolve(value));
  const programs = requiredJson<ProgramMap>("PI_HARNESS_SHELL_PROGRAMS");
  const maxTimeout = Math.max(1, Math.min(300, Number(process.env.PI_HARNESS_SHELL_MAX_TIMEOUT || "120")));

  if (!workspace || writeRoots.length === 0 || Object.keys(programs).length === 0) {
    throw new Error("governed shell requires workspace, write roots, and programs");
  }
  for (const root of writeRoots) {
    if (!inside(workspace, root)) throw new Error(`write root escapes workspace: ${root}`);
  }
  for (const [alias, executable] of Object.entries(programs)) {
    if (!/^[A-Za-z0-9_-]{1,64}$/.test(alias)) throw new Error(`invalid program alias: ${alias}`);
    if (!isAbsolute(executable)) throw new Error(`program path is not absolute: ${alias}`);
    if (process.platform === "win32" && ![".exe", ".com"].includes(extname(executable).toLowerCase())) {
      throw new Error(`program is not a native Windows executable: ${alias}`);
    }
  }

  pi.registerTool({
    name: "bash",
    label: "Governed Shell",
    description: "Run one allowlisted executable with an argument array. No shell command strings, pipes, redirects, or command chaining.",
    promptSnippet: "Run an allowlisted program with explicit arguments and a workspace-confined cwd",
    promptGuidelines: [
      "Use bash only with a program alias from its allowlist and an explicit argument array.",
      "Never encode shell syntax, pipes, redirects, command chaining, or environment assignments in bash arguments.",
    ],
    parameters: Type.Object({
      program: Type.String({ description: "Allowlisted program alias" }),
      args: Type.Array(Type.String(), { maxItems: MAX_ARGS, description: "Literal argument array; not a shell command string" }),
      cwd: Type.Optional(Type.String({ description: "Working directory relative to the workspace" })),
      timeout_seconds: Type.Optional(Type.Number({ minimum: 1, maximum: 300 })),
    }),
    async execute(_toolCallId, params, signal) {
      const executable = programs[params.program];
      if (!executable) throw new Error(`program is not allowlisted: ${params.program}`);
      if (params.args.some((arg: string) => arg.length > MAX_ARG_CHARS)) {
        throw new Error(`argument exceeds ${MAX_ARG_CHARS} characters`);
      }
      const totalChars = params.args.reduce((total: number, arg: string) => total + arg.length, 0);
      if (totalChars > MAX_TOTAL_ARG_CHARS) throw new Error("argument payload is too large");

      const cwd = resolve(workspace, params.cwd || ".");
      if (!inside(workspace, cwd)) throw new Error(`cwd escapes workspace: ${params.cwd}`);
      const timeoutSeconds = Math.max(1, Math.min(maxTimeout, Number(params.timeout_seconds || maxTimeout)));
      const result = await pi.exec(executable, params.args, {
        cwd,
        env: safeEnvironment(),
        signal,
        timeout: timeoutSeconds * 1000,
      });
      const stdout = truncate(result.stdout || "");
      const stderr = truncate(result.stderr || "");
      const text = [
        `program: ${params.program}`,
        `cwd: ${cwd}`,
        `exit_code: ${result.code}`,
        stdout.text ? `stdout:\n${stdout.text}` : "stdout: (empty)",
        stderr.text ? `stderr:\n${stderr.text}` : "stderr: (empty)",
        stdout.truncated || stderr.truncated ? "[output truncated]" : "",
      ].filter(Boolean).join("\n");
      if (result.code !== 0) throw new Error(text);
      return {
        content: [{ type: "text", text }],
        details: {
          program: params.program,
          executable,
          args: params.args,
          cwd,
          exitCode: result.code,
          stdoutTruncated: stdout.truncated,
          stderrTruncated: stderr.truncated,
        },
      };
    },
  });
}
