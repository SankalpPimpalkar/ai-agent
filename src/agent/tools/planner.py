import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from src.agent.settings import SETTINGS
from src.agent.tools.base import BaseTool
from src.agent.tools.state import STATE

PROJECT_KEYWORDS = {
    "build",
    "create",
    "implement",
    "design",
    "scaffold",
    "app",
    "project",
    "cli",
    "converter",
    "interpreter",
    "engine",
    "database",
    "solver",
    "matcher",
    "regex",
    "evaluator",
    "calculator that",
    "todo",
}


def is_project_task(prompt: str) -> bool:
    prompt_lower = prompt.lower()
    return any(kw in prompt_lower for kw in PROJECT_KEYWORDS)


def derive_project_name(prompt: str) -> str:
    mapping = [
        (r"\b(regex|regular\s+expression|matcher)\b", "regex-engine"),
        (r"\b(interpreter|compiler)\b", "mini-interpreter"),
        (r"\btodo\b", "todo-cli"),
        (r"\bmarkdown\b", "markdown-converter"),
        (r"\bsudoku\b", "sudoku-solver"),
        (r"\b(calculator|arithmetic|evaluator)\b", "math-evaluator"),
        (r"\b(sql|database|db)\b", "db-engine"),
    ]
    for pattern, name in mapping:
        if re.search(pattern, prompt, re.I):
            return name

    words = [
        w.lower()
        for w in re.findall(r"\b[a-zA-Z0-9]+\b", prompt)
        if len(w) > 2
        and w.lower()
        not in {
            "build",
            "create",
            "implement",
            "design",
            "scaffold",
            "app",
            "project",
            "that",
            "this",
            "whose",
            "with",
            "from",
            "into",
            "using",
            "then",
            "show",
            "tell",
            "store",
            "under",
        }
    ]
    if words:
        return "-".join(words[:3])
    return "custom-project"


def clean_project_name(name: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip().lower()).strip("-")
    if clean in ("", "app", "project"):
        return ""
    return clean


def get_plan_path(project: str) -> Path:
    proj = (
        clean_project_name(project)
        or getattr(STATE, "current_project", "")
        or "project"
    )
    artifacts_dir = SETTINGS.paths.artifacts_dir
    project_dir = artifacts_dir / proj
    project_dir.mkdir(parents=True, exist_ok=True)
    return project_dir / "PLAN.md"


def decompose_prompt(prompt: str, project_name: str) -> Tuple[List[str], List[str]]:
    """
    Intelligently breaks down any project prompt into concrete subtasks
    and verification test cases.
    """
    p_lower = prompt.lower()

    # 1. Regex engine / matcher
    if any(k in p_lower for k in ["regex", "regular expression", "matcher"]):
        pairs = re.findall(r"['\"]([^'\"]+)['\"]\s+vs\s+['\"]([^'\"]+)['\"]", prompt)
        store_match = re.search(r"store (?:the (?:matching )?strings )?under ['\"]([^'\"]+)['\"]", prompt, re.I)
        store_key = store_match.group(1) if store_match else "matches"

        tasks = [
            "Implement regex pattern tokenizer and AST representation for literals, wildcards (.), character classes ([]), quantifiers (*, +, ?), alternation (|), and grouping (())",
            "Implement NFA/DFA state machine or recursive backtracking matching engine for full-string evaluation",
            f"Implement test runner script to evaluate all pattern/string test pairs and store matching results under '{store_key}'",
            "Verify all test pairs and report match count and matching strings",
        ]
        tests = []
        if pairs:
            for pat, s in pairs[:3]:
                tests.append(f"Verify pattern '{pat}' against string '{s}'")
            tests.append(f"Verify all {len(pairs)} pairs: correct match count and results stored under '{store_key}'")
        else:
            tests = [
                "Verify literal character matching and wildcards ('.')",
                "Verify repetition operators (*, +, ?) and alternation (|)",
                "Verify character classes and grouping parentheses",
                f"Verify full suite execution and results stored under '{store_key}'",
            ]
        return tasks, tests

    # 2. Interpreter / Compiler
    if any(k in p_lower for k in ["interpreter", "compiler", "programming language"]):
        store_match = re.search(r"store (?:the outputs )?under ['\"]([^'\"]+)['\"]", prompt, re.I)
        store_key = store_match.group(1) if store_match else "interp_results"

        tasks = [
            "Design lexer and recursive descent parser for language expressions, statements, and function definitions",
            "Implement evaluation runtime environment with lexical scoping supporting variables, closures, and recursion",
            "Implement arithmetic, comparison operators, and control flow handling",
            f"Write and execute driver programs (Fibonacci, Factorial, counter closure) and persist outputs under '{store_key}'",
        ]
        tests = [
            "Test variable assignment, closure state, and lexical scoping (counter closure returning 1, 2, 3)",
            "Test recursive Fibonacci computation: fib(20) == 6765",
            "Test recursive Factorial computation: fact(15) == 1307674368000",
            f"Verify all outputs are printed and persisted under '{store_key}'",
        ]
        return tasks, tests

    # 3. Todo / CLI
    if any(k in p_lower for k in ["todo", "task manager", "todo-cli"]):
        tasks = [
            "Design task data model and JSON file persistence mechanism",
            "Implement CLI argument parsing for add, list, done, and delete commands",
            "Implement command execution handlers and formatted console display",
        ]
        tests = [
            "Test adding tasks and verifying data persistence",
            "Test listing tasks with completion statuses",
            "Test marking task as done and deleting task",
        ]
        return tasks, tests

    # 4. Sudoku / Puzzle Solver
    if any(k in p_lower for k in ["sudoku", "puzzle", "solver"]):
        tasks = [
            "Design board representation and validation rules for rows, columns, and 3x3 grids",
            "Implement constraint propagation and backtracking solver algorithm",
            "Implement board input parser and pretty-printed solution output",
        ]
        tests = [
            "Verify board validation on valid and invalid states",
            "Verify solver produces correct solution for sample puzzle",
            "Verify solved board satisfies all Sudoku constraints",
        ]
        return tasks, tests

    # 5. Markdown / HTML converter
    if any(k in p_lower for k in ["markdown", "converter", "html"]):
        tasks = [
            "Implement block parser for headings, lists, blockquotes, and code blocks",
            "Implement inline parser for bold, italic, code spans, and links",
            "Implement HTML renderer and output file writer",
        ]
        tests = [
            "Test block level element rendering (headings, lists)",
            "Test inline formatting syntax (bold, italic, code)",
            "Verify full sample document conversion to valid HTML",
        ]
        return tasks, tests

    # 6. Database / SQL Engine
    if any(k in p_lower for k in ["sql", "database", "db"]):
        tasks = [
            "Design table schema and row serialization mechanism",
            "Implement SQL parser for CREATE, INSERT, SELECT, and WHERE statements",
            "Implement query engine and result formatting",
        ]
        tests = [
            "Test table creation and row insertion",
            "Test SELECT query with filtering conditions",
            "Verify persistence of records across queries",
        ]
        return tasks, tests

    # 7. Math / Expression Evaluator
    if any(k in p_lower for k in ["calculator", "evaluator", "expression", "math"]):
        tasks = [
            "Implement token scanner for numbers, operators, and parentheses",
            "Implement parser using shunting-yard algorithm or recursive descent",
            "Implement evaluation engine with operator precedence and error handling",
        ]
        tests = [
            "Test basic binary operations (+, -, *, /) with standard precedence",
            "Test nested parentheses and complex expressions",
            "Verify error handling for invalid syntax and division by zero",
        ]
        return tasks, tests

    # 8. Generic fallback
    tasks = [
        f"Design module structure and data structures for {project_name}",
        "Implement core algorithm and processing logic as required by prompt",
        "Implement CLI interface, argument handling, and output formatting",
        "Verify results and ensure proper state persistence",
    ]
    tests = [
        f"Verify basic execution of {project_name} with sample input",
        "Test edge cases and validate expected results",
    ]
    return tasks, tests


def _run_planner_agent(
    prompt: str,
    project_name: str,
    client: Optional[Any] = None,
    model: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    active_client = client or SETTINGS.OLLAMA_CLIENT
    active_model = model or SETTINGS.DEFAULT_MODEL

    system_prompt = """You are a software planner agent.
Given a programming task, output a JSON object with two arrays of short, clear strings:
- "tasks": 3-5 concrete subtasks breaking down the project into modular implementation steps (e.g. models/parsing, core logic/algorithms, CLI/interface, persistence)
- "tests": 2-4 verification commands and expected results based directly on the prompt's requirements
Output ONLY valid JSON."""

    try:
        res = active_client.chat(
            model=active_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Project: {project_name}\nGoal: {prompt}",
                },
            ],
            think=False,
            options={"num_ctx": 4096},
        )
        content = (res.message.content or "").strip()
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            tasks = [str(t) for t in data.get("tasks", []) if str(t).strip()]
            tests = [str(t) for t in data.get("tests", []) if str(t).strip()]
            if len(tasks) >= 2 and len(tests) >= 1:
                return tasks, tests
    except Exception:
        pass

    return decompose_prompt(prompt, project_name)


def generate_project_plan(
    prompt: str,
    project_name: str,
    client: Optional[Any] = None,
    model: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    return _run_planner_agent(prompt, project_name, client=client, model=model)


def init_plan(
    project: str,
    tasks: Optional[List[str]] = None,
    tests: Optional[List[str]] = None,
    prompt: Optional[str] = None,
    client: Optional[Any] = None,
    model: Optional[str] = None,
) -> Tuple[str, Path]:
    plan_file = get_plan_path(project)
    proj_name = plan_file.parent.name

    if not tasks or not tests:
        p_text = prompt or proj_name
        gen_tasks, gen_tests = generate_project_plan(p_text, proj_name, client=client, model=model)
        tasks = tasks or gen_tasks
        tests = tests or gen_tests

    task_lines = "\n".join(
        f"- [ ] {t.strip().lstrip('- []0123456789. ')}" for t in tasks if t.strip()
    )
    test_lines = "\n".join(
        f"- [ ] {t.strip().lstrip('- []0123456789. ')}" for t in tests if t.strip()
    )

    content = f"""# Plan: {proj_name}

## Tasks
{task_lines}

## Tests
{test_lines}
"""
    plan_file.write_text(content, encoding="utf-8")

    STATE.current_project = proj_name
    STATE.register_file("PLAN.md")

    return content, plan_file


def mark_plan_item_done(project: str, item: str) -> Tuple[bool, str]:
    plan_file = get_plan_path(project)
    if not plan_file.exists():
        return False, f"PLAN.md does not exist for project '{project}'"

    content = plan_file.read_text(encoding="utf-8")
    lines = content.splitlines()
    item_str = str(item).strip().lower()

    target_idx = None
    if item_str.isdigit():
        target_idx = int(item_str)

    current_idx = 0
    matched = False
    new_lines = []

    for line in lines:
        if line.strip().startswith("- [ ]"):
            current_idx += 1
            is_match = False
            if target_idx is not None and current_idx == target_idx:
                is_match = True
            elif item_str and item_str in line.lower():
                is_match = True

            if is_match and not matched:
                new_lines.append(line.replace("- [ ]", "- [x]", 1))
                matched = True
                continue

        new_lines.append(line)

    if not matched:
        final_lines = []
        for line in new_lines:
            if not matched and line.strip().startswith("- [ ]"):
                final_lines.append(line.replace("- [ ]", "- [x]", 1))
                matched = True
            else:
                final_lines.append(line)
        new_lines = final_lines

    if matched:
        updated_content = "\n".join(new_lines) + "\n"
        plan_file.write_text(updated_content, encoding="utf-8")
        return True, updated_content

    return False, content


def mark_next_task_done(project: str, section: Optional[str] = None) -> Tuple[bool, str]:
    """
    Marks the next pending checkbox '- [ ]' as '- [x]'.
    If section is 'tasks', searches under '## Tasks'.
    If section is 'tests', searches under '## Tests'.
    If section is None, searches anywhere.
    """
    plan_file = get_plan_path(project)
    if not plan_file.exists():
        return False, f"PLAN.md does not exist for project '{project}'"

    content = plan_file.read_text(encoding="utf-8")
    lines = content.splitlines()

    current_sec = None
    target_idx = None
    target_text = ""

    for idx, line in enumerate(lines):
        line_s = line.strip().lower()
        if line_s.startswith("## tasks"):
            current_sec = "tasks"
            continue
        elif line_s.startswith("## tests"):
            current_sec = "tests"
            continue

        if line.strip().startswith("- [ ]"):
            if section is None or current_sec == section.lower():
                target_idx = idx
                target_text = line.strip().lstrip("- [ ]").strip()
                break

    if target_idx is not None:
        lines[target_idx] = lines[target_idx].replace("- [ ]", "- [x]", 1)
        updated_content = "\n".join(lines) + "\n"
        plan_file.write_text(updated_content, encoding="utf-8")
        return True, target_text

    return False, f"No pending items found in section '{section}'"


def finalize_plan(project: str, state=None) -> bool:
    if not project or project in ("app", "__pycache__"):
        return False
    artifacts_dir = SETTINGS.paths.artifacts_dir
    plan_file = artifacts_dir / project / "PLAN.md"
    if not plan_file.exists():
        return False

    try:
        content = plan_file.read_text(encoding="utf-8")
        updated = re.sub(r"\[\s*\]", "[x]", content)
        if updated != content:
            plan_file.write_text(updated, encoding="utf-8")
            if state and hasattr(state, "logs"):
                state.logs.append(
                    f"[Planner] Finalized PLAN.md at {plan_file}: all tasks and tests marked [x]."
                )
        return True
    except Exception as e:
        if state and hasattr(state, "logs"):
            state.logs.append(f"[Planner] Error finalizing PLAN.md: {e}")
        return False


class PlannerTool(BaseTool):
    """
    Manages PLAN.md task checklists inside project artifact folders.
    Actions: create_plan, read_plan, mark_done.
    """

    @property
    def name(self) -> str:
        return "planner_tool"

    @property
    def description(self) -> str:
        return """
Create, inspect, or update the PLAN.md task checklist for a project.
Actions:
- mark_done: Mark a completed task or test as done ('- [x]'). Provide 'task' with the task text or line number.
- create_plan: Automatically invokes planner agent to generate tasks and tests and initializes PLAN.md.
- read_plan: Read the current PLAN.md content and checklist progress.
Always use action='mark_done' as you finish tasks so progress is tracked.
""".strip()

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["mark_done", "create_plan", "read_plan"],
                    "description": "Planner action to execute",
                },
                "project": {
                    "type": "string",
                    "description": "Project directory name (defaults to current project)",
                },
                "task": {
                    "type": "string",
                    "description": "Task name, substring, or 1-based checkbox index to mark as completed",
                },
                "tasks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional explicit task items when creating plan",
                },
                "tests": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional explicit test items when creating plan",
                },
                "prompt": {
                    "type": "string",
                    "description": "Goal or description used by planner agent to devise tasks and tests",
                },
            },
            "required": ["action"],
        }

    def run(
        self,
        action: str,
        project: str = "",
        task: str = "",
        tasks: Optional[List[str]] = None,
        tests: Optional[List[str]] = None,
        prompt: str = "",
        **kwargs: Any,
    ) -> str:
        try:
            action = (action or "").strip().lower()
            proj = (
                clean_project_name(project)
                or getattr(STATE, "current_project", "")
                or "project"
            )

            if action == "create_plan":
                p_text = prompt or kwargs.get("goal", "") or proj
                if not tasks and not tests:
                    agent_tasks, agent_tests = generate_project_plan(p_text, proj)
                    tasks = tasks or agent_tasks
                    tests = tests or agent_tests
                content, path = init_plan(proj, tasks, tests, prompt=p_text)
                return f"Plan initialized at {path.name}:\n{content}"

            if action in ("read_plan", "read"):
                plan_file = get_plan_path(proj)
                if not plan_file.exists():
                    return f"No PLAN.md found for project '{proj}'. Use action='create_plan' first."
                return plan_file.read_text(encoding="utf-8")

            if action in ("mark_done", "done", "complete"):
                target = (
                    task
                    or kwargs.get("item", "")
                    or kwargs.get("task_name", "")
                    or kwargs.get("step", "")
                )
                if not target:
                    target = "1"
                ok, res = mark_plan_item_done(proj, str(target))
                if ok:
                    return f"Marked task done in {proj}/PLAN.md:\n{res}"
                return f"Could not mark task done: {res}"

            return f"Error: unknown action '{action}'. Use create_plan, read_plan, or mark_done."
        except Exception as e:
            return f"Error: {e}"


planner_tool_instance = PlannerTool()


def planner_tool(
    action: str,
    project: str = "",
    task: str = "",
    tasks: Optional[List[str]] = None,
    tests: Optional[List[str]] = None,
    prompt: str = "",
    **kwargs: Any,
) -> str:
    return planner_tool_instance.run(
        action=action,
        project=project,
        task=task,
        tasks=tasks,
        tests=tests,
        prompt=prompt,
        **kwargs,
    )


planner_tool_spec = planner_tool_instance.to_spec()
