"""Hybrid analyzer: (1) compiler stage - real JS parser, (2) static rules / scope analysis,
(3) machine-learning line classifier."""
import os, re, joblib, esprima

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")
_model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

GLOBALS = set("""console window document Math JSON Object Array String Number Boolean Date RegExp Error TypeError RangeError
Promise Map Set WeakMap WeakSet Symbol parseInt parseFloat isNaN isFinite setTimeout setInterval clearTimeout clearInterval
fetch alert confirm prompt localStorage sessionStorage navigator location history undefined NaN Infinity require module exports
process Buffer globalThis eval encodeURIComponent decodeURIComponent URL Intl arguments performance requestAnimationFrame
Reflect Proxy BigInt FormData Event Image XMLHttpRequest atob btoa""".split())

ML_MESSAGES = {
    "syntax": "Likely syntax error (malformed statement)",
    "type_error": "Likely runtime TypeError (property/call on invalid value)",
    "logic_error": "Likely logic error (assignment in condition, bad loop bound, NaN compare...)",
    "bad_practice": "Risky or discouraged practice",
}

def err_info(e):
    msg = getattr(e, "description", None) or getattr(e, "message", str(e))
    msg = re.sub(r"^Line \d+: ", "", str(msg))
    return getattr(e, "lineNumber", 1) or 1, max(0, (getattr(e, "column", 1) or 1) - 1), msg

def issue(line, col, sev, kind, msg, source):
    return {"line": line, "column": col, "severity": sev, "type": kind, "message": msg, "source": source}

def walk(node, parent=None, key=None):
    if isinstance(node, dict):
        if "type" in node: yield node, parent, key
        for k, v in node.items():
            yield from walk(v, node, k)
    elif isinstance(node, list):
        for v in node: yield from walk(v, parent, key)

def pattern_names(p):
    if not p: return []
    t = p.get("type")
    if t == "Identifier": return [p["name"]]
    if t == "ObjectPattern": return [n for pr in p["properties"] for n in pattern_names(pr.get("value") or pr.get("argument"))]
    if t == "ArrayPattern": return [n for e in p["elements"] for n in pattern_names(e)]
    if t == "AssignmentPattern": return pattern_names(p["left"])
    if t == "RestElement": return pattern_names(p["argument"])
    return []

def static_analysis(tree):
    out, declared, consts, used, uses = [], set(), set(), {}, []
    for n, parent, key in walk(tree):
        t = n["type"]; loc = n.get("loc", {}).get("start", {"line": 1, "column": 0})
        if t == "VariableDeclarator":
            for nm in pattern_names(n["id"]): declared.add(nm)
        elif t in ("FunctionDeclaration", "FunctionExpression", "ArrowFunctionExpression", "ClassDeclaration"):
            if n.get("id"): declared.add(n["id"]["name"])
            for p in n.get("params", []): declared.update(pattern_names(p))
        elif t == "CatchClause": declared.update(pattern_names(n.get("param")))
        elif t in ("ImportSpecifier", "ImportDefaultSpecifier", "ImportNamespaceSpecifier"):
            declared.add(n["local"]["name"])
        if t == "VariableDeclaration" and n["kind"] == "const":
            for d in n["declarations"]: consts.update(pattern_names(d["id"]))
        if t == "Identifier":
            is_prop = parent and ((parent["type"] == "MemberExpression" and key == "property" and not parent.get("computed"))
                      or (parent["type"] in ("Property", "MethodDefinition") and key == "key" and not parent.get("computed"))
                      or parent["type"] in ("LabeledStatement", "BreakStatement", "ContinueStatement"))
            is_decl = parent and parent["type"] in ("VariableDeclarator", "FunctionDeclaration", "FunctionExpression", "ClassDeclaration",
                      "CatchClause", "ImportSpecifier", "ImportDefaultSpecifier", "ImportNamespaceSpecifier") and key in ("id", "param", "local")
            if not is_prop and not is_decl:
                uses.append((n["name"], loc, parent, key))
    for name, loc, parent, key in uses:
        used[name] = used.get(name, 0) + 1
        if name not in declared and name not in GLOBALS:
            out.append(issue(loc["line"], loc["column"], "error", "ReferenceError",
                             f"'{name}' is not defined", "static-analysis"))
        if name in consts and parent and parent["type"] == "AssignmentExpression" and key == "left":
            out.append(issue(loc["line"], loc["column"], "error", "TypeError",
                             f"Assignment to constant variable '{name}'", "static-analysis"))
        if name in consts and parent and parent["type"] == "UpdateExpression":
            out.append(issue(loc["line"], loc["column"], "error", "TypeError",
                             f"Assignment to constant variable '{name}'", "static-analysis"))
    nulls = {n["id"]["name"] for n, _, _ in walk(tree) if n["type"] == "VariableDeclarator"
             and n["id"]["type"] == "Identifier" and (n["init"] is None or (n["init"]["type"] == "Literal" and n["init"].get("raw") == "null")
             or (n["init"]["type"] == "Identifier" and n["init"]["name"] == "undefined"))}
    for n, _, _ in walk(tree):
        if n["type"] == "MemberExpression" and n["object"]["type"] == "Identifier" and n["object"]["name"] in nulls:
            loc = n["loc"]["start"]
            out.append(issue(loc["line"], loc["column"], "warning", "TypeError",
                             f"'{n['object']['name']}' may be null/undefined here (property access can throw)", "static-analysis"))
        if n["type"] == "VariableDeclarator" and n["id"]["type"] == "Identifier":
            nm = n["id"]["name"]
            if used.get(nm, 0) == 0:
                loc = n["loc"]["start"]
                out.append(issue(loc["line"], loc["column"], "warning", "UnusedVariable",
                                 f"'{nm}' is declared but never used", "static-analysis"))
        if n["type"] in ("IfStatement", "WhileStatement") and n["test"]["type"] == "AssignmentExpression":
            loc = n["test"]["loc"]["start"]
            out.append(issue(loc["line"], loc["column"], "warning", "LogicError",
                             "Assignment inside a condition (did you mean === ?)", "static-analysis"))
        if n["type"] == "BinaryExpression" and n["operator"] in ("==="," ==","!==","==","!=") and \
           any(s.get("type") == "Identifier" and s.get("name") == "NaN" for s in (n["left"], n["right"])):
            loc = n["loc"]["start"]
            out.append(issue(loc["line"], loc["column"], "error", "LogicError",
                             "Comparing with NaN is always false; use Number.isNaN()", "static-analysis"))
        if n["type"] == "CallExpression" and n["callee"].get("type") == "Identifier" and n["callee"]["name"] == "eval":
            loc = n["loc"]["start"]
            out.append(issue(loc["line"], loc["column"], "warning", "Security", "eval() executes arbitrary code", "static-analysis"))
    return out

def ml_analysis(code, threshold=0.8, parsed_ok=True):
    if not _model: return []
    lines = code.split("\n")
    idx = [i for i, l in enumerate(lines) if l.strip() and not l.strip().startswith("//")]
    if not idx: return []
    probs = _model.predict_proba([lines[i] for i in idx])
    classes = list(_model.classes_)
    out = []
    for i, p in zip(idx, probs):
        best = max(range(len(p)), key=lambda j: p[j]); label = classes[best]
        if label != "ok" and p[best] >= threshold and not (parsed_ok and label == "syntax"):
            out.append(issue(i + 1, 0, "info" if label == "bad_practice" else "warning", "ML:" + label,
                             f"{ML_MESSAGES[label]} - confidence {p[best]*100:.0f}%", "machine-learning"))
    return out

def analyze(code):
    issues, stages = [], {"parser": "ok"}
    tree = None
    try:
        parsed = esprima.parseScript(code, {"loc": True, "tolerant": True})
        tree = parsed.toDict()
        for e in getattr(parsed, "errors", []) or []:
            ln, col, msg = err_info(e)
            issues.append(issue(ln, col, "error", "SyntaxError", msg, "parser"))
    except esprima.Error as e:
        try:  # modules (import/export)
            parsed = esprima.parseModule(code, {"loc": True, "tolerant": True}); tree = parsed.toDict()
        except esprima.Error:
            ln, col, msg = err_info(e)
            issues.append(issue(ln, col, "error", "SyntaxError", msg, "parser"))
            stages["parser"] = "failed"
    if tree: issues += static_analysis(tree)
    ml = ml_analysis(code, parsed_ok=not any(i["type"] == "SyntaxError" for i in issues))
    for m in ml:  # keep ML hint only if no rule/parser finding already covers that line
        if not any(i["line"] == m["line"] for i in issues):
            issues.append(m)
    issues.sort(key=lambda i: (i["line"], i["column"]))
    summary = {s: sum(1 for i in issues if i["severity"] == s) for s in ("error", "warning", "info")}
    score = max(0, 100 - summary["error"] * 20 - summary["warning"] * 8 - summary["info"] * 3)
    return {"issues": issues, "summary": summary, "score": score, "stages": stages,
            "lines": len(code.split("\n"))}
