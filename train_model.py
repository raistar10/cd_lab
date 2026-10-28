"""Generates a labelled line-level JavaScript dataset and trains the ML model.
Features: character n-gram TF-IDF. Classifier: Logistic Regression."""
import random, joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

random.seed(7)
V = ["x","y","count","total","user","items","data","name","arr","value","result","index","obj","list"]
F = ["calc","load","render","update","fetchData","init","handle","save","parse"]
N = lambda: random.choice(V)
FN = lambda: random.choice(F)
K = lambda: str(random.randint(0, 99))

TEMPLATES = {
 "syntax": ["function {f}({a} {{", "var {a} = ;", "let {a} = {k}", "if ({a} > {b} {{", "console.log('{a}'",
   "for (let i = 0; i < {k}; i++ {{", "var {a} = [1, 2, 3;", "{a} = {{p: 1, q: }};", "return );", "let = {k};",
   "const {a} == {k};", "{f}({a}, {b};", "}} else if {{", "function ({a}) {f} {{", "var {a} = 'unterminated;",
   "if {a} > {k}) {{", "while ({a} < {k} {{", "{a}.{k}abc = 1;", "else {{ {a}++; }} }}", "var 1{a} = {k};"],
 "type_error": ["{a}.length", "null.{a}", "undefined.{a}()", "let {a} = null; {a}.{b} = {k};", "const {a} = {k}; {a}();",
   "let {a} = 'text'; {a}.push({k});", "{a} = undefined; {a}.{b}", "let {a} = {k}; {a}.toUpperCase();",
   "let {a} = null; {a}[0];", "const {a} = {{}}; {a}.{b}.{f}();", "let {a}; {a}.forEach(fn);", "({k})();",
   "let {a} = {k}; {a}.map(fn);", "{a} = null; {a}.length;", "JSON.parse(undefined);"],
 "logic_error": ["if ({a} = {k}) {{", "while ({a} = {k}) {{", "if ({a} === NaN) {{", "for (var i = 0; i <= {a}.length; i++) {{",
   "if (typeof {a} == 'undefned') {{", "if ({a} = null) {{", "if ({a} > {k} && {a} < {k}) {{ {a} = {b} }}",
   "while (true) {{ {a}++; }}", "for (;;) {{ {a}--; }}", "if ({a} == {b} || {a} = {k}) {{", "return {a}; {b} = {k};",
   "switch ({a}) {{ case {k}: {b}++; case {k}: {b}--; }}", "if ({a} = {b}) return;", "for (let i = {k}; i > 0; i++) {{"],
 "bad_practice": ["eval({a});", "var {a} = {k};", "document.write({a});", "if ({a} == {b}) {{", "console.log({a});",
   "with ({a}) {{", "el.innerHTML = {a};", "setTimeout('{f}()', {k});", "alert({a});", "{a} != {b}",
   "debugger;", "var {a}, {b}, {c};", "new Function({a});", "window.{a} = {k};"],
 "ok": ["let {a} = null;", "let {a};", "let {a} = [];", "const {a} = require('{b}');", "var_{a} = {k};", "console.error({a});", "const {a} = {{}};", "let {a} = {k};", "const {a} = \"{b}\";", "const {a} = {k};", "let {a} = {b} + {k};", "function {f}({a}, {b}) {{", "return {a} * {b};", "if ({a} === {b}) {{",
   "}} else {{", "}}", "for (let i = 0; i < {a}.length; i++) {{", "const {a} = [1, 2, {k}];", "{a}.push({b});",
   "const {a} = {{ {b}: {k} }};", "{f}({a});", "export default {f};", "const {a} = {b}.map(v => v * {k});",
   "try {{", "}} catch (err) {{", "{a} += {k};", "// compute {a}", "async function {f}() {{", "await {f}({a});",
   "const {{ {a}, {b} }} = {c};", "while ({a} < {k}) {{", "{a}.forEach(item => {f}(item));", "return {a} ?? {k};"],
}

def build():
    X, y = [], []
    for label, tpls in TEMPLATES.items():
        for t in tpls:
            for _ in range(14):
                s = t.format(a=N(), b=N(), c=N(), f=FN(), k=K())
                if random.random() < .3: s = "  " * random.randint(1, 3) + s
                X.append(s); y.append(label)
    return X, y

if __name__ == "__main__":
    X, y = build()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=1, stratify=y)
    model = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True),
                          LogisticRegression(max_iter=2000, C=10))
    model.fit(Xtr, ytr)
    print(classification_report(yte, model.predict(Xte)))
    model.fit(X, y)
    joblib.dump(model, "model.joblib")
    print("Saved model.joblib with", len(X), "samples")
