#!/usr/bin/env python3
"""Static coverage check for the JavaScript rule classifiers.

Phrase and ISL both fall back to hand-written finger rules in `index.html`.
If someone later edits either rule function and accidentally orphans a
catalogued sign (as the dead `return "Good"` in ISL used to do until it was
removed), the Dictionary would offer a "Try it" button for a sign the rule
path cannot produce.

This script parses the rule functions in `index.html` and asserts every entry
in `signs.ASL_PHRASES` and `signs.ISL_PHRASES` is reachable by name. Exits 0
on success, 1 on orphans, so it can be wired into CI if we ever add one.

Usage (from repo root or Code/):
    python tools/check_rules.py
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "..", "Sign-Language-Interpreter-using-Deep-Learning-master", "Code")

sys.path.insert(0, CODE)
from signs import ASL_PHRASES, ISL_PHRASES  # noqa: E402


def rule_body(name, html):
    """Return the body of the Recognizer method `_<name>(l) { ... }`."""
    start = html.find(f"_{name}(l) {{")
    if start < 0:
        raise SystemExit(f"could not find _{name} in index.html")
    # Match the enclosing method by looking for `\n  }` which ends it.
    end = html.find("\n  }", start)
    return html[start:end]


def labels_emitted(body):
    """Every quoted literal returned by the rule - a label the rule can produce."""
    return set(re.findall(r"""return\s+["']([^"']+)["']""", body)) | \
           set(re.findall(r"""\?\s*["']([^"']+)["']\s*:\s*["']([^"']+)["']""", body)[0] if False else [])\
           | set(sum(re.findall(r"""\?\s*["']([^"']+)["']\s*:\s*["']([^"']+)["']""", body), ()))


# Known orphans - documented in the Dictionary as approximations the rules
# cannot distinguish. If one is removed from the catalog, remove it here too.
KNOWN_ORPHANS = {
    "isl": {"Good": "shares its hand shape with Yes; Dictionary says so"},
}


def main():
    html_path = os.path.join(CODE, "index.html")
    html = open(html_path, encoding="utf-8").read()

    rc = 0
    for rule, cat, label in [("phrase", ASL_PHRASES, "ASL Phrase"),
                             ("isl", ISL_PHRASES, "ISL")]:
        body = rule_body(rule, html)
        emitted = labels_emitted(body) - {"?"}
        catalog = set(cat)
        orphans_all = catalog - emitted
        known = KNOWN_ORPHANS.get(rule, {})
        orphans = sorted(orphans_all - set(known))
        allowed = sorted(orphans_all & set(known))
        extras = sorted(emitted - catalog)
        ok = not orphans
        marker = "OK" if ok else "FAIL"
        print(f"[{marker}] {label}: catalog {len(catalog)}, rules emit {len(emitted & catalog)}")
        for name in allowed:
            print(f"       known approximation: {name} ({known[name]})")
        if orphans:
            print(f"       missing from rules: {orphans}")
            rc = 1
        if extras:
            print(f"       emitted but not catalogued: {extras}")

    print("All rule-reachable." if rc == 0 else "Fix the orphans above.")
    sys.exit(rc)


if __name__ == "__main__":
    main()
