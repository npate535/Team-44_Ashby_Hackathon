"""Browser-free checks of the plain-language panel's display-only wording pass (interpret.js plainText/markdown) via
node: class keys read as the case view's labels, a class probability never reads 100% or 0%, other numbers are left
alone, the "Validated" trust tag reads "Spot-check passed", and citation chips stay with the word before them.
The cached interpretation JSON is never edited: the server's number and claim checks run on the original text."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import textwrap

import pytest

from core import config

APP = config.REPO_ROOT / "team44-triage-frontend" / "app"
CACHED = config.REPO_ROOT / "artifacts" / "interpretations"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def run_node(body: str, **inputs) -> dict:
    script = textwrap.dedent(f"""
        import {{ readFileSync }} from 'node:fs';
        import * as I from {json.dumps((APP / 'js' / 'views' / 'interpret.js').as_uri())};
        const inp = JSON.parse(readFileSync(0, 'utf8'));
        const out = {{}};
        {body}
        console.log(JSON.stringify(out));
    """)
    res = subprocess.run(["node", "--input-type=module", "-e", script], input=json.dumps(inputs), capture_output=True,
                         text=True, timeout=60)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout.strip().splitlines()[-1])


def test_class_keys_and_extreme_probabilities():
    out = run_node(
        "for (const [k, s] of Object.entries(inp)) out[k] = I.plainText(s);",
        suggests="**Suggests Invasive (100%)** · Ambiguous: Normal vs InSitu",
        top="| Top probabilities | Invasive 100%, InSitu 0%, others 0% | Near-certain |",
        colon="Benign (100.0%); InSitu: 0%",
        mid="InSitu 70%, Normal 17%, Benign 14%, Invasive 0%",
        others="Invasive probability is near 0%; exact CI 88.8–100%; Reliability 100%; OOD percentile 100%",
        partial="Invasive 10%, Benign 0.4%, Normal 100.5%, 0 of 5 InSitu",
    )
    assert out["suggests"] == "**Suggests Invasive (>99%)** · Ambiguous: Normal vs In situ"
    assert out["top"] == "| Top probabilities | Invasive >99%, In situ <1%, others <1% | Near-certain |"
    assert out["colon"] == "Benign (>99%); In situ: <1%"
    assert out["mid"] == "In situ 70%, Normal 17%, Benign 14%, Invasive <1%"
    # not a class probability: reliability, percentiles, interval bounds and prose stay exactly as written
    assert out["others"] == "Invasive probability is near 0%; exact CI 88.8–100%; Reliability 100%; OOD percentile 100%"
    assert out["partial"] == "Invasive 10%, Benign 0.4%, Normal 100.5%, 0 of 5 In situ"


def test_trust_tag_and_citation_chips():
    out = run_node(
        "out.html = I.markdown(I.plainText(inp.md));",
        md=textwrap.dedent("""\
            ## Tissue evidence
            - **Stroma** (region 11) · ↓ invasive · Low score here · Validated [F4]
            - **Pleomorphism** · ↑ invasive [F2] · Validated (19/20)
            - **Mitoses** (regions 2, 5) · ↑ invasive · failed, 0/20 · Unreliable
            - **TILs** · context · above typical · Weak
            - region for pleomorphism (Validated); confirm at the slide [F1, F3].
            """),
    )
    html = out["html"]
    assert ">Validated<" not in html and "(Validated)" not in html
    assert html.count('<span class="ai-tag ok"') == 2 and ">Spot-check passed</span>" in html
    assert "not a pathologist or clinical validation" in html  # the tooltip says what the check is
    assert ">Unreliable</span>" in html and ">Weak</span>" in html
    assert "(spot-check passed)" in html
    # every chip run is bound to the word or tag before it
    assert re.search(r'<span class="ai-nb">slide <button class="ai-cite" data-f="F1"[^>]*>F1</button><button[^>]*>F3</button></span>', html)
    assert re.search(r'<span class="ai-nb"><span class="ai-dir up">↑ invasive</span> <button class="ai-cite" data-f="F2"', html)
    assert re.search(r'<span class="ai-nb"><span class="ai-tag ok"[^>]*>Spot-check passed</span> <button class="ai-cite" data-f="F4"', html)


@pytest.mark.skipif(not CACHED.is_dir(), reason="no cached interpretations")
def test_cached_texts_render_without_raw_keys_or_100_percent():
    texts = {p.name: json.loads(p.read_text()).get("markdown") or "" for p in sorted(CACHED.glob("*/*.json"))}
    assert texts
    out = run_node(
        """
        for (const [k, s] of Object.entries(inp)) {
          const html = I.markdown(I.plainText(s));
          out[k] = html.replace(/<[^>]+>/g, '');
        }
        """,
        **texts,
    )
    for name, text in out.items():
        assert "InSitu" not in text, name
        assert not re.search(r"\b(?:Normal|Benign|In situ|Invasive|others?)\s*\(?\s*(?:100|0)%", text), name
        assert not re.search(r"\bValidated\b", text), name
    # the cached files themselves are untouched: the raw text still carries what the server checked
    assert any("InSitu" in s for s in texts.values()) and any("(100%)" in s for s in texts.values())


def test_panel_applies_the_display_pass():
    out = run_node(
        """
        const ST = {interp: {X: {status: 'ok', data: {markdown: inp.md, verification: {checked: 1, unsupported: []},
                                                      claims: {found: [], remaining: ['InSitu (100%) is not in the set.']}}}}};
        out.html = I.interpretPanel(ST, 'X', 'case');
        """,
        md="## Bottom line\n**Suggests InSitu (100%)**",
    )
    assert "Suggests In situ (&gt;99%)" in out["html"] and "In situ (&gt;99%) is not in the set." in out["html"]
    assert "InSitu" not in out["html"]
