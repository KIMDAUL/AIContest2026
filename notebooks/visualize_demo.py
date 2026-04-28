"""cleaned_html 시각화 데모 — 3가지 방식"""
import pandas as pd
import re
import json
from pathlib import Path
from html import escape

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
train = pd.read_csv(ROOT / 'data' / 'train.csv')
row = train.iloc[0]
html = row['cleaned_html']
candidates = json.loads(row['candidate_elements'])
target_id = row['target_id']

OUT = HERE / 'viz_out'
OUT.mkdir(exist_ok=True)

# 1) 브라우저 렌더용: wireframe 스타일 + 후보 강조
def to_browser_html(html, candidates, target_id, row):
    h = re.sub(r'<text\b', '<span class="t"', html)
    h = re.sub(r'</text>', '</span>', h)

    css = """
    <style>
      body { font: 13px/1.4 -apple-system, sans-serif; margin: 0 340px 0 0; padding: 12px; background: #fff; color: #222; }
      div, header, main, section, nav, ul, ol, li, form, label, p { display: block; box-sizing: border-box; border: 1px dashed #d0d0d0; padding: 4px 6px; margin: 2px 0; border-radius: 3px; }
      header { border-color: #6aa; background: #f4fbfb; }
      main   { border-color: #888; background: #fafafa; }
      ul, ol { padding-left: 18px; }
      button, a, input, select, textarea, option { border: 1px solid #888; padding: 2px 8px; margin: 2px; border-radius: 3px; background: #fff; font: inherit; display: inline-block; }
      a       { color: #1565c0; text-decoration: underline; cursor: pointer; }
      button  { background: #f0f0f0; cursor: pointer; }
      select  { background: #fff8dc; }
      img     { display: inline-block; min-width: 24px; min-height: 24px; border: 1px dashed #aaa; background: #f5f5f5; vertical-align: middle; }
      img::before { content: "🖼"; padding: 0 6px; color: #999; }
      .t      { display: inline; }
      [aria_label]::after { content: " ⓘ" attr(aria_label); color: #999; font-size: 10px; }
      /* 후보 elements 시각 강조 — backend_node_id 매핑 정보 없어 클래스 부여는 패널에만 */
      .panel { position: fixed; right: 0; top: 0; width: 320px; height: 100vh; overflow: auto;
               background: #fafafa; border-left: 1px solid #ccc; padding: 12px; font: 12px ui-monospace, monospace; }
      .panel h3 { margin: 0 0 6px; font-size: 13px; }
      .panel .cand { padding: 3px 4px; margin: 2px 0; border-radius: 3px; }
      .panel .cand.target { background: #ffeb3b; font-weight: bold; }
      .panel hr { border: none; border-top: 1px solid #ddd; margin: 8px 0; }
    </style>
    """

    panel = ['<div class="panel">',
             f'<h3>Task</h3><div>{escape(row["task"])}</div><hr>',
             f'<h3>Label</h3>op = <b>{row["op"]}</b><br>value = <b>{escape(str(row["value"]))}</b><br>'
             f'target = <b>{target_id}</b><hr>',
             '<h3>Candidates</h3>']
    for c in candidates:
        cls = 'cand target' if c['candidate_id'] == target_id else 'cand'
        mark = '★ ' if c['candidate_id'] == target_id else '· '
        panel.append(
            f'<div class="{cls}">{mark}<b>&lt;{c["tag"]}&gt;</b> '
            f'{escape(c["candidate_id"])}<br>'
            f'<span style="color:#555">text: {escape(c["text"][:60]) or "(empty)"}</span><br>'
            f'<span style="color:#999">attrs: {escape(c["attrs"][:60]) or "(none)"}</span></div>'
        )
    panel.append('</div>')
    return f'<!doctype html><html><head><meta charset="utf-8">{css}</head><body>{h}{"".join(panel)}</body></html>'

(OUT / 'demo_browser.html').write_text(
    to_browser_html(html, candidates, target_id, row), encoding='utf-8'
)

# 2) 텍스트 아웃라인: 들여쓰기 트리, 텍스트만 추림
def to_outline(html):
    out, depth = [], 0
    pos = 0
    for m in re.finditer(r'<(/?)(\w+)([^>]*)>([^<]*)', html):
        slash, tag, attrs, text = m.groups()
        if slash:
            depth = max(0, depth - 1)
            continue
        if tag == 'text':
            out.append('  ' * depth + f'· "{text.strip()}"')
        else:
            label = tag
            aria = re.search(r'aria_label="([^"]+)"', attrs)
            if aria:
                label += f' [{aria.group(1)}]'
            out.append('  ' * depth + label)
            depth += 1
        if text.strip() and tag != 'text':
            out.append('  ' * depth + f'· "{text.strip()}"')
    return '\n'.join(out)

(OUT / 'demo_outline.txt').write_text(to_outline(html), encoding='utf-8')

# 3) 후보 element 요약 (모델 디버깅용 가장 유용)
def to_candidate_summary(row, candidates, target_id):
    lines = [f'Task : {row["task"]}',
             f'Op   : {row["op"]}',
             f'Value: {row["value"]}',
             f'Target: {target_id}',
             '-' * 60]
    for c in candidates:
        mark = '>>> ' if c['candidate_id'] == target_id else '    '
        lines.append(f'{mark}{c["candidate_id"]} <{c["tag"]}> '
                     f'text={c["text"][:50]!r} attrs={c["attrs"][:40]!r}')
    return '\n'.join(lines)

(OUT / 'demo_summary.txt').write_text(
    to_candidate_summary(row, candidates, target_id), encoding='utf-8'
)

print('생성됨:')
for p in OUT.iterdir():
    print(' ', p)
