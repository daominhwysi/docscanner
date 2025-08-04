import re
from latex2mathml.converter import convert


def latex_to_inline_mathml(text: str) -> str:
    pattern = re.compile(r"""
        \\\[(.+?)\\\]      |   # group 2: \[...\]
        \\\((.+?)\\\)      |   # group 3: \(...\)
    """, re.VERBOSE | re.DOTALL)

    def repl(m):
        groups = [g for g in m.groups() if g is not None]
        if not groups:
            # Không có gì match, trả nguyên đoạn
            return m.group(0)
        
        latex = groups[0].strip()
        try:
            mathml = convert(latex)
            return f'{mathml}'
        except Exception:
            return m.group(0)


    return pattern.sub(repl, text)

def convert_text_to_html(text):
    lines = text.strip().split('\n')
    html_lines = []

    for line in lines:
        html_lines.append(f'<p>{line}</p>')

    return ''.join(html_lines)

def convert_text2html(input_data: str):
    text_mathml = latex_to_inline_mathml(input_data)
    html = convert_text_to_html(text_mathml)
    return html