import re
from typing import List, Dict, Tuple, Any

def replace_img_to_fig(html: str) -> Tuple[str, List[Dict[str, str]]]:
    image_srcs = []
    index = 0

    def replacer(match):
        nonlocal index
        full_tag = match.group(0)  # toàn bộ thẻ <img ...>
        src = match.group(1)
        attrs = match.group(2) or ""

        img_id = f"fig{index}"
        image_srcs.append({"id": img_id, "src": src})
        index += 1

        # Thay thế bằng <figure id="figX" ... />
        return f'<figure id="{img_id}" {attrs.strip()} />'

    pattern = r'<img\s+[^>]*src=["\']([^"\']+)["\']([^>]*)>'
    text = re.sub(pattern, replacer, html, flags=re.IGNORECASE)

    return text, image_srcs

def replace_fig2img_immutable(json_obj: Any, figures_data: List[Dict[str, str]]) -> Any:
    id_to_src_map = {figure['id']: figure['src'] for figure in figures_data}
    FIGURE_TAG_RE = re.compile(
        r'<figure\b[^>]*\bid\s*=\s*["\']([^"\']+)["\'][^>]*>',
        re.IGNORECASE
    )

    def replace_figures_in_text(text: str) -> str:
        def replacer(match):
            fig_id = match.group(1)
            src = id_to_src_map.get(fig_id)
            if src:
                return f'<img src="{src}">'
            return match.group(0)
        return FIGURE_TAG_RE.sub(replacer, text)

    def _walk_and_rebuild(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: _walk_and_rebuild(v) for k, v in node.items()}
        elif isinstance(node, list):
            return [_walk_and_rebuild(item) for item in node]
        elif isinstance(node, str):
            return replace_figures_in_text(node)
        else:
            return node

    return _walk_and_rebuild(json_obj)

if __name__ == "__main__":
    figures = [{'id': 'fig0', 'src': 'https://pub-81feb243d1ab4aae8e02911485784de0.r2.dev/a25fdf9b.jpg'}, {'id': 'fig1', 'src': 'https://pub-81feb243d1ab4aae8e02911485784de0.r2.dev/29cccdd3.jpg'}, {'id': 'fig2', 'src': 'https://pub-81feb243d1ab4aae8e02911485784de0.r2.dev/e88ed63b.jpg'}, {'id': 'fig3', 'src': 'https://pub-81feb243d1ab4aae8e02911485784de0.r2.dev/7ae81bb9.jpg'}, {'id': 'fig4', 'src': 'https://pub-81feb243d1ab4aae8e02911485784de0.r2.dev/a1dcd944.jpg'}, {'id': 'fig5', 'src': 'https://pub-81feb243d1ab4aae8e02911485784de0.r2.dev/31d6ba4f.jpg'}]
    with open("/teamspace/studios/this_studio/tests/de15.txt", "r", encoding="utf-8") as f:
        text = f.read()
    
