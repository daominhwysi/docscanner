import re
from typing import Any, Dict, Tuple, Union, Optional


def find_last_tag_block(text: str, tag_name: str) -> Optional[str]:
    """
    Finds the content of the last block between <tag_name>...</tag_name> tags.
    Case-insensitive, supports tags with attributes.
    Returns the content between the last matching tags, or None if not found.
    """
    # Escape tag name for regex
    escaped = re.escape(tag_name)
    open_pattern = re.compile(rf"<\s*{escaped}\b[^>]*>", re.IGNORECASE)
    close_pattern = re.compile(rf"</\s*{escaped}\s*>", re.IGNORECASE)

    # Find all closing tags
    closes = list(close_pattern.finditer(text))
    if not closes:
        return None

    last_close = closes[-1]
    close_start = last_close.start()

    # Find opening tags before last close
    opens = []
    for m in open_pattern.finditer(text):
        if m.start() < close_start:
            opens.append(m)
        else:
            break

    if not opens:
        return None

    last_open = opens[-1]
    content_start = last_open.end()
    return text[content_start:close_start].strip()


def extract_and_remove_thinking_block(text: str) -> Tuple[str, str]:
    """
    Extracts the content of the outermost <thinking>...</thinking> block.
    Returns a tuple: (text_without_block, inner_content)
    """
    # Find all closing thinking tags
    close_pattern = re.compile(r"</thinking\s*>", re.IGNORECASE)
    closes = list(close_pattern.finditer(text))
    if not closes:
        return text, ""

    last_close = closes[-1]
    close_start = last_close.start()
    close_end = last_close.end()

    # Find first opening thinking tag
    open_pattern = re.compile(r"<thinking[^>]*>", re.IGNORECASE)
    open_match = open_pattern.search(text)
    if not open_match or open_match.start() >= close_start:
        return text, ""

    open_start = open_match.start()
    open_end = open_match.end()

    inner = text[open_end:close_start]
    outside = text[:open_start] + text[close_end:]
    return outside, inner


class ExtractedResponse:
    def __init__(self, thinking_block: str, document: Optional[str]):
        self.thinking_block = thinking_block
        self.document = document


def extract_response(text: str) -> ExtractedResponse:
    """
    Extracts the thinking block and the final document content from raw text.
    Returns an ExtractedResponse with thinking_block and document.
    """
    cleaned, thinking = extract_and_remove_thinking_block(text)
    block = find_last_tag_block(cleaned, "assessmentmarkuplanguage")
    doc: Optional[str] = None
    doc = block
    return ExtractedResponse(thinking_block=thinking, document=doc)


def replace_image_tags(
    content: str,
    image_dict: dict[str, str]
) -> Tuple[str, str]:
    """
    Replace <graphic tag='IMx' label='...'> with <img src='...'
    alt='...'/> using URLs from image_dict and preserving label from original tag.
    """
    if not isinstance(image_dict, dict):
        return content, 404

    pattern = re.compile(
        r"<graphic\s+tag=['\"]?(IM[0-9O]+)['\"]?"         # group 1: tag
        r"(?:\s+label=['\"](.*?)['\"])?\s*/?>",           # group 2: optional label
        re.IGNORECASE
    )

    used = set()
    missing = False
    extra = False

    def normalize_key(raw: str) -> str:
        return raw.upper().replace('O', '0')

    def repl(match: re.Match) -> str:
        nonlocal missing, extra
        raw_key = match.group(1)
        label = match.group(2)
        key = normalize_key(raw_key)

        url = image_dict.get(key)
        if not url:
            extra = True
            return ''  # hoặc match.group(0) để giữ nguyên nếu muốn

        used.add(key)
        alt = label or key
        return f'<img src="{url}" alt="{alt}"/>'

    try:
        new_content = pattern.sub(repl, content)
    except Exception:
        return content, 404

    # Kiểm tra khóa không dùng tới
    dict_keys = {normalize_key(k) for k in image_dict.keys()}
    if dict_keys - used:
        missing = True

    status = 200 if not (missing or extra) else 404
    return new_content, status


if __name__ == "__main__":
    # Example usage
    sample = "<thinking>Compute something</thinking>..."
    res = extract_response(sample)
    print(res.thinking_block, res.document)
