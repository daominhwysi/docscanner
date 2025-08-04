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
    """
    Tạo một bản sao của đối tượng JSON và thay thế các giá trị 'figX' bằng URL thực tế.
    Đây là phương pháp "bất biến" (immutable) - không làm thay đổi đối tượng đầu vào.

    Args:
        json_obj: Đối tượng JSON (dict hoặc list) cần xử lý.
        figures_data: Danh sách các dictionary chứa mapping {'id': 'figX', 'src': 'url'}.

    Returns:
        Một đối tượng JSON mới đã được thay thế.
    """
    # 1. Tạo một bảng tra cứu (lookup table) để truy cập nhanh id -> src
    id_to_src_map = {figure['id']: figure['src'] for figure in figures_data}

    # 2. Định nghĩa một hàm đệ quy để duyệt và xây dựng lại đối tượng
    def _walk_and_rebuild(node: Any) -> Any:
        # Nếu node là một dictionary...
        if isinstance(node, dict):
            new_dict = {}
            for key, value in node.items():
                # Kiểm tra nếu đây là key 'figures' và giá trị là chuỗi cần thay thế
                if key == 'figures' and isinstance(value, str):
                    fig_ids = [i.strip() for i in value.split(',')]
                    
                    # Lấy các URL tương ứng, bỏ qua nếu ID không tồn tại
                    urls = [id_to_src_map.get(fig_id) for fig_id in fig_ids if id_to_src_map.get(fig_id)]

                    # Nếu chỉ có 1 URL, gán trực tiếp. Nếu nhiều, gán cả danh sách.
                    if len(urls) == 1:
                        new_dict[key] = urls[0]
                    elif len(urls) > 1:
                        new_dict[key] = urls
                    else:
                        # Nếu không tìm thấy URL nào, giữ lại giá trị gốc
                        new_dict[key] = value 
                else:
                    # Nếu không phải key 'figures', gọi đệ quy cho giá trị của nó
                    new_dict[key] = _walk_and_rebuild(value)
            return new_dict

        # Nếu node là một list...
        elif isinstance(node, list):
            # Gọi đệ quy cho từng phần tử trong list và tạo ra một list mới
            return [_walk_and_rebuild(item) for item in node]
        
        # Nếu là các kiểu dữ liệu khác (string, number, bool...), trả về chính nó
        else:
            return node

    # 3. Bắt đầu quá trình từ gốc của đối tượng JSON
    return _walk_and_rebuild(json_obj)

if __name__ == "__main__":
    figures = [{'id': 'fig0', 'src': 'https://pub-81feb243d1ab4aae8e02911485784de0.r2.dev/a25fdf9b.jpg'}, {'id': 'fig1', 'src': 'https://pub-81feb243d1ab4aae8e02911485784de0.r2.dev/29cccdd3.jpg'}, {'id': 'fig2', 'src': 'https://pub-81feb243d1ab4aae8e02911485784de0.r2.dev/e88ed63b.jpg'}, {'id': 'fig3', 'src': 'https://pub-81feb243d1ab4aae8e02911485784de0.r2.dev/7ae81bb9.jpg'}, {'id': 'fig4', 'src': 'https://pub-81feb243d1ab4aae8e02911485784de0.r2.dev/a1dcd944.jpg'}, {'id': 'fig5', 'src': 'https://pub-81feb243d1ab4aae8e02911485784de0.r2.dev/31d6ba4f.jpg'}]
    with open("/teamspace/studios/this_studio/tests/de15.txt", "r", encoding="utf-8") as f:
        text = f.read()
    
