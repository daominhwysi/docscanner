import re
import json
from typing import List, Dict, Any, Union, Optional, Literal, TypedDict

# TypeScript interface: Block
# Được biểu diễn bằng TypedDict để an toàn kiểu
class Block(TypedDict):
    type: str
    content: Union[str, Dict[str, Any]]

# TypeScript interface: State
class State(TypedDict):
    currentBlockType: Optional[str]
    currentBlockObject: Optional[Union[str, Dict[str, Any]]]
    inMultiLine: bool
    multiLineTarget: Optional[Literal['block', 'property']]
    propertyStack: List[str]

class SLURPParser:
    """
    Phân tích cú pháp văn bản định dạng SLURP thành một cấu trúc các khối (blocks).
    SLURP là một định dạng đơn giản, thụt đầu dòng để biểu diễn các khối dữ liệu lồng nhau.
    """
    _BLOCK_REGEX = re.compile(r"^(\w+):\s*\|?(.*)$")
    _PROP_REGEX = re.compile(r"^(>+)\s*([\w-]+):\s*\|?(.*)$")

    def __init__(self):
        self.result: List[Block] = []
        self.current_state: State = self._get_initial_state()

    def _get_initial_state(self) -> State:
        """Trả về một từ điển trạng thái ban đầu."""
        return {
            "currentBlockType": None,
            "currentBlockObject": None,
            "inMultiLine": False,
            "multiLineTarget": None,
            "propertyStack": [],
        }

    def parse(self, text_content: str) -> List[Block]:
        """
        Phân tích cú pháp một chuỗi đầu vào SLURP và trả về một danh sách các khối.
        """
        self.result = []
        self.current_state = self._get_initial_state()

        lines = text_content.replace('\r\n', '\n').split('\n')

        for line in lines:
            if not line.strip() and not self.current_state["inMultiLine"]:
                continue
            self._process_line(line)

        self._finalize_current_block()
        return self.result

    # ========================================================================
    # HÀM ĐÃ ĐƯỢỢC SỬA LỖI BÊN DƯỚI
    # ========================================================================
    def _process_line(self, line: str) -> None:
        """Xử lý một dòng đơn từ đầu vào SLURP."""
        block_match = self._BLOCK_REGEX.match(line)
        prop_match = self._PROP_REGEX.match(line)

        # Phát hiện khối mới
        if not line.startswith('>') and not line.startswith(' ') and block_match:
            self._finalize_current_block()

            block_type, value = block_match.groups()
            self.current_state["currentBlockType"] = block_type
            trimmed_value = value.strip()

            if trimmed_value or '|' in line:
                self.current_state["currentBlockObject"] = trimmed_value
                if '|' in line:
                    self.current_state["inMultiLine"] = True
                    self.current_state["multiLineTarget"] = 'block'
            else:
                self.current_state["currentBlockObject"] = {}
                self.current_state["inMultiLine"] = False

        # Phát hiện thuộc tính
        elif prop_match:
            self.current_state["inMultiLine"] = False

            gt, key, value = prop_match.groups()
            level = len(gt)

            # Cắt ngắn ngăn xếp thuộc tính về mức độ thụt đầu dòng hiện tại
            self.current_state["propertyStack"] = self.current_state["propertyStack"][:level - 1]

            parent_obj = self.current_state["currentBlockObject"]
            if not isinstance(parent_obj, dict):
                # Tự động chuyển đổi chuỗi thành đối tượng nếu cần
                parent_obj = {}
                self.current_state["currentBlockObject"] = parent_obj

            # *** PHẦN SỬA LỖI QUAN TRỌNG ***
            # Đoạn code này đảm bảo rằng chúng ta đang đi vào một từ điển (dict).
            # Nếu đường dẫn tồn tại nhưng chứa một giá trị không phải dict (ví dụ: một chuỗi rỗng),
            # nó sẽ được ghi đè bằng một dict rỗng.
            current_level_ref = parent_obj
            for path_key in self.current_state["propertyStack"]:
                # Kiểm tra xem khóa có tồn tại không và giá trị của nó có phải là dict không
                if not isinstance(current_level_ref.get(path_key), dict):
                    # Nếu không, tạo/ghi đè nó bằng một dict rỗng
                    current_level_ref[path_key] = {}
                # Đi sâu vào cấp độ tiếp theo
                current_level_ref = current_level_ref[path_key]
            # *** KẾT THÚC PHẦN SỬA LỖI ***

            # Bây giờ `current_level_ref` chắc chắn là một dict
            current_level_ref[key] = value.strip()
            self.current_state["propertyStack"].append(key)

            if '|' in line:
                self.current_state["inMultiLine"] = True
                self.current_state["multiLineTarget"] = 'property'

        # Xử lý nội dung nhiều dòng
        elif self.current_state["inMultiLine"]:
            content = line
            if self.current_state["multiLineTarget"] == 'block':
                if isinstance(self.current_state["currentBlockObject"], str):
                    self.current_state["currentBlockObject"] += '\n' + content
            elif self.current_state["multiLineTarget"] == 'property':
                parent = self.current_state["currentBlockObject"]
                if isinstance(parent, dict):
                    stack = self.current_state["propertyStack"]
                    if not stack: return # Bỏ qua nếu không có thuộc tính nào trong ngăn xếp

                    # Đi đến cha của thuộc tính cuối cùng
                    for i in range(len(stack) - 1):
                        # Giả sử đường dẫn đã hợp lệ vì nó được tạo ở trên
                        parent = parent[stack[i]]

                    final_key = stack[-1]
                    if final_key in parent and isinstance(parent[final_key], str):
                        parent[final_key] += '\n' + content
    # ========================================================================
    # KẾT THÚC HÀM ĐÃ ĐƯỢC SỬA LỖI
    # ========================================================================

    def _finalize_current_block(self) -> None:
        """Lưu khối hiện đang được xử lý vào kết quả."""
        current_block_type = self.current_state["currentBlockType"]
        current_block_object = self.current_state["currentBlockObject"]

        if current_block_type and current_block_object is not None:
            final_content = current_block_object

            if isinstance(final_content, str):
                final_content = final_content.strip()

            self.result.append({
                "type": current_block_type,
                "content": final_content
            })

        self.current_state = self._get_initial_state()


def format_output(parsed_array: List[Block]) -> str:
    """
    Chuyển đổi danh sách các khối đã phân tích cú pháp thành định dạng CSV phẳng tùy chỉnh.
    Định dạng: blockType[index].path.to.property,"value"
    """
    output_lines: List[str] = []
    counters: Dict[str, int] = {}

    def flatten(obj: Dict[str, Any], path: str):
        for key, value in obj.items():
            new_path = f"{path}.{key}" if path else key
            if value and isinstance(value, dict):
                flatten(value, new_path)
            else:
                # json.dumps trên một chuỗi sẽ thêm dấu ngoặc kép và thoát các ký tự đặc biệt
                formatted_value = json.dumps(str(value))
                output_lines.append(f"{new_path},{formatted_value}")

    for block in parsed_array:
        block_type = block['type']
        block_content = block['content']

        counters[block_type] = counters.get(block_type, 0) + 1
        index = counters[block_type]
        base_path = f"{block_type}[{index}]"

        if block_content and isinstance(block_content, dict):
            flatten(block_content, base_path)
        else:
            formatted_value = json.dumps(str(block_content))
            output_lines.append(f"{base_path},{formatted_value}")

    return '\n'.join(output_lines)


def split_csv_lines_safe(input_str: str) -> List[str]:
    """
    Tách các dòng CSV một cách an toàn, xử lý các dấu xuống dòng bên trong các chuỗi được trích dẫn.
    """
    lines: List[str] = []
    current_line = ""
    inside_quotes = False

    for i, char in enumerate(input_str):
        if char == '"':
            # Trong JS, input[-1] là undefined. Trong Python, đó là ký tự cuối cùng.
            # Vì vậy, chúng ta phải kiểm tra rõ ràng i > 0.
            is_escaped = i > 0 and input_str[i-1] == '\\'
            if not is_escaped:
                inside_quotes = not inside_quotes

        if char == '\n' and not inside_quotes:
            lines.append(current_line)
            current_line = ""
        else:
            current_line += char

    if current_line:
        lines.append(current_line)

    return lines

# Định nghĩa các loại để làm rõ, tương tự như DataObject và DataMap trong TypeScript
DataMap = Dict[str, List[Dict[str, Any]]]
_CSV_KEY_REGEX = re.compile(r"^(\w+)\[(\d+)\](?:\.(.+))?$")

def parse_csv_to_json(input_str: str) -> DataMap:
    """
    Phân tích cú pháp chuỗi định dạng CSV tùy chỉnh thành một cấu trúc đối tượng lồng nhau,
    đồng thời giữ lại thuộc tính __order để biết thứ tự xuất hiện ban đầu của mỗi khối.
    """
    lines = split_csv_lines_safe(input_str)
    data: DataMap = {}

    global_order = 0

    for line in lines:
        if not line:
            continue

        parts = line.split(',', 1)
        if len(parts) < 2:
            continue
        raw_key, raw_value = parts

        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value  # fallback nếu không phải JSON

        match = _CSV_KEY_REGEX.match(raw_key)
        if not match:
            continue

        obj_name, idx_str, path = match.groups()
        index = int(idx_str)

        if obj_name not in data:
            data[obj_name] = []

        # đảm bảo danh sách đủ dài
        if len(data[obj_name]) < index:
            data[obj_name].extend([None] * (index - len(data[obj_name])))

        list_index = index - 1

        # tạo object mới kèm __order nếu chưa có
        if data[obj_name][list_index] is None:
            data[obj_name][list_index] = {'__order': global_order}
            global_order += 1

        target = data[obj_name][list_index]

        if path:
            keys = path.split('.')
            # tạo nested dict nếu cần
            for key_part in keys[:-1]:
                target = target.setdefault(key_part, {})
            target[keys[-1]] = value
        else:
            target['value'] = value

    # loại bỏ các slot None
    for key in data:
        data[key] = [entry for entry in data[key] if entry is not None]

    return data



def slurp_to_json(slurp: str) -> Dict:
    """
    Chức năng điều phối chính chuyển đổi một chuỗi SLURP thành một chuỗi JSON.
    """
    parser = SLURPParser()
    parsed_blocks = parser.parse(slurp)
    csv_str = format_output(parsed_blocks)
    json_obj = parse_csv_to_json(csv_str)
    return json_obj

# ---- Ví dụ sử dụng ----
# if __name__ == "__main__":
#     BASE_PATH = "/teamspace/studios/this_studio/tests/"
#     with open(f"{BASE_PATH}raw.txt", "r", encoding="utf-8") as f:
#         text = f.read()

#     # Chuyển nội dung sang JSON (dict hoặc list)
#     parsed = slurp_to_json(text)

#     # Ghi ra file JSON đúng cách
#     with open(f"{BASE_PATH}output.json", "w", encoding="utf-8") as f:
#         json.dump(parsed, f, ensure_ascii=False, indent=4)


import re

def autofix_missing_pipes(text: str) -> str:
    lines = text.replace('\r\n', '\n').split('\n')
    fixed_lines = []

    block_regex = re.compile(r"^([^:]+):\s*(?!\|)(.*)$")  # match block chưa có |
    prop_regex = re.compile(r"^(>+)\s*([^:]+):\s*(?!\|)(.*)$")
    new_block_or_prop = re.compile(r"^([^:]+:|>+\s*[^:]+:)")

    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Nếu dòng này đã có | thì giữ nguyên, không xử lý
        if re.match(r"^[^:]+:\s*\|", line):
            fixed_lines.append(line)
            i += 1
            continue

        next_line = lines[i + 1] if i + 1 < len(lines) else ""

        is_block = block_regex.match(line)
        is_prop = prop_regex.match(line)

        if (is_block or is_prop) and next_line.strip() != "":
            if not new_block_or_prop.match(next_line):
                before_colon, after_colon = line.split(':', 1)
                fixed_lines.append(f"{before_colon}: |")
                if after_colon.strip():
                    fixed_lines.append(f"  {after_colon.strip()}")
                i += 1
                while i < len(lines) and (lines[i].startswith(" ") or lines[i].strip() == "") and not new_block_or_prop.match(lines[i]):
                    fixed_lines.append(lines[i])
                    i += 1
                continue

        fixed_lines.append(line)
        i += 1

    return '\n'.join(fixed_lines)


if __name__ == "__main__":
    sample_text = """title: Đây là tiêu đề
noidung: Dòng đầu tiên
  Dòng thứ hai
author: Minh"""

    print("=== Input ===")
    print(sample_text)
    print("\n=== Output ===")
    print(autofix_missing_pipes(sample_text))
    print(slurp_to_json(autofix_missing_pipes(sample_text)))