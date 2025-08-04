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
    Phân tích cú pháp chuỗi định dạng CSV tùy chỉnh thành một cấu trúc đối tượng lồng nhau.
    """
    lines = split_csv_lines_safe(input_str)
    data: DataMap = {}

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
            value = raw_value # Dự phòng nếu không phải là một chuỗi JSON hợp lệ

        match = _CSV_KEY_REGEX.match(raw_key)
        if not match:
            continue

        obj_name, idx_str, path = match.groups()
        index = int(idx_str)

        if obj_name not in data:
            data[obj_name] = []

        # Đảm bảo danh sách đủ dài để chứa chỉ mục
        if len(data[obj_name]) < index:
            data[obj_name].extend([None] * (index - len(data[obj_name])))

        list_index = index - 1
        if data[obj_name][list_index] is None:
            data[obj_name][list_index] = {}

        target = data[obj_name][list_index]

        if path:
            keys = path.split('.')
            # Điều hướng/tạo các đối tượng lồng nhau
            for key_part in keys[:-1]:
                target = target.setdefault(key_part, {})
            target[keys[-1]] = value
        else:
            target['value'] = value

    for key in data:
        data[key] = [entry for entry in data[key] if entry is not None]

    return data


def slurp_to_json(slurp: str):
    """
    Chức năng điều phối chính chuyển đổi một chuỗi SLURP thành một chuỗi JSON.
    """
    parser = SLURPParser()
    parsed_blocks = parser.parse(slurp)
    csv_str = format_output(parsed_blocks)
    json_obj = parse_csv_to_json(csv_str)
    return json_obj

# ---- Ví dụ sử dụng ----
if __name__ == "__main__":
    sample_slurp_input = """
>name: ĐỀ THI THỬ
>subject: VẬT LÍ
>code: ĐỀ 15
>duration: 50 phút

shareinfo:
>id: const
>info: |
\(π = 3,14\); \(T (K) = t (°C) + 273\); \(R = 8,31 J. mol^{-1}. K^{-1}\); \(N_A = 6,02. 10^{23}\) hạt/mol.

sectionHeader: PHẦN I. Thí sinh trả lời từ câu 1 đến câu 18. Mỗi câu hỏi thí sinh chỉ chọn một phương án.

shareinfo:
>id: share-10-11
>info: |
Hình bên mô tả một máy phát điện xoay chiều đơn giản. Máy phát điện xoay chiều gồm hai bộ phận chính là phần cảm và phần ứng.
>figures: fig0

qs:
>dnum: 10
>type: mcq
>shared-info: share-10-11
>qt: Máy phát điện hoạt động dựa trên
>labels:
>>a: hiện tượng cảm ứng điện từ.
>>b: hiện tượng tích điện.
>>c: hiện tượng quang điện.
>>d: hiện tượng nhiễm điện do cọ xát.

qs:
>dnum: 11
>type: mcq
>shared-info: share-10-11
>qt: Phần cảm tạo ra ...(1)..., phần ứng tạo ra ...(2)... khi máy hoạt động. Từ thích hợp điền vào vị trí (1) và (2) lần lượt là
>labels:
>>a: Từ trường, suất điện động cảm ứng.
>>b: Dòng điện, từ trường.
>>c: Suất điện động cảm ứng, từ trường.
>>d: Suất điện động cảm ứng, dòng điện.

qs:
>dnum: 12
>type: mcq
>qt: Một dây dẫn thẳng dài vô hạn có phương vuông góc với mặt phẳng trang giấy. Cho dòng điện chạy qua dây dẫn theo chiều từ trong ra ngoài. Hình nào dưới đây mô tả đúng đường sức từ trên mặt phẳng trang giấy của từ trường của dòng điện chạy trong dây dẫn?
>figures: fig1, fig2, fig3
>labels:
>>a: Hình 2.
>>b: Hình 3.
>>c: Hình 4.
>>d: Hình 1.

sectionHeader: PHẦN II. Thí sinh trả lời từ câu 1 đến câu 4. Trong mỗi ý a), b), c), d) ở mỗi câu, thí sinh chọn đúng hoặc sai.

qs:
>dnum: 3
>type: mtf-2018
>qt: Một thanh kim loại có khối lượng m = 50 g có thể trượt với ma sát không đáng kể trên hai thanh ray song song nằm ngang cách nhau một khoảng d = 4 cm. Đường ray nằm trong một từ trường đều thẳng đứng có độ lớn B = 0,3 T và có hướng như hình bên. Tại thời điểm t = 0 s, điện kế G được kết nối với thanh ray, tạo ra dòng điện không đổi I = 2 A (có chiều như hình) trong dây và thanh ray (kể cả khi dây chuyển động). Biết ban đầu thanh đứng yên.
>figures: fig4
>labels:
>>a: Lực tác dụng lên thanh là lực từ.
>>b: Từ trường do dòng điện tạo ra có hướng hướng theo chiều như từ trường bên ngoài.
>>c: Thanh kim loại chuyển động sang trái và tại lúc t = 1 s vận tốc của thanh có độ lớn là 0,48 m/s.
>>d: Quãng đường thanh đi được sau thời gian 2 s kể từ lúc thiết bị G được kết nối là 0,48 m.

qs:
>dnum: 4
>type: mtf-2018
>qt: Trong y học một đồng vị phóng xạ của Sodium thường được dùng để xác định lượng máu trong cơ thể người là \(^{24}_{11}Na\). Chu kỳ bán rã của \(^{24}_{11}Na\) là 15 giờ. Người ta lấy một lượng \(^{24}_{11}Na\) có độ phóng xạ 2,5 \(\mu\)Ci để tiêm vào một bệnh nhân. Sau 3 giờ, họ lấy ra 1 cm³ máu từ người đó thì thấy nó có 145 phân rã trong 10 giây. Cho biết đồng vị \(^{24}_{11}Na\) phóng xạ tạo ra \(^{24}_{12}Mg\).
>labels:
>>a: Đây là phân rã \(\beta^+\).
>>b: Độ phóng xạ lúc mới tiêm vào cơ thể người là 7,4 · \(10^4\) Bq.
>>c: Số nguyên tử \(^{24}_{11}Na\) trong 1 cm³ máu sau 3 giờ là 3 · \(10^5\) nguyên tử.
>>d: Thể tích máu của người đó là 5,6 lít.

sectionHeader: PHẦN III. Thí sinh trả lời từ câu 1 đến câu 6.

shareinfo:
>id: share-1-2
>info: |
Trong một hệ thống đun nước bằng năng lượng mặt trời, ánh sáng Mặt Trời được hấp thụ bởi nước chảy qua các ống trong một bộ thu nhiệt trên mái nhà. Ánh sáng Mặt Trời đi qua lớp kính trong suốt của bộ thu và làm nóng nước trong ống. Sau đó, nước nóng này được bơm vào bể chứa. Biết nhiệt dung riêng của nước là \(c_{H_2O}\) = 4200 J·\(kg^{-1}\)· \(K^{-1}\), khối lượng riêng của nước là \(D_{H_2O}\) = 1000 kg/m³ .

qs:
>dnum: 1
>type: short-2018
>shared-info: share-1-2
>qt: Biết rằng sự tỏa nhiệt của hệ thống ra không khí là không đáng kể. Năng lượng cần thiết để làm nóng 2 lít nước từ 20°C đến 100°C là x · \(10^6\) J. Tìm x (làm tròn kết quả đến chữ số hàng phần trăm).

qs:
>dnum: 2
>type: short-2018
>shared-info: share-1-2
>qt: Thực tế hệ thống chỉ hoạt động với hiệu suất 30%, nên chỉ 30% năng lượng Mặt Trời được dùng để làm nóng nước. Để làm nóng 2 lít nước từ 20°C đến 100°C thì phải cung cấp nhiệt trong thời gian t. Biết rằng cường độ ánh sáng Mặt Trời chiếu xuống là I = 1000 W·\(m^{-2}\), diện tích của bộ thu là S = 3 m². Công suất bức xạ nhiệt chiếu lên bộ thu nhiệt được cho bởi công thức sau: P = I · S. Tính t theo đơn vị phút (làm tròn kết quả đến chữ số hàng đơn vị).
"""

    json_output = slurp_to_json(sample_slurp_input)
    print(json_output)