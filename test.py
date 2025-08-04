#!/usr/bin/env python3

import os
import sys
import argparse
from pathlib import Path
from termcolor import cprint
import questionary

# Gợi ý ngôn ngữ cho Markdown code block
def get_language_hint(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    ext_map = {
        '.js': 'javascript', '.ts': 'typescript', '.jsx': 'jsx', '.tsx': 'tsx',
        '.py': 'python', '.java': 'java', '.cs': 'csharp', '.php': 'php',
        '.rb': 'ruby', '.go': 'go', '.rs': 'rust', '.swift': 'swift',
        '.kt': 'kotlin', '.scala': 'scala', '.html': 'html', '.css': 'css',
        '.scss': 'scss', '.sh': 'bash', '.ps1': 'powershell', '.sql': 'sql',
        '.xml': 'xml', '.txt': ''
    }
    return ext_map.get(ext, '')

# Xử lý file
def process_file(file_path: Path, base_path: Path) -> str:
    if file_path.stat().st_size > 1_000_000:  # Ví dụ: bỏ qua file > 1MB
        return ''
    relative_path = file_path.relative_to(base_path).as_posix()
    try:
        with open(file_path, encoding='utf-8') as f:
            content = f.read().strip()
        lang = get_language_hint(file_path)
        return f"""---
File: {relative_path}
---
```{lang}
{content}
```\n\n"""
    except Exception as e:
        return f"""---
File: {relative_path}
---
Error reading file: {str(e)}\n\n"""

# Đệ quy duyệt thư mục
def scan_directory(dir_path: Path, base_path: Path, allowed_extensions: list[str]) -> str:
    output = ''
    for entry in dir_path.iterdir():
        if entry.name in ('node_modules', '.git') or entry.name.startswith('.'):
            continue
        if entry.is_dir():
            output += scan_directory(entry, base_path, allowed_extensions)
        elif entry.is_file():
            if not allowed_extensions or entry.suffix.lower() in allowed_extensions:
                output += process_file(entry, base_path)
    return output

# CLI chọn nhiều file/thư mục
def select_files_or_dirs() -> list[str]:
    cwd = Path.cwd()
    entries = [
        {
            'name': e.name + '/' if e.is_dir() else e.name,
            'value': str(e)
        }
        for e in cwd.iterdir()
        if not e.name.startswith('.')
    ]

    selected = questionary.checkbox(
        "Chọn file/thư mục muốn xử lý:",
        choices=entries
    ).ask()

    if not selected:
        print("❌ Phải chọn ít nhất một mục.")
        sys.exit(1)

    return selected

# Main
def main():
    parser = argparse.ArgumentParser(description='Tạo prompt Markdown từ nhiều thư mục và file.')
    parser.add_argument('-o', '--output', default='output.md', help='Tên file markdown đầu ra')
    parser.add_argument('-e', '--ext', default='', help='Chỉ lấy file có phần mở rộng, ví dụ: .js,.ts,.json')
    args = parser.parse_args()

    selected_paths = select_files_or_dirs()
    extensions = [e.strip().lower() for e in args.ext.split(',')] if args.ext else []

    final_output = "# Project Analysis Prompt\n\nAnalyze the following project structure and file contents.\n\n"
    for item in selected_paths:
        path_obj = Path(item)
        base_path = path_obj if path_obj.is_dir() else path_obj.parent
        final_output += f"## From: {path_obj.name}\n\n"
        if path_obj.is_dir():
            final_output += scan_directory(path_obj, path_obj, extensions)
        elif path_obj.is_file():
            final_output += process_file(path_obj, base_path)

    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(final_output)

    cprint(f"✅ Prompt đã được tạo ở: {args.output}", 'green')

if __name__ == "__main__":
    main()
