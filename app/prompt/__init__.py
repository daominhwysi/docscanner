import re
import os
from google.genai import types
BASE_DIR = os.path.dirname(__file__)  
# Tương đương với readTextFile
def read_text_file(dir_path, file_path):
    with open(os.path.join(dir_path, file_path), 'r', encoding='utf-8') as file:
        return file.read()

def process_examples_xml(content: str):
    matches = re.findall(r"<example>([\s\S]*?)<\/example>", content)
    examples = []

    for match in matches:
        user_query_match = re.search(r"<user_query>([\s\S]*?)<\/user_query>", match)
        assistant_response_match = re.search(r"<assistant_response>([\s\S]*?)<\/assistant_response>", match)

        user_query = user_query_match.group(1).strip() if user_query_match else ''
        assistant_response = assistant_response_match.group(1).strip() if assistant_response_match else ''

        examples.append({
            'user_query': user_query,
            'assistant_response': assistant_response
        })

    # Mapping to contents
    contents = []
    for example in examples:
        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=example['user_query']),
                ]
            )
        )

        contents.append(
            types.Content(
                role="model",
                parts=[
                    types.Part.from_text(text=example['assistant_response']),
                ]
            )
        )


    return contents

# Initial Slurp
def get_initial_slurp_prompt():
    return read_text_file(BASE_DIR, './convert2slurp/initial_slurp.md')

def get_initial_slurp_examples():
    content = read_text_file(BASE_DIR, './convert2slurp/initial_example.xml')
    return process_examples_xml(content)

# Continuation Prompt
def get_slurp_continuation_prompt():
    return read_text_file(BASE_DIR, './convert2slurp/continuation_slurp.md')

def get_slurp_continuation_examples():
    initial_example = process_examples_xml(read_text_file(BASE_DIR, './convert2slurp/initial_example.xml'))
    cont_example = process_examples_xml(read_text_file(BASE_DIR, './convert2slurp/continuation_example.xml'))
    return initial_example + cont_example

# Extraction Prompts
def get_extraction_figure_prompt():
    return read_text_file(BASE_DIR, './extraction/figure.txt')

def get_extraction_non_figure_prompt():
    return read_text_file(BASE_DIR, './extraction/non_docfigure.txt')
