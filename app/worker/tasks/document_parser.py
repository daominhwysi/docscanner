import asyncio
import logging

from google.genai import types
import re
from app.worker.instance import worker
from app.utils.agent import GeminiAgent
from app.prompt import (
    get_initial_slurp_prompt,
    get_slurp_continuation_prompt,
    get_initial_slurp_examples,
    get_slurp_continuation_examples
)
import json  
from app.postprocessing.replaceimgfig import replace_img_to_fig, replace_fig2img_immutable
from app.db.client import get_session
from app.db.models import TaskType
from app.services.create_task import create_task, update_task_result
from app.postprocessing.slurp2json import slurp_to_json, autofix_missing_pipes
from app.services.logging_config import get_logger

logger = get_logger()

PARSER_MODEL = "gemini-2.5-flash"
MAX_CONTINUATION_ATTEMPTS = 5


def clean_output(text: str) -> str:
    # Loại bỏ tất cả marker @slurp_...@
    return re.sub(r"@slurp(_resume)?(_incomplete)?_(start|end)@", "", text).strip()

async def llmAsParser(text: str):
    text, figures = replace_img_to_fig(text)

    initial_contents = get_initial_slurp_examples() + [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=f"@mml_start@{text}@mml_end@")]
        )
    ]
    initial_instruction = get_initial_slurp_prompt()
    initial_config = types.GenerateContentConfig(
        system_instruction=[types.Part.from_text(text=initial_instruction)],
        temperature=0.2
    )

    initial_response = await GeminiAgent(
        model=PARSER_MODEL,
        contents=initial_contents,
        config=initial_config
    )
    slurp_content = initial_response.text

    if "@slurp_start@" in slurp_content and "@slurp_end@" in slurp_content:
        return clean_output(slurp_content), figures

    if "@slurp_start@" not in slurp_content:
        raise ValueError("Parser did not return start marker.")

    slurp_content = clean_output(slurp_content)
    cont_instruction = get_slurp_continuation_prompt()
    continuation_config = types.GenerateContentConfig(
        system_instruction=[types.Part.from_text(text=cont_instruction)],
        temperature=0.2
    )
    continuation_examples = get_slurp_continuation_examples()

    for attempt in range(MAX_CONTINUATION_ATTEMPTS):
        cont_contents = continuation_examples + [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=f"@mml_start@{text}@mml_end@\n"
                             f"@slurp_incomplete_start@{slurp_content}@slurp_incomplete_end@"
                    )
                ]
            )
        ]

        cont_response = await GeminiAgent(
            model=PARSER_MODEL,
            contents=cont_contents,
            config=continuation_config,
        )
        cont_output = cont_response.text

        if "@slurp_resume_end@" in cont_output or "@slurp_end@" in cont_output:
            slurp_content += clean_output(cont_output)
            return slurp_content, figures

        slurp_content += clean_output(cont_output)

    logger.info("Parser failed to complete after maximum retries.")
    return slurp_content, figures

@worker.task(name="documentParsing", max_concurrency=20, max_retries=0)
async def parse_document(task_id: str, text: str):
    logger.info(f"Parsing document {task_id}")
    try:
        slurp_content, figures = await llmAsParser(text=text)
        fixed_slurp_content = autofix_missing_pipes(slurp_content)
        parsed_json = slurp_to_json(fixed_slurp_content)
        
        refine_json = replace_fig2img_immutable(parsed_json, figures)
        
        dumped_json = json.dumps({"raw" : slurp_content, "parsed": refine_json }, ensure_ascii=False)
        
        with get_session() as session:
            update_task_result(session=session, task_id=task_id, result=dumped_json)  
        logger.info(f"[{task_id}] Task completed successfully.")
    except Exception as e:
        logger.exception(f"[{task_id}] Failed to parse document : {e}")


if __name__ == "__main__":
    async def main():
        from pathlib import Path
        import json

        # 1. Đọc input
        input_file = Path("tests/de15.txt")
        text = input_file.read_text(encoding="utf-8")

        # 2. Chạy parser
        slurp_content, figures = await llmAsParser(text)
        # 3. Chuyển thành JSON và thay fig → img
        parsed_json = slurp_to_json(autofix_missing_pipes(slurp_content))
        refine_json = replace_fig2img_immutable(parsed_json, figures)

        # 4. Ghi kết quả cuối cùng (có URL) ra file
        output_path = Path("tests/output.json")
        output_path.write_text(
            json.dumps(refine_json, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"✅ Đã ghi kết quả có URL vào {output_path}")

    asyncio.run(main())
