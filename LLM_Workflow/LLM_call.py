from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from LLM_Workflow.response_service import ResponseService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CLI simple para consultar el copiloto analitico con Gemini."
    )
    parser.add_argument(
        "question",
        nargs="+",
        help="Pregunta en lenguaje natural para el copiloto.",
    )
    parser.add_argument(
        "--show-json",
        action="store_true",
        help="Muestra tambien el resultado estructurado del motor analitico.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    question = " ".join(args.question)
    service = ResponseService()
    result = service.generate_answer(question)

    print("\nRespuesta del copiloto:\n")
    print(result["final_answer"])

    if args.show_json:
        print("\nResultado estructurado:\n")
        serializable = dict(result["analytics_result"])
        serializable.pop("support_table", None)
        print(json.dumps(serializable, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
