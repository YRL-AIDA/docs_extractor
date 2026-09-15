import os
import torch
import argparse

from extractor.extractor import ArticleExtractor
from extractor.mineru_compact import parse_doc

def parse_args(desc=''):
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        '-i',
        '--input_path',
        type=str,
        required=True,
        help='Input path to the pdf file'
    )
    parser.add_argument(
        '-o',
        '--output_path',
        type=str,
        default='',
        help='Output path for saving result'
    )
    parser.add_argument(
        '-m',
        '--model_path',
        type=str,
        default='model/MinerU2.5-2509-1.2B',
        help='Path to the model (MinerU)'
    )
    parser.add_argument(
        '-b',
        '--backend',
        type=str,
        default='hybrid-auto-engine',
        help='Backend of the model (MinerU)'
    )
    parser.add_argument(
        '-l',
        '--llm_url',
        type=str,
        default='http://localhost:11434/api/generate',
        help='LLM URL'
    )
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    path_to_file = args.input_path
    output_dir = args.output_path
    path_to_model = args.model_path
    backend = args.backend
    llm_url = args.llm_url
    llm_model_name = "hf.co/ornith-ai/Ornith-1.5-9B-GGUF:Q5_K_M" # get name of model from ollama/api/tags

    file_name, _ext = os.path.splitext(os.path.basename(path_to_file))

    content_list = parse_doc([path_to_file], output_dir, backend=backend, model_path=path_to_model)
    extractor = ArticleExtractor(pdf_path=path_to_file, model_name=llm_model_name, llm_url=llm_url)
    extractor.extract_from_article(content_list, output_dir, file_name)
    extractor.dump_to_json(output_dir)
