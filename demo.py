import os
import torch

from extractor import ArticleExtractor
from mineru_compact import parse_doc

path_to_file = 'articles/demo1.pdf'
output_dir = 'output/'
path_to_model = 'model/MinerU2.5-2509-1.2B'
backend = 'hybrid-auto-engine' # [hybrid-auto-engine, pipeline, vlm-auto-engine]
ner_path = 'nvidia/gliner-pii' # path to gliner-pii (ner) for author`s extraction [default: 'nvidia/gliner-pii' from hf]
ner_device = 'cuda' if torch.cuda.is_available() else 'cpu' # for ner model
print(f'Device: {ner_device}')
ner_batch_size = 16

file_name, _ext = os.path.splitext(os.path.basename(path_to_file))
content_list = parse_doc([path_to_file], output_dir, backend=backend, model_path=path_to_model)

extractor = ArticleExtractor(ner_path = ner_path, device = ner_device, ner_batch_size = ner_batch_size)
extractor.extract_from_article(content_list, output_dir, file_name)
extractor.dump_to_json(output_dir)