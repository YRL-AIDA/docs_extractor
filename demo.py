from extractor import ArticleExtractor
from mineru_compact import parse_doc

path_to_file = 'articles/demo1.pdf'
output_dir = 'output/'
path_to_model = 'model/MinerU2.5-2509-1.2B'

extractor = ArticleExtractor()
extractor.extract_from_article(path_to_file, output_dir, path_to_model)
extractor.dump_to_json(output_dir)