# Система извлечения данных из научных документов
### Установка
Чтобы начать работу, после клонирования репозитория нужно установить необходимые зависимости (используемые версии **python 3.10-3.12, cuda 12.8** => в requirements ставится **torch+cu128**)
````commandline
pip install --upgrade pip
pip install -r requirements.txt
````

Скачать **MinerU** для локального запуска можно по [ссылке](https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B "huggingface MinerU"), разместить ее в папке `model`, также потребуется поставить [Grobid](https://grobid.readthedocs.io/en/latest/Grobid-docker/ "GROBID Documentation: Run with Docker") для извлеченения авторов.

````commandline
docker pull grobid/grobid:0.9.0-crf
docker run --rm --init --ulimit core=0 -p 8070:8070 grobid/grobid:0.9.0-crf
````

### Работа с системой

`mineru_compact.py` -- содержит в себе необходимые функции для инференса модели, возвращает *content_list*

`grobid_extractor.py` -- содержит класс GrobidClient, который обращается к Grobid для извлечения авторов.

`extractor.py` -- содержит класс ArticleExtractor, который реализуют основную логику извлечения данных

`demo.py` -- пример работы (см. ниже):

````python
import os
import torch

from extractor import ArticleExtractor
from mineru_compact import parse_doc

# данные заполнены для примера
path_to_file = 'articles/demo1.pdf'  # путь к pdf-файлу (статье)
output_dir = 'output/' # директория для сохранения результатов
path_to_model = 'model/MinerU2.5-2509-1.2B' # путь к локально установленной модели
backend = 'hybrid-auto-engine' # [hybrid-auto-engine, pipeline, vlm-auto-engine]
grobid_url = 'http://localhost:8070'

file_name, _ext = os.path.splitext(os.path.basename(path_to_file))
content_list = parse_doc([path_to_file], output_dir, backend=backend, model_path=path_to_model)

extractor = ArticleExtractor(pdf_path=path_to_file, grobid_url=grobid_url)
extractor.extract_from_article(content_list, output_dir, file_name) # извлечение данных из документа и создание структуры
extractor.dump_to_json(output_dir) # сохранение результатов в формате json
````
Для извлечения данных необходимо запустить `demo.py` (со всеми аргументами можно ознакомиться с помощью `--help`):
````commandline
python demo.py -i path_to_file.pdf -o output_dir/
````

Выходной файл имеет следующую структуру (**output/demo1.json**):
````javascript
{
	title : str,
	authors: [
		{
			name: str,
			affiliations: str,
			orcid: str
		}
	],
	abstract : str,
	keywords : [str,],
	language: str,
	sections: [ # содержание статьи, разбитое по секциям (главам)
		{
			title : str,
			text : str,
			type : str (enum.Enum),
			page_start : int,
			page_end : int
		},
	],
	references: [ 
		{
			id : int,
			text : str,
			authors : [str,],
			year : int,
			page_end : int
		},
	],
	figures: [
		{
			id : str,
			type : str,
			caption : [str,],
			img_path : [str,],
			page : int
		},
	],
	tables: [
		{
			id : str,
			type : str,
			caption : [str,],
			table_body : str,
			img_path : str,
			page : int
		}
	]
}
````

