# Система извлечения данных из научных документов
### Установка
Чтобы начать работу после клонирования репозитория, нужно установить необходимые зависимости (желательно в виртуальном окружении, с версией **python 3.10-3.12**)
````commandline
pip install --upgrade pip
pip install uv
uv pip install -U "mineru[all]"
pip install langdetect
````
или
````commandline
git clone https://github.com/opendatalab/MinerU.git
cd MinerU
uv pip install -e .[all]
pip install langdetect
````
Скачать модель для локального запуска можно по [ссылке](https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B "huggingface"), разместить ее в папке `model`

### Работа с системой

`mineru_compact.py` -- содержит в себе необходимые функции для инференса модели, возвращает *content_list*

`extractor.py` -- содержит класс ArticleExtractor, который реализуют основную логику извлечения данных

`demo.py` -- пример работы (см. ниже):

````python
import os

from extractor import ArticleExtractor
from mineru_compact import parse_doc

path_to_file = 'articles/demo1.pdf'  # путь к pdf-файлу (статье)
output_dir = 'output/' # директория для сохранения результатов
path_to_model = 'model/MinerU2.5-2509-1.2B' # путь к локально установленной модели
backend = 'hybrid-auto-engine' # [hybrid-auto-engine, pipeline, vlm-auto-engine]

file_name, _ext = os.path.splitext(os.path.basename(path_to_file))
content_list = parse_doc([path_to_file], output_dir, backend=backend, model_path=path_to_model)

extractor = ArticleExtractor()
extractor.extract_from_article(content_list, output_dir, file_name) # извлечение данных из документа и создание структуры
extractor.dump_to_json(output_dir) # сохранение результатов в формате json
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

