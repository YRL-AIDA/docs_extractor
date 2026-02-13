import os
import re
import json
import enum
from langdetect import detect

class sectionType(str, enum.Enum):
    abstract = 'abstract'
    introduction = 'introduction'
    methods = 'methods'
    results = 'results'
    discussion = 'discussion'
    colclusion = 'conclusion'

class ArticleExtractor():
    def __init__(self):
        self.file_name = None
        self.title = None
        self.abstract = None
        self.keywords = None
        self.language = None
        self.sections = None
        self.references = None
        self.figures = None
        self.tables = None

    def extract_from_article(self, data, output_path, file_name):
        self.file_name = file_name

        # поиск ключевых слов
        kwords_pattern = re.compile(r'(keywords|ключевые слова)', flags=re.I)
        for idx, block in enumerate(data):
            text = block.get('text', '')
            if re.search(kwords_pattern, text):
                if block.get('text_level', -1) > 0:
                    temp_kwords = data[idx + 1].get('text', '').strip(' .')
                else:
                    temp_kwords = re.sub(kwords_pattern, '', text).strip(' .')
                temp_kwords = re.split(r'[,;]', temp_kwords)
                self.keywords = [kword.strip() for kword in temp_kwords]
                break

        # исправление уровня подзаголовков: с основного (1) на нижний уровень (2)
        for block in data:
            if block.get('text_level', None) == 1 and re.match(r'\d+\.\d+', block.get('text', '')):
                block['text_level'] = 2
        
        titles = [(idx, block.get('text', '')) for idx, block in enumerate(data) if block.get('text_level', None) == 1]
        self.title = titles[0][1]
        self.language = detect(titles[0][1])

        start_section_idx = 1 # начальный индекс цикла по секциям/главам, пропуская заголовок статьи (и аннотацию, если она не выделена как глава)
        sections_list = []
        
        # поиск аннотации
        abstract = {
                        'title': 'Abstract',
                        'text': None,
                        'type': sectionType.abstract,
                        'page_start': 0,
                        'page_end': 0
                    }
        abs_pattern = re.compile(r'аннотация|abstract', flags=re.I)
        if not re.search(abs_pattern, titles[1][1]): # если аннотация не выделена как глава
            for idx, block in enumerate(data):
                match = re.search(abs_pattern, block.get('text', ''))
                if match:
                    abstract['text'] = block['text']
                    abstract['page_start'] = block['page_idx']
                    abstract['page_end'] = block['page_idx']
                    break
        else:
            start_section_idx = 2
            abstract['text'] = data[titles[1][0] + 1].get('text', '')
            abstract['page_start'] = data[titles[1][0] + 1]['page_idx']
            abstract['page_end'] = data[titles[1][0] + 1]['page_idx']
        
        self.abstract = abstract['text']
        sections_list.append(abstract)

        # обработка списка литературы
        references_list = []

        ref_list = []
        for idx in range(titles[-1][0], len(data)):
            if data[idx].get('sub_type', None) == 'ref_text':
                ref_list += data[idx].get('list_items', [])
            elif data[idx]['type'] == 'ref_text':
                ref_list.append(data[idx].get('text', ''))

        year_pattern = re.compile(r'[//\s\(](\d{4})[\.,;)\s]')
        for idx, ref in enumerate(ref_list):
            match = re.search(year_pattern, ref)
            if match:
                year_ref = match.group(1)
            else:
                year_ref = None

            reference = {
                'id': idx + 1,
                'text': ref,
                'authors': None,
                'year': year_ref
            }
            references_list.append(reference)
        self.references = references_list

        # обработка секций
        for idx in range(start_section_idx, len(titles) - 1):
            section = {'title': titles[idx][1], 'text': '', 'type': None, 'page_start': data[titles[idx][0]]['page_idx']}

            for jdx in range(titles[idx][0] + 1, titles[idx + 1][0]):
                if data[jdx]['type'] in ['text', 'equation']:
                    section['text'] += data[jdx].get('text', '') + '\n'
                elif data[jdx].get('sub_type', None) == 'text':
                    for item in data[jdx].get('list_items', []):
                        section['text'] += item + '\n'
                elif data[jdx]['type'] == 'code':
                    section['text'] += data[jdx].get('code_body', '') + '\n'

            section['page_end'] = data[titles[idx + 1][0] - 1]['page_idx']
            sections_list.append(section)
        self.sections = sections_list

        # обработка визуальных элементов
        ## иллюстрации
        figures_list = []
        idx = 0
        img_counter = 0
        while idx < len(data):
            if data[idx]['type'] == 'image':
                img_counter += 1
                figure = {
                    'id': img_counter,
                    'type': data[idx]['type'],
                    'caption': None,
                    'img_path': None,
                    'page': data[idx]['page_idx'],
                }

                img_path = []
                img_path.append(data[idx]['img_path'])
                if len(data[idx].get('image_caption', '')) == 0:
                    for jdx in range(idx + 1, len(data)):
                        idx = jdx
                        if data[jdx]['type'] == 'image':
                            img_path.append(data[jdx].get('img_path', ''))
                            if len(data[jdx].get('image_caption', '')) != 0:
                                figure['caption'] = data[jdx].get('image_caption', None)
                                break
                        else:
                            break
                else:
                    figure['caption'] = data[idx].get('image_caption', None)

                figure['img_path'] = img_path
                figures_list.append(figure)
            idx += 1
        self.figures = figures_list

        ## таблицы
        tables_list = []
        tables = [block for block in data if block['type'] == 'table']
        for idx, block in enumerate(tables):
            caption = block.get('table_caption', []) + block.get('table_footnote', [])
            table = {
                'id': idx + 1, 
                'type': block['type'], 
                'caption': caption, 
                'table_body': block.get('table_body', None), 
                'img_path': os.path.join(output_path, block.get('img_path', None)), 
                'page': block['page_idx']
            }
            tables_list.append(table)
        self.tables = tables_list
        
        print('Done!')

    def dump_to_json(self, output):
        article = {
            'title': self.title,
            'abstract': self.abstract,
            'keywords': self.keywords,
            'language': self.language,
            'sections': self.sections,
            'references': self.references,
            'figures': self.figures,
            'tables': self.tables,
        }
        with open(os.path.join(output, f'extract_{self.file_name}.json'), 'w', encoding='utf-8') as f:
            json.dump(article, f, ensure_ascii=False, indent=4)