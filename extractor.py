import os
import re
import json
import enum
from langdetect import detect
from gliner import GLiNER

from grobid_extractor import GrobidClient

class sectionType(str, enum.Enum):
    abstract = 'abstract'
    introduction = 'introduction'
    methods = 'methods'
    results = 'results'
    discussion = 'discussion'
    colclusion = 'conclusion'

class ArticleExtractor:
    def __init__(self, pdf_path, grobid_url = 'http://localhost:8070'):
        self.pdf_path = pdf_path
        self.file_name = None
        self.title = None
        self.authors = None
        self.abstract = None
        self.keywords = None
        self.language = None
        self.sections = None
        self.acknowledgements = None
        self.appendix = None
        self.references = None
        self.figures = None
        self.tables = None

        self.grobid = GrobidClient(grobid_url=grobid_url)

    def extract_from_article(self, data, output_path, file_name):
        self.file_name = file_name

        # исправление уровня подзаголовков: с основного (1) на нижний уровень (2)
        for block in data:
            if block.get('text_level', None) == 1 and re.match(r'\d+\.\d+', block.get('text', '')):
                block['text_level'] = 2
        
        titles = [(idx, block.get('text', '')) for idx, block in enumerate(data) if block.get('text_level', None) == 1]

        start_section_idx = 0
        sections_list = []

        # поиск ключевых слов
        end_kwords_idx = 0
        kwords_pattern = re.compile(r'(keywords|index terms|ключевые слова)', flags=re.I)
        for idx, block in enumerate(data):
            text = block.get('text', '')
            if re.search(kwords_pattern, text):
                if block.get('text_level', -1) > 0:
                    temp_kwords = data[idx + 1].get('text', '').strip(' .:-—')
                    end_kwords_idx = idx + 1
                else:
                    temp_kwords = re.sub(kwords_pattern, '', text).strip(' .:-—')
                    end_kwords_idx = idx
                temp_kwords = re.split(r'[,;]', temp_kwords)
                self.keywords = [kword.strip() for kword in temp_kwords]
                break
        
        # поиск аннотации
        end_abs_idx = 0
        abstract = {
                        'title': 'Abstract',
                        'text': None,
                        'type': sectionType.abstract,
                        'page_start': 0,
                        'page_end': 0
                    }
        abs_pattern = re.compile(r'аннотация|abstract', flags=re.I)
        for idx, block in enumerate(data):
            match = re.search(abs_pattern, block.get('text', ''))
            if match:
                end_title_idx = idx
                if block.get('text_level', -1) > 0:
                    end_abs_idx = idx + 1
                else:
                    end_abs_idx = idx

                abstract['text'] = ''
                abstract['page_start'] = block['page_idx']
                for i in range(end_abs_idx, len(data)):
                    if data[i].get('text_level', -1) > 0 or re.search(kwords_pattern, data[i].get('text', '')):
                        break
                    else:
                        end_abs_idx = i
                        abstract['text'] += data[i].get('text', '')
                        abstract['page_end'] = data[i]['page_idx']
        
        self.abstract = abstract['text']
        sections_list.append(abstract)
        self.language = detect(abstract['text'].split('.')[0])

        # извлечение заголовка (первый заголовок перед аннотацией)
        self.title = data[max([idx for idx, title in enumerate(titles) if title[0] < end_title_idx])].get('text', '')
        
        # обработка секций с первого заголовка после ключевых слов
        start_section_idx = max(end_kwords_idx, end_abs_idx) + 1

        # grobid: авторы и список литературы
        self.authors, self.references = self.grobid.process_pdf(pdf_path=self.pdf_path)

        # поиск индексов ключевых точек статьи: приложение, список источников и т.д.
        end_section_idx = len(data) 

        start_app_idx = 131313
        app_pattern = re.compile(r'приложение|appendix', flags=re.I)
        for idx, title in enumerate(titles):
            match = re.search(app_pattern, title[1])
            if match:
                start_app_idx = title[0]
        
        start_ack_idx = 131313
        ack_pattern = re.compile(r'благодарности|acknowledgements|acknowledgments', flags=re.I)
        for idx, block in enumerate(data):
            match = re.search(ack_pattern, block.get('text', ''))
            if match:
                if start_section_idx < idx:
                    start_ack_idx = idx
                if block.get('text_level', -1) > 0:
                    self.acknowledgements = data[idx + 1].get('text', '')
                else:
                    self.acknowledgements = re.sub(ack_pattern, '', data[idx].get('text', '')).strip(' .:-—')

        start_refs_idx = 131313
        refs_pattern = re.compile(r'список источников|список литературы|references', flags=re.I)
        for idx, title in enumerate(titles):
            match = re.search(refs_pattern, title[1])
            if match:
                start_refs_idx = title[0]

        end_section_idx = min(end_section_idx, start_ack_idx, start_app_idx, start_refs_idx)

        # обработка секций
        section_flag = False
        for idx in range(start_section_idx, end_section_idx):
            if section_flag and data[idx].get('text_level', -1) != 1:
                if data[idx]['type'] in ['text', 'equation']:
                    section['text'] += data[idx].get('text', '') + '\n'
                elif data[idx].get('sub_type', None) == 'text':
                    for item in data[idx].get('list_items', []):
                        section['text'] += item + '\n'
                elif data[idx]['type'] == 'code':
                    section['text'] += data[idx].get('code_body', '') + '\n'
                section['page_end'] = data[idx]['page_idx']
            
            if data[idx].get('text_level', -1) == 1:
                if section_flag:
                    sections_list.append(section)
                section = {
                    'title': data[idx].get('text', ''),
                    'text': '',
                    'type': None,
                    'page_start': data[idx]['page_idx'],
                    'page_end': data[idx]['page_idx']
                }
                section_flag = True
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
        
        print(f'-- {self.file_name} is done!')

    def dump_to_json(self, output):
        article = {
            'title': self.title,
            'authors': self.authors,
            'abstract': self.abstract,
            'keywords': self.keywords,
            'language': self.language,
            'sections': self.sections,
            'acknowledgements': self.acknowledgements,
            'appendix': self.appendix,
            'references': self.references,
            'figures': self.figures,
            'tables': self.tables,
        }
        with open(os.path.join(output, f'extract_{self.file_name}.json'), 'w', encoding='utf-8') as f:
            json.dump(article, f, ensure_ascii=False, indent=4)