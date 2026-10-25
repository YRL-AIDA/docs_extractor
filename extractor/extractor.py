import os
import re
import json
import enum
from langdetect import detect
from gliner import GLiNER

from extractor.grobid_extractor import GrobidClient
from extractor.llm_extractor import LLMExtractor, PromptType


pattern = {
    'kword': {
        'ru': r'ключевые слова',
        'en': r'keywords|index terms',
        None: r'ключевые слова|keywords|index terms'
    },
    'abs': {
        'ru': r'аннотация',
        'en': r'abstract',
        None: r'аннотация|abstract'
    },
    'app': {
        'ru': r'приложение',
        'en': r'appendix',
        None: r'приложение|appendix'
    },
    'ack': {
        'ru': r'благодарности',
        'en': r'acknowledgements|acknowledgments',
        None: r'благодарности|acknowledgements|acknowledgments'
    },
    'refs': {
        'ru': r'список источников|список литературы|литература|список использованных источников|библиографический список',
        'en': r'references',
        None: r'список источников|список литературы|литература|список использованных источников|библиографический список|references'
    },
    'authors': {
        'ru': r'об авторах|сведения об авторах',
        'en': r'about the authors|note on authors|contributors',
        None: r'об авторах|сведения об авторах|about the authors|note on authors|contributors'
    },
}

class SectionType(str, enum.Enum):
    ABSTRACT = 'abstract'
    INTRODUCTION = 'introduction'
    METHODS = 'methods'
    RESULTS = 'results'
    DISCUSSION = 'discussion'
    CONCLUSION = 'conclusion'

class ArticleExtractor:

    def __init__(self, pdf_path, model_name, llm_url = 'http://localhost:11434/api/generate'):
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

        self.llm_extractor = LLMExtractor(
            url = llm_url,
            model_name = model_name
        )

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
        end_title_idx = 0
        end_kwords_idx = 0
        kwords_pattern = re.compile(pattern["kword"][self.language], flags=re.I)
        for idx, block in enumerate(data):
            text = block.get('text', '')
            if re.search(kwords_pattern, text):
                end_title_idx = idx
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
        end_abs_idx = 1
        abstract = {
            'title': 'Abstract',
            'text': None,
            'type': SectionType.ABSTRACT,
            'page_start': 0,
            'page_end': 0
        }
        abs_pattern = re.compile(pattern["abs"][None], flags=re.I)
        stop_fl = False
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
                        stop_fl = True
                        break
                    else:
                        end_abs_idx = i
                        abstract['text'] += data[i].get('text', '')
                        abstract['page_end'] = data[i]['page_idx']
            if stop_fl:
                break

        if abstract['text'] is None:
            block_lst = []
            end_abs_search_idx = 0
            if end_kwords_idx != 0:
                end_abs_search_idx = min([title[0] for idx, title in enumerate(titles) if title[0] > end_kwords_idx])
            else:
                for idx, block in enumerate(data):
                    if block.get('page_idx', float('inf')) > 1:
                        end_abs_search_idx = min(end_abs_search_idx, idx)
            for idx in range(end_abs_search_idx):
                    block_lst.append(data[idx])
            abstract_text = json.dumps(block_lst, ensure_ascii=False, indent=4)
            temp_abstract = self.llm_extractor.extract(type=PromptType.ABSTRACT, text=abstract_text, language='None')
            abstract['text'] = temp_abstract['abstract']
            if len(temp_abstract['index']) > 0:
                start_abs_idx = temp_abstract['index'][0]
                end_abs_idx = temp_abstract['index'][-1]
            else:
                start_abs_idx = None
                end_abs_idx = None
            abstract['page_start'] = data[start_abs_idx]['page_idx']
            abstract['page_end'] = data[end_abs_idx]['page_idx']

        self.abstract = abstract['text']
        sections_list.append(abstract)
        if abstract['text'] is not None:
            temp_lang = detect(abstract['text'])
            self.language = temp_lang if temp_lang in ('ru', 'en') else None

        # поиск индексов ключевых точек статьи: приложение, список источников и т.д.
        end_section_idx = len(data)
        end_refs_idx = len(data)

        start_app_idx = 131313
        app_pattern = re.compile(pattern['app'][self.language], flags=re.I)
        for idx, title in enumerate(titles):
            match = re.search(app_pattern, title[1])
            if match:
                start_app_idx = title[0]
                break

        start_ack_idx = 131313
        ack_pattern = re.compile(pattern['ack'][self.language], flags=re.I)
        for idx, block in enumerate(data):
            match = re.search(ack_pattern, block.get('text', ''))
            if match:
                if start_section_idx < idx:
                    start_ack_idx = idx
                if block.get('text_level', -1) > 0:
                    self.acknowledgements = data[idx + 1].get('text', '')
                else:
                    self.acknowledgements = re.sub(ack_pattern, '', data[idx].get('text', '')).strip(' .:-—')
                break

        start_refs_idx = 131313
        refs_pattern = re.compile(pattern['refs'][self.language], flags=re.I)
        for idx, title in enumerate(titles):
            match = re.search(refs_pattern, title[1])
            if match:
                start_refs_idx = title[0]
                break

        # обработка списка литературы
        references_text = ''
        titles_idx = [x[0] for x in titles] + [start_app_idx, start_ack_idx, len(data)] # возможны заголовки, которые и закончат раздел
        end_refs_idx = min([x for x in titles_idx if x > start_refs_idx])
        temp_refs_idx = 0 # фактический конец списка источников (в цикле уточнится как последний добавленный элемент)
        for idx in range(start_refs_idx + 1, end_refs_idx):
            if data[idx].get('sub_type', '') == 'ref_text':
                references_text += '\n'.join(data[idx]['list_items']) + '\n'
                temp_refs_idx = idx
            elif data[idx]['type'] == 'ref_text':
                references_text += data[idx].get('text', '')
                temp_refs_idx = idx
        end_refs_idx = temp_refs_idx
        self.references = self.llm_extractor.extract(type=PromptType.REFERENCES, text=references_text, language=self.language)

        # маркировка блоков на иностранном для статьи языке
        email_orcid_pattern = re.compile(r'([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,7}|\d{4}-\d{4}-\d{4}-\d{4})', flags=re.I) # дополнительная информация, которая нужна (независимо от языка)
        if self.language is not None:
            for block in data:
                # обработка первых двух страниц (название, аннотация, ключевые слова)
                if block['page_idx'] >= 2:
                    break
                temp_text = ''
                block_lang = None
                if block['type'] == 'text':
                    temp_text += block.get('text', 'empty')
                elif block.get('sub_type', '') == 'text':
                    temp_text += ' '.join(block.get('list_items', []))
                if len(temp_text) != 0 and not re.search(email_orcid_pattern, temp_text):
                    try:
                        block_lang = detect(temp_text)
                        if block_lang != self.language and block_lang not in ('bg', 'mk'): # иногда путает ru с bg
                            block['skip'] = True
                    except Exception as e:
                        print('Ошибка перевода: ', e)

            # все, что после списка источников
            for idx in range(end_refs_idx, len(data)):
                temp_text = ''
                block_lang = None
                if data[idx]['type'] in ('text', 'ref_text'):
                    temp_text += data[idx].get('text', 'empty')
                elif data[idx].get('sub_type', '') in ('text', 'ref_text'):
                    temp_text += ' '.join(data[idx].get('list_items', []))
                if len(temp_text) != 0 and not re.search(email_orcid_pattern, temp_text):
                    try:
                        block_lang = detect(temp_text)
                        if block_lang != self.language and block_lang not in ('bg', 'mk'):
                            data[idx]['skip'] = True
                    except Exception as e:
                        print('Ошибка перевода: ', e)

        # извлечение заголовка (первый заголовок перед аннотацией
        temp_title = None
        for idx in range(end_title_idx):
            if not data[idx].get('skip', False) and data[idx].get('text_level', None) == 1:
                temp_title = data[idx]['text']
        self.title = temp_title


        # обработка секций с первого заголовка после ключевых слов
        start_section_idx = max(end_kwords_idx, end_abs_idx) + 1

        header_text = ''
        # от заголовка до начала аннотации
        for idx in range(end_title_idx):
            if data[idx]['type'] == 'list':
                header_text += '\n'.join(data[idx]['list_items']) + '\n'
            else:
                header_text += data[idx].get('text', '') + '\n'
        # footnote (может содержаться дополнительная информация по авторам)
        for idx in range(end_abs_idx, len(data)):
            if data[idx]['type'] == 'page_footnote':
                header_text += data[idx].get('text', '')
            if data[idx]['page_idx'] > 0:
                break

        # об авторах (дополнительный блок в конце статьи)
        authors_pattern = re.compile(pattern["authors"][self.language], flags=re.I)
        temp_authors_text = ''
        if end_refs_idx + 1 != len(data): # если список литературы не является последним, что есть в статье
            for idx in range(end_refs_idx + 1, len(data)):
                if not data[idx].get('skip', False):
                    if data[idx]['type'] == 'list':
                        temp_authors_text += '\n'.join(data[idx]['list_items'])
                    if data[idx]['type'] in ('text', 'ref_text'):
                        temp_authors_text += data[idx]['text']
                if data[idx].get('text_level', 0) == 1 and re.search(authors_pattern, data[idx].get('text', '')):
                    temp_authors_text = ''
        header_text += temp_authors_text

        temp_header = self.llm_extractor.extract(type=PromptType.HEADER, text=header_text, language=self.language)
        self.authors = temp_header['authors']
        if temp_header['title'] is not None or temp_header['title'] != '':
            self.title = temp_header['title']
        if self.acknowledgements is None:
            self.acknowledgements = temp_header['acknowledgements']

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
                data[idx]['link'] = figure['id']
                if len(data[idx].get('image_caption', '')) == 0:
                    for jdx in range(idx + 1, len(data)):
                        idx = jdx
                        if data[jdx]['type'] == 'image':
                            data[jdx]['link'] = figure['id']
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

        end_section_idx = min(end_section_idx, start_ack_idx, start_app_idx, start_refs_idx)

        # обработка секций
        section = {
            'title': None,
            'text': [],
            'type': None,
            'page_start': data[start_section_idx]['page_idx'],
            'page_end': None
        }
        table_counter = 0
        for idx in range(start_section_idx, end_section_idx):
            if data[idx].get('text_level', -1) != 1 and not data[idx].get('skip', False):
                if data[idx]['type'] in ['text', 'equation']:
                    paragraph = {
                        'type': data[idx]['type'],
                        'value': data[idx].get('text', '')
                    }
                    section['text'].append(paragraph)
                elif data[idx].get('sub_type', None) == 'text':
                    paragraph = {
                        'type': 'list',
                        'value': data[idx].get('list_items', '')
                    }
                    section['text'].append(paragraph)
                elif data[idx]['type'] == 'code':
                    paragraph = {
                        'type': data[idx]['type'],
                        'value': data[idx].get('code_body', '')
                    }
                    section['text'].append(paragraph)
                elif data[idx]['type'] == 'image':
                    paragraph = {
                        'type': 'figure',
                        'value': data[idx].get('link', None)
                    }
                    if paragraph not in section['text']:
                        section['text'].append(paragraph)
                elif data[idx]['type'] == 'table':
                    table_counter += 1
                    paragraph = {
                        'type': 'table',
                        'value': table_counter
                    }
                    section['text'].append(paragraph)
                section['page_end'] = data[idx]['page_idx']

            if data[idx].get('text_level', -1) == 1 and not data[idx].get('skip', False):
                print(data[idx].get('text', ''))
                if len(section['text']) != 0:
                    sections_list.append(section)
                section = {
                    'title': data[idx].get('text', ''),
                    'text': [],
                    'type': None,
                    'page_start': data[idx]['page_idx'],
                    'page_end': data[idx]['page_idx']
                }
        sections_list.append(section)
        self.sections = sections_list

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
