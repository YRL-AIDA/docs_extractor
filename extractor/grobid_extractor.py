import re
import requests
from lxml import etree

class GrobidClient:
    ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
    def __init__(self, grobid_url='http://localhost:8070'):
        self.grobid_url = grobid_url

    def process_pdf(self, pdf_path):
        authors = self._extract_authors(pdf_path)
        refs = self._extract_refs(pdf_path)
        return authors, refs

    def _extract_authors(self, pdf_path):
        with open(pdf_path, 'rb') as pdf_file:
            files = {'input': (pdf_path, pdf_file)}
            data = {'includeRawAffiliations': 1}
            url = f"{self.grobid_url}/api/processHeaderDocument"
            response = requests.post(
                url=url,
                files=files,
                data=data,
                timeout=300,
                headers={'Accept': 'application/xml'}
            )

            if response.status_code == 200:
                xml_text = response.text
            else:
                print(f'Grobid error: {response.status_code}')

        root = etree.fromstring(xml_text.encode('utf-8'))

        authors = []
        sourceDesc = root.find('.//tei:sourceDesc', self.ns)
        if sourceDesc is not None:
            for author in sourceDesc.findall('.//tei:author', self.ns):
                affs = []
                for aff in author.findall('.//tei:affiliation', self.ns):
                    text = ' '.join(aff.find('.//tei:note', self.ns).itertext()).strip()
                    affs.append(text)
                persName = author.find('.//tei:persName', self.ns)
                if persName is not None:
                    author_name = ' '.join(persName.itertext()).strip()
                    if author_name:
                        authors.append({'name':author_name, 'affiliation': affs})

        return authors
    
    def _extract_refs(self, pdf_path):
        with open(pdf_path, 'rb') as pdf_file:
            files = {'input': (pdf_path, pdf_file)}
            data = {'includeRawCitations': 1}
            url = f"{self.grobid_url}/api/processReferences"
            response = requests.post(
                url=url,
                files=files,
                data=data,
                timeout=300,
                headers={'Accept': 'application/xml'}
            )

            if response.status_code == 200:
                xml_text = response.text
            else:
                print(f'Grobid error: {response.status_code}')

        root = etree.fromstring(xml_text.encode('utf-8'))

        refs = []
        listBibl = root.find('.//tei:listBibl', self.ns)
        if listBibl is not None:
            for idx, bibl in enumerate(listBibl.findall('.//tei:biblStruct', self.ns)):
                text = bibl.find('.//tei:note[@type="raw_reference"]', self.ns).text

                year_pattern = re.compile(r'[^\d][//\s\(-](\d{4})[\.,;)\s]')
                match = re.search(year_pattern, text)
                if match:
                    year_ref = match.group(1)
                else:
                    year_ref = None

                ref = {
                    'id': idx + 1,
                    'text': text,
                    'authors': [],
                    'year': year_ref
                }
                for author in bibl.findall('.//tei:author', self.ns):
                     persName = author.find('.//tei:persName', self.ns)
                     if persName is not None:
                         author_name = ' '.join(persName.itertext()).strip()
                         if author_name:
                             ref['authors'].append(author_name)
                refs.append(ref)

        return refs
