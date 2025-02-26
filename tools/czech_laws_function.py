from typing import Dict, List, Tuple
from pydantic import BaseModel, Field
from lxml import html
import requests
import re

laws_dict = {
    "Zákon o daních z příjmů": "https://www.zakonyprolidi.cz/cs/1992-586",
    "Občanský zákoník": "https://www.zakonyprolidi.cz/cs/2012-89",
    "Zákoník práce": "https://www.zakonyprolidi.cz/cs/2006-262",
    "Stavební zákon": "https://www.zakonyprolidi.cz/cs/2006-183",
    "Zákon o zadávání veřejných zakázek": "https://www.zakonyprolidi.cz/cs/2016-134",
    "Trestní zákoník": "https://www.zakonyprolidi.cz/cs/2009-40",
    "Správní řád": "https://www.zakonyprolidi.cz/cs/2004-500",
    "Stavební zákon": "https://www.zakonyprolidi.cz/cs/2021-283",
    "Vyhláška o dokumentaci staveb": "https://www.zakonyprolidi.cz/cs/2006-499",
    "Zákon o pobytu cizinců na území České republiky": "https://www.zakonyprolidi.cz/cs/1999-326",
    "Občanský soudní řád": "https://www.zakonyprolidi.cz/cs/1963-99",
    "Zákon o dani z přidané hodnoty": "https://www.zakonyprolidi.cz/cs/2004-235",
    "Zákon o obchodních korporacích": "https://www.zakonyprolidi.cz/cs/2012-90",
    "Trestní řád": "https://www.zakonyprolidi.cz/cs/1961-141",
    "Živnostenský zákon": "https://www.zakonyprolidi.cz/cs/1991-455",
    "Nařízení vlády, kterým se stanoví podmínky ochrany zdraví při práci": "https://www.zakonyprolidi.cz/cs/2007-361",
    "Zákon o zajištění dalších podmínek bezpečnosti a ochrany zdraví při práci": "https://www.zakonyprolidi.cz/cs/2006-309",
}

def _scrape_law(url: str) -> Tuple[Dict, List[Dict]]:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:97.0) Gecko/20100101 Firefox/97.0',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'DNT': '1', # Do Not Track Request Header
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }

    #Get page by url
    try:
        response = requests.get(url, timeout=30, headers=headers)
    except Exception as e:
        print(f'Error occurred while loading url {url}. Error: {str(e)}')
        return None
    
    tree = html.fromstring(response.text)

    law_data = {}
    #Get the name of the law (number)
    law_html = tree.xpath('//div[@class="doc-header"]/h1')
    if len(law_html) > 0:
        law_data['law_number'] = law_html[0].text
    #Get the name of the law (name)
    law_html = tree.xpath('//div[@class="doc-header"]/h1/span')
    if len(law_html) > 0:
        law_data['law_title'] = law_html[0].text
    # Platnost od
    law_html = tree.xpath('//div[@class="doc-header"]/table/tr/td[1]/div/table/tbody/tr[2]/td[2]/span')
    if len(law_html) > 0:
        law_data['platnost_od'] = law_html[0].text
    # Ucinnost od
    law_html = tree.xpath('//div[@class="doc-header"]/table/tr/td[1]/div/table/tbody/tr[3]/td[2]/span')
    if len(law_html) > 0:
        law_data['ucinnost_od'] = law_html[0].text

    # Get the page title
    paragraphs = []
    law_content = tree.xpath('//div[@class="Frags"]/*')
    paragraph = {'cast': None,
                 'cast_name': None,
                 'hlava': None,
                 'hlava_name': None,
                 'dil': None,
                 'dil_name': None,
                 'cislo_odstavce': None,
                 'obsah_odstavce': None,
                 'odkaz': None,
                 }
    for each in law_content:
        pattern = r'^(\w+)?\s*(L)(\d+)\s*(\w+)?'
        searched_pattern_from_class_tag = re.search(pattern, each.get('class'))
        if searched_pattern_from_class_tag is None:
            continue
        first_word, L, number, other_symbols = searched_pattern_from_class_tag.groups()
        if L != 'L':
            print(f"L is not L, but {L}")
        if first_word == 'CAST':
            paragraph['cast'] = each.text_content()
            paragraph['cast_name'] = ''
            paragraph['hlava'] = ''
            paragraph['hlava_name'] = ''
            paragraph['dil'] = ''
            paragraph['dil_name'] = ''
            paragraph['cislo_odstavce'] = ''
            paragraph['obsah_odstavce'] = ''
            paragraph['odkaz'] = ''
        elif (first_word == 'NADPIS') and (number == '2'):
            paragraph['cast_name'] = each.text_content()
            paragraph['hlava'] = ''
            paragraph['hlava_name'] = ''
            paragraph['dil'] = ''
            paragraph['dil_name'] = ''
            paragraph['cislo_odstavce'] = ''
            paragraph['obsah_odstavce'] = ''
            paragraph['odkaz'] = ''
        elif first_word == 'HLAVA':
            paragraph['hlava'] = each.text_content()
            paragraph['hlava_name'] = ''
            paragraph['dil'] = ''
            paragraph['dil_name'] = ''
            paragraph['cislo_odstavce'] = ''
            paragraph['obsah_odstavce'] = ''
            paragraph['odkaz'] = ''
        elif (first_word == 'NADPIS') and (number == '3'):
            paragraph['hlava_name'] = each.text_content()
            paragraph['dil'] = ''
            paragraph['dil_name'] = ''
            paragraph['cislo_odstavce'] = ''
            paragraph['obsah_odstavce'] = ''
            paragraph['odkaz'] = ''
        elif first_word == 'DIL':
            paragraph['dil'] = each.text_content()
            paragraph['dil_name'] = ''
            paragraph['cislo_odstavce'] = ''
            paragraph['obsah_odstavce'] = ''
            paragraph['odkaz'] = ''
        elif (first_word == 'NADPIS') and (number == '4'):
            paragraph['dil_name'] = each.text_content()
            paragraph['cislo_odstavce'] = ''
            paragraph['obsah_odstavce'] = ''
            paragraph['odkaz'] = ''
        elif first_word == 'PARA':
            paragraph['cislo_odstavce'] = each.text_content()
            paragraph['obsah_odstavce'] = ''
            paragraph['odkaz'] = ''
        elif (first_word is None) and (other_symbols is None):
            paragraph['obsah_odstavce'] = each.text_content()
            paragraph['odkaz'] = ''
            paragraphs.append(paragraph.copy())

    return law_data, paragraphs

def _convert_paragraphs_to_string(paragraphs: List[Dict]) -> str:

    previous_paragraph = {'cast': None,
                    'cast_name': None,
                    'hlava': None,
                    'hlava_name': None,
                    'dil': None,
                    'dil_name': None,
                    'cislo_odstavce': None,
                    'obsah_odstavce': None,
                    'odkaz': None,
    }

    paragraphs_string = ''
    for paragraph in paragraphs:
        for paragraph_part in previous_paragraph:
            if paragraph[paragraph_part] != previous_paragraph[paragraph_part]:
                paragraphs_string += f"{paragraph[paragraph_part]}\n"
                previous_paragraph[paragraph_part] = paragraph[paragraph_part]

    return paragraphs_string

def get_czech_law(law_name: str) -> Dict:
    """Get the text of the law of Czech Republic.

    Args:
        law_name: The name of the law in Czech Republic.
                   One of the following values:
                   - 'Zákon o daních z příjmů'
                   - 'Občanský zákoník'
                   - 'Zákoník práce'
                   - 'Stavební zákon'
                   - 'Zákon o zadávání veřejných zakázek'
                   - 'Trestní zákoník'
                   - 'Správní řád'
                   - 'Vyhláška o dokumentaci staveb'
                   - 'Zákon o pobytu cizinců na území České republiky'
                   - 'Občanský soudní řád'
                   - 'Zákon o dani z přidané hodnoty'
                   - 'Zákon o obchodních korporacích'
                   - 'Trestní řád'
                   - 'Živnostenský zákon'
                   - 'Nařízení vlády, kterým se stanoví podmínky ochrany zdraví při práci'
                   - 'Zákon o zajištění dalších podmínek bezpečnosti a ochrany zdraví při práci'

    Returns:
        A dictionary containing the law text or an error message.
    """
    if law_name in laws_dict:
        url = laws_dict[law_name]
        _, paragraphs = _scrape_law(url)
        law_text = _convert_paragraphs_to_string(paragraphs)
        return {"law_text": law_text}

    else:
        return {"error": f"Law {law_name} not found."}