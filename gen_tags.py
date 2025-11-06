import re
import json

def insert_tags(text, tags):
    tags = reversed(tags)
    for tag, pos in tags:
        text = text[:pos] + tag + text[pos:]
    return text

def parse_black_num(node):
    value = node['value']
    num  = ord(value)
    if ord('❶') <= num <= ord('❿'):
        num -= ord('❶') - 1
    elif ord('⓫') <= num <= ord('⓴'):
        num -= ord('⓫') - 11
    elif ord('⓵') <= num <= ord('⓾'):
        num -= ord('⓵') - 1
    elif ord('㉑') <= num <= ord('㉟'):
        num -= ord('㉑') - 21
    return [(f"<num value='{num}'>", node['start_pos']), (f"</num>", node['end_pos'])]

def parse_link_word(node):
    word = node['value']
    return [(f"<a href=entry://'{word}'>", node['start_pos']), (f"</a>", node['end_pos'])]

tag_map = {
    'start': '',
    'word': 'entry',
    'pos': '',
    'POS': 'pos',
    'usage': 'usage',
    'definition': 'def',
    'wordhead':'head',
    'wordhead.WORD':'word',
    'ZH': 'zh',
    'FR': 'fr',
    'BLACK_NUM': parse_black_num,
    'explanation': 'def',
    'explanation_num': 'def',
    'explanation_num.explanation': '',
    'phrase': 'phrase',
    'example': 'example',
    'rem': 'rem',
    'GRAMMAR': 'grammar',
    'pron': '',
    'PRON': 'pron',
    'CATEGORY_FR':'cat_fr',
    'CATEGORY_ZH':'cat_zh',
    'etymology.PAREN_ENCLOSED':'etym',
    'ETYM_ZH':'etym_zh',
    'link.FR_WORD': parse_link_word,
}


def tree_to_tags(node,parent_tag=None):
    node_type = node['type']
    if node_type == 'RULE':
        tag = node['value']
    else:
        tag = node_type
    if parent_tag and f'{parent_tag}.{tag}' in tag_map:
        tag_val = tag_map.get(f'{parent_tag}.{tag}', '')
    else:
        tag_val = tag_map.get(tag, '')
    if isinstance(tag_val, str):
        if tag_val:
            open_tag = f"<{tag_val}>"
            close_tag = f"</{tag_val}>"
        else:
            open_tag = ''
            close_tag = ''
        children = node.get('children', [])
        result = []
        if open_tag:
            result.append((open_tag, node['start_pos']))
        for child in children:
            result.extend(tree_to_tags(child,tag))
        if close_tag:
            result.append((close_tag,node['end_pos']))
        return result
    elif callable(tag_val):
        return tag_val(node)
    return []
    


if __name__ == "__main__":
    with open('./拉鲁斯法汉双解词典_xml.json','r',encoding='utf8') as f:
        data = json.load(f)
    with open('./285_xml.json','r',encoding='utf8') as f:
        ai_xml = json.load(f)
    with open('./data/error_parse.json','r',encoding='utf8') as f:
        errors = json.load(f)
    ai_map = {entry['headword']:entry['xml'] for entry in ai_xml}
    for entry in data:
        if entry['headword'] in ai_map:
            entry['xml'] = ai_map[entry['headword']]
            entry['tag_stats'] = 'ai_gen'
        elif entry['id'] in errors and 'xml' in entry:
            entry['tag_stats'] = 'has_error'
        entry['xml'] = entry['xml'].replace('<fr> ',' <fr>').replace(' </fr>','</fr> ')
    with open('./拉鲁斯法汉双解词典_xml1.json','w',encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    with open('./拉鲁斯法汉双解词典_xml.json','r',encoding='utf8') as f:
        data = json.load(f)
    with open('./拉鲁斯法汉双解词典_parsed.json','r',encoding='utf8') as f:
        data = json.load(f)
    with open('./拉鲁斯法汉双解词典_expanded.json','r',encoding='utf8') as f:
        data1 = json.load(f)
    for entry,entry1 in zip(data,data1):
        if 'parse_tree' not in entry:
            continue
        tags = tree_to_tags(entry['parse_tree'])
        entry['xml'] = insert_tags(entry['text'], tags)
        if entry1.get('main_word'):
            entry['main_word'] = entry1['main_word']
        if entry1.get('expanded_words'):
            entry['expanded_words'] = entry1['expanded_words']
        if entry1.get('normalized_words'):
            entry['normalized_words'] = entry1['normalized_words']
        del entry['parse_tree']
    with open('./拉鲁斯法汉双解词典_xml.json','w',encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)