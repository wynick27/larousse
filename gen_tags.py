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


tag_map = {
    'start': '',
    'word': 'entry',
    'pos': '',
    'POS': 'pos',
    'usage': 'usage',
    'definition': 'def',
    'ZH': 'zh',
    'FR': 'fr',
    'BLACK_NUM': parse_black_num,
    'explanation': 'def',
    'phrase': 'phrase',
    'example': 'example',
    'rem': 'rem',
    'GRAMMAR': 'grammar',
    'pron': '',
    'PRON': 'pron',
}


def tree_to_tags(node):
    node_type = node['type']
    if node_type == 'RULE':
        tag = node['value']
    else:
        tag = node_type
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
            result.extend(tree_to_tags(child))
        if close_tag:
            result.append((close_tag,node['end_pos']))
        return result
    elif callable(tag_val):
        return tag_val(node)
    return []
    


if __name__ == "__main__":
    with open('./拉鲁斯法汉双解词典_parsed.json','r',encoding='utf8') as f:
        data = json.load(f)
    for entry in data:
        if 'parse_tree' not in entry:
            continue
        tags = tree_to_tags(entry['parse_tree'])
        entry['xml'] = insert_tags(entry['text'], tags)
        del entry['parse_tree']
    with open('./拉鲁斯法汉双解词典_xml.json','w',encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)