import json
import re
import os

def check_brackets(s: str):
    # 定义所有括号的对应关系
    brackets = {
        '(': ')', '[': ']', '{': '}',
        '（': '）', '【': '】', '｛': '｝',
        '《': '》', '「': '」', '『': '』',
        '〔': '〕', '〖': '〗',
        '⟪': '⟫'
    }

    opening = set(brackets.keys())
    closing = set(brackets.values())
    stack = []

    for idx, char in enumerate(s, start=1):  # 下标从1开始，便于提示
        if char in opening:  # 左括号入栈
            stack.append((char, idx))
        elif char in closing:  # 遇到右括号
            if not stack:
                return f"第 {idx} 个字符 '{char}' 没有匹配的左括号"
            last, pos = stack.pop()
            if brackets[last] != char:
                return f"第 {idx} 个字符 '{char}' 与第 {pos} 个字符 '{last}' 不匹配"

    if stack:
        last, pos = stack[-1]
        return f"第 {pos} 个字符 '{last}' 没有匹配的右括号"

    return True



def parse_entries(path:str,errorlog=None):
    with open(path,'r',encoding='utf8') as f:
        text = f.read()

    
    first_pos = text.find('〈1〉')
    text = text[first_pos:]
    words = []
    cur_word = None
    cur_page = 0
    page_start = False
    lines = text.splitlines()
    cur_no = 0
    cur_page_no = 0
    unmatched=0
    for line in lines:
        if not line.strip():
            continue
        if match := re.match(r'〈(\d+)〉',line):
            cur_page = int(match.group(1))
            page_start = True
            cur_page_no = 0
            
        elif page_start and re.fullmatch(r'[A-Z]+',line):
            pass
        #elif #re.match(r'(?i)^(\d+\.\s*)?\*?[a-zàâçéèêëîïôöûùüÿñæœ \.,\-\']+(\(.{2,10}\)\s*)?\[|^[a-zàâçéèêëîïôöûùüÿñæœ\-]+,?\s*(préfixe|préf.)|[A-Z][a-zàâçéèêëîïôöûùüÿñæœ]+\s*\([a-zàâçéèêëîïôöûùüÿñæœ ]+\)',line) or\
            #not page_start and re.match(r'^(\d+\.\s*)[a-zàâçéèêëîïôöûùüÿñæœ]+|^[a-zàâçéèêëîïôöûùüÿñæœ]+ (adj\.|n\.)|^([A-Z]\.[ ,]){2,}',line):
        elif re.match(r'(?i)^(\d+\.\s*)?\*? *(([A-Za-zàâçéèêëîïôöûùüÿñæœ][a-zàâçéèêëîïôöûùüÿñæœ\-\']*( [a-zàâçéèêëîïôöûùüÿñæœ\-\']+){,2} *(, *[a-zéèêëîïôöûùü]+ *){,2})( *(ou|et) *\*?[a-zàâçéèêëîïôöûùüÿñæœ\-\']+( [a-zàâçéèêëîïôöûùüÿñæœ\-\']+)? *(, *[a-zéèêëîïôöûùü]+ *){,2})?( \(de\) *)?\[[^\u4e00-\u9fff]+|(\-?[A-Z]\. ?){2,5}(\[[^\u4e00-\u9fff]+|(, )?sigle)|[a-zàâçéèêëîïôöûùüÿñæœ\-\']+(?: ?(n|[mf]|adj|v|[it]|adv|inv|pl|pr|prép|ind|loc|pron|poss|indéf|relat|et interr|dém|déf|interj|art|impers)\.){,4}(, [a-zàâçéèêëîïôöûùüÿñæœ\-\']+(?: *(n|[mf]|adj|v|[it]|adv|inv|pl|pr|prép|ind|loc|pron|poss|indéf|relat|et interr|dém|déf|interj|art|impers)\.){,4}){,3} *→|[A-Z][a-zàâçéèêëîïôöûùüÿñæœ\-\']+ ?\([a-zàâçéèêëîïôöûùüÿñæœ ]+?d[e\']\)|[a-zàâçéèêëîïôöûùüÿñæœ\-\']+\-, (préf\.|préfixe)|T\. G\. V\.)',line)\
            or not page_start:
            cur_no += 1
            cur_page_no += 1
            word = {'text': line, 'page': cur_page, 'no': cur_no, 'id':f"{cur_page}.{cur_page_no}"}
            headword= re.match(r'(?i)^((\d+\.\s*)?\*? *([a-zàâçéèêëîïôöûùüÿñæœ\-\']+(?: [a-zàâçéèêëîïôöûùüÿñæœ\-\']+){,4}|(?:\-?[A-Z]\. ?){2,5}|b\.a\.\-ba))(?=,| *sigle| *\(| *\[| [a-zàâçéèêëîïôöûùüÿñæœ]+\.| ou| →)',line)
            if headword:
                word['headword'] = headword.group(1).strip()
            else:
                print(f"警告：无法解析词头: {line} (page {cur_page})")
            cur_word = word
            words.append(word)
            page_start = False
        else:
            if not page_start:
                print(cur_page)
                print(line)
                unmatched += 1
            elif errorlog:
                errorlog.write(f"{cur_page}\n{line}\n")
            cur_word['text'] += ' ' + line
            if isinstance(cur_word['page'],int):
                cur_word['page'] = [cur_word['page']]
            if not cur_page in cur_word['page']:
                cur_word['page'].append(cur_page)
            page_start = False
            
    print("unmatched",unmatched)
    return words

def match_image_pos(words):
    errorlist = []
    start_page = 72
    for page,words_in_page in enumerate(words,1):
        if not os.path.exists(f'./json/page_{page+start_page:04}.json'):
            print(f'page {page} 标注未找到')
        with open(f'./json/page_{page+start_page:04}.json') as f:
            annotation = json.load(f)
        entries = []

        for index, entry in enumerate(annotation['entries']):
            position_info = {'page':page+70, 'bbox':entry['coords']}
            if entry['is_headword']:
                entries.append([position_info])
            elif index == 0:
                entries.append([position_info])
            else:
                last_entry = entries.pop()
                if isinstance(last_entry,list):
                    last_entry.append(position_info)
                else:
                    last_entry = [last_entry,position_info]
                entries.append(last_entry)
        
            

        if len(words_in_page) == len(entries):
            for word, position in zip(words_in_page, entries):
                if 'position' in word:
                    word['position'].extend(position)
                else:
                    word['position'] = position
        elif len(words_in_page)  + 1 == len(entries) and isinstance(words_in_page[0]['page'],int) and not annotation['entries'][0]['is_headword']:
            if 'position' in words[page-2][-1]:
                words[page-2][-1]['position'].extend(entries[0])
            for word, position in zip(words_in_page, entries[1:]):
                if 'position' in word:
                    word['position'].extend(position)
                else:
                    word['position'] = position
        #elif len(words_in_page)  + 1 == len(entries):
        #    print(f"错误： page {page} ({page+start_page}) 的单词数{len(words_in_page)}和标记数{len(entries)}不匹配")
        #    pass
        else:
            print(f"错误： page {page} ({page+start_page}) 的单词数{len(words_in_page)}和标记数{len(entries)}不匹配")
            errorlist.append(page+start_page-1)

    print(errorlist)

def combine_image_pos():
    word_pos = []
    start_page = 72
    for page in range(1,2058):
        if not os.path.exists(f'./json/page_{page+start_page:04}.json'):
            print(f'page {page} 标注未找到')
        with open(f'./json/page_{page+start_page:04}.json') as f:
            annotation = json.load(f)
        entries = []
        cur_no = 1
        for index, entry in enumerate(annotation['entries']):
            position_info = {'page':page+start_page, 'bbox':entry['coords']}
            if entry['is_headword']:
                word_pos.append({"id":f"{page}.{cur_no}","page":page,"position":[position_info]})
                cur_no += 1
            else:
                
                if word_pos:
                    last_entry = word_pos[-1]
                    last_entry["position"].append(position_info)
                else:
                    word_pos.append({"id":f"{page}.0","page":page,"position":[position_info]})
    return word_pos

def split_page(words):
    current_page = 0
    word_in_page = []
    pages = []
    for word in words:
        while True:
            if word['page'] == current_page:
                word_in_page.append(word)
                break
            elif word['page'] == current_page + 1:
                
                current_page += 1
                word_in_page = []
                pages.append(word_in_page)
                word_in_page.append(word)
                break
            elif isinstance(word['page'],list):
                if current_page in word['page']:
                    word_in_page.append(word)
                    
                if current_page + 1 in word['page']:
                    current_page += 1
                    word_in_page = []
                    pages.append(word_in_page)
                    continue
                break
            else:
                break
    return pages


def write_brackets_check_results():
    with open('brackets_check_result.txt','w',encoding='utf8') as f:
        
        unmatched = 0
        for word in words:
            result = check_brackets(word['text'])
            if result != True:
                f.write(f"{word['page']}\n")
                f.write(word['text']+ '\n')
                f.write(result+ '\n')
                unmatched += 1
        f.write(f'括号匹配错误数目：{unmatched}\n')
       

def write_word_pos():
    word_pos = combine_image_pos()
    with open('image_pos.json','w',encoding='utf8') as f:
        json.dump(word_pos,f, ensure_ascii=False, indent=2)
def add_suffix_by_first_letter(word: str, suffix: str) -> str:
    """
    根据用户指定的规则拼接后缀：
    - 如果 word 以 'e' 结尾，直接返回 word + suffix。
    - 否则，找 word 中最后一次出现 suffix[0] 的位置 pos，
      从 pos 开始（包含 pos 所在字母）截断原词，然后接上整个 suffix。
    - 如果没找到相同字母，则直接 word + suffix。

    例子:
    - add_suffix_by_first_letter("votif", "ive") -> "votive":
        suffix[0] == 'i'，在 "votif" 中最后出现 'i' 的位置是索引 3，
        截断到该位置前（即保留 word[:3] -> "vot"）再加 "ive" -> "votive"
    """
    if not suffix:
        return word  # 没有后缀，返回原词
    
    if len(suffix) > 6 :
        return suffix

    # 若以 'e' 结尾，直接加后缀
    if suffix == 'e':
        return word + suffix

    first = suffix[0]
    pos = word.rfind(first)  # 找最后一个匹配字母的位置

    if pos == -1:
        # 没找到相同字母，直接拼接
        return word + suffix
    else:
        # 截断到 pos（不保留 pos 及其之后的字符），再加后缀
        return word[:pos] + suffix

mapping = {
        'à': 'a', 'â': 'a', 'ä': 'a',
        'á': 'a', 'ã': 'a',
        'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
        'í': 'i', 'ì': 'i', 'î': 'i', 'ï': 'i',
        'ó': 'o', 'ò': 'o', 'ô': 'o', 'ö': 'o', 'õ': 'o',
        'ú': 'u', 'ù': 'u', 'û': 'u', 'ü': 'u',
        'ç': 'c',
        'œ': 'oe', 'æ': 'ae',
        'À': 'A', 'Â': 'A', 'Ä': 'A',
        'Á': 'A', 'Ã': 'A',
        'É': 'E', 'È': 'E', 'Ê': 'E', 'Ë': 'E',
        'Í': 'I', 'Ì': 'I', 'Î': 'I', 'Ï': 'I',
        'Ó': 'O', 'Ò': 'O', 'Ô': 'O', 'Ö': 'O', 'Õ': 'O',
        'Ú': 'U', 'Ù': 'U', 'Û': 'U', 'Ü': 'U',
        'Ç': 'C',
        'Œ': 'OE', 'Æ': 'AE',
        'ñ': 'n', 'Ñ': 'N'
    }
maketrans_table = str.maketrans(mapping)
# ---------- 法语特殊字母转普通字母的函数 ----------
def normalize_french(word: str) -> str:
    """
    将法语中的重音字母 / 特殊字母转换为普通字母。
    """
    
    return word.translate(maketrans_table)

def add_bold_tags(text: str, positions: list[tuple[int, int]]) -> str:
    """
    给字符串中指定的多个区间加上<b>...</b>标签。
    positions 为 [(start1, end1), (start2, end2), ...]，end 不包含。
    自动合并重叠和相邻区间，避免标签嵌套错误。
    """

    if not positions:
        return text

    # 1. 按起点排序
    positions = sorted(positions, key=lambda x: x[0])

    # 2. 合并重叠或相邻区间
    merged = []
    current_start, current_end = positions[0]

    for start, end in positions[1:]:
        if start <= current_end:  # 重叠或相邻
            current_end = max(current_end, end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end
    merged.append((current_start, current_end))

    # 3. 从后往前插入标签，避免索引偏移
    result = text
    for start, end in reversed(merged):
        result = result[:end] + "</b>" + result[end:]
        result = result[:start] + "<b>" + result[start:]
    
    return result

def headword_expand():
    with open('./拉鲁斯法汉双解词典.json','r',encoding='utf8') as f:
        data = json.load(f)
    with open('./data/larousse_grammar.txt','r',encoding='utf8') as f:
        from lark import Lark, UnexpectedInput, Token
        grammar_text = f.read()
        errors = []
        larousse = Lark(grammar_text)
    parsed = []
    def clean(token):
        text, start, end = token.value, token.start_pos, token.end_pos
        if re.match(r'^[0-9]\. ',text):
            text = text[3:]
            start += 3
        
        if text.startswith('*'):
            text = text[1:]
            start += 1
        original_text = text
        if text.startswith('à la '):
            text = text[5:]
            start += 5
        if text.startswith('à l\''):
            text = text[4:]
            start += 4
        if text.startswith('à '):
            text = text[2:]
            start += 2
        if text.startswith('s\''):
            text = text[2:]
            start += 2
        if text.startswith('se '):
            text = text[3:]
            start += 3
        
        return original_text,text, (start,end)
    for word in data:
        try:
            parse_result = larousse.parse(word['text'])
            parsed.append((word,parse_result))
            expand_words = []
            normalized_words = []
            bold_ranges = []
            root_word = ''
            for token in parse_result.scan_values(lambda v: isinstance(v, Token) and v.type == 'WORD'):
                text, text1, range = clean(token)
                if text != text1:
                    expand_words.append(text1)
                bold_ranges.append(range)
                parts = [part.strip() for part in text.split(',')]
                if root_word == '' or parts[0] not in ['aux','als','one']:
                    if root_word == '':
                        word['main_word'] = parts[0]
                    root_word = parts[0]
                    if root_word != word['headword']:
                        print(f"警告：词头不匹配 {word['headword']} vs {root_word} (page {word['page']})")
                        expand_words.append(root_word)
                    if len(parts) > 1:
                        #print(text)
                        suffixes = parts[1:]
                        for suffix in suffixes:
                            new_word = add_suffix_by_first_letter(root_word, suffix)
                            #print('->',new_word)
                            expand_words.append(new_word)
                else:
                    print(text)
                    new_word = add_suffix_by_first_letter(root_word, text)
                    print('->',new_word)
                    expand_words.append(new_word)
            if word['main_word'] in expand_words:
                expand_words.remove(word['main_word'])
            if expand_words:
                word['expanded_words'] = expand_words
            normalized_word = normalize_french(root_word)
            if normalized_word != root_word:
                normalized_words.append(normalized_word)
                for ew in expand_words:
                    normalized_ew = normalize_french(ew)
                    if normalized_ew != ew:
                        normalized_words.append(normalized_ew)
            if normalized_words:
                word['normalized_words'] = list(set(normalized_words))
            word['text'] = add_bold_tags(word['text'], bold_ranges)

            #print(parse_result.pretty())
        except UnexpectedInput as e:
            print(f"{word['headword']}({word['page']})\n{e}")
            errors.append((word,e))

    with open('./拉鲁斯法汉双解词典_expanded.json','w',encoding='utf8') as f:
        json.dump(data,f, ensure_ascii=False, indent=2)

    return errors

from lark import Tree, Token
def tree_to_dict(node):
    if isinstance(node, Tree):
        return {
            "type": node.data.type,
            "type": node.data.value,
            "start_pos": getattr(node.meta, "start_pos", None),
            "end_pos": getattr(node.meta, "end_pos", None),
            "children": [tree_to_dict(child) for child in node.children]
        }
    elif isinstance(node, Token):
        return {
            "type": node.type,
            "value": node.value,
            "start_pos": getattr(node, "start_pos", None),
            "end_pos": getattr(node, "end_pos", None)
        }
def grammar_check():
    with open('./拉鲁斯法汉双解词典.json','r',encoding='utf8') as f:
        data = json.load(f)
    with open('./data/larousse_grammar.txt','r',encoding='utf8') as f:
        from lark import Lark, UnexpectedInput, Token
        grammar_text = f.read()
        errors = []
        #larousse = Lark(grammar_text,propagate_positions=True)
        larousse = Lark(grammar_text)
    parsed = []
        
    for word in data:
        try:
            parse_result = larousse.parse(word['text'])
            parsed.append((word,parse_result))
            #word['parse_tree'] = tree_to_dict(parse_result)
            #print(parse_result.pretty())
        except UnexpectedInput as e:
            print(f"{word['headword']}({word['page']})\n{e}")
            errors.append((word,e))
    with open('error_parse.txt','w',encoding='utf8') as f:
        for word,error in errors:
            f.write(f"{word['text']}\n")
    with open('./data/error_parse.json','w',encoding='utf8') as f:
        error_map = {}
        for word,error in errors:
            text = word['text'][:error.pos_in_stream] + '^' + word['text'][error.pos_in_stream:] 
            error_map[word['id']] ={'text':text} 
        json.dump(error_map,f, ensure_ascii=False, indent=2)
    with open('./拉鲁斯法汉双解词典_parsed.json','w',encoding='utf8') as f:
        json.dump(data,f, ensure_ascii=False, indent=2)
    #import pickle
    #with open('./parsed_data.pickle','wb') as f:
    #    pickle.dump(parsed,f)
    return errors

def load_parsed_data():
    import pickle
    with open('./parsed_data.pickle','rb') as f:
        parsed,errors = pickle.load(f)
    return parsed,errors

def gen_diff_list(words,words_fr):
    wordset_zh = {}
    for w in words:
        if w['headword'] in wordset_zh:
            print(f"警告：重复词头 {w['headword']} (page {wordset_zh[w['headword']]['page']} {w['page']})")
        wordset_zh[w['headword']] = w
    wordset_fr = {}
    for w in words_fr:
        if w['headword'] in wordset_fr:
            print(f"警告：重复词头 {w['headword']} (page  {wordset_fr[w['headword']]['page']} {w['page']})")
        wordset_fr[w['headword']] = w
    wordset_zh_extra = wordset_zh.keys() - wordset_fr.keys()
    wordset_fr_extra = wordset_fr.keys() - wordset_zh.keys()
    word_list_zh = []
    word_list_fr = []
    with open('wordset_zh.txt','w',encoding='utf8') as f:
        for headword,word in wordset_zh.items():
            if headword in wordset_zh_extra:
                f.write(f"{headword}\t{word['page']}\n")
                f.write(f"{word['text']}\n")
                if headword not in ['1. bille','1. blanc']:
                    word_list_zh.append(word)
    with open('wordset_fr.txt','w',encoding='utf8') as f:
        for headword,word in wordset_fr.items():
            if headword in wordset_fr_extra:
                f.write(f"{headword}\t{word['page']}\n")
                f.write(f"{word['text']}\n")
                if headword not in ['vacillement','vaciller','vacuité','vacuole']:
                    word_list_fr.append(word)

    wordmap_fr = {}
    for zh,fr in zip(word_list_zh,word_list_fr):
        wordmap_fr[zh['id']] = fr
    with open('word_diff_list_fr.json','w',encoding='utf8') as f:
        json.dump(wordmap_fr,f, ensure_ascii=False, indent=2)

#with open('temp.txt','w',encoding='utf8') as f:
#words = parse_entries('./拉鲁斯法汉双解词典 文本.txt')
#words_fr = parse_entries('./dictionnaire de la langue française.txt')
#gen_diff_list(words,words_fr)
def change_num(match):
    circled_num = "❶❷❸❹❺❻❼❽❾❿⓫⓬⓭⓮⓯⓰⓱⓲⓳⓴㉑㉒㉓㉔㉕㉖㉗㉘㉙㉚㉛㉜㉝㉞㉟㊱㊲㊳㊴㊵㊶㊷㊸㊹㊺㊻㊼㊽㊾㊿"
    num = int(match.group(1))
    if num > 50:
        return match.group(0)
    return circled_num[num-1]
#for word in words_fr:
#    word['text'] = re.sub(r'\-\s*(\d+)\.',change_num,word['text'])
#    word['text'] = word['text'].replace("□","◇")

#word_by_page_fr = split_page(words_fr)

with open('./拉鲁斯法汉双解词典.json','r',encoding='utf8') as f:
    words = json.load(f)

with open('./data/french.json','r',encoding='utf8') as f:
    words_fr = json.load(f)
word_by_page = split_page(words)
match_image_pos(word_by_page)
#write_word_pos()

def replace_prons(match):
    text = match.group(0)
    prons = match.group(1) or match.group(2)
    newtext = prons.translate(str.maketrans({'r':'ʀ'}))
    return text.replace('['+prons+']','['+newtext+']')

#for word in words:
#    word['text'] = re.sub(r'(?i)^(?:\d+\.\s*)?\*? *(?:[A-Za-zàâçéèêëîïôöûùüÿñæœ][a-zàâçéèêëîïôöûùüÿñæœ\-\']*(?: [a-zàâçéèêëîïôöûùüÿñæœ\-\']+){,2} *(?:, *[a-zéèêëîïôöûùü]+ *){,2})(?: *(?:ou|et) *\*?[a-zàâçéèêëîïôöûùüÿñæœ\-\']+(?: [a-zàâçéèêëîïôöûùüÿñæœ\-\']+)? *(?:, *[a-zéèêëîïôöûùü]+ *){,2})?(?: \(de\) *)?\[([^\u4e00-\u9fff]+?)\]|(?:\-?[A-Z]\. ?){2,5}\[([^\u4e00-\u9fff]+)\]',replace_prons,word['text'])
#for word in words_fr:
#    word['text'] = re.sub(r'(?i)^(?:\d+\.\s*)?\*? *(?:[A-Za-zàâçéèêëîïôöûùüÿñæœ][a-zàâçéèêëîïôöûùüÿñæœ\-\']*(?: [a-zàâçéèêëîïôöûùüÿñæœ\-\']+){,2} *(?:, *[a-zéèêëîïôöûùü]+ *){,2})(?: *(?:ou|et) *\*?[a-zàâçéèêëîïôöûùüÿñæœ\-\']+(?: [a-zàâçéèêëîïôöûùüÿñæœ\-\']+)? *(?:, *[a-zéèêëîïôöûùü]+ *){,2})?(?: \(de\) *)?\[([^\u4e00-\u9fff]+?)\]|(?:\-?[A-Z]\. ?){2,5}\[([^\u4e00-\u9fff]+)\]',replace_prons,word['text'])
#with open('./拉鲁斯法汉双解词典1.json','w',encoding='utf8') as f:
#    json.dump(words,f, ensure_ascii=False, indent=2)
#with open('./data/french1.json','w',encoding='utf8') as f:
#    json.dump(words_fr,f, ensure_ascii=False, indent=2)

headword_expand()
errors = grammar_check()
#parsed,errors = load_parsed_data()
#for word,error in errors:
#    print(f"{word['headword']}({word['page']})\n{error}")

#write_brackets_check_results()

#with open('拉鲁斯法汉双解词典.json','w',encoding='utf8') as f:
#    json.dump(words,f, ensure_ascii=False, indent=2)

#with open('french.json','w',encoding='utf8') as f:
#    json.dump(words_fr,f, ensure_ascii=False, indent=2)
