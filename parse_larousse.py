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

    for page,words_in_page in enumerate(words,1):
        if not os.path.exists(f'./json/page_{page+70:04}.json'):
            print(f'page {page} 标注未找到')
        with open(f'./json/page_{page+70:04}.json') as f:
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
        elif len(words_in_page)  + 1 == len(entries):
            print(f"错误： page {page} ({page+70}) 的单词数{len(words_in_page)}和标记数{len(entries)}不匹配")
            pass
        else:
            print(f"错误： page {page} ({page+70}) 的单词数{len(words_in_page)}和标记数{len(entries)}不匹配")
            errorlist.append(page+69)

    print(errorlist)

def combine_image_pos():
    word_pos = []
    for page in range(1,2058):
        if not os.path.exists(f'./json/page_{page+70:04}.json'):
            print(f'page {page} 标注未找到')
        with open(f'./json/page_{page+70:04}.json') as f:
            annotation = json.load(f)
        entries = []
        cur_no = 1
        for index, entry in enumerate(annotation['entries']):
            position_info = {'page':page+70, 'bbox':entry['coords']}
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

def grammar_check():
    with open('./拉鲁斯法汉双解词典.json','r',encoding='utf8') as f:
        data = json.load(f)
    with open('./data/larousse_grammar.txt','r',encoding='utf8') as f:
        from lark import Lark, UnexpectedInput
        grammar_text = f.read()
        errors = []
        larousse = Lark(grammar_text)
    parsed = []
    for word in data:
        try:
            parse_result = larousse.parse(word['text'])
            parsed.append((word,parse_result))
            #print(parse_result.pretty())
        except UnexpectedInput as e:
            errors.append((word,e))
    import pickle
    with open('./parsed_data.pickle','wb') as f:
        pickle.dump((parsed,errors),f)
    return errors

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
#word_by_page = split_page(words)
#word_by_page_fr = split_page(words_fr)
#match_image_pos(word_by_page)
with open('./拉鲁斯法汉双解词典.json','r',encoding='utf8') as f:
    words = json.load(f)
with open('./data/french.json','r',encoding='utf8') as f:
    words_fr = json.load(f)
def replace_prons(match):
    text = match.group(0)
    prons = match.group(1) or match.group(2)
    newtext = prons.translate(str.maketrans({'r':'ʀ'}))
    return text.replace('['+prons+']','['+newtext+']')

for word in words:
    word['text'] = re.sub(r'(?i)^(?:\d+\.\s*)?\*? *(?:[A-Za-zàâçéèêëîïôöûùüÿñæœ][a-zàâçéèêëîïôöûùüÿñæœ\-\']*(?: [a-zàâçéèêëîïôöûùüÿñæœ\-\']+){,2} *(?:, *[a-zéèêëîïôöûùü]+ *){,2})(?: *(?:ou|et) *\*?[a-zàâçéèêëîïôöûùüÿñæœ\-\']+(?: [a-zàâçéèêëîïôöûùüÿñæœ\-\']+)? *(?:, *[a-zéèêëîïôöûùü]+ *){,2})?(?: \(de\) *)?\[([^\u4e00-\u9fff]+?)\]|(?:\-?[A-Z]\. ?){2,5}\[([^\u4e00-\u9fff]+)\]',replace_prons,word['text'])
for word in words_fr:
    word['text'] = re.sub(r'(?i)^(?:\d+\.\s*)?\*? *(?:[A-Za-zàâçéèêëîïôöûùüÿñæœ][a-zàâçéèêëîïôöûùüÿñæœ\-\']*(?: [a-zàâçéèêëîïôöûùüÿñæœ\-\']+){,2} *(?:, *[a-zéèêëîïôöûùü]+ *){,2})(?: *(?:ou|et) *\*?[a-zàâçéèêëîïôöûùüÿñæœ\-\']+(?: [a-zàâçéèêëîïôöûùüÿñæœ\-\']+)? *(?:, *[a-zéèêëîïôöûùü]+ *){,2})?(?: \(de\) *)?\[([^\u4e00-\u9fff]+?)\]|(?:\-?[A-Z]\. ?){2,5}\[([^\u4e00-\u9fff]+)\]',replace_prons,word['text'])
with open('./拉鲁斯法汉双解词典1.json','w',encoding='utf8') as f:
    json.dump(words,f, ensure_ascii=False, indent=2)
with open('./data/french1.json','w',encoding='utf8') as f:
    json.dump(words_fr,f, ensure_ascii=False, indent=2)
grammar_check()

#write_brackets_check_results()

#with open('拉鲁斯法汉双解词典.json','w',encoding='utf8') as f:
#    json.dump(words,f, ensure_ascii=False, indent=2)

#with open('french.json','w',encoding='utf8') as f:
#    json.dump(words_fr,f, ensure_ascii=False, indent=2)
