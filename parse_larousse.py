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
        elif re.match(r'^(\d+\.\s*)?\*? *(([a-zàâçéèêëîïôöûùüÿñæœ\-\']+( [a-zàâçéèêëîïôöûùüÿñæœ\-\']+){,2} *(, *[a-zéèêëîïôöûùü]+ *){,2})( *(ou|et) *\*?[a-zàâçéèêëîïôöûùüÿñæœ\-\']+( [a-zàâçéèêëîïôöûùüÿñæœ\-\']+)? *(, *[a-zéèêëîïôöûùü]+ *){,2})?( \(de\) *)?\[[ptkbdgfvszʃʒlʀmnɲxŋieɛaɑɔouyœøəjɥwɑ̃ɛ̃ɔ̃œ̃]+|(\-?[A-Z]\. ?){2,5}(\[[ptkbdgfvszʃʒlʀmnɲxŋieɛaɑɔouyœøəjɥwɑ̃ɛ̃ɔ̃œ̃]+|(, )?sigle)|[a-zàâçéèêëîïôöûùüÿñæœ\-\']+(?: ?(n|[mf]|adj|v|[it]|adv|inv|pl|pr|prép|ind|loc|pron|poss|indéf|relat|et interr|dém|déf|interj|art|impers)\.){,4}(, [a-zàâçéèêëîïôöûùüÿñæœ\-\']+(?: +(n|[mf]|adj|v|[it]|adv|inv|pl|pr|prép|ind|loc|pron|poss|indéf|relat|et interr|dém|déf|interj|art|impers)\.){,4}){,3} *→|[A-Z][a-zàâçéèêëîïôöûùüÿñæœ\-\']+ ?\([a-zàâçéèêëîïôöûùüÿñæœ ]+?d[e\']\)|[a-zàâçéèêëîïôöûùüÿñæœ\-\']+\-, (préf\.|préfixe))',line)\
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
    with open('./data/larousse_grammar.txt','r',encoding='utf8') as f:
        from lark import Lark, UnexpectedInput
        grammar_text = f.read()
        errors = []
        larousse = Lark(grammar_text)
        for word in words:
            try:

                parse_result = larousse.parse(word['text'])
                #print(parse_result.pretty())
            except UnexpectedInput as e:
                errors.append((word,e))
    return errors



#with open('temp.txt','w',encoding='utf8') as f:
words = parse_entries('./拉鲁斯法汉双解词典 文本.txt')
words_fr = parse_entries('./dictionnaire de la langue française.txt')
wordset_zh = {w['headword']:w for w in words if 'headword' in w}
wordset_fr = {w['headword']:w for w in words_fr if 'headword' in w}
wordset_zh_extra = wordset_zh.keys() - wordset_fr.keys()
wordset_fr_extra = wordset_fr.keys() - wordset_zh.keys()
with open('wordset_zh.txt','w',encoding='utf8') as f:
    for headword,word in wordset_zh.items():
        if headword in wordset_zh_extra:
            f.write(f"{headword}\t{word['page']}\n")
            f.write(f"{word['text']}\n")
with open('wordset_fr.txt','w',encoding='utf8') as f:
    for headword,word in wordset_fr.items():
        if headword in wordset_fr_extra:
            f.write(f"{headword}\t{word['page']}\n")
            f.write(f"{word['text']}\n")
print("中文多余词头",wordset_zh_extra)
print("法文多余词头",wordset_fr_extra)

word_by_page = split_page(words)
match_image_pos(word_by_page)
grammar_check()

#write_brackets_check_results()

with open('拉鲁斯法汉双解词典_gemini.json','w',encoding='utf8') as f:
    json.dump(words,f, ensure_ascii=False, indent=2)

