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



def parse_entries(text:str):
    words = []
    cur_word = None
    cur_page = 0
    page_start = False
    lines = text.splitlines()
    cur_no = 0
    cur_page_no = 0
    for line in lines:
        if not line.strip():
            continue
        if match := re.match('〈(\d+)〉',line):
            cur_page = int(match.group(1))
            page_start = True
            cur_page_no = 0
            
        elif page_start and re.fullmatch(r'(?i)^\*?[a-zàâçéèêëîïöôûùüÿñæœ \.\-\']+',line):
            pass
        elif re.match(r'(?i)^(\d+\.\s*)?\*?[a-zàâçéèêëîïôöûùüÿñæœ \.,\-\']+(\(.{2,10}\)\s*)?\[|^[a-zàâçéèêëîïôöûùüÿñæœ\-]+,?\s*(préfixe|préf.)|[A-Z][a-zàâçéèêëîïôöûùüÿñæœ]+\s*\([a-zàâçéèêëîïôöûùüÿñæœ ]+\)',line) or\
            not page_start and re.match(r'^(\d+\.\s*)[a-zàâçéèêëîïôöûùüÿñæœ]+|^[a-zàâçéèêëîïôöûùüÿñæœ]+ (adj\.|n\.)|^([A-Z]\.[ ,]){2,}',line):
            cur_no += 1
            cur_page_no += 1
            word = {'text': line, 'page': cur_page, 'no': cur_no, 'id':f"{cur_page}.{cur_page_no}"}
            cur_word = word
            words.append(word)
            page_start = False
        elif '→' in line:
            cur_no += 1
            cur_page_no += 1
            word = {'text': line, 'page': cur_page, 'no': cur_no, 'id':f"{cur_page}.{cur_page_no}"}
            cur_word = word
            words.append(word)
            page_start = False
        else:
            if not page_start:
                print(cur_page)
                print(line)
            cur_word['text'] += ' ' + line
            if isinstance(cur_word['page'],int):
                cur_word['page'] = [cur_word['page']]
            if not cur_page in cur_word['page']:
                cur_word['page'].append(cur_page)
            page_start = False
            

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
       

with open('./拉鲁斯法汉双解词典 文本.txt','r',encoding='utf8') as f:
    text = f.read()
first_pos = text.find('〈1〉')
text = text[first_pos:]

words = parse_entries(text)

word_by_page = split_page(words)

match_image_pos(word_by_page)
 
write_brackets_check_results()

with open('拉鲁斯法汉双解词典_gemini.json','w',encoding='utf8') as f:
    json.dump(words,f, ensure_ascii=False, indent=2)

