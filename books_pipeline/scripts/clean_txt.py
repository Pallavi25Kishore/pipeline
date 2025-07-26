import os
import json
import re
from multiprocessing import Pool
from tqdm import tqdm
import sys
import time
from itertools import permutations
from numpy import nan
from datetime import datetime
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import NLTK components if available
NLTK_AVAILABLE = False
try:
    import nltk
    try:
        # Test if NLTK data is properly initialized
        from nltk.corpus import wordnet
        from nltk.stem import WordNetLemmatizer
        from nltk.corpus import stopwords
        from english_words import get_english_words_set

        # Test initialization
        lemmatizer = WordNetLemmatizer()
        test_word = lemmatizer.lemmatize("testing")

        # If we get here, NLTK is working
        lower_words_set = get_english_words_set(['web2','gcide'], lower=True)
        stop_words = set(stopwords.words('english'))
        NLTK_AVAILABLE = True
        print("✓ NLTK initialized successfully")
    except Exception as e:
        print(f"⚠️  NLTK initialization failed: {e}")
        print("⚠️  Continuing without NLTK features")
        NLTK_AVAILABLE = False
except ImportError:
    print("⚠️  NLTK not installed - some advanced cleaning features disabled")
    NLTK_AVAILABLE = False

def replace_html_parsing_escape_chars(text):
    """Pre bs4 get rid of MS word generated html escape chars"""
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&amp;', '&')
    text = text.replace('&quot;', '"')
    text = text.replace('&apos;', "'")
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('\u20b9', '₹')
    text = text.replace('0xa0', ' ')
    text = text.replace('\xa0', ' ')
    text = text.replace('\ra0', ' ')
    text = text.replace('\u2013', '-')
    text = text.replace('\u2014', '-')
    text = text.replace('\u2019', "'")
    text = text.replace('\u201c', '"')
    text = text.replace('\u201d', '"')
    text = text.replace('\u2026', '...')
    text = text.replace('\u2018', "'")
    text = text.replace('&mdash;', '-')
    text = text.replace('&ndash;', '-')
    text = text.replace('&lsquo;', "'")
    text = text.replace('&rsquo;', "'")
    text = text.replace('&ldquo;', '"')
    text = text.replace('&rdquo;', '"')
    text = text.replace('&sbquo;', "'")
    text = text.replace('&bdquo;', '"')
    text = text.replace('&bull;', '•')
    text = text.replace('&prime;', '′')
    text = text.replace('&Prime;', '″')
    text = text.replace('&lsaquo;', '‹')
    text = text.replace('&rsaquo;', '›')
    text = text.replace('&oline;', '‾')
    text = text.replace('&frasl;', '/')
    text = text.replace('&rupee;', '₹')
    text = text.replace('&dollars;', '$')
    diamonds = ['&#9671;', '&#x25C7;', '◇', '◆', '♢', '♦', '&#9826;', '◊', '<>']
    for d in diamonds:
        text = text.replace(d, '◆')
    text = text.replace('„', '"')
    text = text.replace('"', '"')
    text = text.replace('"', '"')
    text = text.replace('…', '...')
    text = text.replace('–', '-')
    text = text.replace('—', '-')
    text = text.replace(''', "'")
    text = text.replace(''', "'")
    text = text.replace('‟', '"')
    text = text.replace('„', '"')
    text = text.replace('‛', "'")
    text = re.sub(r"(?:\'\s*\'|\`\s*\`)", '"', text)
    text = re.sub(r"(Hon)\s*(')\s*(ble)", r"\1\2\3", text)
    text = re.sub(r"(?:![BIUH]\d?!\s*)*[\:\-\\\/\s]*TRUE\s*COPY[\:\-\\\/\s]*(?:!=[BIUH]\d?!\s*)*", "", text)
    text = re.sub(r"(?:![BIUH]\d?!\s*)*[\:\-\\\/]+\s*[Tt]rue\s*[Cc]opy\s*[\:\-\\\/]+(?:!=[BIUH]\d?!\s*)*", "", text)
    return text

def premature_yeets(text):
    """Remove premature content that should be yeeted"""
    initals = re.compile(r"(?<=\n\n)\s*(?:![BIUH]\d?!\s*)*(?:\s*D[Rr]\.?\s*)?(?:\s*C\.?J\.?I?\.?\s*)?[A-Z\.]+\s*\,\s*(?:JUDGE|C\.?J\.?I\.?|C\.?J\.?|J)\s*\&?(?:\s*!=[BIUH]\d?!)*")
    yeets_init = []
    yeets = []
    yeets_init.extend(re.findall(initals, text))
    if len(yeets_init) > 0:
        yeets = [x for x in yeets_init if len(x) > 0]
        for y in yeets:
            text = text.replace(y, '')
            print(f"\t\t*PREMATURE YEET: {y}")
    yeets = "\n|\n".join([y.strip() for y in yeets])
    return text, yeets

def replace_blockquote(match):
    """Replace blockquote formatting"""
    firstline_indent = match.group(1)
    blockquote = match.group(2)
    lines = blockquote.split('\n\n')
    line_indents = [re.match(r"\A *(\t*) *", line).group(1) for line in lines]
    stripped_lines = [line.strip() for line in lines]

    if re.match(r"\A(?:![BIUH]\d?!)*\s*(?:\"|\')\s*\Z", stripped_lines[0]):
        stripped_lines[0] = stripped_lines[0] + stripped_lines[1]
        stripped_lines.pop(1)
        line_indents.pop(1)

    if len(stripped_lines) > 1:
        if re.match(r"\A(?:\"|\')\s*(?:!=[BIUH]\d?!)*\Z", stripped_lines[-1]) or re.match(r"(?i)\A\s*(?:!=?[BIUH]\d?!\s*)*[\[\(]?\s*emphasis\s*added\s*[\[\)]?\s*(?:!=?[BIUH]\d?!\s*)*\Z", stripped_lines[-1]):
            stripped_lines[-2] = stripped_lines[-2] + stripped_lines[-1]
            stripped_lines.pop(-1)
            line_indents.pop(-1)
        stripped_lines[-1] = re.sub(r"(\S)\s+(['\"])", r"\1\2", stripped_lines[-1])

    indented_lines = []
    for i, line in enumerate(stripped_lines):
        if len(line_indents[i]) >= len(firstline_indent):
            indented_lines.append(f'\t\t{line_indents[i]}{line}')
        else:
            indented_lines.append(f'\t\t{firstline_indent}{line}')
    return '\n\n[BEGIN BLOCKQUOTE]\n\n' + '\n\n'.join(indented_lines) + '\n\n[END BLOCKQUOTE]\n\n'


def is_english_word(word):
    """Check if word is English using NLTK"""
    if not NLTK_AVAILABLE:
        return True
    try:
        lemmatized_word = lemmatizer.lemmatize(word.lower())
        return wordnet.synsets(lemmatized_word) or word in stop_words or word in lower_words_set
    except Exception:
        return True

def replace_broken_lowercase_words(match, s2_dash=False):
    """Replace broken lowercase words across line breaks"""
    s1 = match.group(1)
    s2 = match.group(4)
    if s2.startswith('-'):
        s2 = s2[1:].lstrip()
        s2_dash = True
    if is_english_word(s1) and is_english_word(s2):
        if s2_dash:
            return f'{s1}{match.group(2)}{match.group(3)}-{s2}'
        return f'{s1}{match.group(2)} {match.group(3)}{s2}'
    elif is_english_word(s1 + s2):
        if s2_dash:
            return f'{s1}{match.group(2)}{match.group(3)}-{s2}'
        return f'{s1}{match.group(2)}{match.group(3)}{s2}'
    if s2_dash:
        return f'{s1}{match.group(2)}{match.group(3)}-{s2}'
    return f'{s1}{match.group(2)} {match.group(3)}{s2}'

def titled_case(match):
    """Handle titled name/initials case"""
    name_components = [match.group(1) + match.group(2), match.group(3)]
    name_components = [x for x in name_components if x != None]
    name_components = [x.strip() for x in name_components if x.strip() != '']
    name_components = [re.sub(r"[\t\n]+", " ", x) for x in name_components]
    name_components = [re.sub(r" +", " ", x) for x in name_components]
    return ' '.join(name_components)

def replace_lowercase_word_followed_only(match):
    """Replace lowercase word followed by specific patterns"""
    start_word = match.group(1)
    whitespace_in_bw = match.group(3)
    close_markup = match.group(2)
    new_markup_open = match.group(4)
    if close_markup == None:
        close_markup = ''
    else:
        close_markup = close_markup.strip()
    if new_markup_open == None:
        new_markup_open = ''
    else:
        new_markup_open = new_markup_open.strip()
    if start_word.lower() == start_word:
        return f'{start_word}{close_markup} {new_markup_open}'
    else:
        return match.group(0)

def name_regex_replacer(match):
    """2-4 word long capitalized series line br fixes"""
    open_markups = [match.group(1), match.group(4), match.group(8), match.group(12)]
    names = [match.group(2), match.group(5), match.group(9), match.group(13)]
    close_markups = [match.group(3), match.group(6), match.group(10), '']
    header_false_matches_indicators = ["versus", "v", "vs", "vs.", "v.", "respondents", "respondent", "petitioners", "petitioner", "petitioner(s)", "respondent(s)", "coram", "judge", "bench", "date"]
    names_lower = [x.lower() for x in names]
    if any(x in names_lower for x in header_false_matches_indicators):
        return match.group(0)
    else:
        opening_spaces = [match.group(7), match.group(11)]
        result = ''
        non_empty_names = [x for x in names if x != None and x != '']
        final_space = None
        if len(non_empty_names) == 2 and opening_spaces[0].strip() != opening_spaces[0]:
            final_space = opening_spaces[0]
        if len(non_empty_names) == 3 and opening_spaces[1].strip() != opening_spaces[1]:
            final_space = opening_spaces[1]
        for i in range(len(names)):
            if names[i] != None and names[i] != '' and i != len(non_empty_names) - 1:
                result += f'{open_markups[i]}{names[i]}{close_markups[i]} '
            else:
                result += f'{open_markups[i]}{names[i]}{close_markups[i]}'.strip()
        if final_space:
            result += final_space
        return result

def replace_broken_case_num(match):
    """Replace broken case numbers"""
    result = match.group(1).strip()
    if not result:
        result = ''
    if match.group(2) != None and match.group(2) != '':
        result += match.group(2) + ' '
    if match.group(3) != None and match.group(3) != '':
        if match.group(3).endswith('.'):
            result += match.group(3) + ' '
        else:
            result += match.group(3)
    if match.group(4) != None and match.group(4) != '':
        result += match.group(4) + ' '
    result += match.group(5) + match.group(6) + match.group(7) + ' ' + match.group(8) + ' ' + match.group(9) + ' ' +  match.group(10) + ' ' + match.group(11)
    if match.group(12) != None and match.group(12) != '':
        result += ' ' + match.group(12)
    result += match.group(13)
    result = result.replace('  ', ' ')
    if match.group(0).lstrip() != match.group(0):
        result = ' ' + result
    if match.group(0).rstrip() != match.group(0):
        result += ' '
    return result

def rupee_cleaner(m):
    """Clean rupee number formatting"""
    if m.group(1) == '`':
        ans = f'₹ {m.group(2)}{m.group(3)}{m.group(4)}'
    else:
        ans = f'{m.group(1)} {m.group(2)}{m.group(3)}{m.group(4)}'
    if m.group(0).lstrip() != m.group(0):
        ans = ' ' + ans
    if m.group(0).rstrip() != m.group(0):
        ans += ' '
    return ans

def letter_to_int(s: str) -> int:
    """Convert letter to number"""
    return ord(s.lower()) - 96

def roman_to_int(s: str) -> int:
    """Convert Roman numerals to integers"""
    values = {'i': 1, 'v': 5, 'x': 10, 'l': 50, 'c': 100, 'd': 500, 'm': 1000}
    int_val = 0
    s = s.lower()
    for i in range(len(s)):
        if s[i] not in values.keys():
            return 0
        if i > 0 and values[s[i]] > values[s[i - 1]]:
            int_val += values[s[i]] - 2 * values[s[i - 1]]
        else:
            int_val += values[s[i]]
    return int_val

def section_num_replace(m):
    """Replace section number formatting"""
    if m.group(3) != '' and m.group(3) != None:
        ans = f"{m.group(1)} {m.group(2)} {m.group(3)}"
    else:
        ans = f"{m.group(1)} {m.group(2)}"
    if m.group(0).lstrip() != m.group(0):
        ans = ' ' + ans
    if m.group(0).rstrip() != m.group(0):
        ans += ' '
    return ans

def broken_date_replacer(m):
    """Fix broken date formatting"""
    result = f"{m.group(1)}{m.group(2)}{m.group(3)}{m.group(4)}{m.group(5)}{m.group(6)}{m.group(7)}{m.group(8)}{m.group(9)}{m.group(10)}{m.group(11)}{m.group(12)}"
    if m.group(0).rstrip() != m.group(0):
        result += ' '
    if m.group(0).lstrip() != m.group(0):
        result = ' ' + result
    return result

def broken_citation_fixer(m):
    """Fix broken citations"""
    ans = ''
    if m.group(0).lstrip() != m.group(0):
        ans = ' '
    ans += f"{m.group(1)}{m.group(2)} {m.group(3)}{m.group(4)} {m.group(5)}{m.group(6)} {m.group(7)}{m.group(8)}{m.group(9)}"
    if m.group(0).rstrip() != m.group(0):
        ans += ' '
    return ans

def broken_air_fixer(m):
    """Fix broken AIR citations"""
    ans = ''
    if m.group(0).lstrip() != m.group(0):
        ans = ' '
    ans += re.sub(r"\s+", "", m.group(1)) + re.sub(r"\s+", "", m.group(2)) + ' ' + re.sub(r"\s+", "", m.group(3)) + re.sub(r"\s+", "", m.group(4)) + ' ' + re.sub(r"\s+", "", m.group(5)) + re.sub(r"\s+", "", m.group(6)) + ' ' + re.sub(r"\s+", "", m.group(7)) + re.sub(r"\s+", "", m.group(8)) + ' ' + re.sub(r"\s+", "", m.group(9)) + re.sub(r"\s+", "", m.group(10)) + re.sub(r"\s+", "", m.group(11))
    if m.group(0).rstrip() != m.group(0):
        ans += ' '
    return ans

def clean_txt(doc):
    """
    comprehensive legal document cleaning
    """
    # spaces cleanup
    paras = doc.split('\n\n')
    paras = [re.sub(r"\A(\s*)(\d{1,2}[\.\-\/\s]+\d{1,2}[\s\.\-\/]+\d{2,5}\.?)[ \t]{2,}([A-Z])", r"\1\2\n\1\3", p) for p in paras]
    doc = '\n\n'.join(paras)
    doc = re.sub(r' +', ' ', doc)
    doc = doc.replace(": -", ":-")
    doc = re.sub(r"(\w),(\w[^\d])", r"\1, \2", doc)

    # replacing these into all caps
    doc = re.sub(r'(?i)J\s+U\s+D\s+G\s+M\s+E\s+N\s+T', 'JUDGMENT', doc)
    doc = re.sub(r'(?i)O\s+R\s+D\s+E\s+R', 'ORDER', doc)
    doc = re.sub(r'(?i)B\s+E\s+F\s+O\s+R\s+E', 'BEFORE', doc)
    doc = re.sub(r'(?i)C\s+O\s+R\s+A\s+M', 'CORAM', doc)
    doc = re.sub(r'(?i)V\s+E\s+R\s+S\s+U\s+S', 'VERSUS', doc)
    doc = re.sub(r'(?i)E\s*m\s*p\s*h\s*a\s*s\s*i\s*s\s*A\s*d\s*d\s*e\s*d', 'Emphasis Suppplied', doc)
    doc = re.sub(r'(?i)E\s*m\s*p\s*h\s*a\s*s\s*i\s*s\s*S\s*u\s+p\s*p\s*l\s*i\s*e\s*d', 'Emphasis Suppplied', doc)
    doc = doc.replace("_A_G_A_R_T_A_L_A_", "AGARTALA")
    doc = re.sub(r"([Ee])(xtra [Oo]rdinary)", r"\1xtraordinary", doc)
    doc = doc.replace("EXTRA ORDINARY", "EXTRAORDINARY")
    doc = doc.replace("COMMOM", "COMMON")

    # hard coded order tbl format header row cleaning
    doc = re.sub(r'(?i)"(!B!)?\s*Sl.\s*(!=B!)?\s*(!B!)?\s*No.\s*(!=B!)?\s*",', '!B!Sl. No.!=B!,', doc)
    doc = re.sub(r'(?i),"(!B!)?\s*OFFICE\s*(!=B!)?\s*(!B!)?\s*NOTE\s*(!=B!)?\s*"', ',!B!OFFICE NOTE!=B!', doc)
    doc = re.sub(r"(?i)Contd\.\.*\s*\d+", "", doc)

    # generic whitespace repair
    doc = re.sub(r"\n+\s*\n+", "\n\n", doc)
    doc = re.sub(r" +", " ", doc)

    # word sometimes fails at superscripts
    superscripts = [
        r"(\s+)(?<!\!SUP\!)th(?!\!\=SUP\!)(\s+)",
        r"(\s+)(?<!\!SUP\!)st(?!\!\=SUP\!)(\s+)",
        r"(\s+)(?<!\!SUP\!)nd(?!\!\=SUP\!)(\s+)",
        r"(\s+)(?<!\!SUP\!)rd(?!\!\=SUP\!)(\s+)"
    ]
    for s in superscripts:
        doc = re.sub(s, r"\1", doc)

    """
    Indentation and whitespace repair:
    """
    # Move paragraphs up to same line as para number
    doc = re.sub(r"(?<=\n\n)([ \t]*)((?:![BIUH]\d?!)*)([ \t]*)(\[?\(?(?:\d+\.?\d*\.?\d*|[a-z]+)\.?\]?\.?\)?)\s*((?:!=[BIUH]\d?!)*)[ \t]*\n+([ \t]*)", r"\n\1\3\2\4\5 ", doc)

    # If numbered para is in-line AFTER TAB+ (enforcement), push it to newline
    doc = re.sub(r"([A-Za-z\[\]\(\)\.\,\;\'\"\:\{\}\?\/\<\>\-\—\*\&\%]) *(\t+) *((?:![BIUH]\d?!)*)([ \t]*)(\[?\(?(?:\d+\.?\d*\.?\d*|[a-z]+)\.?\]?\)?\.?)[ \t]*((?:!=[BIUH]\d?!)*)([ \t]*)(\S)", r"\1\n\n\7\2\4\3\5\6 \8", doc)

    # missing line breaks before hrules
    doc = re.sub(r"(\S)([ \t]+(?:![BIUH]\d?!)*[ \t]*\={7,})", r"\1\n\n\2", doc)
    doc = re.sub(r"(\S)([ \t]+(?:![BIUH]\d?!)*[ \t]*\_{7,})", r"\1\n\n\2", doc)
    doc = re.sub(r"(\S)([ \t]+(?:![BIUH]\d?!)*[ \t]*\-{7,})", r"\1\n\n\2", doc)

    # missing line breaks after hrules
    doc = re.sub(r"(\={7,}(?:[ \t]*!=[BIUH]\d?!)*)([ \t]+\S)", r"\1\n\n\2", doc)
    doc = re.sub(r"(\-{7,}(?:[ \t]*!=[BIUH]\d?!)*)([ \t]+\S)", r"\1\n\n\2", doc)
    doc = re.sub(r"(\_{7,}(?:[ \t]*!=[BIUH]\d?!)*)([ \t]+\S)", r"\1\n\n\2", doc)

    # line breaks misrepresented as spaces by word due to heading styles
    doc = re.sub(r"(\S)([ \t]+!H\d!)", r"\1\n\n\2", doc)

    # first convert all "correct" in-line tabs to spaces
    doc = re.sub(r"(\S) *\t+ *((?:!=?[BIUH]\d?!)*[\:\-])", r"\1\2", doc)
    doc = re.sub(r"([\)\]\.\,\;\:\?\-\/\\\!\w]) *(\t+) *(\S)", r"\1 \3", doc)

    # all other in-line tabs become line breaks
    doc = re.sub(r"(\S[^\*\=\-\_\:\;]) *(\t+) *(\S[^\*\=\-\_])", r"\1\n\n\2\3", doc)

    # standardize line breaks to \n\n
    doc = re.sub(r"\n\n+", "\n\n", doc)
    doc = re.sub(r"([^\n])\n([^\n])", r"\1\n\n\2", doc)
    doc = re.sub(r"[ \t]+\n", "\n", doc)

    # no spaces in between, before or after tab indent
    doc = re.sub(r"(\t+) +", r"\1", doc)
    doc = re.sub(r" +(\t+)", r"\1", doc)
    doc = re.sub(r"\A(\n+)", "", doc)
    doc = re.sub(r"(\s*)\Z", "", doc)
    doc = re.sub(r"(\t+) +(\t+)", r"\1\2", doc)
    doc = re.sub(r" +(\,)", r"\1", doc)
    doc = re.sub(r"((?:!=(?:SUP|SUB)!)+)(\w)", r"\1 \2", doc)

    """
    FALSE LINE BREAKS - Complex legal document specific fixes
    """
    # GENERAL CASES - first lower case word \n\n second lower case word
    if NLTK_AVAILABLE:
        doc = re.sub(r"\b([a-z]+)((?:!=?(?:[BIUH]\d?|SUB|SUP)!)*)[ \t]*\n\n[ \t]*((?:!=?(?:[BIUH]\d?|SUB|SUP)!)*)[ \t]*((?:\- ?)?[a-z]+(?=[\s\,]))", replace_broken_lowercase_words, doc)

    # line break but newline starts with lowercase series
    doc = re.sub(r"(?<![\.\:\;\-\"\'\?])[ \t]*\n\n[ \t]*(?=(?:!=?(?:[BIUH]\d?|SUB|SUP)!)*(?:[a-z]+[^\)\]\}\.\-\>]\b|a ))", r" ", doc)

    # lowercase beginning newline after an inline quote
    doc = re.sub(r'(?<=\S)(\"[\w ]+)\" *\n\n[ \t]*([a-z]+ )', r'\1 \2', doc)

    # newline starting with punctuation but not after punctuation that denotes bullet points
    doc = re.sub(r"(?<![\.\:\;\-\"\'\,\?])[ \t]*\n\n[ \t]*([\.\,])", r" \1", doc)

    # acronym period newline lowercase word or number
    doc = re.sub(r"([A-Z\.\,]+[A-Z]\.(?:!=?[BIUH]\d?![ \t]*)*)[ \t]*\n\n[ \t]*((?:!=?[BIUH]\d?![ \t]*)*[\w\(\[\,]+[^\.\]\)])", r"\1 \2", doc)

    # any paranthetical piece
    doc = re.sub(r'\n\n[ \t]*((?:!=?[BIUH]\d?!)*)[ \t]*([\(\[](?:[\w\"\,\.]+\s+[\w\"\,\.]+\s+|[A-Za-z]{6,}))', r' \1\2', doc)

    # any time there is text \n\n punctuation(s) \n\n text --> text(punctuations) \n\n text
    doc = re.sub(r"([^\n]+)\n\n[ \t]*(((?:!=?(?:[BIUH]\d?|SUB|SUP)!)*)[\.\,\:\;\-\?]+((?:!=?(?:[BIUH]\d?|SUB|SUP)!)*))[ \t]*\n\n[ \t]*([^\n]+)", r"\1\2\n\n\3", doc)

    # SPECIAL CASES
    doc = re.sub(r"(No.|Govt.|i.e.)[ \t]*\n\n[ \t]*", r"\1 ", doc)
    doc = re.sub(r"\n\n[ \t]*(i\.e[\.\,\s])", r" \1", doc)
    doc = re.sub(r"([IC][Rr\.]*P\.?C\.?[ \t*](?:!=?[BIUH]\d?![ \t]*)*)\n\n[ \t]*([\(\[\w\,]+)", r"\1 \2", doc)
    doc = re.sub(r"(ORDER)([ \t]+)([A-Z][a-z]+)", r"\1\n\n\2\3", doc)
    doc = re.sub(r"(?i)((?:respond[ae]nts?|applicants?|petitioners|parties|party|appell[ae]nts?|prosecution|prosecutrix|victims?|complain[ae]nts|defence|defense|accused|defend[ae]nts?|plaintiffs?)-)[ \t]*\n\n[ \t]*((?:respondents?|applicants?|petitioners|parties|party|appellants?|prosecution|prosecutrix|victims?|complainants|defence|defense|accused|defendants?|plaintiffs?))", r"\1 \2", doc)
    doc = re.sub(r"\n\n\s*((?:!=?(?:[BIUH]\d?|SUB|SUP)!\s*)*\([Ss][Uu][Pp][Rr][Aa]\))", r" \1", doc)
    doc = re.sub(r"(\d)\n\n(\s*(?:!=?(?:[BIUH]\d?|SUB|SUP)!\s*)*\([a-z]\))", r"\1 \2", doc)
    doc = re.sub(r"(in)[ \t\n]+([A-Z]{1,4}\. ?[A-Z]{1,4}\.)", r"\1 \2", doc)
    doc = re.sub(r"(Civil|Criminal|District Court|Trial Court|Single|Govt.|Advocate|Solicitor|Public|Special|Learned|Constitution|Sr\.|Divn\.|Division|Senior)\s+(Judge|Bench|Advocate|General|Pleader|Prosecutor|Lawyer|Counsel|Division|Divn\.)", r"\1 \2", doc)

    # broken dates
    doc = re.sub(r"\s*((?:!(?:[BIUH]\d?|SUP|SUB)!)*)\s*[^\d](\d)\s*(\d?)\s*([\-\.\/])\s*(\d?)\s*(\d)\s*([\-\.\/])\s*(\d?)\s*(\d?)\s*(\d?)\s*(\d)(\d)\s*", broken_date_replacer, doc)
    doc = re.sub(r"^(\s*\d{1,2}[\.\-\/\s]+\d{1,2}[\s\.\-\/]+\d{2,5}\.?)[ \t]{2,}([A-Za-z])", r"\1\n\n\2", doc)
    doc = re.sub(r"([a-z])[ \t\n]+((?:[A-Z][a-z]+ ){0,4}Act)\s*(,|of|\-)?\s*(\d{4})?", r"\1 \2\3 \4", doc)
    doc = re.sub(r"(\S\s)((?:(?:[Ss]ub\-?)?[Ss]ection|[Aa]rticle|[Cc]lause|pp\.?)+(?:s)?(?:!=?(?:[BIUH]\d?|SUB|SUP)!)*)\s*((?:!=?(?:[BIUH]\d?|SUB|SUP)!)*\(?\[?[a-z\d]+\)?\]?)\s*((?:[IC][Rr]?\.?P\.?C)?)", section_num_replace, doc)
    doc = re.sub(r"(\d+(?:!=?(?:[BIUH]\d?|SUB|SUP)!)*)\s*((?:!=?(?:[BIUH]\d?|SUB|SUP)!)*\/\-)", r"\1\2", doc)

    # rupee cleaning
    doc = re.sub(r"(?:[\s\b])((?:(?:`)|[Rr][SsEe]\.?|I\.?N\.?R\.?|₹|[Rr]upee(?:s)?|\u20b9|\u0060)(?:!=?(?:[BIUH]\d?|SUB|SUP)!)*)\s*((?:!=?(?:[BIUH]\d?|SUB|SUP)!)*)((?:\d+\,?)+(?!\.))(\/?\-?)", rupee_cleaner, doc)
    doc = re.sub(r"([a-z]+\,?)\s*\n\n\s*((?:!=?(?:[BIUH]\d?|SUB|SUP)!)*(?:\`|[Rr][Ss]\.?|I\.?N\.?R\/?|₹|[Rr]upee(?:s)?|₹))", r"\1 \2", doc)

    # line breaks before and after 'vs'
    doc = re.sub(r"\n\n[ \t]*((?:(?:!=?(?:[BIUH]\d?|SUP|SUB)!)*(?:[Vv][Ss]?|[Vv]ersus)\.?(?:!=?(?:[BIUH]\d?|SUP|SUB)!)*))(?=(?:[ \t]+\S|[ \t]*\n\n))", r" \1", doc)
    doc = re.sub(r"((?:(?:!=?(?:[BIUH]\d?|SUP|SUB)!)*(?:[Vv][Ss]?|[Vv]ersus)\.?(?:!=?(?:[BIUH]\d?|SUP|SUB)!)*))(?:\t|\s{2,}|\n\n)+", r"\1 ", doc)
    doc = re.sub(r"(esp\.(?:!=?(?:[BIUH]\d?|SUP|SUB)!)*)\s*(\S)", r"\1 \2", doc)
    doc = re.sub(r"(\S)[ \t\n]+([\,\)\]])", r"\1 \2", doc)
    doc = re.sub(r"([\(\[])[ \t\n]+((?:!=?(?:[BIUH]\d?|SUP|SUB)!)*[a-zA-Z\d])", r"\1\2", doc)

    # titled case
    titled = r"(?=\b|\s)((?:Mr|Mrs|Miss|[Mm]\/[Ss]|Shri|Smt|Dr|Prof|Hon|Rev|Fr|Ms|Hon'ble|Justice|Judge|Master|Addl|Learned|P\.?P|S\.?G|G\.?P|A\.?G|P\.?W\d?|D\.?W\d?|P\d)(?:!=?(?:[BIUH]\d?|SUP|SUB)!)*)\s*(\.?(?:!=?(?:[BIUH]\d?|SUP|SUB)!)*)((?:(?:!=?(?:[BIUH]\d?|SUP|SUB)!)*(?:\s*[A-Z\&][a-z]+|\s*[A-Z\&]\s*\.?)){2,7})"
    doc = re.sub(titled, titled_case, doc)

    # case number pattern
    case_n_pattern = re.compile(r"((?:!(?:[BIUH]\d?|SUP|SUB)!\s*)*)(\d*)\s*([A-Za-z]*\.?)\s*([A-Za-z]*\.?)\s*([A-Za-z]+\.?)\s*(\(?[A-Za-z]*\)?)\s*(\.?)\s*((?:No|\/|\-)\.?)\s*(\d+)\s*(of|\/)\s*(\d{2,4})\s*(\(?\d{0,5}\)?)\s*((?:\.?!=(?:[BIUH]\d?|SUP|SUB)!)*)")
    doc = re.sub(case_n_pattern, replace_broken_case_num, doc)

    doc = re.sub(r"([\w\.\,])(?:!=?(?:[BIUH]\d?|SUP|SUB)!)*\s*(\.{3,12})\s*((?:!=?(?:[BIUH]\d?|SUP|SUB)!)*)\s*((?:!=?(?:[BIUH]\d?|SUP|SUB)!)*(?:[Ff]or|[Cc]ounsel|[Aa]dv|[Rr]espondent|[Pp]etitioner|[Pp]laintiff|[Dd]efendant))", r"\1\2\3 \4", doc)

    # named entities without titles
    named_pattern = r"(?=\S *)((?:!=?(?:[BIUH]\d?|SUP|SUB)!)*)([A-Z][a-z]+)((?:!=?(?:[BIUH]\d?|SUP|SUB)!)*)\s*((?:!=?(?:[BIUH]\d?|SUP|SUB)!)*)([A-Z][a-z]+)((?:!=?(?:[BIUH]\d?|SUP|SUB)!)*)(\s*)((?:!=?(?:[BIUH]\d?|SUP|SUB)!)*)((?:[A-Z][a-z]+)?)((?:!=?(?:[BIUH]\d?|SUP|SUB)!)*)(\s*)((?:!=?(?:[BIUH]\d?|SUP|SUB)!)*)((?:[A-Z][a-z]+)?)(?= *\S)"
    doc = re.sub(named_pattern, name_regex_replacer, doc)

    # mid_citation_line_breaks
    spaced_markup = r"\s*((?:!=?(?:[BIUH]|SUB|SUP)!\s*)*)"
    air = spaced_markup + r"([\[\)]?AIR)" + spaced_markup + r"(\d{4})" + spaced_markup + r"([A-Za-z]+|[A-Z]+)" + spaced_markup + "((?:(?:Court)? ?(?:of [A-za-z]+)?)?)" + spaced_markup + r"(\d+[\]\)]?)" + spaced_markup
    doc = re.sub(re.compile(air), broken_air_fixer, doc)
    scc = spaced_markup + r"([\[\)]?\d+)" + spaced_markup + r"(\d+)" + spaced_markup + r"(SCC)" + spaced_markup + r"(\d+[\]\)]?)" + spaced_markup
    scr = spaced_markup + r"([\[\)]?\d+)" + spaced_markup + r"(\d+)" + spaced_markup + r"(SCR)" + spaced_markup + r"(\d+[\]\)]?)" + spaced_markup
    for pat in [re.compile(scc), re.compile(scr)]:
        doc = re.sub(pat, broken_citation_fixer, doc)

    # between X.Y. (minimum 2 initials) and name
    doc = re.sub(r"((?:[A-Z]\. ?(?:!=?(?:[BIUH]\d?|SUP|SUB)!)*){2,})[ \t]*\n\n[ \t]*((?:!=?(?:[BIUH]\d?|SUP|SUB)!)*[A-Z][a-z]+)", r"\1 \2", doc)

    # MORE GENERAL CASES
    lower_then_upper = r"([A-Za-z\/\,]+)((?:!=?(?:[BIUH]\d?|SUB|SUP)!)*)(\s*\n\n\s*)((?:!=?(?:[BIUH]\d?|SUB|SUP)!)*)(?=[A-Z][a-z]+|[A-Z]+ |[A-Z\.\d]+|\d{1,2}[\.\-\/\s]+\d{1,2}[\s\.\-\/]+\d{2,5}\.?|\d+[\w\(\[ \)\]]+)"
    doc = re.sub(lower_then_upper, replace_lowercase_word_followed_only, doc)

    doc = re.sub(r"((?:[a-z]+|[A-Z]{3,}|\)|\,)\)?(?:!=?(?:[BIUH]\d?|SUB|SUP)!)*)[ \t]*\n\n[ \t]*((?:!=?(?:[BIUH]\d?|SUB|SUP)!)*\d[^\.\)\]\,\-\\\}\>\:\;])", r"\1 \2", doc)

    # NUMBERED PARAS complex processing
    paras = doc.split('\n\n')

    # fix errors like The points are: (i) should be in next line indented
    for i in range(len(paras)-1):
        inline_point = re.search(r"(?i)(\S)[ \t]*\(([ivxlcdm]+)\)", paras[i])
        if inline_point:
            roman_in = inline_point.group(2).lower()
            next_point = re.search(r"(?i)\A(\s*)\(([ivxlcdm]+)\)", paras[i+1])
            if next_point:
                roman_nxt = next_point.group(2).lower()
                if roman_to_int(roman_nxt) == roman_to_int(roman_in) + 1:
                    indent_nxt = next_point.group(1)
                    paras[i] = paras[i].replace(inline_point.group(0), f"{inline_point.group(1)}\n\n{indent_nxt}({inline_point.group(2)})")

    for i in range(len(paras)-1):
        inline_point = re.search(r"(?i)(\S)[ \t]*([\(\[]?)([a-z](?![a-z]))([\.\)\]]+)", paras[i])
        if inline_point:
            letter_in = inline_point.group(3).lower()
            next_point = re.search(r"(?i)\A(\s*)([\(\[]?)([a-z](?![a-z]))([\.\)\]]+)", paras[i+1])
            if next_point:
                letter_nxt = next_point.group(3).lower()
                if letter_to_int(letter_nxt) == letter_to_int(letter_in) + 1:
                    indent_nxt = next_point.group(1)
                    paras[i] = paras[i].replace(inline_point.group(0), f"{inline_point.group(1)}\n\n{indent_nxt}{inline_point.group(2)}{inline_point.group(3)}{inline_point.group(4)})")

    for i in range(len(paras) - 1):
        inline_point = re.search(r"(?i)(\S)[ \t]*([\(\[]?)(\d+(?:\.\d+){0,2})([\.\)\]]*)", paras[i])
        if inline_point:
            number_in = inline_point.group(3)
            next_point = re.search(r"(?i)\A(\s*)[\(\[]?(\d+(\.\d+){0,2})[\.\)\]]*", paras[i + 1])
            if next_point:
                number_nxt = next_point.group(2)
                num_parts_in = [int(part) for part in number_in.split('.')]
                num_parts_nxt = [int(part) for part in number_nxt.split('.')]
                if len(num_parts_in) == len(num_parts_nxt):
                    if len(num_parts_in) == 1 and num_parts_nxt[0] == num_parts_in[0] + 1:
                        indent_nxt = next_point.group(1)
                        paras[i] = paras[i].replace(inline_point.group(0), f"{inline_point.group(1)}\n\n{indent_nxt}{inline_point.group(2)}{inline_point.group(3)}{inline_point.group(4)}")
                    elif len(num_parts_in) == 2 and ((num_parts_nxt[0] == num_parts_in[0] and num_parts_nxt[1] == num_parts_in[1] + 1) or num_parts_nxt[0] == num_parts_in[0] + 1):
                        indent_nxt = next_point.group(1)
                        paras[i] = paras[i].replace(inline_point.group(0), f"{inline_point.group(1)}\n\n{indent_nxt}{inline_point.group(2)}{inline_point.group(3)}{inline_point.group(4)}")
                    elif len(num_parts_in) == 3 and ((num_parts_nxt[0] == num_parts_in[0] and num_parts_nxt[1] == num_parts_in[1] and num_parts_nxt[2] == num_parts_in[2] + 1) or num_parts_nxt[1] == num_parts_in[1] + 1 or num_parts_nxt[0] == num_parts_in[0] + 1):
                        indent_nxt = next_point.group(1)
                        paras[i] = paras[i].replace(inline_point.group(0), f"{inline_point.group(1)}\n\n{indent_nxt}{inline_point.group(2)}{inline_point.group(3)}{inline_point.group(4)}")

    # consequent paragraph numbers should take same indent amount
    for i, p in enumerate(paras[1:]):
        levels = re.search(r"\A(\s*)(?:![BIUH]\d?!)*(\s*)[\(\[]?(\d+|\d+\.\d+|[a-z]{1,5}(?![a-z]))[\.\)\]]*", p)
        if not levels or len(levels.groups()) != 3:
            continue
        prev_levels = re.search(r"\A(\s*)(?:![BIUH]\d?!)*(\s*)[\(\[]?(\d+|\d+\.\d+|[a-z]{1,5}(?![a-z]))[\.\)\]]*", paras[i])
        if not prev_levels or len(levels.groups()) != 3:
            continue
        p_num = levels.group(3)
        p_num_prev = prev_levels.group(3)
        if p_num and p_num_prev:
            idk = True
            if re.match(r"\A[a-z]+\Z", p_num) and re.match(r"\A[a-z]+\Z", p_num_prev):
                if re.match(r'\A[ivxlcdm]+\Z', p_num_prev) and re.match(r'\A[ivxlcdm]+\Z', p_num):
                    if len(p_num) > 1 and len(p_num_prev) > 1:
                        p_num = roman_to_int(p_num)
                        p_num_prev = roman_to_int(p_num_prev)
                        idk = False
                else:
                    if len(p_num) == 1 and len(p_num_prev) == 1:
                        p_num = letter_to_int(p_num)
                        p_num_prev = letter_to_int(p_num_prev)
                        idk = False
                    elif len(p_num) == 2 and len(p_num_prev) == 2:
                        if p_num[0] == p_num_prev[0]:
                            p_num = letter_to_int(p_num[1])
                            p_num_prev = letter_to_int(p_num_prev[1])
                            idk = False
                        elif p_num[1] == 'a' and p_num_prev[1] == 'z' and ord(p_num[0]) == ord(p_num_prev[0]) + 1:
                            p_num = 27
                            p_num_prev = 26
                            idk = False
                    elif len(p_num) > 2 or len(p_num_prev) > 2:
                        continue
                if idk:
                    r_p_num = roman_to_int(p_num)
                    r_p_num_prev = roman_to_int(p_num_prev)
                    if r_p_num != 0 and r_p_num_prev != 0:
                        p_num = r_p_num
                        p_num_prev = r_p_num_prev
                        idk = False
                    else:
                        try:
                            p_num = letter_to_int(p_num)
                            p_num_prev = letter_to_int(p_num_prev)
                            idk = False
                        except:
                            continue
            if idk == False:
                if float(p_num) == float(p_num_prev) + 1 or (float(p_num) == float(p_num_prev) + 0.1 and float(p_num) % 1 != 0):
                    indent_prev_1 = prev_levels.group(1)
                    indent_prev_2 = prev_levels.group(2)
                    paras[i+1] = re.sub(r"\A\s*(![BIUH]\d?!)*\s*", indent_prev_1 + indent_prev_2 + r"\1", p)

    # BLOCK QUOTES
    paras = [re.sub(r"\A(\s*(?:![BIUH]\d?!)*)(\s*)(\"|\')(\s*)", r"\1\2\4\3", p) for p in paras]

    for i, p in enumerate(paras[1:]):
        if re.match(r"\A\s*(?:![BIUH]\d?!)*\s*(?:\"|\')", p):
            indent_i = re.search(r"\A(\s*)(?:![BIUH]\d?!)*(\s*)(?:\"|\')", p)
            if not indent_i or len(indent_i.groups()) != 2:
                continue
            indent_i0 = indent_i.group(1)
            indent_i1 = indent_i.group(2)
            indent_prev = re.search(r"\A(\s*)(?:![BIUH]\d?!)*(\s*)", paras[i])
            if not indent_prev or len(indent_prev.groups()) != 2:
                continue
            indent_prev0 = indent_prev.group(1)
            indent_prev1 = indent_prev.group(2)
            ith = indent_i0 + indent_i1.replace(' ', '')
            prev = indent_prev0 + indent_prev1.replace(' ', '')
            if not len(ith) > len(prev):
                paras[i+1] = re.sub(r"\A\s*(![BIUH]\d?!)*\s*", r"\t" + indent_prev0 + indent_prev1 + r"\1", p)

    doc = '\n\n'.join(paras)

    # block quote patterns
    pattern = r'(?i)(?<=\n\n)(\t*)((?:![BIUH]\d?!\s*)*\"(?:.|\n)*?\"[ \t]*(?:!=?[BIUH]\d?!)?\s*(?:[\(\[]\s*[Ee]mphasis [Ss]upplied\s*[\]\)])?(?=[ \t]*\n\n))'
    single_quoted_pattern = r"(?i)(?<=\n\n)(\t*)((?:![BIUH]\d?!\s*)*\'(?:.|\n)*?\'[ \t]*(?:!=?[BIUH]\d?!)?\s*(?:[\(\[]\s*[Ee]mphasis [Ss]upplied\s*[\]\)])?(?=[ \t]*\n\n))"
    doc = re.sub(single_quoted_pattern, replace_blockquote, doc)
    doc = re.sub(pattern, replace_blockquote, doc)

    # FINALLY, one more space removal
    doc = re.sub(r'(\S)\s*(\([A-Z\s\.\,]+, J\.?\))', r'\1\n\n\2', doc)
    doc = re.sub(r"\n\n\n+", "\n\n", doc)
    doc = re.sub(r"([\)\,\.\;\:\?])([A-Za-z])", r"\1 \2", doc)
    doc = re.sub(r" +", r" ", doc)
    return doc

def call_all(doc):
    """Main cleaning function """
    res = {}
    res['pure_ocr_doc'] = doc

    lazy_clean = doc.split('\n\n')
    lazy_clean = [re.sub(r"\s{2,}", " ", para) for para in lazy_clean]
    res['lazy_clean_doc'] = '\n\n\t\t'.join(lazy_clean)

    doc = replace_html_parsing_escape_chars(doc)
    doc, yeeted_footers = premature_yeets(doc)
    doc = clean_txt(doc)

    res['legacy_clean_doc'] = doc

    return res

def process_single_page_text(args):
    """Process a single page text - for parallel processing"""
    page_path, page_num = args

    try:
        with open(page_path, 'r', encoding='utf-8') as f:
            page_text = f.read()

        # Clean the page text
        cleaned_data = call_all(page_text)

        return page_num, cleaned_data

    except Exception as e:
        print(f"  ⚠️  Error processing page {page_num}: {e}")
        return page_num, {
            'pure_ocr_doc': '',
            'lazy_clean_doc': '',
            'legacy_clean_doc': ''
        }

def process_book_texts_sequential(input_dir, output_dir, book_name):
    """Process all text files for a single book using ThreadPoolExecutor for pages"""
    try:
        print(f"\nProcessing: {book_name}")
        print(f"Current memory usage: {psutil.Process().memory_info().rss / 1024 / 1024 / 1024:.2f} GB")

        book_txt_dir = os.path.join(input_dir, book_name)
        if not os.path.exists(book_txt_dir):
            print(f"Directory not found: {book_txt_dir}")
            return None

        # Get all page text files
        page_files = [f for f in os.listdir(book_txt_dir) if f.startswith('page_') and f.endswith('.txt')]
        if not page_files:
            print(f"No page files found in {book_txt_dir}")
            return None

        # Sort by page number
        page_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

        # Prepare args for parallel processing
        page_args = []
        for page_file in page_files:
            page_path = os.path.join(book_txt_dir, page_file)
            page_num = int(page_file.split('_')[1].split('.')[0])
            page_args.append((page_path, page_num))

        # Process pages in parallel with ThreadPoolExecutor
        book_data = {}
        successful_pages = 0

        # Determine number of workers based on available memory
        available_mem = psutil.virtual_memory().available / 1024 / 1024 / 1024
        max_workers = min(4, int(available_mem / 0.5))
        max_workers = max(1, max_workers)
        print(f"Using {max_workers} workers for page processing")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all page processing tasks
            future_to_page = {
                executor.submit(process_single_page_text, args): args[1]
                for args in page_args
            }

            with tqdm(total=len(page_args), desc=f"Pages in {book_name}") as pbar:
                for future in as_completed(future_to_page):
                    page_num = future_to_page[future]
                    try:
                        result_page_num, cleaned_data = future.result()
                        book_data[result_page_num] = cleaned_data
                        successful_pages += 1
                    except Exception as e:
                        print(f"Error on page {page_num}: {e}")
                    pbar.update(1)

                    # Monitor memory periodically
                    if successful_pages % 50 == 0 and successful_pages > 0:
                        current_mem = psutil.Process().memory_info().rss / 1024 / 1024 / 1024
                        if current_mem > 8.0:
                            print(f"\nWarning: High memory usage ({current_mem:.2f} GB)")

        # Save cleaned book JSON
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'{book_name}.json')

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(book_data, f, ensure_ascii=False, indent=2, sort_keys=True)

        print(f"Completed: {book_name} ({successful_pages}/{len(page_args)} pages)")
        gc.collect()

        return book_name

    except Exception as e:
        print(f"Error processing {book_name}: {e}")
        return None

def main():
    """Main text cleaning function"""
    print("Starting Text Cleaning for Legal Books Pipeline")

    # Setup paths
    input_dir = "intermediate/ocr/txts"
    output_dir = "intermediate/cleaned/jsons"

    # Check input directory
    if not os.path.exists(input_dir):
        print(f"Input directory not found: {input_dir}")
        print("Run OCR processing first: python scripts/01_ocr_books.py")
        return

    # Get list of books to process
    book_dirs = [d for d in os.listdir(input_dir)
                 if os.path.isdir(os.path.join(input_dir, d))]

    if not book_dirs:
        print(f" No book directories found in {input_dir}")
        print(" Run OCR processing first: python scripts/01_ocr_books.py")
        return

    print(f"Found {len(book_dirs)} books")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Check for already processed books
    if os.path.exists(output_dir):
        processed_books = [f.replace('.json', '') for f in os.listdir(output_dir)
                          if f.endswith('.json')]
        book_dirs = [b for b in book_dirs if b not in processed_books]

    if not book_dirs:
        print("All books have already been cleaned!")
        print(f"Check results in: {output_dir}")
        return

    print(f"Cleaning {len(book_dirs)} books...")
    print(f"Initial memory usage: {psutil.Process().memory_info().rss / 1024 / 1024 / 1024:.2f} GB")
    print(f"Available memory: {psutil.virtual_memory().available / 1024 / 1024 / 1024:.2f} GB")
    print("\nTIP: Use 'htop' or 'top' in another terminal to monitor system resources\n")

    # Process books sequentially
    start_time = time.time()
    successful = 0
    failed = 0

    for book_name in tqdm(book_dirs, desc="Processing books"):
        result = process_book_texts_sequential(input_dir, output_dir, book_name)
        if result is not None:
            successful += 1
        else:
            failed += 1

        gc.collect()

        # Print memory status every 5 books
        if successful % 5 == 0 and successful > 0:
            print(f"\nMemory check - Usage: {psutil.Process().memory_info().rss / 1024 / 1024 / 1024:.2f} GB")

    # Summary
    total_time = time.time() - start_time

    print(f"\nText Cleaning Complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"\nOutput location:")
    print(f"Cleaned JSON files: {output_dir}")

    if failed > 0:
        print(f"\n⚠️  {failed} files failed processing")
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()