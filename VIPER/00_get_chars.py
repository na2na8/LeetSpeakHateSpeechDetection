from PIL import Image, ImageDraw, ImageFont
from unidecode import unidecode

# Arial-Unicode-MS.ttf
arial_unicode_ms = {
    'group1' : (int("0021", 16), int("052F", 16)),
    "gujarati" : (int("0A80", 16), int("0AFF", 16)),
    'hangul1' : (int("1100", 16), int("11FF", 16)),
    'hangul2' : (int("AC00", 16), int("D7FF", 16)),
    'group2' : (int("1E00", 16), int("1FFF", 16)),
    'math' : (int("2200", 16), int("22FF", 16)),
    'group3' : (int("2300", 16), int("23E7", 16)),
    'group4' : (int("2460", 16), int("2647", 16)),
    'group7' : (int("3000", 16), int("32FF", 16)),
}

# 한자 PingFang-SC-Regular.ttf
# 3400 ~ 9FD5
pingfang = {
    'hanja' : (int("3400", 16), int("9FD5", 16)),
    'sp1' : (int("1F110", 16), int("1F189", 16)),
    'group5' : (int("2E80", 16), int("2EF3", 16)),
    'group6' : (int("2F00", 16), int("2FD5", 16)),
    # 'group8' : (int("3200", 16), int("32FE", 16))
    
}

def get_chars(uni_dict, fname) :
    chars = []
    for key in uni_dict :
        start, end = uni_dict[key][0], uni_dict[key][1]
        for idx in range(start, end + 1) :
            character = chr(idx)
            if unidecode(character) == unidecode(chr(int("0091", 16))) :
                continue
            else :
                chars.append(character)
            
    strs = '\n'.join(chars)
    f = open(fname, 'w')
    f.write(strs)
    f.close()
    
if __name__ == "__main__" :
    # arial
    get_chars(arial_unicode_ms, '00_arial_chars.txt')
    # pingfang
    get_chars(pingfang, '00_pingfang_chars.txt')
    print("generated new chars")