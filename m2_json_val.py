import json

with open('val.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# 使用列表推导式将字符串分割为单个字符的列表
chinese_characters = [char for char in text]

character_data = []
unicode_characters = []

# 生成BIG5编码和Unicode编码
for character in chinese_characters:
    unicode_code = "\\u" + hex(ord(character))[2:].upper().zfill(4)
    unicode_characters.append(unicode_code)

unicode_characters = sorted(unicode_characters, key=lambda x: int(x[2:], 16))
data_dict2 = {"twVal": unicode_characters}

json_text = json.dumps(data_dict2, ensure_ascii=False, separators=(', ', ':')).replace("\\\\", "\\")

with open("cjk_val.json", "w", encoding="utf-8") as json_file:
    json_file.write(json_text)