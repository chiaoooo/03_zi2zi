import json

with open('train.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# 使用列表推导式将字符串分割为单个字符的列表
chinese_characters = [char for char in text]

character_data = []
unicode_characters = []

# 生成BIG5编码和Unicode编码
for character in chinese_characters:
    unicode_code = "0x" + hex(ord(character))[2:].upper().zfill(4)
    unicode_code2 = "\\u" + hex(ord(character))[2:].upper().zfill(4)
    character_data.append({
        #"Character": character,
        "UNICODE": unicode_code
    })

    unicode_characters.append(unicode_code2)

character_data = sorted(character_data, key=lambda x: int(x["UNICODE"], 16))
unicode_characters = sorted(unicode_characters, key=lambda x: int(x[2:], 16))
data_dict = {"CP950": character_data}
data_dict2 = {"twTrain": unicode_characters}

# 保存文件
#with open("./script_ntut/CP950.json", "w", encoding="utf-8") as json_file:
    #json.dump(data_dict, json_file, ensure_ascii=False, indent=2)

json_text = json.dumps(data_dict2, ensure_ascii=False, separators=(', ', ':')).replace("\\\\", "\\")

with open("cjk_train.json", "w", encoding="utf-8") as json_file:
    json_file.write(json_text)
