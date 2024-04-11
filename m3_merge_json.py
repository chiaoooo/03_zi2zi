import json

# 读取 cjk_train.json 文件
with open('cjk_train.json', 'r', encoding='utf-8') as train_file:
    cjk_train_data = json.load(train_file)

# 读取 cjk_val.json 文件
with open('cjk_val.json', 'r', encoding='utf-8') as val_file:
    cjk_val_data = json.load(val_file)

# 合并数据
merged_data = {}
merged_data.update(cjk_train_data)
merged_data.update(cjk_val_data)

# 将合并后的数据写入新的 JSON 文件，保留 Unicode 编码
with open('cjk.json', 'w', encoding='utf-8') as merged_file:
    #json.dump(merged_data, merged_file, ensure_ascii=True, indent=2)
    json.dump(merged_data, merged_file, ensure_ascii=True, indent=None)
