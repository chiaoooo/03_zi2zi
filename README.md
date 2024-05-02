
## zi2zi: Master Chinese Calligraphy with Conditional Adversarial Networks

- **WEB**

https://kaonashi-tyc.github.io/2017/04/06/zi2zi.html

- **官方**

https://github.com/kaonashi-tyc/zi2zi

- **修正後** 

https://github.com/chiaoooo/zi2zi_tensorflow


## 使用 Anaconda Prompt (Anaconda3) - 若要直接使用可以跳到程式執行部分

#### Requirements

```
* Python = 3.7
* CUDA
* cudnn
* Tensorflow = 1.14.0
* Pillow
* numpy
* scipy = 1.2.1
* imageio = 2.9.0
```

#### 建議使用虛擬環境
*  電腦要有**NVIDIA GPU**
*  VRAM 要大於 8GB
*  顯卡不能太新，要支援 CUDA 10.0  

#### 測試 tensorflow-gpu
```
>python
>>>import tensorflow as tf
>>>tf.test.is_gpu_available()
```




---
<br>

### 前置：製作 charset，指定你想生成的字
將要 train 的字放入 train.txt
![image](https://hackmd.io/_uploads/B1t44SLxA.png)

將要 val 的字放入 val.txt
![image](https://hackmd.io/_uploads/SJH8VrUxR.png)


* 做 train 的 json 檔
```
python m1_json_train.py.py
```
* 做 val 的 json 檔
```
python m2_val_train.py.py
```
* 合併兩個 json 檔
```
python m3_merge_json.py.py
```
**執行完會得到 cjk.json 就代表成功！**



---

<br>

<h2 style="color:green;">程式執行</h2>

### 建立環境

使用下面指令可以直接生成環境！！！！！
```
conda env create -f environment.yml
git clone https://github.com/chiaoooo/zi2zi_tensorflow.git
cd zi2zi_tensorlow
```

####建立 sample 資料夾

```
mkdir image_train
mkdir image_val
```

*--srcfont: 來源字體路徑位置
--dstfont: 目標字體路徑位置
--charset: 要讀取的字集 e.g. CN、CNT、JP、KR、<font color=red>TWTrain</font>、<font color=red>TWVal</font>
--samplecount:取幾張圖訓練（數字）
--sampledir:圖片存放位置（對應 package.py 的 --dir）
--label: 類別編號，在<font color=red>同模型訓練多字體</font>時需更換，ex: 2、3...
--shuffle: 是否重新排序字集中文字的排序 e.g. 0: false, 1: true*

這裡設定<font color=hotpink>**來源字體為源樣黑體，目標字體為 CircleFont，訓練字數 1000 字**</font>。

```
python font2img.py --src_font=font/GenYoGothicTW-EL-01.ttf --dst_font=font/CircleFont.ttf --charset=TWTrain --sample_count=1000 --sample_dir=image_train --label=1 --filter=1 --shuffle=1
python font2img.py --src_font=font/GenYoGothicTW-EL-01.ttf --dst_font=font/CircleFont.ttf --charset=TWVal --sample_count=670 --sample_dir=image_val --label=1 --filter=1 --shuffle=0
```

### 建立訓練、驗證資料 object

**得到 train.obj 和 val.obj 在 save_dir 資料夾**

得到 train.obj
save_dir 預設 `experiment/data`

```
python package.py --dir=image_train --save_dir=experiment/data --split_ratio=0.1
```

得到 val.obj 會在最後驗證步驟 infer.py 用到
（這裡 --save_dir 與 infer.py 的 --source_obj 相同）

```
python package.py --dir=image_val --save_dir=experiment/data/val --split_ratio=1
```

### TRAIN

*--experimentdir: 訓練要存的資料夾（已存在），會在內建立 checkpoint、log、sample 資料夾
--experimentid: 模型編號（數字）
--batchsize: 設定 1 epoch ? batch（數字）*

```
python train.py --experiment_dir=experiment --experiment_id=1 --batch_size=16 --lr=0.001 --epoch=500 --sample_steps=50 --schedule=20 --L1_penalty=100 --Lconst_penalty=15
```

### 推論結果 INFER

*--modeldir: 訓練後的 checkpoint 資料夾
--batchsize: 圖片中的文字列數
--experimentids: 對應 font2img 的 --label 數字（預設 1 代表要推論出 label=1 的驗證資料集）*

```
python infer.py --model_dir=experiment/checkpoint/experiment_1_batch_16 --batch_size=1 --source_obj=experiment/data/val/val.obj --embedding_ids=1 --save_dir=experiment/infer_1
```

<p style="color:green;font-weight:bold;">如果要推論沒訓練過的字（沒看過的字）:</p>

把46-56行改成下面這樣

```
def draw_example(ch, src_font, dst_font, canvas_size, x_offset, y_offset, filter_hashes):
    dst_img = draw_single_char(ch, dst_font, canvas_size, x_offset, y_offset)
    # check the filter example in the hashes or not
    dst_hash = hash(dst_img.tobytes())
    if dst_hash in filter_hashes:
        src_img = draw_single_char(ch, src_font, canvas_size, x_offset, y_offset)
        example_img = Image.new("RGB", (canvas_size * 2, canvas_size), (255, 255, 255))
        example_img.paste(src_img, (canvas_size, 0))
        return example_img
    src_img = draw_single_char(ch, src_font, canvas_size, x_offset, y_offset)
    example_img = Image.new("RGB", (canvas_size * 2, canvas_size), (255, 255, 255))
    example_img.paste(dst_img, (0, 0))
    example_img.paste(src_img, (canvas_size, 0))
    return example_img
```

並重新執行　
* python font2img.py --src_font=font/GenYoGothicTW-EL-01.ttf --dst_font=font/CircleFont.ttf --charset=TWVal --sample_count=670 --sample_dir=image_val --label=1 --filter=1 --shuffle=0
* python package.py --dir=image_val --save_dir=experiment/data/val --split_ratio=1


