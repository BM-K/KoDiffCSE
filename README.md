# KoDiffCSE
Difference-based Contrastive Learning for Korean Sentence Embeddings <br>
   - [DiffCSE-[NAACL 2022]](https://arxiv.org/abs/2204.10298) <br>
   - [[Github]](https://github.com/voidism/DiffCSE) Official implementation of DiffCSE <br>
<img src=https://user-images.githubusercontent.com/55969260/201829550-9674a3ac-cb9b-4e17-b777-7d96fdf5c633.png>

## Quick tour
```python
import torch
from transformers import AutoModel, AutoTokenizer

def cal_score(a, b):
    if len(a.shape) == 1: a = a.unsqueeze(0)
    if len(b.shape) == 1: b = b.unsqueeze(0)

    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    return torch.mm(a_norm, b_norm.transpose(0, 1)) * 100

model = AutoModel.from_pretrained('BM-K/KoDiffCSE-RoBERTa')
tokenizer = AutoTokenizer.from_pretrained('BM-K/KoDiffCSE-RoBERTa')

sentences = ['치타가 들판을 가로 질러 먹이를 쫓는다.',
             '치타 한 마리가 먹이 뒤에서 달리고 있다.',
             '원숭이 한 마리가 드럼을 연주한다.']

inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
embeddings, _ = model(**inputs, return_dict=False)

score01 = cal_score(embeddings[0][0], embeddings[1][0])  # 84.56
# '치타가 들판을 가로 질러 먹이를 쫓는다.' @ '치타 한 마리가 먹이 뒤에서 달리고 있다.'
score02 = cal_score(embeddings[0][0], embeddings[2][0])  # 48.06
# '치타가 들판을 가로 질러 먹이를 쫓는다.' @ '원숭이 한 마리가 드럼을 연주한다.'
```

## Setups
[![Python](https://img.shields.io/badge/python-3.8.5-blue?logo=python&logoColor=FED643)](https://www.python.org/downloads/release/python-385/)
[![Pytorch](https://img.shields.io/badge/pytorch-1.7.1-red?logo=pytorch)](https://pytorch.org/get-started/previous-versions/)

## Encoder Models
Baseline encoders used for korean sentence embedding - [KLUE-PLMs](https://github.com/KLUE-benchmark/KLUE/blob/main/README.md)

| Model                | Embedding size | Hidden size | # Layers | # Heads |
|----------------------|----------------|-------------|----------|---------|
| KLUE-BERT-base            | 768            | 768         | 12       | 12      |
| KLUE-RoBERTa-base         | 768            | 768         | 12       | 12      |

> **Warning** <br>
> Large pre-trained models need a lot of GPU memory to train

## Datasets
The data must exist in the "--path_to_data" folder
- [wiki-corpus](https://github.com/jeongukjae/korean-wikipedia-corpus) (Unsupervised Training)
- [KorSTS](https://github.com/kakaobrain/KorNLUDatasets) (Validation & Testing)

## Training - unsupervised 
```
python main.py \
    --model klue/roberta-base \
    --generator_name klue/roberta-small \
    --multi_gpu True \
    --train True \
    --test False \
    --max_len 64 \
    --batch_size 256 \
    --epochs 1 \
    --eval_steps 125 \
    --lr 0.00005 \
    --masking_ratio 0.15 \
    --lambda_weight 0.005 \
    --warmup_ratio 0.05 \
    --temperature 0.05 \
    --path_to_data Dataset/ \
    --train_data wiki_corpus_examples.txt \
    --valid_data valid_sts.tsv \
    --ckpt best_checkpoint.pt
```
```
bash run_diff.sh
```
> **Note** <br>
> Using roberta as an encoder is beneficial for training because the KoBERT model cannot build a small-sized generator. 

## Evaluation
```
python main.py \
    --model klue/roberta-base \
    --generator klue/roberta-small \
    --train False \
    --test True \
    --max_len 64 \
    --batch_size 256 \
    --path_to_data Dataset/ \
    --test_data test_sts.tsv \
    --path_to_saved_model output/best_checkpoint.pt
```

## Performance - unsupervised

| Model                  | Average | Cosine Pearson | Cosine Spearman | Euclidean Pearson | Euclidean Spearman | Manhattan Pearson | Manhattan Spearman | Dot Pearson | Dot Spearman |
|------------------------|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| KoSRoBERTa-base<sup>†</sup>    | N/A | N/A | 48.96 | N/A | N/A | N/A | N/A | N/A | N/A |
| KoSRoBERTa-large<sup>†</sup>    | N/A | N/A | 51.35 | N/A | N/A | N/A | N/A | N/A | N/A |
| | | | | | | | | | |
| KoSimCSE-BERT    | 74.08 | 74.92 | 73.98 | 74.15 | 74.22 | 74.07 | 74.07 | 74.15 | 73.14 |
| KoSimCSE-RoBERTa    | 75.27 | 75.93 | 75.00 | 75.28 | 75.01 | 75.17 | 74.83 | 75.95 | 75.01 |
| | | | | | | | | | |
| KoDiffCSE-RoBERTa    | 77.17 | 77.73 | 76.96 | 77.21 | 76.89 | 77.11 | 76.81 | 77.74 | 76.97 |

- [Korean-SRoBERTa<sup>†</sup>](https://arxiv.org/abs/2004.03289)

## License
This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />

## References

```bibtex
@inproceedings{chuang2022diffcse,
   title={{DiffCSE}: Difference-based Contrastive Learning for Sentence Embeddings},
   author={Chuang, Yung-Sung and Dangovski, Rumen and Luo, Hongyin and Zhang, Yang and Chang, Shiyu and Soljacic, Marin and Li, Shang-Wen and Yih, Wen-tau and Kim, Yoon and Glass, James},
   booktitle={Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL)},
   year={2022}
}
@misc{park2021klue,
      title={KLUE: Korean Language Understanding Evaluation},
      author={Sungjoon Park and Jihyung Moon and Sungdong Kim and Won Ik Cho and Jiyoon Han and Jangwon Park and Chisung Song and Junseong Kim and Yongsook Song and Taehwan Oh and Joohong Lee and Juhyun Oh and Sungwon Lyu and Younghoon Jeong and Inkwon Lee and Sangwoo Seo and Dongjun Lee and Hyunwoo Kim and Myeonghwa Lee and Seongbo Jang and Seungwon Do and Sunkyoung Kim and Kyungtae Lim and Jongwon Lee and Kyumin Park and Jamin Shin and Seonghyun Kim and Lucy Park and Alice Oh and Jungwoo Ha and Kyunghyun Cho},
      year={2021},
      eprint={2105.09680},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
@article{ham2020kornli,
  title={KorNLI and KorSTS: New Benchmark Datasets for Korean Natural Language Understanding},
  author={Ham, Jiyeon and Choe, Yo Joong and Park, Kyubyong and Choi, Ilji and Soh, Hyungjoon},
  journal={arXiv preprint arXiv:2004.03289},
  year={2020}
}
```
