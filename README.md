# KoDiffCSE
Difference-based Contrastive Learning for Korean Sentence Embeddings <br>
   - [DiffCSE-[NAACL 2022]](https://arxiv.org/abs/2204.10298) <br>
   - [[Github]](https://github.com/voidism/DiffCSE) Official implementation of DiffCSE. <br>
<img src=https://user-images.githubusercontent.com/55969260/201829550-9674a3ac-cb9b-4e17-b777-7d96fdf5c633.png>

## Setups
[![Python](https://img.shields.io/badge/python-3.8.5-blue?logo=python&logoColor=FED643)](https://www.python.org/downloads/release/python-385/)
[![Pytorch](https://img.shields.io/badge/pytorch-1.7.1-red?logo=pytorch)](https://pytorch.org/get-started/previous-versions/)

## Datasets
- [wiki-corpus](https://github.com/jeongukjae/korean-wikipedia-corpus) (Unsupervised Training)
- [KorSTS](https://github.com/kakaobrain/KorNLUDatasets) (Validation & Testing)

## Encoder Models
Baseline encoders used for korean sentence embedding - [KLUE-PLMs](https://github.com/KLUE-benchmark/KLUE/blob/main/README.md)

| Model                | Embedding size | Hidden size | # Layers | # Heads |
|----------------------|----------------|-------------|----------|---------|
| KLUE-BERT-base            | 768            | 768         | 12       | 12      |
| KLUE-RoBERTa-base         | 768            | 768         | 12       | 12      |

> **Warning** <br>
> Large pre-trained models need a lot of GPU memory to train

## Training - unsupervised 
```
python main.py \
    --model klue/roberta-base \
    --generator_name klue/roberta-small \
    --multi_gpu True \
    --train True \
    --test False \
    --max_len 64 \
    --batch_size 128 \
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
    --batch_size 128 \
    --path_to_data Dataset/ \
    --test_data test_sts.tsv \
    --path_to_saved_model output/best_checkpoint.pt
```

## Performance-unsupervised

| Model                  | Average | Cosine Pearson | Cosine Spearman | Euclidean Pearson | Euclidean Spearman | Manhattan Pearson | Manhattan Spearman | Dot Pearson | Dot Spearman |
|------------------------|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| KoSRoBERTa-base<sup>†</sup>    | N/A | N/A | 48.96 | N/A | N/A | N/A | N/A | N/A | N/A |
| KoSRoBERTa-large<sup>†</sup>    | N/A | N/A | 51.35 | N/A | N/A | N/A | N/A | N/A | N/A |
| | | | | | | | | | |
| KoSimCSE-BERT    | 71.97 | 72.99 | 71.94 | 71.84 | 72.20 | 71.68 | 72.00 | 72.07 | 71.05 |
| KoSimCSE-RoBERTa    | 73.57 | 74.32 | 73.29 | 73.44 | 73.29 | 73.37 | 73.20 | 74.37 | 73.32 |
| | | | | | | | | | |
| KoDiffCSE-RoBERTa    | 75.65 | 76.58 | 75.42 | 75.66 | 75.49 | 75.63 | 75.49 | 76.51 | 75.30 |

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
