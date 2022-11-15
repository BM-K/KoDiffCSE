from transformers import (
        AutoTokenizer,
        AutoModelForMaskedLM,
)

model = AutoModelForMaskedLM.from_pretrained('klue/bert-base')
tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')

inputs = tokenizer("대한민국의 수도는 [MASK] 이다."ensors="pt")
print(inputs)
exit()
outputs = model(**inputs)
gen= outputs[0].argmax(-1)
