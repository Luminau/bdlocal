from transformers import T5Tokenizer, T5Model, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("cahya/t5-base-indonesian-summarization-cased")
model = T5ForConditionalGeneration.from_pretrained("cahya/t5-base-indonesian-summarization-cased")

#
ARTICLE_TO_SUMMARIZE = "When Russian President Vladimir Putin dials into the virtual BRICS summit hosted by Beijing " \
                       "on Thursday, it will be his first time attending a forum with the heads of major economies " \
                       "since launching an invasion of Ukraine earlier this year.For Putin, this could offer a " \
                       "welcome picture: his face beamed onscreen alongside other leaders whose countries make up " \
                       "this acronymous grouping: China's Xi Jinping, India's Narendra Modi, Brazil's Jair Bolsonaro, " \
                       "and South Africa's Cyril Ramaphosa -- a signal that Russia, though battered by sanctions and " \
                       "remonstrations for the invasion, is not alone.It's a message that may resonate even more " \
                       "clearly as China and Russia, weeks before the invasion, declared their own relationship to " \
                       "have no limits, and as each of the BRICS leaders have avoided condemning Russia outright, " \
                       "even as they hold varying levels of interest in not being seen to endorse its actions -- and " \
                       "run foul of Western friends. "

# generate summary
input_ids = tokenizer.encode(ARTICLE_TO_SUMMARIZE, return_tensors='pt')
summary_ids = model.generate(input_ids,
                             min_length=20,
                             max_length=80,
                             num_beams=10,
                             repetition_penalty=2.5,
                             length_penalty=1.0,
                             early_stopping=True,
                             no_repeat_ngram_size=2,
                             use_cache=True,
                             do_sample=True,
                             temperature=0.8,
                             top_k=50,
                             top_p=0.95)

summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print(summary_text)
