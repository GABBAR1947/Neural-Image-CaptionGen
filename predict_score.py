
def calc_score(img_name_val, cap_val, tokenizer):

  ORIG = []
  PRED = []
  IMG_ID = []
  for rid in range(1):#len(img_name_val)):
    if(rid%100==0):
      print(rid)
    image = img_name_val[rid]
    real_caption = ' '.join([tokenizer.index_word[i] for i in cap_val[rid] if i not in [0]])
    result, attention_plot = evaluate(image)
    result = ' '.join(result)
    result = '<start> ' + result
  #print(real_caption)
  #print("--")
  #print(result)
    ORIG.append(real_caption)
    PRED.append(result)
    IMG_ID.append(img_name_val[rid])

  print("DONE")


  METEORscore = 0
  for i in range(len(ORIG)):
    METEORscore = METEORscore + nltk.translate.meteor_score.meteor_score([ORIG[i]], PRED[i])
  print(METEORscore/float(len(ORIG)))

  BLEUscore = 0.
  for i in range(len(ORIG)):
    BLEUscore += sentence_bleu([ORIG[i].strip().split()], PRED[i].strip().split())

  BLEUscore /= float(len(ORIG))
  print(BLEUscore)

