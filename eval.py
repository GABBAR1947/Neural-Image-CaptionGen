from include import *
from model import *
from CNN import *
from COCO_process import *

def restore_model(checkpoint_path, vocab_size):
    image_features_extract_model = CNN()
    encoder = CNN_Encoder(embedding_dim)
    decoder = RNN_Decoder(embedding_dim, units, vocab_size)
    optimizer = tf.keras.optimizers.Adam()

    ckpt = tf.train.Checkpoint(encoder=encoder,
                               decoder=decoder,
                               optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!')

    return image_features_extract_model, encoder, decoder



def evaluate(image):

    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)
        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    return result


def main():

    global embedding_dim, units
    parser = argparse.ArgumentParser(description='Eval Model')
    parser.add_argument('--model', type=str, help="size of batch", default='train_basic')
    parser.add_argument('--batch-size', type=int, help="size of batch", default=64)
    parser.add_argument('--epochs', type=int, help="num epochs", default=20)
    parser.add_argument('--embed-dim', type=int, help="size of embeddings", default=256)
    parser.add_argument('--buffer-size', type=int, help="size of buffer", default=1000)
    parser.add_argument('--units', type=int, help="units", default=512)
    parser.add_argument('--vocab', type=int, help="vocabulary size", default=5001)
    parser.add_argument('--feat', type=int, help="shape of features", default=2048)
    parser.add_argument('--att', type=int, help="shape of attention features", default=64)


    args = parser.parse_args()
    
    BATCH_SIZE = args.batch_size
    BUFFER_SIZE = args.buffer_size
    embedding_dim = args.embed_dim
    units = args.units
    vocab_size = args.vocab
    features_shape = args.feat
    EPOCHS = args.epochs
    attention_features_shape = args.att

    train_captions, img_name_vector = data_process_COCO('./annotations/captions_train2014.json','./train2014/')
    
    #print("2 done")
    #image_features_extract_model = CNN()
    #print("3 done")
    global tokenizer,max_length
    img_name_train, cap_train, img_name_val, cap_val, tokenizer,max_length = preprocess_feat_cap(img_name_vector, train_captions, 5000)
    #print("4 done")

    global image_features_extract_model, encoder, decoder
    image_features_extract_model, encoder, decoder = restore_model('./checkpoint/'+args.model,5001)
    ORIG = []
    PRED = []
    IMG_ID = []
    for rid in range(len(img_name_val)):
        #if(rid%100==0):
        #    print(rid)
        image = img_name_val[rid]
        real_caption = ' '.join([tokenizer.index_word[i] for i in cap_val[rid] if i not in [0]])
        result = evaluate(image)
        result = ' '.join(result)
        result = '<start> ' + result
        #print(real_caption)
        #print("--")
        #print(result)
        ORIG.append(real_caption)
        PRED.append(result)
        IMG_ID.append(img_name_val[rid])

    METEORscore = 0
    for i in range(len(ORIG)):
        METEORscore = METEORscore + nltk.translate.meteor_score.meteor_score([ORIG[i]], PRED[i])
    print(METEORscore/float(len(ORIG)))

    BLEUscore = 0.
    for i in range(len(ORIG)):
        BLEUscore += sentence_bleu([ORIG[i].strip().split()], PRED[i].strip().split())

    BLEUscore /= float(len(ORIG))
    print(BLEUscore)

    with open('./orig.txt','a') as f:
        for i in range(len(ORIG)):
            f.write(ORIG[i])
            print(i)
            f.write('\n')

    with open('./pred.txt','w') as f:
        for i in PRED:
            f.write(i)
            f.write('\n')

    with open('./img_id.txt','w') as f:
        for i in IMG_ID:
            f.write(i)
            f.write('\n')

if __name__ == "__main__":
    main()

