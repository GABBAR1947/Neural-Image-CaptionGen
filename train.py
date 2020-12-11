from include import *
from data_download import *
from COCO_process import *
from CNN import *
from model import *
from eval import *
#from score import *

@tf.function
def train_step(img_tensor, target):
	loss = 0
	hidden = decoder.reset_state(batch_size=target.shape[0])

	dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)

	with tf.GradientTape() as tape:
		features = encoder(img_tensor)

		for i in range(1, target.shape[1]):
			predictions, hidden, _ = decoder(dec_input, features, hidden)
			loss += loss_function(target[:, i], predictions, loss_object)
			dec_input = tf.expand_dims(target[:, i], 1)

	total_loss = (loss / int(target.shape[1]))
	trainable_variables = encoder.trainable_variables + decoder.trainable_variables
	gradients = tape.gradient(loss, trainable_variables)
	optimizer.apply_gradients(zip(gradients, trainable_variables))

	return loss, total_loss

def train_fin(dataset, EPOCHS):
	checkpoint_path = "./checkpoints/train_advanced"
	ckpt = tf.train.Checkpoint(encoder=encoder,
    	                       decoder=decoder,
        	                   optimizer = optimizer)
	ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

	start_epoch = 0
	if ckpt_manager.latest_checkpoint:
		start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
	ckpt.restore(ckpt_manager.latest_checkpoint)
	
	for epoch in range(0, EPOCHS):
		loss_plot = []
		start = time.time()
		total_loss = 0

		for (batch, (img_tensor, target)) in enumerate(dataset):
			batch_loss, t_loss = train_step(img_tensor, target)
			total_loss += t_loss

			if batch % 100 == 0:
				print ('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))
		loss_plot.append(total_loss / num_steps)

		if epoch % 5 == 0:
			ckpt_manager.save()

		print ('Epoch {} Loss {:.6f}'.format(epoch + 1,
                                         total_loss/num_steps))
		print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


def main():

	parser = argparse.ArgumentParser(description='Train Advanced Model')
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

	annotation_file, PATH = download_COCO()
	
	#annotation_file = './annotations/captions_train2014.json'
	#PATH = './train2014/'

	print("COCO Downloaded")
	train_captions, img_name_vector = data_process_COCO(annotation_file, PATH)
	
	#print("2 done")
	#image_features_extract_model = CNN()
	print("COCO processed")
	global tokenizer
	img_name_train, cap_train, img_name_val, cap_val, tokenizer, max_length = preprocess_feat_cap(img_name_vector, train_captions, 5000)
	print("Data split done")
	#return 
	global num_steps
	num_steps = len(img_name_train) // BATCH_SIZE
	dataset = model_util(img_name_train, cap_train)
	print("dataset ready")
	
	global encoder, decoder, optimizer, loss_object
	encoder, decoder, optimizer, loss_object = train_init(embedding_dim, units, vocab_size)
	train_fin(dataset, EPOCHS)

if __name__ == "__main__":
    main()














