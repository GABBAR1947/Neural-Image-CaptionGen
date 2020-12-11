#sno 3
from include import *

def calc_max_length(tensor):
    return max(len(t) for t in tensor)


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.resize(tf.image.decode_jpeg(img, channels=3), (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


def CNN():
	image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
	new_input = image_model.input
	hidden_layer = image_model.layers[-1].output
	image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

	return image_features_extract_model


def preprocess_feat_cap(img_name_vector, train_captions, top_k = 5000):
	encode_train = sorted(set(img_name_vector))
	print("caption_length")
	print(len(encode_train))

	image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
	image_dataset = image_dataset.map(load_image).batch(16)
	#print(list(image_dataset.as_numpy_iterator()))
	image_features_extract_model = CNN()
	print(len(image_dataset))
	for img, path in image_dataset:
		#print("hi")
		batch_features = image_features_extract_model(img)
		batch_features = tf.reshape(batch_features, (batch_features.shape[0], -1, batch_features.shape[3]))

		for bf, p in zip(batch_features, path):
			#print("yello")
			path_of_feature = p.numpy().decode("utf-8")
			np.save(path_of_feature, bf.numpy(),allow_pickle=True)

	global tokenizer
	tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k, oov_token="<unk>", filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
	tokenizer.fit_on_texts(train_captions)
	train_seqs = tokenizer.texts_to_sequences(train_captions)

	tokenizer.word_index['<pad>'] = 0
	tokenizer.index_word[0] = '<pad>'
	train_seqs = tokenizer.texts_to_sequences(train_captions)

	cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')
	max_length = calc_max_length(train_seqs)


	img_to_cap_vector = collections.defaultdict(list)
	for img, cap in zip(img_name_vector, cap_vector):
		img_to_cap_vector[img].append(cap)
	img_keys = list(img_to_cap_vector.keys())
	random.shuffle(img_keys)

	slice_index = int(len(img_keys)*0.8)
	img_name_train_keys, img_name_val_keys = img_keys[:slice_index], img_keys[slice_index:]

	img_name_train = []
	cap_train = []
	for imgt in img_name_train_keys:
		capt_len = len(img_to_cap_vector[imgt])
		img_name_train.extend([imgt] * capt_len)
		cap_train.extend(img_to_cap_vector[imgt])

	img_name_val = []
	cap_val = []
	for imgv in img_name_val_keys:
		capv_len = len(img_to_cap_vector[imgv])
		img_name_val.extend([imgv] * capv_len)
		cap_val.extend(img_to_cap_vector[imgv])


	return img_name_train, cap_train, img_name_val, cap_val, tokenizer, max_length
