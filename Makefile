work/.folder_structure_sentinel: 
	mkdir -p data/articles/twenty_newsgroups
	mkdir data/word_embeddings

	mkdir -p work/preds
	mkdir work/weights

	touch work/.folder_structure_sentinel

folders: work/.folder_structure_sentinel

##################################
# Wikipedia + Gigaword embedding # 
##################################

data/word_embeddings/glove.6B.50d.txt: 
	curl http://nlp.stanford.edu/data/glove.6B.zip -O
		
	unzip glove.6B.zip

	# The embeddings require the number of words and dimenions in the first line.
	echo '400000 50' | cat - glove.6B.50d.txt > temp && mv temp glove.6B.50d.txt
	echo '400000 100' | cat - glove.6B.100d.txt > temp && mv temp glove.6B.100d.txt
	echo '400000 200' | cat - glove.6B.200d.txt > temp && mv temp glove.6B.200d.txt
	echo '400000 300' | cat - glove.6B.300d.txt > temp && mv temp glove.6B.300d.txt

	mv *.txt data/word_embeddings
	rm glove.6B.zip

########################################
# Twenty news pickled bodies/headlines #
########################################

bodies.pkl: headline_generation/data_setup/twenty_news_gen.py
	python headline_generation/data_setup/twenty_news_gen.py	

headlines.pkl: headline_generation/data_setup/twenty_news_gen.py
	python headline_generation/data_setup/twenty_news_gen.py	

word_embeddings: data/word_embeddings/glove.6B.50d.txt 
data: bodies.pkl headlines.pkl
