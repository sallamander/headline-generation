work/.folder_structure_sentinel: 
	mkdir -p data/articles/twenty_newsgroups
	mkdir data/word_embeddings

	mkdir -p work/articles/twenty_newsgroups
	
	touch work/.folder_structure_sentinel

folders: work/.folder_structure_sentinel
