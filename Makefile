.PHONY: clean preprocess

clean:
	rm source_data/*.tsv
	rm source_data/*.fasta
	rm source_data/*.pkl

clean_tensor:
	rm preprocessed_data/train_set/*.pt
	rm preprocessed_data/test_set/*.pt

preprocess:
	python preprocess.py source_data/metadata.tsv.xz source_data/sequences.fasta.xz