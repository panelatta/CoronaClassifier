.PHONY: clean preprocess

clean:
	rm source_data/*.tsv
	rm source_data/*.fasta
	rm source_data/*.pkl

preprocess:
	python -u preprocess.py source_data/metadata.tsv.xz source_data/sequences.fasta.xz