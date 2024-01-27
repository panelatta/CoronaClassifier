.PHONY: clean clean_tensor preprocess train output_requirements install_requirements

clean:
	rm source_data/*.tsv
	rm source_data/*.fasta
	rm source_data/*.pkl

clean_tensor:
	rm preprocessed_data/train_set/*.pt
	rm preprocessed_data/test_set/*.pt

preprocess:
	python preprocess.py source_data/metadata.tsv.xz source_data/sequences.fasta.xz

train:
	python -m torch.distributed.launch --nproc_per_node=8 train.py

output_requirements:
	pip freeze > requirements.txt

install_requirements:
	pip install -r requirements.txt