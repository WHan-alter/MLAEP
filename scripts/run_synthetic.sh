## This shell script perform synthetic process for inuput sequences

conda activate covid_predict
python ./src/synthetic.py 0.8 data/pVNT_seq.csv result/success_seq.txt result/failed_seq.txt