#!/bin/bash
echo "Running Dutch BERTje on [DUMMY] Dataset"
python ./models/transformer/main.py --task dummy --model_dir transformer_bertje_dummy --model_type transformer_bertje --train_batch_size 512 --eval_batch_size 512 --do_eval --logging_steps 9000 --save_steps 9000
echo "Running Dutch BERTje on [DUMMY] Dataset + CRF"
python ./models/transformer/main.py --task dummy --model_dir transformer_bertje_dummy_crf --model_type transformer_bertje --train_batch_size 512 --eval_batch_size 512 --use_crf --do_eval --logging_steps 9000 --save_steps 9000
#
echo "Running Dutch BERTje on [DUMMY - MASK] Dataset"
python ./models/transformer/main.py --task dummy_masked --model_dir transformer_bertje_dummy_masked --model_type transformer_bertje --train_batch_size 512 --eval_batch_size 512 --do_eval --logging_steps 9000 --save_steps 9000
echo "Running Dutch BERT on [DUMMY - MASK] Dataset + CRF"
python ./models/transformer/main.py --task dummy_masked --model_dir transformer_bertje_dummy_masked_crf --model_type transformer_bertje --train_batch_size 512 --eval_batch_size 512 --use_crf --do_eval --logging_steps 9000 --save_steps 9000
#
echo "Running Dutch BERTje on [DUMMY - AUG] Dataset"
python ./models/transformer/main.py --task dummy_aug --model_dir transformer_bertje_dummy_aug --model_type transformer_bertje --train_batch_size 512 --eval_batch_size 512 --do_eval --logging_steps 9000 --save_steps 9000
echo "Running Dutch BERT on [DUMMY - AUG] Dataset + CRF"
python ./models/transformer/main.py --task dummy_aug --model_dir transformer_bertje_dummy_aug_crf --model_type transformer_bertje --train_batch_size 512 --eval_batch_size 512 --use_crf --do_eval --logging_steps 9000 --save_steps 9000
#
echo "Running Dutch BERTje on [DUMMY - MASK & AUG] Dataset"
python ./models/transformer/main.py --task dummy_masked_aug --model_dir transformer_bertje_dummy_masked_aug --model_type transformer_bertje --train_batch_size 512 --eval_batch_size 512 --do_eval --logging_steps 9000 --save_steps 9000
echo "Running Dutch BERT on [DUMMY - MASK & AUG] Dataset + CRF"
python ./models/transformer/main.py --task dummy_masked_aug --model_dir transformer_bertje_dummy_masked_aug_crf --model_type transformer_bertje --train_batch_size 512 --eval_batch_size 512 --use_crf --do_eval --logging_steps 9000 --save_steps 9000
#
#
echo "Running Dutch robBERT on [DUMMY] Dataset"
python ./models/transformer/main.py --task dummy --model_dir transformer_robbert_dummy --model_type transformer_robbert --train_batch_size 512 --eval_batch_size 512 --do_eval --logging_steps 9000 --save_steps 9000
echo "Running Dutch robBERT on [DUMMY] Dataset + CRF"
python ./models/transformer/main.py --task dummy --model_dir transformer_robbert_dummy_crf --model_type transformer_robbert --train_batch_size 512 --eval_batch_size 512 --use_crf --do_eval --logging_steps 9000 --save_steps 9000
#
echo "Running Dutch robBERT on [DUMMY - MASK] Dataset"
python ./models/transformer/main.py --task dummy_masked --model_dir transformer_robbert_dummy_masked --model_type transformer_robbert --train_batch_size 512 --eval_batch_size 512 --do_eval --logging_steps 9000 --save_steps 9000
echo "Running Dutch robBERT on [DUMMY - MASK] Dataset + CRF"
python ./models/transformer/main.py --task dummy_masked --model_dir transformer_robbert_dummy_masked_crf --model_type transformer_robbert --train_batch_size 512 --eval_batch_size 512 --use_crf --do_eval --logging_steps 9000 --save_steps 9000
#
echo "Running Dutch robBERT on [DUMMY - AUG] Dataset"
python ./models/transformer/main.py --task dummy_aug --model_dir transformer_robbert_dummy_aug --model_type transformer_robbert --train_batch_size 512 --eval_batch_size 512 --do_eval --logging_steps 9000 --save_steps 9000
echo "Running Dutch robBERT on [DUMMY - AUG] Dataset + CRF"
python ./models/transformer/main.py --task dummy_aug --model_dir transformer_robbert_dummy_aug_crf --model_type transformer_robbert --train_batch_size 512 --eval_batch_size 512 --use_crf --do_eval --logging_steps 9000 --save_steps 9000
#
echo "Running Dutch robBERT on [DUMMY - MASKED & AUG] Dataset"
python ./models/transformer/main.py --task dummy_masked_aug --model_dir transformer_robbert_dummy_masked_aug --model_type transformer_robbert --train_batch_size 512 --eval_batch_size 512 --do_eval --logging_steps 9000 --save_steps 9000
echo "Running Dutch robBERT on [DUMMY - MASKED & AUG] Dataset + CRF"
python ./models/transformer/main.py --task dummy_masked_aug --model_dir transformer_robbert_dummy_masked_aug_crf --model_type transformer_robbert --train_batch_size 512 --eval_batch_size 512 --use_crf --do_eval --logging_steps 9000 --save_steps 9000
#
#
echo "Running Multi BERT on [DUMMY] Dataset"
python ./models/transformer/main.py --task dummy --model_dir transformer_multibert_dummy --model_type transformer_bert-multi --train_batch_size 512 --eval_batch_size 512 --do_eval --logging_steps 9000 --save_steps 9000
echo "Running Multi BERT on [DUMMY] Dataset + CRF"
python ./models/transformer/main.py --task dummy --model_dir transformer_multibert_dummy_crf --model_type transformer_bert-multi --train_batch_size 512 --eval_batch_size 512 --use_crf --do_eval --logging_steps 9000 --save_steps 9000
#
echo "Running Multi BERT on [DUMMY - MASK] Dataset"
python ./models/transformer/main.py --task dummy_masked --model_dir transformer_multibert_dummy_masked --model_type transformer_bert-multi --train_batch_size 512 --eval_batch_size 512 --do_eval --logging_steps 9000 --save_steps 9000
echo "Running Multi BERT on [DUMMY - MASK] Dataset + CRF"
python ./models/transformer/main.py --task dummy_masked --model_dir transformer_multibert_dummy_masked_crf --model_type transformer_bert-multi --train_batch_size 512 --eval_batch_size 512 --use_crf --do_eval --logging_steps 9000 --save_steps 9000
#
echo "Running Multi BERT on [DUMMY - AUG] Dataset"
python ./models/transformer/main.py --task dummy_aug --model_dir transformer_multibert_dummy_aug --model_type transformer_bert-multi --train_batch_size 512 --eval_batch_size 512 --do_eval --logging_steps 9000 --save_steps 9000
echo "Running Multi BERT on [DUMMY - AUG] Dataset + CRF"
python ./models/transformer/main.py --task dummy_aug --model_dir transformer_multibert_dummy_aug_crf --model_type transformer_bert-multi --train_batch_size 512 --eval_batch_size 512 --use_crf --do_eval --logging_steps 9000 --save_steps 9000
#
echo "Running Multi BERT on [DUMMY - MASK & AUG] Dataset"
python ./models/transformer/main.py --task dummy_masked_aug --model_dir transformer_multibert_dummy_masked_aug --model_type transformer_bert-multi --train_batch_size 512 --eval_batch_size 512 --do_eval --logging_steps 9000 --save_steps 9000
echo "Running Multi BERT on [DUMMY - MASK & AUG] Dataset + CRF"
python ./models/transformer/main.py --task dummy_masked_aug --model_dir transformer_multibert_dummy_masked_aug_crf --model_type transformer_bert-multi --train_batch_size 512 --eval_batch_size 512 --use_crf --do_eval --logging_steps 9000 --save_steps 9000