# Slot-filling
 Backend code for the MSc thesis trajectory at The University of Amsterdam - Data Science programme. The Topic concerns automated slot-filling on Dutch registration forms.
 
**Shortened abstract**
*"In this work, we compile a data set that originates from Dutch law enforcement report forms, which we annotate by means of pattern-based matching. Additional processing steps such as masking and augmentation are applied with the aim of enlarging diversity and generalizability. We adopt, fine-tune and optionally stack three state-of-the-art Dutch pre- trained Transformer models with a CRF layer. Results show that Transformer architectures combined with pre- trained information, outperform prior deep learning architectures and learning from scratch. Out of all models, Multilingual BERT generalizes to our data set foremost, achieving a F1 score of 85.30% and frame-level accuracy of 73.10%. Despite promising results, in- depth analysis reveals that our model encounters difficulties when dealing with several challenges within the field of Natural Language Understanding (NLU)."*

In short, a threefold of learning techniques are implemented: (i) baseline approach - regex & dictionaries obtained from a scraper (ii) BiLSTM and (iii) Transformers. For more information the thesis can be sent upon request.

## Dependencies
Either install each of the listed packages below manual or ```pip install -r requirements.txt```, note that the requirements text file is located in the main folder. 
- python>=3.5
- torch==1.4.0
- transformers==2.7.0
- seqeval==0.0.12
- pytorch-crf==0.7.2

## Data preparation
The data used for this research cannot be disclosed with any parties. To mimick the research, one needs to find data comparable to the Dutch law enforcement registration forms or introduce a topic themselves. In case of the latter, please mention that masking and augmenting the data can be achieved by reusing the ```data_transformation.py``` script located in the *utilities* folder. Type ```python utilities/data_transformation.py --help``` for an overview of the expected arguments. Additionally, one needs to provide a list of the target labels or *slots* in the ```data_transformation.py``` file. To achieve this, open the file in an editor such as Notepad++.  

## Training & Evaluation
It is helpful to know that this repository is optimized for offline usage meaning all pretrained models were downloaded and stored into an offline folder. This only applies for transformer models. Either copy this approach or refrain from offline usage. Please check the *help* argument of *main.py* with ```python models/transformer/main.py --help``` . The default behaviour will try to make use of the internet, however this behaviour is not tested!. 

The required step prior to training models is to provide some configurations. Please provide these at ```config/global_config.py``` and replace values with your own information. Please note that this repository was largely meant to run in an offline environment.

The second step is to define a task in the *data_loader.py* files. In this way the data loader object knows how to handle a particular data set. Please update the **processors** dictionary variable of the two *data_loader* scripts, residing in ```models/rnn/data_loader.py``` and ```models/transformer/data_loader.py```. The key of this dictionary represents the name of your of your data (sub) set (e.g. *set_1* or *set_1_masked*) and the value needs to be set to *Processor* unless you want to implement a custom loading process for your data.  

### Training
It is advised to construct bash files to automate training procedures, see ```jobs/``` folder for an example.

### Evaluation
See ```jobs/``` folder for an example.

## Prediction
Navigate to root of project and execute:
```
python models/[type of model (rnn | transformer)]/predict.py --input_file x.txt --output_file y.txt --model_dir z
```
replace x, y and z with your own values. *z* is the path to the output directory of a singular saved model. Please note that ```config/global_config.py``` holds the default location of all saved models (```output_models```). In addition just refer to the help parameter if not sure what to provide (e.g. ```python models/transformer/predict.py --help```).

## References
- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [pytorch-crf](https://github.com/kmkurn/pytorch-crf)
- [Code inspired by](https://github.com/monologg/JointBERT)