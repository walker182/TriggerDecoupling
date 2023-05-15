## The Official Implementation of The Paper "Trigger Decoupling-based Backdoor Defense for Pretrained Language Models"


## Attention
Only a part of the code is available at the moment. We will make the full code publicly available once the paper's blind review process is completed.




## Requirement
We implement this code with python3.8. Please use the follwing code to prepare the environment:

```shell
pip install -r requirements.txt
```

## Datasets

Please use the following code to obatin the SST-2, AG New's and HSOL datasets:
```bash
cd datasets
bash download_sentiment_analysis.sh
bash download_text_classification.sh
bash download_toxic.sh
cd ..
```


## Backdoor Attack
Due to the numerous attack methods and datasets, we choose `SST-2` and `stylebkd` as examples to demonstrate the running commands. The same applies to other configurations.

```shell
python -u demo_attack.py --config_path configs/my_config/attack/sst2_stylebkd.json --basemodel sst2_stylebkd
```

## Backdoor Defense of Baselines
We take the BFClass defense method as an example:

```shell
python -u convert_defend.py --config_path configs/my_config/defense/bfclass_stylebkd_sst2.json --basemodel sst2_stylebkd
```
## Backdoor Defense of VAETD

### pretraining
```shell
python -u pretrainVAE.py --config_path configs/my_config/attack/sst2_stylebkd.json --basemodel sst2_stylebkd
```
### trigger decoupling training

```shell
python -u convert_defend.py --config_path configs/my_config/defense/converter_stylebkd_sst2.json --basemodel sst2_stylebkd

```

## Acknowledges

We use the [OpenBackdoor](https://github.com/thunlp/OpenBackdoor/tree/main) as a united framework for a fair comparison. We implement the BFClass, BDDR-DD, BDDR-DR according to their papers ourselves because there are no released codes about these works. In addition, we incorporate POR, NeuBA, GPTD, VAETD into this framework.
