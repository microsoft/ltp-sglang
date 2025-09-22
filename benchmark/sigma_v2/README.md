# Overview
This benchmark pipeline aims to measure the latency of the different inference models.
The benchmark will work correctly after PR42&43 are merged.

There are three purpose to creating this pipeline rather than using the official commands.

* It can benchmark multiple different input sizes in one input. It overrides the SGLang's scheduling policy and always run in the given batch sizes.
* It only measures the model forward time, excluding any preprocessing and postprocessing of the requests.
* I try to avoid long cmdlines. Official script needs one command to launch the inference server with excessively-long arguments and another command to send a request.

## Applicability
It works these models (GPT-3, Grok, Qwen, DeepSeek, Sigma) on H200 nodes.
## Instructions
### Prepare
1. Create a model folder under `model_conf` folder. Take qwen as example.
2. Put necessary files under the folder `model_conf/qwen` (e.g. `config.json`, `tokenizer.json` and `tokenizer_config.json`).
3. [Necessary] Rename the `config.json` to `config_full.json`.
    * You could change the number of layers to 2 in the JSON file and name `config_2l.json` if you want.
    * This enables benchmark of different architectures (e.g. layer number, expert scale) of the same model.
4. Inside `scripts/bsz_seq.csv` file, change the batch size and sequence length based on your need.
    * The format is {bsz1,bsz2};{seq1,seq2}.
    * The benchmark will loop from all the combinations of {bsz} and {seq}.
### Run the benchmark
#### Single node
```bash
# benchmark full model Qwen with TP+EP
bash scripts/run.sh --model qwen --mconf full --deploy tp8ep8
# benchmark 2 layer Qwen with DP+EP
bash scripts/run.sh --model qwen --mconf 2l --deploy tp8ep8dp8
# benchmark and save results to a csv
bash scripts/run.sh --model qwen --mconf 2l --deploy tp8ep8 --save
# profile both prefill and decode
bash scripts/run.sh --model qwen --mconf 2l --deploy tp8ep8 --profile both
# profile decode
bash scripts/run.sh --model qwen --mconf 2l --deploy tp8ep8 --profile decode
# In case you want to customize a different base.yaml.
bash scripts/run.sh --model qwen --base {customize_base} --mconf 2l --deploy tp8ep8 --profile decode
```
#### Multi node
Create a multi-node YAML file under `conf/deploy` first. Let's say, tp16ep16.
```bash
# On node 1
bash scripts/run.sh --model qwen --mconf full --deploy tp16ep16 --ip {master_ip}
# On node 2
bash scripts/run.sh --model qwen --mconf full --deploy tp16ep16 --ip {master_ip} --rank 1
```
### Results
1. `results/raw_output`: raw logs
2. `results/csv`: benchmark results in CSV if `--save` is on.
3. `results/qwen_profile`: NSight profile and SQLite file.

## File Structure
1. `conf`: includes YAML file which maps to the SGLang's server arguments. The existing YAML can be viewed as a template or a standard of my benchmark. It could produce stable and input-agnostic best results.
    * `base.yaml`: basic arguments.
    * `deploy`: this folder includes different deployment decisions (TP, TP+EP, DP+EP).
    * `backend`: currently it only includes two backend of Attention layer (FlashAttention3 and FlashInfer).
        * SGLang is working on having backend options for MoE layer. It will be more useful in the future versions.
    * `optional`: it contains other SGLang supported arguments. The `default.yaml` is the default decision used by SGLang.
2. `model_conf`: here it includes all the model configuration JSON file.
    * Each model has its own folder. Inside it, it should at least has a `config.json`, `tokenizer.json` and `tokenizer_config.json` to run the inference process.
    * Since most models have repetitive layers, benchmark usually uses fewer layers. So it could have different config files, like `config_full.json`, `config_2l.json` depends on your needs. And later when run bash scripts, you could select which config to use.
3. `python/benchmark`: benchmark Python script
4. `python/tools`: Python script different model counters and analysis of the benchmark results with MFU computed.
5. `scripts/run.sh`: bash scripts to run the benchmark
6. `scripts/bsz_seq.csv`: file of different input size.
