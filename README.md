Replication of Toxic Chat eval for gpt-oss-safeguard

https://openai.com/index/introducing-gpt-oss-safeguard/
https://huggingface.co/openai/gpt-oss-safeguard-20b
lmsys/toxic-chat


first, start vllm. This works on 2 Nvidia L40s (96GB vram)
```bash
vllm serve openai/gpt-oss-safeguard-20b --tensor-parallel-size 2 --max-num-seqs 128
```
we halve max number of sequences from 256 to prevent out of memory errors


in a separate terminal, run the eval with 
```bash
python main.py
```

view with
```bash
inspect view --port <any port>
```