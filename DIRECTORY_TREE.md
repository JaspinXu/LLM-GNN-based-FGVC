```
├── .vscode/
│   └── launch.json
├── checkpoint/
│   └── CUB/
│       ├── logdir/
│       │   ├── events.out.tfevents.1738337189.autodl-container-58da118cfa-4da764b9
│       │   ├── events.out.tfevents.1738339518.autodl-container-58da118cfa-4da764b9
│       │   ├── events.out.tfevents.1738339547.autodl-container-58da118cfa-4da764b9
│       │   ├── events.out.tfevents.1738340096.autodl-container-58da118cfa-4da764b9
│       │   └── events.out.tfevents.1739708694.autodl-container-7dda1180fa-663c94fa
│       ├── epoch1000.pth
│       ├── epoch5.pth
│       ├── epoch10.pth
│       ├── epoch15.pth
│       ├── epoch20.pth
│       ├── epoch25.pth
│       └── epoch30.pth
├── my_method/
│   ├── clash-for-linux/
│   │   ├── bin/
│   │   │   ├── clash-linux-amd64
│   │   │   ├── clash-linux-arm64
│   │   │   └── clash-linux-armv7
│   │   ├── conf/
│   │   │   ├── Country.mmdb
│   │   │   ├── cache.db
│   │   │   └── config.yaml
│   │   ├── dashboard/
│   │   │   └── public/

│   │   ├── logs/
│   │   │   ├── clash.log
│   │   │   └── subconverter.log
│   │   ├── scripts/
│   │   │   ├── clash_profile_conversion.sh
│   │   │   └── get_cpu_arch.sh
│   │   ├── temp/
│   │   │   ├── clash.yaml
│   │   │   ├── clash_config.yaml
│   │   │   ├── config.yaml
│   │   │   ├── proxy.txt
│   │   │   └── templete_config.yaml
│   │   ├── tools/
│   │   │   └── subconverter/

│   │   ├── .env
│   │   ├── README.md
│   │   ├── restart.sh
│   │   ├── shutdown.sh
│   │   ├── start.sh
│   │   └── index.html
│   ├── data/
│   │   ├── cub2011/
│   │   │   ├── CUB_200_2011/

│   │   │   └── des_and_concept/

│   │   └── aircraft/
│   │       ├── fgvc-aircraft-2013b/

│   │       └── des_and_concept/

│   ├── longclip/
│   │   ├── SDXL/
│   │   │   ├── SDXL.md
│   │   │   ├── SDXL_img2img.py
│   │   │   ├── SDXL_pipeline.py
│   │   │   ├── demo_SDXL.png
│   │   │   ├── encode_prompt.py
│   │   │   └── sdxl.py
│   │   ├── checkpoints/
│   │   │   ├── longclip-B.pt
│   │   │   └── model_zoo.md
│   │   ├── eval/
│   │   │   ├── classification/

│   │   │   └── retrieval/

│   │   ├── img/
│   │   │   ├── demo.png
│   │   │   ├── demo_SDXL.png
│   │   │   ├── framework.PNG
│   │   │   ├── generation.png
│   │   │   └── retrieval.png
│   │   ├── model/
│   │   │   ├── __pycache__/

│   │   │   ├── __init__.py
│   │   │   ├── bpe_simple_vocab_16e6.txt.gz
│   │   │   ├── longclip.py
│   │   │   ├── model_longclip.py
│   │   │   └── simple_tokenizer.py
│   │   ├── open_clip_long/
│   │   │   ├── model_configs/

│   │   │   ├── __init__.py
│   │   │   ├── big_vision.py
│   │   │   ├── bpe_simple_vocab_16e6.txt.gz
│   │   │   ├── coca_model.py
│   │   │   ├── constants.py
│   │   │   ├── factory.py
│   │   │   ├── hf_configs.py
│   │   │   ├── hf_model.py
│   │   │   ├── loss.py
│   │   │   ├── model.py
│   │   │   ├── modified_resnet.py
│   │   │   ├── openai.py
│   │   │   ├── pos_embed.py
│   │   │   ├── pretrained.py
│   │   │   ├── push_to_hf_hub.py
│   │   │   ├── timm_model.py
│   │   │   ├── tokenizer.py
│   │   │   ├── transform.py
│   │   │   ├── transformer.py
│   │   │   ├── utils.py
│   │   │   ├── version.py
│   │   │   ├── zero_shot_classifier.py
│   │   │   └── zero_shot_metadata.py
│   │   ├── train/
│   │   │   ├── scheduler.py
│   │   │   ├── sharegpt4v.py
│   │   │   ├── train.md
│   │   │   ├── train.py
│   │   │   ├── train_slurm.sh
│   │   │   └── utils.py
│   │   ├── LICENSE
│   │   ├── README.md
│   │   └── demo.py
│   ├── save_path/
│   │   ├── model_epoch_10.pth
│   │   ├── model_epoch_20.pth
│   │   └── model_epoch_30.pth
│   ├── chaos/
│   │   ├── __pycache__/
│   │   │   ├── cub_dataset.cpython-310.pyc
│   │   │   └── train.cpython-310.pyc
│   │   ├── dataset_cub.py
│   │   ├── final_model.pth
│   │   ├── llm_gpt.py
│   │   ├── net.py
│   │   ├── test.py
│   │   ├── train_enhanced.py
│   │   ├── train.py
│   │   └── cub_dataset.py
│   ├── fgvc_dataset/
│   │   ├── __pycache__/
│   │   │   ├── cub2011.cpython-310.pyc
│   │   │   └── aircraft.cpython-310.pyc
│   │   ├── LICENSE
│   │   ├── README.md
│   │   ├── aircraft.py
│   │   ├── cars.py
│   │   ├── cub2011.py
│   │   ├── dogs.py
│   │   ├── inat2017.py
│   │   ├── nabirds.py
│   │   └── tiny_imagenet.py
│   ├── __pycache__/
│   │   ├── cub_dataset.cpython-310.pyc
│   │   ├── model.cpython-310.pyc
│   │   ├── options.cpython-310.pyc
│   │   ├── train_.cpython-310.pyc
│   │   ├── view_ex.cpython-310.pyc
│   │   ├── enhanced.cpython-310.pyc
│   │   ├── cate.cpython-310.pyc
│   │   ├── infonce.cpython-310.pyc
│   │   ├── resnet.cpython-310.pyc
│   │   └── transforms.cpython-310.pyc
│   ├── gen_des_and_concept/
│   │   ├── q.py
│   │   ├── w_aircraft.py
│   │   ├── w_cub.py
│   │   ├── lala.json
│   │   ├── 6f47aa74a7c157d5bade8042451299d6.jpg
│   │   ├── cropped_result.jpg
│   │   └── clip_concept.py
│   ├── main.py
│   ├── model.py
│   ├── options.py
│   ├── resnet.py
│   ├── resnet50-19c8e357.pth
│   ├── view_ex.py
│   ├── train_.py
│   ├── cate.py
│   ├── infonce.py
│   ├── enhanced.py
│   ├── proxy_utils.sh
│   ├── transforms.py
│   └── gen_structure.py
├── old_method/
│   ├── src/
│   │   ├── clip/
│   │   │   ├── __pycache__/

│   │   │   ├── __init__.py
│   │   │   ├── bpe_simple_vocab_16e6.txt.gz
│   │   │   ├── clip.py
│   │   │   ├── model.py
│   │   │   └── simple_tokenizer.py
│   │   ├── models/
│   │   │   ├── disjoint_encoding.py
│   │   │   ├── global_only.py
│   │   │   ├── gnn_agg_hausdorff.py
│   │   │   ├── gnn_agg_ondisk.py
│   │   │   ├── gnn_agg_online.py
│   │   │   ├── holistic_encoding.py
│   │   │   ├── multiview_hausdorff.py
│   │   │   ├── proxy_graph.py
│   │   │   ├── relational_proxies.py
│   │   │   └── transformer_agg.py
│   │   ├── networks/
│   │   │   ├── ast.py
│   │   │   ├── class_proxy.py
│   │   │   ├── encoder.py
│   │   │   ├── gcn.py
│   │   │   ├── relation_net.py
│   │   │   └── resnet.py
│   │   ├── utils/
│   │   │   ├── auto_load_resume.py
│   │   │   ├── cate.py
│   │   │   ├── constants.py
│   │   │   ├── enhanced.py
│   │   │   ├── factory.py
│   │   │   ├── infonce.py
│   │   │   ├── misc.py
│   │   │   ├── options.py
│   │   │   ├── organize_datasets.py
│   │   │   ├── transforms.py
│   │   │   └── view_extractor.py
│   │   ├── Laysan_Albatross_0003_1033.jpg
│   │   ├── cub_dataset_.py
│   │   ├── data_split.json
│   │   ├── main.py
│   │   └── t.py
│   ├── .gitignore
│   ├── LICENSE
│   ├── README.md
│   └── run_expt.sh
└── .gitignore
```
