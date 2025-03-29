## ChannelMAE (Masked Autoencoders + Channel Vision Transformer): A PyTorch Implementation

This is a fun repo, it combines Masked AutoEncoders (MAE) with Channel Vision Transformers (ChannelVit) to Channel Masked AutoEncoders (ChannelMAE). It is essentially channels getting rolled out during MAE pretraining. This repos also has support for subsequent fine-tunning.

**Masked AutoEncoders (MAE):**
Are a powerfull pretraining model were we mask out often 75% of the images and make the model predict the rest.

**Channel Vision Transformers (ChannelVit):**
Are useful in non-traditional image applications like cell-painting images or satellite images, where each channels conveys very different information and it doesnt makes sense stacking them up.

**Channel Masked AutoEncoders (ChannelMAE):**
Combining them can be useful for pretraining models for non-traditional image applications like cell-painting images or satellite images.
This is based on `MAE` and `Vit` implementation https://github.com/facebookresearch/mae, modified to add `ChannelVit` and `ChannelMAE`

### Contributions

- [x] Pre-training code for MAE and ChannelMAE
- [x] Fine-tuning code for Vit and ChannelVit (encoders of MAE and ChannelMAE)
- [x] Linprobe code for Vit and ChannelVit (encoders of MAE and ChannelMAE)

We implemented `ChannelMAE` in `models_chamae.py`
We implemented `ChannelVit` in `models_vit.py`


### Pre-training

Sample testing implementation use:
```python main_pretrain.py```

For submitting jobs: The instruction is in [PRETRAIN.md](PRETRAIN.md).

### Fintunning and Linear Probing

Sample testing implementation use:
```python main_finetune.py```
```python main_lineprobe.py```

For submitting jobs: The instruction is in [FINETUNE.md](FINETUNE.md).


### Masked Autoencoders
Masked Autoencoders Are Scalable Vision Learners

<p align="center">
  <img src="https://user-images.githubusercontent.com/11435359/146857310-f258c86c-fde6-48e8-9cee-badd2b21bd2c.png" width="480">
</p>


### Channel Vit

Channel Vision Transformer: An Image Is Worth C x 16 x 16 Words

<figure>
  <p align="center">
  <img src="assets/channelvit.jpg" width=90% align="center" alt="my alt text"/>
  </p>
</figure>
<br/>




### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.
