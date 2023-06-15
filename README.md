# Reproducing results

```
python test.py checkpoint_path='./diffusion.ckpt'
```


# Training Models

## Diffusion model

```
python train.py model=TConv128Diff
```

## Encoder-Decoder model
```
python train.py model=TConv128
```

## Unet model
```
python train.py model=UNet
```

## Multi-modal loss

```
python train.py model=TConv128Spec
```