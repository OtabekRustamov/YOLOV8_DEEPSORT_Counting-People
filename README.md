People Counter

### Install

```
conda create --name PyTorch python=3.8.12
conda remove --name PyTorch --all
conda clean --all

conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts
pip install ultralytics
```

### Run

```
python main.py
```

### Result

* Check `./runs/` folder

#### Reference

* https://github.com/Matskevichivan/Counting-People
* https://github.com/MuhammadMoinFaisal/YOLOv8-DeepSORT-Object-Tracking
