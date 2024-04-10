# BiDiE_FG2024
Official code for paper: Multi-View Consistent 3D GAN Inversion via Bidirectional Encoder (The 18th IEEE International Conference on Automatic Face and Gesture Recognition)
The abbreviation of this project is BiDiE (Bidirectional Encoder)

## Environment
The environment used in this paper: [env.txt](https://github.com/WuHaoZhan2000/FG2024_Bidirectional_Encoder/blob/main/environment/env.txt)  
We strongly recommend that you successfully build [EG3D](https://github.com/NVlabs/eg3d) env first, and then build this paper's env based on it.

## Train and test Bidirectional Encoder
Train: [/Bidirectional_Encoder/scripts/train_hybrid.py](https://github.com/WuHaoZhan2000/FG2024_Bidirectional_Encoder/blob/main/scripts/train_hybrid.py)  
Test Single-View Reconstruction: [/Bidirectional_Encoder/scripts/test_psp20_encode_loop.py](https://github.com/WuHaoZhan2000/FG2024_Bidirectional_Encoder/blob/main/scripts/test_psp20_encode_loop.py)  
Test Multi-View Consistency: [/Bidirectional_Encoder/scripts/test_psp20_3D_multi_loop.py](https://github.com/WuHaoZhan2000/FG2024_Bidirectional_Encoder/blob/main/scripts/test_psp20_3D_multi_loop.py)

## Sketch Synthesis Algorithm and Synthesized Sketch Dataset
Coming soon ! (Don’t worry, We promise to upload this part of code and data before July 2024)

## Acknowledgments
This code borrows from: [EG3D](https://github.com/NVlabs/eg3d), [pSp](https://github.com/eladrich/pixel2style2pixel), [e4e](https://github.com/omertov/encoder4editing)

## Citation
If you use this "Bidirectional_Encoder" / "Sketch Dataset" / "Sketch Synthesis Algorithm" for your research, please cite our paper
```
@inproceedings{wu_2024_Bidirectional_Encoder,
  title={Multi-View Consistent 3D GAN Inversion via Bidirectional Encoder},
  author={Haozhan Wu, Hu Han, Shiguang Shan, Xilin Chen},
  booktitle={The 18th IEEE International Conference on Automatic Face and Gesture Recognition, 2024. Proceedings.},
  pages={???--???},
  year={2024},
  organization={IEEE}
}
```
