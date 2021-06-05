# Wav2Lip-HQ: high quality lip-sync

This is unofficial extension of [Wav2Lip: Accurately Lip-syncing Videos In The Wild](https://github.com/Rudrabha/Wav2Lip) repository. We use image super resolution and face segmentation for improving visual quality of lip-synced videos.

## References
Our work is to a great extent based in the code from the following repositories:

1. Clearly, [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) repository, that is a core model of our algorithm that performs lip-sync.
1. Moreover, [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch.git) repository provides us with a model for face segmentation.
1. We also use extremely useful [BasicSR](https://github.com/xinntao/BasicSR.git) respository for super resolution.
1. Finally, Wav2Lip heavily depends on [face_alignment](https://github.com/1adrianb/face-alignment) repository for detection.

## The algorithm
Our algorithm consists of the following steps:

1. Pretrain ESRGAN on a video with some speech of a target person.
1. Apply Wav2Lip model to the source video and target audio, as it is done in official Wav2Lip repository.
1. Upsample the output of Wav2Lip with ESRGAN.
1. Use BiSeNet to change only relevant pixels in video.

You can learn more about the method in [this article](https://drive.google.com/file/d/1ptTFVNc1v9kzr-V3OK8DJEywziVMKh68/view?usp=sharing) (in russian).

## Results
Our approach is definetly not at all flawless, and some of the frames produced with it contain artifacts or weird mistakes. However, it can be used to perform lip-sync to high quality videos with plausible output.

![comparison](./images/comparison.png)